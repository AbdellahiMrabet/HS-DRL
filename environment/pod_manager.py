# environment/pod_manager.py - Pod Deployment Manager

import time
import subprocess
import traceback
import numpy as np
from typing import Tuple, Dict, Optional
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from config import POD_NAMESPACE, POD_IMAGE, K8S_AVAILABLE, POD_TTL_MIN, POD_TTL_MAX


class PodDeploymentManager:
    """Manages actual nginx pod deployment with Kubernetes API latency measurement"""
    
    def __init__(self):
        self.deployed_pods: Dict[str, dict] = {}  # pod_name -> {start_time, api_time, ttl_seconds, image}
        self.api_latencies = []
        self.core_v1 = None
        self.deployed_pods_count = 0
        self._init_k8s_client()
        
        # Base TTL range for generation if no custom TTL is provided
        self.base_ttl_min = POD_TTL_MIN
        self.base_ttl_max = POD_TTL_MAX
        
        # FIX: Removed self.pod_image from init - now generated per-pod
    
    def _init_k8s_client(self):
        if not K8S_AVAILABLE:
            return
        
        try:
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
                print("[✓] Kubernetes client initialized")
            except Exception as e:
                print(f"[!] Could not load Kubernetes config: {e}")
                return
        
        self.core_v1 = client.CoreV1Api()
    
    def _generate_pod_ttl(self) -> float:
        """Generate a random TTL for a new pod (30-45 seconds base, with variation)"""
        base_ttl = np.random.randint(self.base_ttl_min, self.base_ttl_max)
        # Add ±20% variation
        variation = np.random.uniform(0.8, 1.2)
        return float(base_ttl * variation)
    
    def _generate_pod_image(self) -> str:
        """
        Generate a random pod image from POD_IMAGE list.
        Each pod gets an independently selected image (matching TTL behavior).
        """
        return np.random.choice(POD_IMAGE)
    
    def deploy_nginx_pod(
        self, 
        node_name: str, 
        cpu_request: float, 
        mem_request: int, 
        custom_ttl: Optional[float] = None,
        custom_image: Optional[str] = None
    ) -> Tuple[bool, str, float]:
        """
        Deploy a pod to a specific node.
        
        Args:
            node_name: Target node name
            cpu_request: CPU request in cores (e.g., 0.1)
            mem_request: Memory request in Mi (e.g., 128)
            custom_ttl: Optional custom TTL in seconds. If None, generates a random one.
            custom_image: Optional custom image. If None, generates a random one.
            
        Returns:
            Tuple of (success, pod_name, api_response_time_ms)
        """
        if self.core_v1 is None:
            return False, "", 0.0
        
        # FIX: Generate image per-pod (matching TTL behavior)
        if custom_image is not None:
            pod_image = custom_image
        else:
            pod_image = self._generate_pod_image()
        
        pod_name = f"{pod_image.split(':')[1].lower()}-{int(time.time()*1000)}-{np.random.randint(1000, 9999)}"
        cpu_millicores = int(cpu_request * 1000)
        
        # Use custom_ttl if provided, otherwise generate a new one
        if custom_ttl is not None:
            pod_ttl = float(custom_ttl)
        else:
            pod_ttl = self._generate_pod_ttl()
        
        pod_manifest = {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name': pod_name,
                'namespace': POD_NAMESPACE,
                'labels': {'app': 'benchmark', 'scheduler': 'test'}
            },
            'spec': {
                'nodeName': node_name,
                'imagePullSecrets': [{'name': 'my-sec'}],
                'containers': [{
                    'name': 'nginx',
                    'image': pod_image,  # Use the per-pod image
                    'imagePullPolicy': 'IfNotPresent',
                    'resources': {
                        'requests': {'cpu': f'{cpu_millicores}m', 'memory': f'{mem_request}Mi'},
                        'limits': {'cpu': f'{cpu_millicores}m', 'memory': f'{mem_request}Mi'}
                    }
                }],
                'restartPolicy': 'Never'
            }
        }
        
        api_start = time.perf_counter()
        
        try:
            self.core_v1.create_namespaced_pod(namespace=POD_NAMESPACE, body=pod_manifest)
            api_response_time = (time.perf_counter() - api_start) * 1000
            
            self.api_latencies.append(api_response_time)
            
            # Store individual pod TTL and image in the dict
            self.deployed_pods[pod_name] = {
                'start_time': time.time(), 
                'api_time': api_response_time,
                'ttl_seconds': pod_ttl,  # Unique TTL for this pod
                'image': pod_image       # Unique image for this pod
            }
            self.deployed_pods_count += 1
            
            print(f"  📦 Deployed '{pod_name}' to '{node_name}' (Image: {pod_image}, API: {api_response_time:.1f}ms, TTL: {pod_ttl:.1f}s)")
            
            return True, pod_name, api_response_time
            
        except ApiException as e:
            print(f"  ❌ Failed to deploy: {e}")
            return False, "", 0.0
    
    def get_avg_api_latency(self) -> float:
        if self.api_latencies:
            return np.mean(self.api_latencies)
        return 0.0
    
    def cleanup_old_pods(self):
        """Clean up pods that have exceeded their individual TTL"""
        current_time = time.time()
        pods_to_remove = []
        
        # First pass: identify expired pods and capture their age for logging
        expired_pods_info = {}
        for pod_name, info in self.deployed_pods.items():
            pod_age = current_time - info['start_time']
            if pod_age > info['ttl_seconds']:
                pods_to_remove.append(pod_name)
                expired_pods_info[pod_name] = pod_age
        
        # Second pass: delete pods using kubectl (more reliable than API)
        for pod_name in pods_to_remove:
            try:
                # Try kubectl first (more forgiving)
                result = subprocess.run(
                    ['kubectl', 'delete', 'pod', pod_name, '-n', POD_NAMESPACE, '--grace-period=5'],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0 or 'NotFound' in result.stderr:
                    # Successfully deleted or already gone
                    if pod_name in self.deployed_pods:
                        del self.deployed_pods[pod_name]
                    pod_age = expired_pods_info.get(pod_name, 0)
                    print(f"  🗑️ Cleaned up expired pod: {pod_name} (Age: {pod_age:.1f}s)")
                else:
                    print(f"  ⚠️ kubectl delete failed for {pod_name}: {result.stderr.strip()}")
                    
            except subprocess.TimeoutExpired:
                print(f"  ⚠️ Timeout deleting pod {pod_name}")
            except Exception as e:
                print(f"  ⚠️ Unexpected error deleting pod {pod_name}: {e}")
    
    def cleanup_all(self):
        """Clean up all benchmark pods regardless of TTL"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'pod', '-l', 'app=benchmark', '-o', 'name'],
                capture_output=True, 
                text=True, 
                check=True
            )
            lines = result.stdout.strip().split('\n')
            count = 0
            for line in lines:
                if line.strip():
                    pod_name = line.replace('pod/', '')
                    subprocess.run(['kubectl', 'delete', 'pod', pod_name], check=False)
                    count += 1
            if count > 0:
                print(f"  🧹 Cleaned up {count} benchmark pods")
        except Exception as e:
            print(f"  ⚠️ Cleanup warning: {e}")
            pass
