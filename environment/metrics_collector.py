# environment/metrics_collector.py - Kubernetes Metrics Collector

import time
import subprocess
from typing import List, Dict
from config import NUM_NODES
from environment.pod_manager import PodDeploymentManager


class K8sCollector:
    """Real-time Kubernetes metrics collector"""
    
    __slots__ = ['cache', 'cache_time', 'pod_manager', 'disk_io_cache', 'node_accessible']
    
    def __init__(self):
        self.cache = None
        self.cache_time = 0
        self.pod_manager = PodDeploymentManager()
        self.disk_io_cache = {}
        self.node_accessible = {}
    
    def _get_actual_disk_io(self, node_name: str) -> float:
        try:
            pod_result = subprocess.run(
                ["kubectl", "get", "pods", "--all-namespaces", 
                 "--field-selector", f"spec.nodeName={node_name},status.phase=Running",
                 "-o", "jsonpath={.items[0].metadata.name}"],
                capture_output=True, text=True, timeout=5
            )
            pod_name = pod_result.stdout.strip()
            namespace_result = subprocess.run(
                ["kubectl", "get", "pods", "--all-namespaces", 
                 "--field-selector", f"spec.nodeName={node_name},status.phase=Running",
                 "-o", "jsonpath={.items[0].metadata.namespace}"],
                capture_output=True, text=True, timeout=5
            )
            namespace = namespace_result.stdout.strip()
            
            if not pod_name or not namespace:
                return 0.05
            
            result = subprocess.run(
                ["kubectl", "exec", "-n", namespace, pod_name, "--", "cat", "/proc/diskstats"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return 0.05
            
            total_read_sectors = 0
            total_write_sectors = 0
            
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 14:
                    device_name = parts[2]
                    if device_name.startswith('loop') or device_name.startswith('ram'):
                        continue
                    try:
                        read_sectors = int(parts[5])
                        write_sectors = int(parts[9])
                        total_read_sectors += read_sectors
                        total_write_sectors += write_sectors
                    except (ValueError, IndexError):
                        continue
            
            return self._calculate_disk_io_rate(node_name, total_read_sectors, total_write_sectors)
            
        except subprocess.TimeoutExpired:
            return 0.05
        except Exception:
            return 0.05
    
    def _calculate_disk_io_rate(self, node_name: str, read_sectors: int, write_sectors: int) -> float:
        current_time = time.time()
        cache_key = f"{node_name}_disk"
        
        if cache_key in self.disk_io_cache:
            prev_time, prev_read, prev_write = self.disk_io_cache[cache_key]
            time_diff = current_time - prev_time
            
            if time_diff > 0 and time_diff < 5:
                read_bytes = (read_sectors - prev_read) * 512
                write_bytes = (write_sectors - prev_write) * 512
                total_mb = (read_bytes + write_bytes) / (1024 * 1024)
                mb_per_sec = total_mb / time_diff
                disk_io = min(mb_per_sec / 100, 1.0)
                self.disk_io_cache[cache_key] = (current_time, read_sectors, write_sectors)
                return disk_io
        
        self.disk_io_cache[cache_key] = (current_time, read_sectors, write_sectors)
        return 0.05
    
    def get_metrics(self) -> List[Dict]:
        now = time.time()
        
        if self.cache and (now - self.cache_time) < 2:
            return self.cache
        
        try:
            result = subprocess.run(["kubectl", "top", "nodes", "--no-headers"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return self._get_fallback_metrics()
            
            nodes_status = subprocess.run(
                ["kubectl", "get", "nodes", "-o", "jsonpath='{range .items[*]}{.metadata.name}{\" \"}{.status.conditions[?(@.type==\"Ready\")].status}{\"\\n\"}{end}'"],
                capture_output=True, text=True, timeout=5
            )
            
            node_ready = {}
            for line in nodes_status.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.strip().strip("'").split()
                    if len(parts) >= 2:
                        node_ready[parts[0]] = parts[1] == 'True'
            
            metrics = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        node_name = parts[0]
                        cpu_percent = float(parts[2].replace('%', '')) / 100
                        mem_percent = float(parts[4].replace('%', '')) / 100
                        disk_io = self._get_actual_disk_io(node_name)
                        is_ready = node_ready.get(node_name, True)
                        
                        metrics.append({
                            'name': node_name,
                            'cpu': min(cpu_percent, 1.0),
                            'mem': min(mem_percent, 1.0),
                            'disk_io': min(disk_io, 1.0),
                            'pods': 0,
                            'ready': is_ready
                        })
            
            if metrics:
                self.cache = metrics[:NUM_NODES]
                self.cache_time = now
                return self.cache
            return self._get_fallback_metrics()
                
        except Exception:
            return self._get_fallback_metrics()
    
    def _get_fallback_metrics(self) -> List[Dict]:
        return [
            {'name': 'minikube', 'cpu': 0.03, 'mem': 0.26, 'disk_io': 0.08, 'pods': 0, 'ready': True},
            {'name': 'minikube-m02', 'cpu': 0.01, 'mem': 0.09, 'disk_io': 0.03, 'pods': 0, 'ready': True},
            {'name': 'minikube-m03', 'cpu': 0.01, 'mem': 0.22, 'disk_io': 0.05, 'pods': 0, 'ready': True},
        ]
    
    def cleanup(self):
        self.pod_manager.cleanup_all()
        self.disk_io_cache.clear()
        self.cache = None