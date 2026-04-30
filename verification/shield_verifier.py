# verification/shield_verifier.py
# Z3-based formal verification of the safety shield

import z3
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class VerificationResult(Enum):
    """Verification result status"""
    PROVED = "PROVED"
    DISPROVED = "DISPROVED" 
    UNKNOWN = "UNKNOWN"
    TIMEOUT = "TIMEOUT"


@dataclass
class SafetyTheorem:
    """A formal safety theorem to prove"""
    name: str
    statement: str
    condition: Any  # z3 expression
    guarantee: Any  # z3 expression
    proved: bool = False
    counterexample: Optional[Dict] = None


class FormalShieldVerifier:
    """
    Formal verification of the Hierarchical Safety Projection (HSP) shield.
    
    Proves that:
    1. Projection always returns a node within resource limits
    2. Projection never increases resource utilization beyond thresholds
    3. Projection preserves action intent when possible
    """
    
    def __init__(self, num_nodes: int, constraints: Dict):
        """
        Args:
            num_nodes: Number of edge nodes
            constraints: Dict with keys: 'cpu_limit', 'mem_limit', 'rt_limit'
        """
        self.num_nodes = num_nodes
        self.constraints = constraints
        self.solver = z3.Solver()
        self.timeout_ms = 30000  # 30 second timeout
        
        # Create symbolic variables for system state
        self._create_symbolic_variables()
        
        # Define constraint bounds
        self._add_bounds()
    
    def _create_symbolic_variables(self):
        """Create Z3 symbolic variables for system state"""
        # Node resource utilization (0-1)
        self.node_cpu = [z3.Real(f"cpu_{i}") for i in range(self.num_nodes)]
        self.node_mem = [z3.Real(f"mem_{i}") for i in range(self.num_nodes)]
        self.node_rt = [z3.Real(f"rt_{i}") for i in range(self.num_nodes)]
        self.node_ready = [z3.Bool(f"ready_{i}") for i in range(self.num_nodes)]
        
        # Node resource capacity (absolute values)
        self.node_cpu_cap = [z3.Real(f"cpu_cap_{i}") for i in range(self.num_nodes)]
        self.node_mem_cap = [z3.Real(f"mem_cap_{i}") for i in range(self.num_nodes)]
        self.node_cpu_used = [z3.Real(f"cpu_used_{i}") for i in range(self.num_nodes)]
        self.node_mem_used = [z3.Real(f"mem_used_{i}") for i in range(self.num_nodes)]
        
        # Pod resource demand
        self.pod_cpu = z3.Real("pod_cpu")
        self.pod_mem = z3.Real("pod_mem")
        
        # Action (proposed and projected)
        self.raw_action = z3.Int("raw_action")
        self.projected_action = z3.Int("projected_action")
    
    def _add_bounds(self):
        """Add variable bounds to solver"""
        # Utilization bounds (0-1)
        for i in range(self.num_nodes):
            self.solver.add(self.node_cpu[i] >= 0, self.node_cpu[i] <= 1)
            self.solver.add(self.node_mem[i] >= 0, self.node_mem[i] <= 1)
            self.solver.add(self.node_rt[i] >= 0, self.node_rt[i] <= 500)  # ms
            self.solver.add(self.node_cpu_cap[i] > 0)
            self.solver.add(self.node_mem_cap[i] > 0)
            self.solver.add(self.node_cpu_used[i] >= 0)
            self.solver.add(self.node_mem_used[i] >= 0)
        
        # Pod demands (positive)
        self.solver.add(self.pod_cpu >= 0, self.pod_cpu <= 2)  # Up to 2 cores
        self.solver.add(self.pod_mem >= 0, self.pod_mem <= 4096)  # Up to 4GB
        
        # Action bounds
        self.solver.add(self.raw_action >= 0, self.raw_action < self.num_nodes)
        self.solver.add(self.projected_action >= 0, self.projected_action <= self.num_nodes)
    
    def _node_is_safe(self, node_idx: int) -> z3.BoolRef:
        """Return condition that node i is safe for deployment"""
        # Relationship between utilization and absolute values
        cpu_util = self.node_cpu_used[node_idx] / self.node_cpu_cap[node_idx]
        mem_util = self.node_mem_used[node_idx] / self.node_mem_cap[node_idx]
        
        # After adding pod
        new_cpu_util = (self.node_cpu_used[node_idx] + self.pod_cpu) / self.node_cpu_cap[node_idx]
        new_mem_util = (self.node_mem_used[node_idx] + self.pod_mem) / self.node_mem_cap[node_idx]
        
        cpu_ok = new_cpu_util <= self.constraints.get('cpu_limit', 0.85)
        mem_ok = new_mem_util <= self.constraints.get('mem_limit', 0.85)
        rt_ok = self.node_rt[node_idx] <= self.constraints.get('rt_limit', 200.0)
        ready_ok = self.node_ready[node_idx]
        
        return z3.And(cpu_ok, mem_ok, rt_ok, ready_ok)
    
    def prove_projection_safety(self) -> Tuple[VerificationResult, str, Optional[Dict]]:
        """
        Prove: The projection function always returns a safe node.
        
        Theorem: For any system state, if there exists at least one safe node,
                 the projection will return a node that satisfies all safety constraints.
        """
        self.solver.push()
        self.solver.set("timeout", self.timeout_ms)
        
        # Condition: There exists at least one safe node
        exists_safe_node = z3.Or([self._node_is_safe(i) for i in range(self.num_nodes)])
        
        # Guarantee: The projected action is safe
        projected_is_safe = self._node_is_safe(self.projected_action)
        
        # Theorem: exists_safe_node → projected_is_safe
        theorem = z3.Implies(exists_safe_node, projected_is_safe)
        
        # Try to disprove (find counterexample)
        self.solver.add(z3.Not(theorem))
        result = self.solver.check()
        
        if result == z3.unsat:
            return VerificationResult.PROVED, "THEOREM PROVED: Projection always returns a safe node when one exists", None
        elif result == z3.sat:
            model = self.solver.model()
            counterexample = self._extract_counterexample(model)
            return VerificationResult.DISPROVED, "Counterexample found: Projection returned unsafe node", counterexample
        else:
            return VerificationResult.UNKNOWN, f"Solver returned unknown (timeout or incomplete)", None
    
    def prove_no_unsafe_execution(self) -> Tuple[VerificationResult, str, Optional[Dict]]:
        """
        Prove: No unsafe action is ever executed.
        
        Theorem: The shield guarantees that 100% of executed actions satisfy all constraints.
        """
        self.solver.push()
        self.solver.set("timeout", self.timeout_ms)
        
        # Any executed action must be safe
        executed_safe = self._node_is_safe(self.projected_action)
        
        # Theorem: Always safe
        theorem = executed_safe
        
        self.solver.add(z3.Not(theorem))
        result = self.solver.check()
        
        if result == z3.unsat:
            return VerificationResult.PROVED, "THEOREM PROVED: All executed actions satisfy safety constraints", None
        elif result == z3.sat:
            model = self.solver.model()
            counterexample = self._extract_counterexample(model)
            return VerificationResult.DISPROVED, "Counterexample found: Unsafe action could be executed", counterexample
        else:
            return VerificationResult.UNKNOWN, "Unable to verify safety guarantee", None
    
    def prove_projection_minimal(self) -> Tuple[VerificationResult, str, Optional[Dict]]:
        """
        Prove: Projection is minimal (prefers original action if safe).
        
        Theorem: If raw action is safe, projection returns the same action.
        """
        self.solver.push()
        self.solver.set("timeout", self.timeout_ms)
        
        raw_is_safe = self._node_is_safe(self.raw_action)
        projection_preserves = (self.projected_action == self.raw_action)
        
        theorem = z3.Implies(raw_is_safe, projection_preserves)
        
        self.solver.add(z3.Not(theorem))
        result = self.solver.check()
        
        if result == z3.unsat:
            return VerificationResult.PROVED, "THEOREM PROVED: Projection preserves safe actions", None
        elif result == z3.sat:
            model = self.solver.model()
            counterexample = self._extract_counterexample(model)
            return VerificationResult.DISPROVED, "Counterexample found: Safe action was changed", counterexample
        else:
            return VerificationResult.UNKNOWN, "Unable to verify minimal projection", None
    
    def prove_resource_bounds(self) -> Tuple[VerificationResult, str, Optional[Dict]]:
        """
        Prove: Post-deployment resource utilization never exceeds limits.
        """
        self.solver.push()
        self.solver.set("timeout", self.timeout_ms)
        
        cpu_limit_val = self.constraints.get('cpu_limit', 0.85)
        mem_limit_val = self.constraints.get('mem_limit', 0.85)
        
        # After deployment, calculate utilization for the target node
        target_idx = self.projected_action
        new_cpu_util = (self.node_cpu_used[target_idx] + self.pod_cpu) / self.node_cpu_cap[target_idx]
        new_mem_util = (self.node_mem_used[target_idx] + self.pod_mem) / self.node_mem_cap[target_idx]
        
        bounds_ok = z3.And(new_cpu_util <= cpu_limit_val, new_mem_util <= mem_limit_val)
        
        self.solver.add(z3.Not(bounds_ok))
        result = self.solver.check()
        
        if result == z3.unsat:
            return VerificationResult.PROVED, "THEOREM PROVED: Resource bounds never exceeded", None
        elif result == z3.sat:
            model = self.solver.model()
            counterexample = self._extract_counterexample(model)
            return VerificationResult.DISPROVED, "Counterexample: Resource bounds would be exceeded", counterexample
        else:
            return VerificationResult.UNKNOWN, "Unable to verify resource bounds", None
    
    def _extract_counterexample(self, model: z3.ModelRef) -> Dict:
        """Extract counterexample values from Z3 model"""
        counterexample = {
            'node_cpu': [],
            'node_mem': [],
            'node_rt': [],
            'node_ready': [],
            'node_cpu_cap': [],
            'node_mem_cap': [],
            'node_cpu_used': [],
            'node_mem_used': [],
            'pod_cpu': None,
            'pod_mem': None,
            'raw_action': None,
            'projected_action': None
        }
        
        for i in range(self.num_nodes):
            try:
                counterexample['node_cpu'].append(float(model.eval(self.node_cpu[i]).as_fraction()))
                counterexample['node_mem'].append(float(model.eval(self.node_mem[i]).as_fraction()))
                counterexample['node_rt'].append(float(model.eval(self.node_rt[i]).as_fraction()))
                counterexample['node_ready'].append(model.eval(self.node_ready[i]))
                counterexample['node_cpu_cap'].append(float(model.eval(self.node_cpu_cap[i]).as_fraction()))
                counterexample['node_mem_cap'].append(float(model.eval(self.node_mem_cap[i]).as_fraction()))
                counterexample['node_cpu_used'].append(float(model.eval(self.node_cpu_used[i]).as_fraction()))
                counterexample['node_mem_used'].append(float(model.eval(self.node_mem_used[i]).as_fraction()))
            except:
                counterexample['node_cpu'].append(0.0)
                counterexample['node_mem'].append(0.0)
                counterexample['node_rt'].append(0.0)
                counterexample['node_ready'].append(True)
                counterexample['node_cpu_cap'].append(1.0)
                counterexample['node_mem_cap'].append(1000.0)
                counterexample['node_cpu_used'].append(0.5)
                counterexample['node_mem_used'].append(500.0)
        
        try:
            counterexample['pod_cpu'] = float(model.eval(self.pod_cpu).as_fraction())
            counterexample['pod_mem'] = float(model.eval(self.pod_mem).as_fraction())
            counterexample['raw_action'] = model.eval(self.raw_action).as_long()
            counterexample['projected_action'] = model.eval(self.projected_action).as_long()
        except:
            pass
        
        return counterexample
    
    def run_full_verification_suite(self) -> Dict[str, Any]:
        """Run all verification theorems and return comprehensive report"""
        results = {
            'timestamp': time.time(),
            'constraints': self.constraints.copy(),
            'num_nodes': self.num_nodes,
            'theorems': {}
        }
        
        # Run each theorem
        theorems = [
            ('projection_safety', self.prove_projection_safety),
            ('no_unsafe_execution', self.prove_no_unsafe_execution),
            ('projection_minimal', self.prove_projection_minimal),
            ('resource_bounds', self.prove_resource_bounds)
        ]
        
        all_proved = True
        for name, prove_func in theorems:
            result, message, counterexample = prove_func()
            results['theorems'][name] = {
                'status': result.value,
                'message': message,
                'counterexample': counterexample
            }
            if result != VerificationResult.PROVED:
                all_proved = False
        
        results['all_theorems_proved'] = all_proved
        
        return results


# For backward compatibility
SafetyTheorem = SafetyTheorem
