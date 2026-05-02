# verification/z3_validator.py
# PURE Z3 validation BEFORE and AFTER safety projection - NO empirical fallback

import z3
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CPU_LIMIT, MEM_LIMIT, RESPONSE_TIME_THRESHOLD_FULL, NUM_NODES


@dataclass
class BeforeAfterStats:
    """Statistics comparing before vs after safety projection"""
    before_total: int = 0
    before_safe: int = 0
    before_unsafe: int = 0
    before_z3_proved: int = 0
    before_z3_disproved: int = 0
    
    after_total: int = 0
    after_safe: int = 0
    after_unsafe: int = 0
    after_z3_proved: int = 0
    after_z3_disproved: int = 0
    
    projections_triggered: int = 0
    projections_saved: int = 0
    
    cpu_violations_before: int = 0
    mem_violations_before: int = 0
    rt_violations_before: int = 0
    
    episode_history: List[Dict] = field(default_factory=list)
    
    episode_before_safe: int = 0
    episode_before_unsafe: int = 0
    episode_after_safe: int = 0
    episode_after_unsafe: int = 0
    episode_projections: int = 0
    
    def reset_episode(self):
        self.episode_before_safe = 0
        self.episode_before_unsafe = 0
        self.episode_after_safe = 0
        self.episode_after_unsafe = 0
        self.episode_projections = 0
    
    def finalize_episode(self, episode_num: int):
        before_total = self.episode_before_safe + self.episode_before_unsafe
        after_total = self.episode_after_safe + self.episode_after_unsafe
        
        self.episode_history.append({
            'episode': episode_num,
            'before_safe': self.episode_before_safe,
            'before_unsafe': self.episode_before_unsafe,
            'before_safety_rate': (self.episode_before_safe / max(before_total, 1)) * 100,
            'after_safe': self.episode_after_safe,
            'after_unsafe': self.episode_after_unsafe,
            'after_safety_rate': (self.episode_after_safe / max(after_total, 1)) * 100,
            'projections': self.episode_projections,
            'unsafe_prevented': self.episode_before_unsafe - self.episode_after_unsafe,
            'shield_effectiveness': ((self.episode_before_unsafe - self.episode_after_unsafe) / max(self.episode_before_unsafe, 1)) * 100
        })
        self.reset_episode()
    
    def get_summary(self) -> Dict:
        before_total = self.before_safe + self.before_unsafe
        after_total = self.after_safe + self.after_unsafe
        
        return {
            'total_checks_before': self.before_total,
            'safe_before': self.before_safe,
            'unsafe_before': self.before_unsafe,
            'safety_rate_before': (self.before_safe / max(before_total, 1)) * 100,
            'z3_proved_before': self.before_z3_proved,
            'z3_disproved_before': self.before_z3_disproved,
            'total_checks_after': self.after_total,
            'safe_after': self.after_safe,
            'unsafe_after': self.after_unsafe,
            'safety_rate_after': (self.after_safe / max(after_total, 1)) * 100,
            'z3_proved_after': self.after_z3_proved,
            'z3_disproved_after': self.after_z3_disproved,
            'projections_triggered': self.projections_triggered,
            'projections_saved': self.projections_saved,
            'unsafe_prevented': self.before_unsafe - self.after_unsafe,
            'shield_effectiveness': ((self.before_unsafe - self.after_unsafe) / max(self.before_unsafe, 1)) * 100,
            'cpu_violations_prevented': self.cpu_violations_before,
            'mem_violations_prevented': self.mem_violations_before,
            'rt_violations_prevented': self.rt_violations_before,
        }


class Z3Validator:
    """
    PURE Z3 validation for safety constraints.
    
    Uses SMT solver to formally verify:
    1. CPU utilization after deployment ≤ CPU_LIMIT
    2. Memory utilization after deployment ≤ MEM_LIMIT
    3. Response time < RESPONSE_TIME_THRESHOLD_FULL
    
    Projection formula: projected = current_util * (new_request / current_request)
    """
    
    def __init__(self, num_nodes: int = NUM_NODES, timeout_ms: int = 5000):
        self.num_nodes = num_nodes
        self.timeout_ms = timeout_ms
        self.stats = BeforeAfterStats()
        
        self.z3_available = False
        try:
            self.solver = z3.Solver()
            self.solver.set("timeout", timeout_ms)
            self.z3_available = True
            print("  ✓ Z3 solver initialized for formal verification")
        except Exception as e:
            print(f"  ❌ Z3 not available: {e}")
            raise RuntimeError("Z3 solver is required for validation")
    
    def _create_symbolic_state(self, node: Dict, pod_cpu: float, pod_mem: float) -> Tuple[z3.Solver, Any, Any, Any]:
        """
        Create Z3 symbolic variables for a node and pod.
        Returns (solver, projected_cpu, projected_mem, rt)
        """
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        
        # Symbolic variables
        current_cpu_req = z3.Real('current_cpu_req')
        current_mem_req = z3.Real('current_mem_req')
        current_cpu_util = z3.Real('current_cpu_util')
        current_mem_util = z3.Real('current_mem_util')
        rt = z3.Real('rt')
        
        # Set actual values
        solver.add(current_cpu_req == node.get('cpu', 0.0))
        solver.add(current_mem_req == node.get('mem', 0.0))
        solver.add(current_cpu_util == node.get('cpu_percent', 0.0))
        solver.add(current_mem_util == node.get('mem_percent', 0.0))
        solver.add(rt == node.get('response_time', 100.0))
        
        # Pod demands
        pod_cpu_val = z3.Real('pod_cpu')
        pod_mem_val = z3.Real('pod_mem')
        solver.add(pod_cpu_val == pod_cpu)
        solver.add(pod_mem_val == pod_mem)
        
        # New requests after deployment
        new_cpu_req = current_cpu_req + pod_cpu_val
        new_mem_req = current_mem_req + pod_mem_val
        
        # Project new utilization using the formula: projected = current_util * (new_request / current_request)
        # Handle division by zero: if current_request = 0, projected = 1.0 if new_request > 0 else 0
        cpu_projection = z3.If(current_cpu_req > 0,
                              current_cpu_util * (new_cpu_req / current_cpu_req),
                              z3.If(new_cpu_req > 0, z3.RealVal(1.0), z3.RealVal(0.0)))
        
        mem_projection = z3.If(current_mem_req > 0,
                              current_mem_util * (new_mem_req / current_mem_req),
                              z3.If(new_mem_req > 0, z3.RealVal(1.0), z3.RealVal(0.0)))
        
        # Clamp to 1.0
        projected_cpu = z3.If(cpu_projection > 1, z3.RealVal(1.0), cpu_projection)
        projected_mem = z3.If(mem_projection > 1, z3.RealVal(1.0), mem_projection)
        
        return solver, projected_cpu, projected_mem, rt
    
    def validate_action(self, nodes: List[Dict], action: int, 
                        pod_cpu: float, pod_mem: float) -> Tuple[bool, List[str]]:
        """
        Validate a single action using Z3.
        
        Returns:
            (is_safe, list_of_violations)
        """
        # Delay action is always safe
        if action == self.num_nodes:
            return True, []
        
        # Out of range action
        if action >= len(nodes):
            return False, ["action_out_of_range"]
        
        target_node = nodes[action]
        
        # Node not ready
        if not target_node.get('ready', True):
            return False, ["node_not_ready"]
        
        # Create symbolic state
        solver, projected_cpu, projected_mem, rt = self._create_symbolic_state(
            target_node, pod_cpu, pod_mem
        )
        
        # Define safety constraints
        cpu_ok = projected_cpu <= CPU_LIMIT
        mem_ok = projected_mem <= MEM_LIMIT
        rt_ok = rt < RESPONSE_TIME_THRESHOLD_FULL
        
        all_ok = z3.And(cpu_ok, mem_ok, rt_ok)
        
        # Try to find a counterexample (violation)
        solver.add(z3.Not(all_ok))
        result = solver.check()
        
        if result == z3.unsat:
            # No counterexample found - action is safe
            return True, []
        elif result == z3.sat:
            # Counterexample found - action is unsafe
            model = solver.model()
            violations = []
            
            try:
                proj_cpu_val = model.eval(projected_cpu)
                proj_mem_val = model.eval(projected_mem)
                rt_val = model.eval(rt)
                
                if float(proj_cpu_val.as_fraction()) > CPU_LIMIT:
                    violations.append(f"cpu:{float(proj_cpu_val.as_fraction()):.3f}>{CPU_LIMIT}")
                if float(proj_mem_val.as_fraction()) > MEM_LIMIT:
                    violations.append(f"mem:{float(proj_mem_val.as_fraction()):.3f}>{MEM_LIMIT}")
                if float(rt_val.as_fraction()) >= RESPONSE_TIME_THRESHOLD_FULL:
                    violations.append(f"rt:{float(rt_val.as_fraction()):.0f}>={RESPONSE_TIME_THRESHOLD_FULL}")
            except:
                violations.append("unsafe_by_z3")
            
            return False, violations
        else:
            # Solver error (timeout, unknown)
            raise RuntimeError(f"Z3 solver returned {result} - cannot determine safety")
    
    def validate_before_projection(self, env, raw_action: int) -> Tuple[bool, str]:
        """
        Validate the RAW agent action BEFORE safety projection using Z3.
        Shows what WOULD happen without the shield.
        """
        if not hasattr(env, 'current_pod') or env.current_pod is None:
            return True, "no_pod"
        
        pod_cpu = env.current_pod['cpu']
        pod_mem = env.current_pod['mem']
        
        self.stats.before_total += 1
        
        is_safe, violations = self.validate_action(env.nodes, raw_action, pod_cpu, pod_mem)
        
        if is_safe:
            self.stats.before_safe += 1
            self.stats.before_z3_proved += 1
            self.stats.episode_before_safe += 1
            return True, "z3_proved_safe"
        else:
            self.stats.before_unsafe += 1
            self.stats.before_z3_disproved += 1
            self.stats.episode_before_unsafe += 1
            
            # Track violation types
            for v in violations:
                if v.startswith('cpu'):
                    self.stats.cpu_violations_before += 1
                elif v.startswith('mem'):
                    self.stats.mem_violations_before += 1
                elif v.startswith('rt'):
                    self.stats.rt_violations_before += 1
            
            return False, "; ".join(violations)
    
    def validate_after_projection(self, env, safe_action: int) -> Tuple[bool, str]:
        """
        Validate the PROJECTED action AFTER safety projection using Z3.
        Shows what ACTUALLY executes.
        """
        if not hasattr(env, 'current_pod') or env.current_pod is None:
            return True, "no_pod"
        
        pod_cpu = env.current_pod['cpu']
        pod_mem = env.current_pod['mem']
        
        self.stats.after_total += 1
        
        if safe_action == env.num_nodes:
            self.stats.after_safe += 1
            self.stats.after_z3_proved += 1
            self.stats.episode_after_safe += 1
            return True, "delay_action"
        
        is_safe, violations = self.validate_action(env.nodes, safe_action, pod_cpu, pod_mem)
        
        if is_safe:
            self.stats.after_safe += 1
            self.stats.after_z3_proved += 1
            self.stats.episode_after_safe += 1
            return True, "z3_proved_safe_after_projection"
        else:
            self.stats.after_unsafe += 1
            self.stats.after_z3_disproved += 1
            self.stats.episode_after_unsafe += 1
            print(f"  🔴🔴 Z3 CRITICAL: Projected action {safe_action} is UNSAFE! {violations}")
            return False, f"Z3_UNSAFE_AFTER_PROJECTION: {violations}"
    
    def record_projection(self, raw_action: int, safe_action: int):
        """Record that a projection occurred"""
        self.stats.projections_triggered += 1
        self.stats.episode_projections += 1
        self.stats.projections_saved += 1
    
    def get_episode_stats(self) -> Dict:
        """Get statistics for the current episode"""
        before_total = self.stats.episode_before_safe + self.stats.episode_before_unsafe
        after_total = self.stats.episode_after_safe + self.stats.episode_after_unsafe
        
        return {
            'z3_before_safe': self.stats.episode_before_safe,
            'z3_before_unsafe': self.stats.episode_before_unsafe,
            'z3_before_safety_rate': (self.stats.episode_before_safe / max(before_total, 1)) * 100,
            'z3_after_safe': self.stats.episode_after_safe,
            'z3_after_unsafe': self.stats.episode_after_unsafe,
            'z3_after_safety_rate': (100 - self.stats.episode_after_unsafe) * 100,
            'z3_projections': self.stats.episode_projections,
            'z3_unsafe_prevented': self.stats.episode_before_unsafe - self.stats.episode_after_unsafe,
            'z3_shield_effectiveness': ((self.stats.episode_before_unsafe - self.stats.episode_after_unsafe) / max(self.stats.episode_before_unsafe, 1)) * 100
        }
    
    def reset_episode(self, episode_num: int):
        """Call at end of episode"""
        self.stats.finalize_episode(episode_num)
    
    def get_summary(self) -> Dict:
        """Get overall summary"""
        return self.stats.get_summary()
    
    def get_comparison_report(self) -> str:
        """Generate a report comparing before vs after"""
        summary = self.get_summary()
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Z3 FORMAL VERIFICATION REPORT                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Projection Formula: projected = current_util * (new_request / current_request)              ║
║                                                                              ║
║  BEFORE SAFETY PROJECTION (Raw Agent Actions):                               ║
║    ├─ Total Actions Checked: {summary['total_checks_before']:<10}                     ║
║    ├─ Z3 Proved Safe: {summary['z3_proved_before']:<10}                                 ║
║    ├─ Z3 Disproved (Unsafe): {summary['z3_disproved_before']:<10}                           ║
║    ├─ Safe Rate: {summary['safety_rate_before']:.1f}%                                         ║
║    └─ UNSAFE Rate: {100 - summary['safety_rate_before']:.1f}%                                      ║
║                                                                              ║
║  AFTER SAFETY PROJECTION (What Actually Executes):                           ║
║    ├─ Total Actions Checked: {summary['total_checks_after']:<10}                     ║
║    ├─ Z3 Proved Safe: {summary['z3_proved_after']:<10}                                  ║
║    ├─ Z3 Disproved (Unsafe): {summary['z3_disproved_after']:<10}                            ║
║    ├─ Safe Rate: {summary['safety_rate_after']:.1f}%                                         ║
║    └─ UNSAFE Rate: {100 - summary['safety_rate_after']:.1f}%                                      ║
║                                                                              ║
║  SHIELD STATISTICS:                                                          ║
║    ├─ Projections Triggered: {summary['projections_triggered']:<10}                     ║
║    ├─ Unsafe Actions Prevented: {summary['unsafe_prevented']:<10}                      ║
║    └─ Shield Effectiveness: {summary['shield_effectiveness']:.1f}%                                 ║
║                                                                              ║
║  VIOLATIONS PREVENTED BY TYPE:                                               ║
║    ├─ CPU Limit Violations: {summary['cpu_violations_prevented']:<10}                      ║
║    ├─ Memory Limit Violations: {summary['mem_violations_prevented']:<10}                     ║
║    └─ Response Time Violations: {summary['rt_violations_prevented']:<10}                     ║
║                                                                              ║
║  Z3 CONCLUSION:                                                              ║
║    {'✓ ALL PROJECTED ACTIONS ARE Z3-PROVED SAFE' if summary['z3_disproved_after'] == 0 else f'✗ {summary["z3_disproved_after"]} UNSAFE ACTIONS FOUND'}                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return report


_global_validator: Optional[Z3Validator] = None


def get_validator() -> Z3Validator:
    """Get or create global validator instance"""
    global _global_validator
    if _global_validator is None:
        _global_validator = Z3Validator(enabled=True)
    return _global_validator


def reset_validator():
    """Reset the global validator"""
    global _global_validator
    _global_validator = Z3Validator()