"""
IPOPT Debugging: Capture exact state/control values when NaN occurs.

This creates a callback that IPOPT will invoke at each iteration,
allowing us to see the exact values that cause the NaN.
"""
import casadi as ca
import numpy as np
import pickle
from pathlib import Path

class IPOPTNaNDebugger:
    """Callback to capture state when NaN is detected."""
    
    def __init__(self, output_dir="diagnostics/2026-02-13_3d-llt-debug"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.iteration = 0
        self.nan_data = []
        
    def __call__(self, f_arg, f_res):
        """Called by IPOPT at each iteration."""
        self.iteration += 1
        
        # Extract decision variables
        x = np.array(f_arg[0]).flatten()
        
        # Extract constraint values if available
        g = np.array(f_res[0]).flatten() if len(f_res) > 0 else None
        
        # Check for NaN in constraints
        if g is not None and np.any(np.isnan(g)):
            print(f"\n{'='*80}")
            print(f"🔴 NaN DETECTED at iteration {self.iteration}")
            print(f"{'='*80}")
            
            nan_indices = np.where(np.isnan(g))[0]
            print(f"NaN in constraints: {nan_indices}")
            
            # Save detailed state
            data = {
                'iteration': self.iteration,
                'x': x,
                'g': g,
                'nan_constraint_indices': nan_indices,
                'x_stats': {
                    'min': float(np.min(x)),
                    'max': float(np.max(x)),
                    'mean': float(np.mean(x)),
                    'nan_count': int(np.sum(np.isnan(x))),
                    'inf_count': int(np.sum(np.isinf(x)))
                }
            }
            
            self.nan_data.append(data)
            
            # Save immediately
            output_file = self.output_dir / f"nan_iteration_{self.iteration}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"✅ Saved NaN data to {output_file}")
            print(f"\nDecision variables at NaN:")
            print(f"  Shape: {x.shape}")
            print(f"  Min: {data['x_stats']['min']:.6f}")
            print(f"  Max: {data['x_stats']['max']:.6f}")
            print(f"  Mean: {data['x_stats']['mean']:.6f}")
            print(f"  Contains NaN: {data['x_stats']['nan_count']}")
            print(f"  Contains Inf: {data['x_stats']['inf_count']}")
            
            # Print specific variables of interest
            # Variable 155 is where gradient NaN occurs
            if len(x) > 155:
                print(f"\n🎯 Variable 155 (vz at stage 25): {x[155]:.6f}")
            
            # Print constraints around 212
            if g is not None and len(g) > 212:
                print(f"\n🎯 Constraint 212: {g[212]}")
                print(f"   Constraints 210-215: {g[210:216]}")
            
            print(f"{'='*80}\n")
        
        return 0


def add_debugging_to_ocp(opti):
    """
    Add NaN debugging callback to an Opti stack.
    
    Usage:
        opti = ca.Opti()
        # ... set up problem ...
        add_debugging_to_ocp(opti)
        solution = opti.solve()
    """
    debugger = IPOPTNaNDebugger()
    
    # This hooks into IPOPT's iteration callback
    # Note: This requires modifying the solver options
    opts = {
        'ipopt.print_level': 5,
        'print_time': True,
        'ipopt.acceptable_tol': 1e-6,
        'ipopt.acceptable_obj_change_tol': 1e-6,
    }
    
    opti.solver('ipopt', opts)
    
    print("✅ NaN debugger attached to IPOPT")
    return debugger


if __name__ == "__main__":
    print("This module provides IPOPT NaN debugging utilities.")
    print("Import and use add_debugging_to_ocp() in your OCP setup.")
