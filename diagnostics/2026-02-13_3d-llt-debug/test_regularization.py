"""
Test different dynamics regularization settings to find which prevents NaN.

Tests:
1. Baseline (v_floor=0.1, eps=1e-6) - Already FAILS
2. Higher velocity floor (v_floor=0.5, eps=1e-6)
3. Much higher velocity floor (v_floor=1.0, eps=1e-6)  
4. Larger epsilon (v_floor=0.1, eps=1e-4)
5. Both (v_floor=0.5, eps=1e-4)
"""

import subprocess
import sys
from pathlib import Path

# Test configurations
tests = [
    {"name": "baseline", "v_floor": 0.1, "eps": 1e-6, "desc": "Baseline (already tested - FAILS)"},
    {"name": "vfloor_0.5", "v_floor": 0.5, "eps": 1e-6, "desc": "Higher velocity floor 0.5 m/s"},
    {"name": "vfloor_1.0", "v_floor": 1.0, "eps": 1e-6, "desc": "Much higher velocity floor 1.0 m/s"},
    {"name": "eps_1e-4", "v_floor": 0.1, "eps": 1e-4, "desc": "Larger symbolic epsilon 1e-4"},
    {"name": "both", "v_floor": 0.5, "eps": 1e-4, "desc": "Both: v_floor=0.5, eps=1e-4"},
]

results = []

for test in tests[1:]:  # Skip baseline, already tested
    print(f"\n{'='*80}")
    print(f"TEST: {test['desc']}")
    print(f"  v_floor = {test['v_floor']} m/s")
    print(f"  symbolic_epsilon = {test['eps']}")
    print(f"{'='*80}\n")
    
    # Modify glider_jinenv.py to set parameters
    jinenv_path = Path("glider_optimization/utils/glider_jinenv.py")
    with open(jinenv_path) as f:
        content = f.read()
    
    # Find and replace the config lines
    original = content
    
    # Add configuration right after __init__
    if f"self._velocity_floor = {test['v_floor']}" not in content:
        # Add after _clamp_coeffs line
        content = content.replace(
            "self._clamp_coeffs = True      # Enable coefficient clamping to prevent extreme forces",
            f"self._clamp_coeffs = True      # Enable coefficient clamping to prevent extreme forces\n        self._velocity_floor = {test['v_floor']}  # TEST: velocity floor\n        self._symbolic_epsilon = {test['eps']}  # TEST: symbolic epsilon"
        )
    
    # Write modified file
    with open(jinenv_path, 'w') as f:
        f.write(content)
    
    # Run test
    log_file = f"diagnostics/2026-02-13_3d-llt-debug/test_{test['name']}.log"
    cmd = f"WANDB_MODE=offline conda run -n general glider-opt --config conf/test.yaml 2>&1 | tee {log_file}"
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    # Check result
    with open(log_file) as f:
        log_content = f.read()
    
    success = "Singular matrix" not in log_content and "NaN detected" not in log_content
    
    result_dict = {
        "name": test["name"],
        "desc": test["desc"],
        "v_floor": test["v_floor"],
        "eps": test["eps"],
        "success": success
    }
    results.append(result_dict)
    
    # Restore original file
    with open(jinenv_path, 'w') as f:
        f.write(original)
    
    print(f"\n{'='*80}")
    print(f"RESULT: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*80}\n")
    
    if success:
        print(f"🎉 FOUND WORKING CONFIGURATION!")
        print(f"  v_floor = {test['v_floor']} m/s")
        print(f"  symbolic_epsilon = {test['eps']}")
        break

# Summary
print(f"\n\n{'='*80}")
print("SUMMARY OF ALL TESTS")
print(f"{'='*80}\n")

for r in results:
    status = "✅ SUCCESS" if r["success"] else "❌ FAILED"
    print(f"{status}: {r['desc']}")
    print(f"         v_floor={r['v_floor']}, eps={r['eps']}\n")

# Find working solution
working = [r for r in results if r["success"]]
if working:
    print(f"\n🎯 RECOMMENDED CONFIGURATION:")
    print(f"   v_floor = {working[0]['v_floor']} m/s")
    print(f"   symbolic_epsilon = {working[0]['eps']}")
else:
    print(f"\n⚠️  NO CONFIGURATION WORKED - Need alternative approach")
    print("   Consider: different solver, constraint reformulation, or energy formulation")
