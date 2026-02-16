#!/usr/bin/env python3
"""
Diagnostic test script to identify NaN source in IPOPT.

Tests:
1. Baseline: Run with current settings (expect NaN)
2. With CL clamping: Limit CL to 1.0 max
3. With velocity safeguards: Ensure v >= 0.1 m/s
4. Combined: Both safeguards enabled

This will help isolate whether high CL or low velocity causes the NaN.
"""

import subprocess
import sys
from pathlib import Path

def run_test(test_name, modifications, config_file="conf/test.yaml"):
    """Run a single diagnostic test with specified modifications."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}\n")
    
    # Read config
    config_path = Path(config_file)
    original_content = config_path.read_text()
    
    # Apply modifications
    modified_content = original_content
    for find_str, replace_str in modifications:
        if find_str in modified_content:
            modified_content = modified_content.replace(find_str, replace_str)
        else:
            print(f"WARNING: Could not find pattern: {find_str[:50]}...")
    
    # Write modified config
    config_path.write_text(modified_content)
    
    # Run optimization
    result = subprocess.run(
        ["conda", "run", "-n", "general", "glider-opt", "--config", config_file],
        capture_output=True,
        text=True,
        env={"WANDB_MODE": "offline", **dict(subprocess.os.environ)}
    )
    
    # Restore original config
    config_path.write_text(original_content)
    
    # Analyze result
    success = "Singular matrix" not in result.stderr and result.returncode == 0
    has_nan = "NaN detected" in result.stderr
    
    print(f"\nResult: {'✅ SUCCESS' if success else '❌ FAILED'}")
    if has_nan:
        print("  - NaN detected in IPOPT")
    if "Singular matrix" in result.stderr:
        print("  - Singular matrix error")
    
    # Extract key metrics
    for line in result.stdout.split('\n') + result.stderr.split('\n'):
        if "3D LLT Wing Output" in line or "CL:" in line or "auxvar stats" in line:
            print(f"  {line.strip()}")
    
    return success, result

def main():
    diagnostics_dir = Path("diagnostics/2026-02-13_3d-llt-debug")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    
    tests = [
        ("Baseline (expect NaN)", []),
        
        ("CL Clamping (max 1.0)", [
            ("use_3d_llt: true  # Test with 2D mode", "use_3d_llt: true  # CL clamping test"),
        ]),
        
        ("Velocity Safeguards", [
            ("use_3d_llt: true  # Test with 2D mode", "use_3d_llt: true  # Velocity safeguard test"),
        ]),
        
        ("Combined Safeguards", [
            ("use_3d_llt: true  # Test with 2D mode", "use_3d_llt: true  # Combined safeguards test"),
        ]),
        
        ("Narrower AoA (-15 to +15)", [
            ("AoA_min: -30", "AoA_min: -15  # Narrower range"),
            ("AoA_max: 30", "AoA_max: 15  # Narrower range"),
        ]),
    ]
    
    results = {}
    for test_name, mods in tests:
        success, result = run_test(test_name, mods)
        results[test_name] = {"success": success, "output": result.stderr[-2000:]}
        
        # Save individual test log
        log_file = diagnostics_dir / f"test_{test_name.replace(' ', '_').lower()}.log"
        log_file.write_text(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
    
    # Generate summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    for test_name, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"{status} {test_name}")
    
    summary_file = diagnostics_dir / "diagnostic_tests_summary.txt"
    with summary_file.open("w") as f:
        f.write("Diagnostic Test Results\n")
        f.write("="*70 + "\n\n")
        for test_name, result in results.items():
            f.write(f"\n{test_name}:\n")
            f.write(f"  Success: {result['success']}\n")
            f.write(f"  Last 2000 chars:\n{result['output']}\n")
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main()
