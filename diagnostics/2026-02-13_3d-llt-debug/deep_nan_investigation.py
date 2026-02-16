"""
Deep NaN Investigation: Run 2D and 3D side-by-side with maximum logging.

This script will:
1. Run both 2D and 3D with IPOPT print_level=12 (maximum verbosity)
2. Extract trajectory data at each iteration
3. Compare where 2D and 3D diverge
4. Identify the exact values causing NaN in 3D
"""

import subprocess
import sys
from pathlib import Path
import re

def run_with_full_logging(config_file, output_log, mode_name):
    """Run glider-opt with maximum IPOPT verbosity."""
    
    print(f"\n{'='*80}")
    print(f"Running {mode_name} with full IPOPT logging")
    print(f"Config: {config_file}")
    print(f"Output: {output_log}")
    print(f"{'='*80}\n")
    
    cmd = [
        "conda", "run", "-n", "general",
        "glider-opt", "--config", config_file
    ]
    
    env = {"WANDB_MODE": "offline"}
    
    with open(output_log, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**subprocess.os.environ, **env}
        )
        
        for line in process.stdout:
            f.write(line)
            # Also print to console for immediate feedback
            if any(keyword in line for keyword in ['NaN', 'Inf', 'CRITICAL', 'iteration', 'Objective']):
                print(line.rstrip())
        
        process.wait()
    
    print(f"\n✅ {mode_name} complete. Log saved to {output_log}\n")
    return process.returncode


def parse_ipopt_iteration(log_file):
    """Extract iteration-by-iteration data from IPOPT log."""
    
    iterations = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # IPOPT iteration pattern:
    # iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
    pattern = r'^\s*(\d+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([-\d.e+]+)\s+([\d.e+-]+)'
    
    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            it_num, obj, inf_pr, inf_du, lg_mu, d_norm = match.groups()
            iterations.append({
                'iteration': int(it_num),
                'objective': float(obj),
                'inf_pr': float(inf_pr),  # Primal infeasibility
                'inf_du': float(inf_du),  # Dual infeasibility
                'log_mu': float(lg_mu),
                'd_norm': float(d_norm)
            })
    
    return iterations


def find_nan_location(log_file):
    """Find where NaN first appears in the log."""
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if 'NaN detected' in line:
                # Extract row/col info
                row_match = re.search(r'row (\d+)', line)
                col_match = re.search(r'col (\d+)', line)
                
                row = int(row_match.group(1)) if row_match else None
                col = int(col_match.group(1)) if col_match else None
                
                return {
                    'line_number': line_num,
                    'line_content': line.strip(),
                    'row': row,
                    'col': col
                }
    
    return None


def compare_trajectories(log_2d, log_3d):
    """Compare 2D and 3D IPOPT trajectories."""
    
    print(f"\n{'='*80}")
    print("TRAJECTORY COMPARISON: 2D vs 3D")
    print(f"{'='*80}\n")
    
    iter_2d = parse_ipopt_iteration(log_2d)
    iter_3d = parse_ipopt_iteration(log_3d)
    
    print(f"2D: {len(iter_2d)} IPOPT iterations")
    print(f"3D: {len(iter_3d)} IPOPT iterations")
    
    if len(iter_3d) == 0:
        print("\n🔴 3D failed immediately (0 iterations)")
        nan_info = find_nan_location(log_3d)
        if nan_info:
            print(f"\n📍 NaN detected at:")
            print(f"   Line {nan_info['line_number']}: {nan_info['line_content']}")
            print(f"   Row: {nan_info['row']}, Col: {nan_info['col']}")
    else:
        print(f"\n📊 2D Trajectory:")
        for it in iter_2d[:5]:
            print(f"   Iter {it['iteration']:3d}: obj={it['objective']:12.6f}, inf_pr={it['inf_pr']:10.3e}, inf_du={it['inf_du']:10.3e}")
        
        print(f"\n📊 3D Trajectory:")
        for it in iter_3d[:5]:
            print(f"   Iter {it['iteration']:3d}: obj={it['objective']:12.6f}, inf_pr={it['inf_pr']:10.3e}, inf_du={it['inf_du']:10.3e}")
        
        # Find divergence point
        for i, (it2d, it3d) in enumerate(zip(iter_2d, iter_3d)):
            obj_diff = abs(it2d['objective'] - it3d['objective'])
            if obj_diff > 1e-6:
                print(f"\n🔍 Divergence at iteration {i}:")
                print(f"   2D obj: {it2d['objective']:.6f}")
                print(f"   3D obj: {it3d['objective']:.6f}")
                print(f"   Diff: {obj_diff:.6e}")
                break


def extract_constraint_evaluation(log_file, around_line):
    """Extract constraint/variable values around NaN detection."""
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Look at context around the NaN line
    start = max(0, around_line - 50)
    end = min(len(lines), around_line + 10)
    
    print(f"\n{'='*80}")
    print(f"CONTEXT AROUND NaN (lines {start}-{end})")
    print(f"{'='*80}\n")
    
    for i in range(start, end):
        line = lines[i]
        # Highlight important lines
        if any(keyword in line for keyword in ['NaN', 'Inf', 'constraint', 'variable', 'WARNING', 'ERROR']):
            print(f">>> {i+1:5d}: {line.rstrip()}")
        elif i == around_line - 1:
            print(f"*** {i+1:5d}: {line.rstrip()}")  # The NaN line


if __name__ == "__main__":
    output_dir = Path("diagnostics/2026-02-13_3d-llt-debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run 2D with full logging
    log_2d = output_dir / "detailed_2d_full.log"
    run_with_full_logging("conf/test_2d.yaml", log_2d, "2D NeuralFoil")
    
    # Run 3D with full logging
    log_3d = output_dir / "detailed_3d_full.log"
    run_with_full_logging("conf/test.yaml", log_3d, "3D LLT")
    
    # Analyze results
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")
    
    # Compare trajectories
    compare_trajectories(log_2d, log_3d)
    
    # Find NaN in 3D
    nan_info = find_nan_location(log_3d)
    if nan_info:
        extract_constraint_evaluation(log_3d, nan_info['line_number'])
    
    print(f"\n{'='*80}")
    print("✅ Analysis complete. Review the logs above.")
    print(f"{'='*80}\n")
    print(f"Full logs:")
    print(f"  2D: {log_2d}")
    print(f"  3D: {log_3d}")
