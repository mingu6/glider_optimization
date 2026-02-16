#!/usr/bin/env python3
"""
Extract diagnostic data from logs and create CSV summaries.
"""

import re
import csv
from pathlib import Path

def extract_llt_outputs(log_file):
    """Extract LLT coefficient outputs from log."""
    with open(log_file) as f:
        content = f.read()
    
    # Find all LLT output lines
    wing_pattern = r"🔍 3D LLT Wing Output.*?- (CL|CD|CM): min=([-\d.]+), max=([-\d.]+), mean=([-\d.]+)"
    elev_pattern = r"🔍 3D LLT Elevator Output.*?- (CL_e|CD_e|CM_e): min=([-\d.]+), max=([-\d.]+), mean=([-\d.]+)"
    
    wing_matches = re.findall(wing_pattern, content)
    elev_matches = re.findall(elev_pattern, content)
    
    wing_data = []
    for coeff, min_val, max_val, mean_val in wing_matches:
        wing_data.append({
            'coefficient': coeff,
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean_val)
        })
    
    elev_data = []
    for coeff, min_val, max_val, mean_val in elev_matches:
        elev_data.append({
            'coefficient': coeff,
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean_val)
        })
    
    return wing_data, elev_data

def extract_chebyshev_coeffs(log_file):
    """Extract Chebyshev polynomial coefficients from log."""
    with open(log_file) as f:
        content = f.read()
    
    pattern = r"🔍 Chebyshev coefficients: (phi_CL|phi_CD|phi_CM) range=\[([-\d.e+-]+), ([-\d.e+-]+)\]"
    matches = re.findall(pattern, content)
    
    data = []
    for coeff, min_val, max_val in matches:
        data.append({
            'coefficient': coeff,
            'min': float(min_val),
            'max': float(max_val)
        })
    
    return data

def extract_auxvar_stats(log_file):
    """Extract OCP auxvar statistics."""
    with open(log_file) as f:
        content = f.read()
    
    pattern = r"🔍 OCP auxvar stats - min=([-\d.]+), max=([-\d.]+), mean=([-\d.]+)"
    matches = re.findall(pattern, content)
    
    data = []
    for min_val, max_val, mean_val in matches:
        data.append({
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean_val)
        })
    
    return data

def extract_convergence_info(log_file):
    """Extract LLT convergence information."""
    with open(log_file) as f:
        content = f.read()
    
    # Look for convergence messages
    conv_pattern = r"🔍 LLT converged at iteration (\d+)/(\d+).*?rel_diff=([\d.e+-]+)"
    residual_pattern = r"🔍 Residual gradients: recent=([\d.e+-]+), mid.*?=([\d.e+-]+)"
    
    conv_matches = re.findall(conv_pattern, content)
    residual_matches = re.findall(residual_pattern, content)
    
    data = []
    for iteration, max_iter, rel_diff in conv_matches:
        data.append({
            'iteration': int(iteration),
            'max_iter': int(max_iter),
            'final_residual': float(rel_diff)
        })
    
    if residual_matches:
        for recent, mid in residual_matches:
            if data:
                data[-1]['residual_gradient_recent'] = float(recent)
                data[-1]['residual_gradient_mid'] = float(mid)
    
    return data

def main():
    diag_dir = Path("diagnostics/2026-02-13_3d-llt-debug")
    
    # Process all log files
    log_files = [
        "glider_debug_output.log",
        "test_with_clamping.log"
    ]
    
    for log_name in log_files:
        log_path = diag_dir / log_name
        if not log_path.exists():
            print(f"⚠️  Log not found: {log_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {log_name}")
        print(f"{'='*70}")
        
        base_name = log_name.replace('.log', '')
        
        # Extract LLT outputs
        wing_data, elev_data = extract_llt_outputs(log_path)
        if wing_data:
            csv_path = diag_dir / f"{base_name}_llt_wing_coeffs.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['coefficient', 'min', 'max', 'mean'])
                writer.writeheader()
                writer.writerows(wing_data)
            print(f"✓ Wing coefficients: {csv_path.name}")
        
        if elev_data:
            csv_path = diag_dir / f"{base_name}_llt_elevator_coeffs.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['coefficient', 'min', 'max', 'mean'])
                writer.writeheader()
                writer.writerows(elev_data)
            print(f"✓ Elevator coefficients: {csv_path.name}")
        
        # Extract Chebyshev coefficients
        cheb_data = extract_chebyshev_coeffs(log_path)
        if cheb_data:
            csv_path = diag_dir / f"{base_name}_chebyshev_coeffs.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['coefficient', 'min', 'max'])
                writer.writeheader()
                writer.writerows(cheb_data)
            print(f"✓ Chebyshev coefficients: {csv_path.name}")
        
        # Extract auxvar stats
        auxvar_data = extract_auxvar_stats(log_path)
        if auxvar_data:
            csv_path = diag_dir / f"{base_name}_auxvar_stats.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['min', 'max', 'mean'])
                writer.writeheader()
                writer.writerows(auxvar_data)
            print(f"✓ Auxvar statistics: {csv_path.name}")
        
        # Extract convergence info
        conv_data = extract_convergence_info(log_path)
        if conv_data:
            csv_path = diag_dir / f"{base_name}_llt_convergence.csv"
            with open(csv_path, 'w', newline='') as f:
                fieldnames = list(conv_data[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(conv_data)
            print(f"✓ LLT convergence: {csv_path.name}")
    
    print(f"\n{'='*70}")
    print("Summary saved to CSVs in diagnostics/2026-02-13_3d-llt-debug/")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
