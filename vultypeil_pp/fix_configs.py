"""
Fix all config files to use correct relative paths
"""
import os
import re

config_files = [
    "configs/1_vultypeil_fixed200.yaml",
    "configs/2_vultypeil_ratio_allold.yaml",
    "configs/3_vultypeilpp_er.yaml",
    "configs/4_vultypeilpp_mcss.yaml",
    "configs/5_vultypeilpp_gcr_approx.yaml",
    "configs/6_vultypeilpp_mcss_cons.yaml",
]

for config_file in config_files:
    print(f"Fixing {config_file}...")
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Fix data_dir
    content = re.sub(r'data_dir: "/admin/dataset"', 'data_dir: "../incremental_tasks_csv"', content)
    
    # Fix output paths - remove vultypeil_pp/ prefix
    content = re.sub(r'checkpoint_dir: "vultypeil_pp/outputs/', 'checkpoint_dir: "outputs/', content)
    content = re.sub(r'results_dir: "vultypeil_pp/outputs/', 'results_dir: "outputs/', content)
    content = re.sub(r'metrics_dir: "vultypeil_pp/outputs/', 'metrics_dir: "outputs/', content)
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Fixed {config_file}")

print("\nAll config files fixed!")
