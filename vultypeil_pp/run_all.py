"""
Run all VulTypeIL++ ablation experiments
"""
import argparse
import os
import time
from run import run_experiment


CONFIGS = [
    "configs/1_vultypeil_fixed200.yaml",
    "configs/2_vultypeil_ratio_allold.yaml",
    "configs/3_vultypeilpp_er.yaml",
    "configs/4_vultypeilpp_mcss.yaml",
    "configs/5_vultypeilpp_gcr_approx.yaml",
    "configs/6_vultypeilpp_mcss_cons.yaml",
]

CONFIG_NAMES = [
    "VulTypeIL Fixed 200 (Mahalanobis)",
    "VulTypeIL Ratio All-Old (20%)",
    "VulTypeIL++ Random ER",
    "VulTypeIL++ MCSS",
    "VulTypeIL++ GCR-Approx",
    "VulTypeIL++ MCSS + Consolidation",
]


def main():
    parser = argparse.ArgumentParser(description="Run all VulTypeIL++ ablations")
    parser.add_argument('--ablations', type=int, nargs='+', default=None,
                       help='Specific ablations to run (1-6). Default: all')
    parser.add_argument('--skip', type=int, nargs='+', default=[],
                       help='Ablations to skip (1-6)')
    args = parser.parse_args()
    
    # Determine which ablations to run
    if args.ablations is not None:
        ablations_to_run = [i - 1 for i in args.ablations if 1 <= i <= 6]
    else:
        ablations_to_run = list(range(6))
    
    # Remove skipped ablations
    ablations_to_run = [i for i in ablations_to_run if (i + 1) not in args.skip]
    
    print(f"\n{'='*80}")
    print("VulTypeIL++ Ablation Study")
    print(f"{'='*80}")
    print(f"Running {len(ablations_to_run)} ablations:")
    for i in ablations_to_run:
        print(f"  {i+1}. {CONFIG_NAMES[i]}")
    print(f"{'='*80}\n")
    
    results = {}
    total_start_time = time.time()
    
    for idx in ablations_to_run:
        config_path = CONFIGS[idx]
        config_name = CONFIG_NAMES[idx]
        
        print(f"\n{'='*80}")
        print(f"Running Ablation {idx+1}/{len(CONFIGS)}: {config_name}")
        print(f"Config: {config_path}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            metrics = run_experiment(config_path)
            elapsed_time = time.time() - start_time
            
            summary = metrics.get_summary()
            results[config_name] = {
                'success': True,
                'time': elapsed_time,
                'avg_forgetting': summary['avg_forgetting'],
                'task1_final_acc': summary['task1_final_acc'],
                'final_avg_acc': summary['final_avg_acc'],
                'backward_transfer': summary['backward_transfer'],
            }
            
            print(f"\n{'='*80}")
            print(f"Ablation {idx+1} Completed Successfully")
            print(f"Time: {elapsed_time/60:.2f} minutes")
            print(f"Avg Forgetting: {summary['avg_forgetting']:.4f}")
            print(f"Task 1 Final Acc: {summary['task1_final_acc']:.4f}")
            print(f"Final Avg Acc: {summary['final_avg_acc']:.4f}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            results[config_name] = {
                'success': False,
                'time': elapsed_time,
                'error': str(e)
            }
            
            print(f"\n{'='*80}")
            print(f"Ablation {idx+1} Failed")
            print(f"Error: {e}")
            print(f"{'='*80}\n")
    
    # Print summary
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total Time: {total_time/3600:.2f} hours\n")
    
    print(f"{'Method':<40} {'Status':<10} {'Time (min)':<12} {'Avg Forg':<10} {'T1 Final':<10} {'Final Avg':<10}")
    print(f"{'-'*40} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    
    for name, result in results.items():
        if result['success']:
            print(f"{name:<40} {'✓':<10} {result['time']/60:<12.2f} "
                  f"{result['avg_forgetting']:<10.4f} "
                  f"{result['task1_final_acc']:<10.4f} "
                  f"{result['final_avg_acc']:<10.4f}")
        else:
            print(f"{name:<40} {'✗':<10} {result['time']/60:<12.2f} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    print(f"{'='*80}\n")
    
    # Save summary to file
    summary_path = "vultypeil_pp/outputs/experiment_summary.txt"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("VulTypeIL++ Ablation Study Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Time: {total_time/3600:.2f} hours\n\n")
        
        f.write(f"{'Method':<40} {'Status':<10} {'Time (min)':<12} {'Avg Forg':<10} {'T1 Final':<10} {'Final Avg':<10}\n")
        f.write(f"{'-'*40} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}\n")
        
        for name, result in results.items():
            if result['success']:
                f.write(f"{name:<40} {'✓':<10} {result['time']/60:<12.2f} "
                       f"{result['avg_forgetting']:<10.4f} "
                       f"{result['task1_final_acc']:<10.4f} "
                       f"{result['final_avg_acc']:<10.4f}\n")
            else:
                f.write(f"{name:<40} {'✗':<10} {result['time']/60:<12.2f} {'N/A':<10} {'N/A':<10} {'N/A':<10}\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
