"""
Compare results across all VulTypeIL++ ablations
Generate comparison tables and plots
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_metrics(method_name, base_dir="outputs"):
    """Load metrics for a specific method."""
    metrics_dir = Path(base_dir) / method_name / "metrics"
    
    if not metrics_dir.exists():
        return None
    
    # Find summary file
    summary_files = list(metrics_dir.glob("*_summary.csv"))
    if not summary_files:
        return None
    
    summary = pd.read_csv(summary_files[0])
    
    # Find accuracy matrix
    acc_files = list(metrics_dir.glob("*_acc_matrix.csv"))
    acc_matrix = None
    if acc_files:
        acc_matrix = pd.read_csv(acc_files[0], index_col=0)
    
    # Find forgetting file
    forg_files = list(metrics_dir.glob("*_forgetting.csv"))
    forgetting = None
    if forg_files:
        forgetting = pd.read_csv(forg_files[0])
    
    return {
        'summary': summary,
        'acc_matrix': acc_matrix,
        'forgetting': forgetting
    }


def create_comparison_table(methods_data):
    """Create comparison table of key metrics."""
    rows = []
    
    for method_name, data in methods_data.items():
        if data is None:
            continue
        
        summary = data['summary']
        row = {
            'Method': method_name,
            'Avg Forgetting': summary['avg_forgetting'].values[0],
            'Task 1 Final Acc': summary['task1_final_acc'].values[0],
            'Final Avg Acc': summary['final_avg_acc'].values[0],
            'Backward Transfer': summary['backward_transfer'].values[0],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Avg Forgetting')  # Sort by forgetting (lower is better)
    
    return df


def plot_forgetting_comparison(methods_data, output_path):
    """Plot forgetting comparison across methods."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_labels = []
    forgetting_values = []
    
    for method_name, data in methods_data.items():
        if data is None or data['summary'] is None:
            continue
        
        x_labels.append(method_name)
        forgetting_values.append(data['summary']['avg_forgetting'].values[0])
    
    colors = ['#d62728' if 'fixed200' in m or 'ratio_allold' in m else '#2ca02c' 
              for m in x_labels]
    
    bars = ax.bar(range(len(x_labels)), forgetting_values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Average Forgetting', fontsize=12)
    ax.set_title('Catastrophic Forgetting Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, forgetting_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved forgetting comparison to {output_path}")
    plt.close()


def plot_accuracy_heatmaps(methods_data, output_path):
    """Plot accuracy matrices as heatmaps."""
    n_methods = sum(1 for d in methods_data.values() if d is not None and d['acc_matrix'] is not None)
    
    if n_methods == 0:
        print("No accuracy matrices found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (method_name, data) in enumerate(methods_data.items()):
        if data is None or data['acc_matrix'] is None:
            continue
        
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        acc_matrix = data['acc_matrix'].values
        
        sns.heatmap(acc_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Accuracy'})
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Task Evaluated', fontsize=10)
        ax.set_ylabel('After Training Task', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy heatmaps to {output_path}")
    plt.close()


def plot_task1_trajectory(methods_data, output_path):
    """Plot Task 1 accuracy trajectory across training."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_name, data in methods_data.items():
        if data is None or data['acc_matrix'] is None:
            continue
        
        # Extract Task 1 accuracy after each task
        task1_acc = data['acc_matrix'].iloc[:, 0].values
        tasks = range(1, len(task1_acc) + 1)
        
        linestyle = '--' if 'fixed200' in method_name or 'ratio_allold' in method_name else '-'
        ax.plot(tasks, task1_acc, marker='o', label=method_name, 
               linestyle=linestyle, linewidth=2, markersize=6)
    
    ax.set_xlabel('Training Task', fontsize=12)
    ax.set_ylabel('Task 1 Accuracy', fontsize=12)
    ax.set_title('Task 1 Performance Over Time (Forgetting)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 6))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Task 1 trajectory to {output_path}")
    plt.close()


def main():
    # Define methods to compare
    methods = {
        '1_fixed200': 'VulTypeIL Fixed 200',
        '2_ratio_allold': 'VulTypeIL Ratio All-Old',
        '3_er': 'VulTypeIL++ ER',
        '4_mcss': 'VulTypeIL++ MCSS',
        '5_gcr': 'VulTypeIL++ GCR',
        '6_mcss_cons': 'VulTypeIL++ MCSS+Cons',
    }
    
    print("Loading metrics from all methods...")
    methods_data = {}
    for method_dir, method_name in methods.items():
        data = load_metrics(method_dir)
        if data is not None:
            methods_data[method_name] = data
            print(f"  ✓ Loaded {method_name}")
        else:
            print(f"  ✗ No data for {method_name}")
    
    if not methods_data:
        print("\nNo metrics found. Please run experiments first.")
        return
    
    # Create output directory
    output_dir = Path("outputs/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison table
    print("\nGenerating comparison table...")
    comparison_df = create_comparison_table(methods_data)
    comparison_df.to_csv(output_dir / "comparison_table.csv", index=False)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved to {output_dir / 'comparison_table.csv'}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_forgetting_comparison(methods_data, output_dir / "forgetting_comparison.png")
    plot_accuracy_heatmaps(methods_data, output_dir / "accuracy_heatmaps.png")
    plot_task1_trajectory(methods_data, output_dir / "task1_trajectory.png")
    
    print(f"\n{'='*80}")
    print("Comparison complete! Check outputs/comparison/ for results.")
    print(f"{'='*80}\n")
    
    # Print key findings
    print("KEY FINDINGS:")
    print("-" * 80)
    
    best_forgetting = comparison_df.loc[comparison_df['Avg Forgetting'].idxmin()]
    print(f"✓ Best (Lowest) Forgetting: {best_forgetting['Method']}")
    print(f"  Avg Forgetting: {best_forgetting['Avg Forgetting']:.4f}")
    
    best_task1 = comparison_df.loc[comparison_df['Task 1 Final Acc'].idxmax()]
    print(f"\n✓ Best Task 1 Retention: {best_task1['Method']}")
    print(f"  Task 1 Final Acc: {best_task1['Task 1 Final Acc']:.4f}")
    
    best_overall = comparison_df.loc[comparison_df['Final Avg Acc'].idxmax()]
    print(f"\n✓ Best Overall Performance: {best_overall['Method']}")
    print(f"  Final Avg Acc: {best_overall['Final Avg Acc']:.4f}")
    
    print("-" * 80)


if __name__ == "__main__":
    main()
