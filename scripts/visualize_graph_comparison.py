"""
Visualize the difference between Fixed and Learned Graph topologies.
Creates figures for presentation.
ALL VALUES LOADED DYNAMICALLY FROM comparison_results.json
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_results():
    """Load results from the comparison experiment."""
    results_path = Path("outputs/comparison_results.json")
    
    if not results_path.exists():
        print(f"ERROR: {results_path} not found!")
        print("Run: python scripts/run_comparison.py --epochs 5 first")
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract final epoch metrics
    no_graph_final = results["no_graph"][-1]
    fixed_final = results["fixed_graph_k8"][-1]
    learned_final = results["learned_graph_k4"][-1]
    
    # CRITICAL FIX: edges_per_node is already the correct value (edges/node)
    # Don't divide by 10000 again!
    learned_edges_per_node = learned_final.get("edges_per_node", 0)
    
    # Also get total edges for reference
    learned_total_edges = learned_final.get("avg_edges", 0)
    
    print("\n" + "="*60)
    print("LOADED RESULTS:")
    print("="*60)
    print(f"No Graph Recall: {no_graph_final['recall']:.3f}")
    print(f"Fixed Graph Recall: {fixed_final['recall']:.3f}")
    print(f"Learned Graph Recall: {learned_final['recall']:.3f}")
    print(f"Learned Total Edges: {learned_total_edges:.0f}")
    print(f"Learned Edges/Node: {learned_edges_per_node:.2f}")
    print("="*60 + "\n")
    
    return {
        "no_graph_recall": no_graph_final["recall"],
        "fixed_recall": fixed_final["recall"],
        "learned_recall": learned_final["recall"],
        "learned_edges_per_node": learned_edges_per_node,  # FIXED: use directly
        "learned_total_edges": learned_total_edges,
    }


def visualize_fixed_vs_learned(results):
    """Create side-by-side visualization of graph topologies."""
    
    if results is None:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Grid parameters (showing a small 6x6 section for clarity)
    grid_size = 6
    
    # ========== No Graph ==========
    ax = axes[0]
    ax.set_title("No Graph\n(Baseline)", fontsize=14, fontweight='bold')
    
    # Draw grid nodes
    for i in range(grid_size):
        for j in range(grid_size):
            circle = plt.Circle((j, grid_size-1-i), 0.15, color='steelblue', alpha=0.7)
            ax.add_patch(circle)
    
    ax.set_xlim(-0.5, grid_size-0.5)
    ax.set_ylim(-0.5, grid_size-0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(grid_size/2-0.5, -0.8, "No connections\nbetween nodes", ha='center', fontsize=10)
    
    # ========== Fixed Graph (k=8) ==========
    ax = axes[1]
    ax.set_title("Fixed Graph (k=8)\n8 neighbors per node", fontsize=14, fontweight='bold')
    
    # Draw all 8-neighbor connections for center nodes
    for i in range(grid_size):
        for j in range(grid_size):
            y, x = grid_size-1-i, j
            # Draw edges to 8 neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        ny, nx = grid_size-1-ni, nj
                        ax.plot([x, nx], [y, ny], 'gray', linewidth=0.5, alpha=0.3)
    
    # Draw nodes on top
    for i in range(grid_size):
        for j in range(grid_size):
            circle = plt.Circle((j, grid_size-1-i), 0.15, color='steelblue', alpha=0.7)
            ax.add_patch(circle)
    
    ax.set_xlim(-0.5, grid_size-0.5)
    ax.set_ylim(-0.5, grid_size-0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.text(grid_size/2-0.5, -0.8, f"~8 edges/node\nMany redundant", ha='center', fontsize=10)
    
    # ========== Learned Graph (k=max 8) ==========
    ax = axes[2]
    ax.set_title("Learned Graph (k=max 8)\nAdaptive connections", fontsize=14, fontweight='bold')
    
    # Get actual learned edges per node from results
    learned_edges = results["learned_edges_per_node"]
    
    # Place some "objects" (cars, pedestrians)
    np.random.seed(42)
    objects = [(1, 2), (2, 3), (4, 1), (4, 4)]
    
    # Adjust number of edges shown based on actual learned value
    # Scale from ~1-2 edges (very sparse) to ~5-6 edges (moderate)
    if learned_edges < 2:
        # Very sparse
        learned_edge_pairs = [
            ((1, 2), (2, 3)),
            ((4, 1), (4, 2)),
            ((4, 4), (4, 3)),
        ]
    elif learned_edges < 4:
        # Sparse to moderate
        learned_edge_pairs = [
            ((1, 2), (2, 2)),
            ((1, 2), (2, 3)),
            ((2, 3), (3, 3)),
            ((4, 1), (4, 2)),
            ((4, 4), (4, 3)),
            ((2, 3), (2, 2)),
        ]
    else:
        # Moderate connectivity (4-8 edges)
        learned_edge_pairs = [
            ((1, 2), (2, 2)),
            ((1, 2), (2, 3)),
            ((2, 3), (2, 2)),
            ((2, 3), (3, 3)),
            ((4, 1), (4, 2)),
            ((4, 1), (3, 1)),
            ((4, 4), (4, 3)),
            ((4, 4), (3, 4)),
            ((2, 3), (4, 4)),
            ((1, 2), (0, 2)),
        ]
    
    for (i1, j1), (i2, j2) in learned_edge_pairs:
        y1, x1 = grid_size-1-i1, j1
        y2, x2 = grid_size-1-i2, j2
        ax.plot([x1, x2], [y1, y2], 'orangered', linewidth=2, alpha=0.7)
    
    # Draw nodes
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) in objects:
                color = 'green'
                size = 0.2
            else:
                color = 'steelblue'
                size = 0.15
            circle = plt.Circle((j, grid_size-1-i), size, color=color, alpha=0.7)
            ax.add_patch(circle)
    
    ax.set_xlim(-0.5, grid_size-0.5)
    ax.set_ylim(-0.5, grid_size-0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Use actual value from results
    ax.text(grid_size/2-0.5, -0.8, 
            f"~{learned_edges:.1f} edges/node\nOnly meaningful", 
            ha='center', fontsize=10)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='green', label='Object (car/pedestrian)'),
        mpatches.Patch(color='steelblue', label='Empty cell'),
        plt.Line2D([0], [0], color='orangered', linewidth=2, label='Learned edge'),
        plt.Line2D([0], [0], color='gray', linewidth=1, alpha=0.5, label='Fixed edge'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save
    output_path = Path("outputs/graph_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_results_bar_chart(results):
    """Create bar chart comparing methods."""
    
    if results is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['No Graph', 'Fixed Graph\n(k=8)', 'Learned Graph\n(k=max 8)']
    recalls = [
        results["no_graph_recall"],
        results["fixed_recall"],
        results["learned_recall"]
    ]
    edges = [0, 8, results["learned_edges_per_node"]]
    
    # ========== Recall Comparison ==========
    ax = axes[0]
    colors = ['#ff7f7f', '#7fbf7f', '#7f7fff']
    bars = ax.bar(methods, recalls, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Detection Recall Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')
    
    # ========== Efficiency Comparison ==========
    ax = axes[1]
    
    # Edges per node
    bars = ax.bar(methods, edges, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Edges per Node', fontsize=12)
    ax.set_title('Graph Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10)
    
    for bar, val in zip(bars, edges):
        label = 'N/A' if val == 0 else f'{val:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                label, ha='center', fontsize=11, fontweight='bold')
    
    # Add annotation for efficiency
    reduction_pct = (8 - results["learned_edges_per_node"]) / 8 * 100
    ax.annotate(f'{reduction_pct:.0f}% fewer\nedges!', 
                xy=(2, results["learned_edges_per_node"]), 
                xytext=(2.5, 6),
                fontsize=11, color='blue', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.tight_layout()
    
    output_path = Path("outputs/results_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_key_insight(results):
    """Create the key insight figure as bar graphs for presentation."""
    
    if results is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['Fixed Graph\n(k=8)', 'Learned Graph\n(k=max 8)']
    recalls = [results["fixed_recall"], results["learned_recall"]]
    
    # CRITICAL FIX: Convert edges/node to total edges (in thousands)
    # 10,000 nodes * edges_per_node / 1000 = thousands of edges
    learned_edges_thousands = results["learned_edges_per_node"] * 10000 / 1000
    edges = [80, learned_edges_thousands]  # Thousands
    
    # ========== Recall Bar Chart ==========
    ax = axes[0]
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax.bar(methods, recalls, color=colors, edgecolor='black', linewidth=2, width=0.6)
    
    ax.set_ylabel('Recall', fontsize=14, fontweight='bold')
    ax.set_title('Detection Performance', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontsize=14, fontweight='bold')
    
    # Add comparison annotation
    recall_diff = recalls[1] - recalls[0]
    if abs(recall_diff) < 0.03:  # Within 3%
        ax.annotate('Comparable\nperformance', xy=(0.5, max(recalls) + 0.05), 
                    fontsize=12, ha='center', color='green', fontweight='bold')
    elif recall_diff > 0:
        ax.annotate('Learned is\nBETTER!', xy=(1, recalls[1]), 
                    fontsize=12, ha='center', color='green', fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ========== Edges Bar Chart ==========
    ax = axes[1]
    bars = ax.bar(methods, edges, color=colors, edgecolor='black', linewidth=2, width=0.6)
    
    ax.set_ylabel('Edges (thousands)', fontsize=14, fontweight='bold')
    ax.set_title('Graph Complexity', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, val in zip(bars, edges):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}K', ha='center', fontsize=14, fontweight='bold')
    
    # Add reduction annotation with arrow
    reduction_pct = (edges[0] - edges[1]) / edges[0] * 100
    ax.annotate(f'{reduction_pct:.0f}% Fewer\nEdges!', 
                xy=(1, edges[1]), 
                xytext=(1.3, 50),
                fontsize=12, ha='center', color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Main title
    fig.suptitle(f'Key Insight: Comparable Performance, {reduction_pct:.0f}% Fewer Edges!', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = Path("outputs/key_insight.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Creating visualizations for presentation...")
    print("Loading results from outputs/comparison_results.json...\n")
    
    # Load results once
    results = load_results()
    
    if results is None:
        print("\nERROR: Cannot create visualizations without results!")
        print("Please run: python scripts/run_comparison.py --epochs 5")
        exit(1)
    
    # Create all visualizations
    visualize_fixed_vs_learned(results)
    visualize_results_bar_chart(results)
    visualize_key_insight(results)
    
    print("\n✓ All visualizations saved to outputs/ folder!")
    print("\nFiles created:")
    print("  - outputs/graph_comparison.png    (Fixed vs Learned topology)")
    print("  - outputs/results_comparison.png  (Bar charts)")
    print("  - outputs/key_insight.png         (Key takeaway)")