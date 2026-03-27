import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from bsp_parser import BSPParser
from nav_graph import build_nav_graph
from astar import astar
from learned_heuristics import LearnedHeuristic

PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_correction_factor_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # histogram of all correction factors
    axes[0].hist(df['correction_factor'].clip(upper=6), bins=60,
                 color='steelblue', edgecolor='white', linewidth=0.3)
    axes[0].set_xlabel('correction factor (true cost / euclidean)')
    axes[0].set_ylabel('count')
    axes[0].set_title('correction factor distribution\n(all maps combined)')
    axes[0].axvline(1.0, color='red', linestyle='--', linewidth=1, label='cf = 1.0 (perfect heuristic)')
    axes[0].axvline(df['correction_factor'].mean(), color='orange', linestyle='--',
                    linewidth=1, label=f"mean = {df['correction_factor'].mean():.2f}")
    axes[0].legend()

    # median cf per map - shows which maps have the hardest geometry
    per_map = df.groupby('map_name')['correction_factor'].median().sort_values()
    colors  = ['coral' if m.startswith('e4') or m.startswith('dm') or m in ['start','end']
               else 'steelblue' for m in per_map.index]
    per_map.plot(kind='barh', ax=axes[1], color=colors)
    axes[1].set_xlabel('median correction factor')
    axes[1].set_title('median correction factor per map\n(higher = euclidean more misleading)')
    axes[1].axvline(1.0, color='red', linestyle='--', linewidth=1)

    train_patch = mpatches.Patch(color='steelblue', label='train/val maps (e1-e3)')
    test_patch  = mpatches.Patch(color='coral',     label='test maps (e4, dm, start, end)')
    axes[1].legend(handles=[train_patch, test_patch])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'correction_factor_dist.png', dpi=150)
    print("saved: correction_factor_dist.png")
    plt.show()


def plot_euclidean_vs_true_cost(df):
    # sample 5000 points so the plot isn't too dense
    sample = df.sample(min(5000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(sample['euclidean_dist'], sample['true_cost'],
               alpha=0.1, s=3, c='steelblue')

    # perfect heuristic line
    max_val = max(sample['euclidean_dist'].max(), sample['true_cost'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='perfect heuristic (cf=1.0)')

    ax.set_xlabel('euclidean distance')
    ax.set_ylabel('true path cost (dijkstra)')
    ax.set_title('euclidean distance vs true path cost\n(points above the line = underestimation)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'euclidean_vs_true_cost.png', dpi=150)
    print("saved: euclidean_vs_true_cost.png")
    plt.show()


def plot_benchmark_summary(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # box plot of nodes expanded distributions
    train_maps = [m for m in results_df['map_name'].unique()
                  if m.startswith('e1') or m.startswith('e2')]
    val_maps   = [m for m in results_df['map_name'].unique() if m.startswith('e3')]
    test_maps  = [m for m in results_df['map_name'].unique()
                  if m.startswith('e4') or m.startswith('dm') or m in ['start','end']]

    for split, maps, color in [('train (e1+e2)', train_maps, 'steelblue'),
                                ('val (e3)',      val_maps,   'orange'),
                                ('test (e4+dm)',  test_maps,  'coral')]:
        subset = results_df[results_df['map_name'].isin(maps)]
        axes[0].hist(subset['nodes_reduction_pct'], bins=40, alpha=0.6,
                     color=color, label=f'{split} (mean={subset["nodes_reduction_pct"].mean():.1f}%)')

    axes[0].axvline(0, color='black', linewidth=1)
    axes[0].set_xlabel('nodes expanded reduction (%)')
    axes[0].set_ylabel('count')
    axes[0].set_title('distribution of node reduction\nby train/val/test split')
    axes[0].legend()

    # suboptimality distribution
    axes[1].hist(results_df['learned_subopt'].clip(upper=1.2), bins=50,
                 color='steelblue', edgecolor='white', linewidth=0.3)
    axes[1].axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='optimal (ratio=1.0)')
    axes[1].axvline(results_df['learned_subopt'].mean(), color='orange', linestyle='--',
                    linewidth=1.5, label=f"mean = {results_df['learned_subopt'].mean():.4f}")
    axes[1].set_xlabel('path cost ratio (learned / optimal)')
    axes[1].set_ylabel('count')
    axes[1].set_title('suboptimality distribution\n(how much longer are learned paths?)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'benchmark_summary.png', dpi=150)
    print("saved: benchmark_summary.png")
    plt.show()


def plot_astar_comparison_map(map_name="e1m1"):
    bsp_path = BASE_DIR / "data" / "maps" / f"{map_name}.bsp"
    data     = bsp_path.read_bytes()
    bsp      = BSPParser(data, map_name=map_name).parse()
    graph    = build_nav_graph(bsp)

    main  = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(main).copy()

    node_list      = list(graph.nodes)
    node_positions = {n: graph.nodes[n]['pos'] for n in graph.nodes}

    # pick a pair that's far apart for a dramatic comparison
    rng  = np.random.default_rng(99)
    best_pair = None
    best_dist = 0
    for _ in range(200):
        s = node_list[rng.integers(len(node_list))]
        g = node_list[rng.integers(len(node_list))]
        if s == g:
            continue
        d = np.linalg.norm(node_positions[s] - node_positions[g])
        if d > best_dist:
            best_dist = d
            best_pair = (s, g)

    src, goal = best_pair

    def euclid_h(n, g):
        return float(np.linalg.norm(node_positions[n] - node_positions[g]))

    def zero_h(n, g):
        return 0.0

    learned_h = LearnedHeuristic(graph, node_positions)

    path_e, cost_e, expanded_e = astar(graph, src, goal, euclid_h)
    path_l, cost_l, expanded_l = astar(graph, src, goal, learned_h)
    _,      cost_d, _          = astar(graph, src, goal, zero_h)

    print(f"\n{map_name} comparison (src={src}, goal={goal})")
    print(f"  euclidean {expanded_e} nodes, cost {cost_e:.0f}")
    print(f"  learned  {expanded_l} nodes, cost {cost_l:.0f}")
    print(f"  optimal   {cost_d:.0f}")
    print(f"  reduction {(expanded_e - expanded_l) / expanded_e * 100:.1f}%")

    all_pos = np.array([node_positions[n] for n in graph.nodes])

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(f"a* comparison on {map_name}", color='white', fontsize=14)

    for ax, (path, expanded, cost, title) in zip(axes, [
        (path_e, expanded_e, cost_e, f"euclidean a*\n{expanded_e} nodes expanded"),
        (path_l, expanded_l, cost_l, f"learned a*\n{expanded_l} nodes expanded  ({(expanded_e-expanded_l)/expanded_e*100:.1f}% fewer)"),
    ]):
        ax.set_facecolor('#1a1a1a')

        for u, v in graph.edges():
            pu = node_positions[u]
            pv = node_positions[v]
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]],
                    color='gray', alpha=0.15, linewidth=0.4)

        ax.scatter(all_pos[:, 0], all_pos[:, 1], s=5, c='steelblue', zorder=2)

        if path:
            px = [node_positions[n][0] for n in path]
            py = [node_positions[n][1] for n in path]
            ax.plot(px, py, color='orange', linewidth=2.5, zorder=4, label=f'path (cost {cost:.0f})')

        ax.scatter(*node_positions[src][:2],  c='lime', s=200, zorder=5, marker='*', label='start')
        ax.scatter(*node_positions[goal][:2], c='red',  s=200, zorder=5, marker='X', label='goal')

        ax.set_title(title, color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#333333', labelcolor='white')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{map_name}_astar_comparison.png', dpi=150,
                facecolor=fig.get_facecolor())
    print(f"saved {map_name}_astar_comparison.png")
    plt.show()





def print_summary_table(results_df):

    summary = results_df.groupby('map_name').agg(
        euclid_nodes  = ('euclid_nodes',       'mean'),
        learned_nodes = ('learned_nodes',       'mean'),
        reduction_pct = ('nodes_reduction_pct', 'mean'),
        suboptimality = ('learned_subopt',      'mean'),
    ).round(2)

    summary['episode'] = summary.index.map(lambda x:
        'e1' if x.startswith('e1') else
        'e2' if x.startswith('e2') else
        'e3' if x.startswith('e3') else
        'e4' if x.startswith('e4') else
        'dm' if x.startswith('dm') else 'other'
    )

    print(summary.sort_values('reduction_pct', ascending=False).to_string())
    print(f"\noverall mean reduction {results_df['nodes_reduction_pct'].mean():.1f}%")
    print(f"overall suboptimality:  {results_df['learned_subopt'].mean():.4f}")

    # per episode summary
    print(summary.groupby('episode')[['euclid_nodes', 'learned_nodes', 'reduction_pct', 'suboptimality']].mean().round(2))


if __name__ == "__main__":
    print("loading data.")
    gt_df      = pd.read_parquet(BASE_DIR / "data" / "ground_truth.parquet")
    results_df = pd.read_parquet(BASE_DIR / "data" / "benchmark_results.parquet")

    print(f"ground truth {len(gt_df)} pairs")
    print(f"benchmark    {len(results_df)} queries\n")


    print(" correction factor distribution")
    plot_correction_factor_distribution(gt_df)


    print(" euclidean vs true cost")
    plot_euclidean_vs_true_cost(gt_df)


    print(" benchmark summary")
    plot_benchmark_summary(results_df)


    print(" a* comparison on e1m1")
    plot_astar_comparison_map("e1m1")

    print_summary_table(results_df)

    """

    "C:\Users\rahul\OneDrive\Desktop\LEIDEN SEM 2\games\assignment 3\.venv\Scripts\python.exe" "C:\Users\rahul\OneDrive\Desktop\LEIDEN SEM 2\games\assignment 3\scripts\visualize.py"
loading data.
ground truth 625740 pairs
benchmark    38000 queries

 correction factor distribution
saved: correction_factor_dist.png
 euclidean vs true cost
saved: euclidean_vs_true_cost.png
 benchmark summary
saved: benchmark_summary.png
 a* comparison on e1m1

building nav graph for e1m1
  walkable faces- 853 / 5516
  nav nodes after merge 608
  edges: 6416
  connected componentss 4
  main component 598 / 608 nodes

e1m1 comparison (src=64, goal=562)
  euclidean 195 nodes, cost 3788
  learned  50 nodes, cost 3825
  optimal   3788
  reduction 74.4%
saved e1m1_astar_comparison.png

=== final results table ===

          euclid_nodes  learned_nodes  reduction_pct  suboptimality episode
map_name
e2m4             99.74          38.30          57.40           1.02      e2
e2m1             87.27          39.98          50.17           1.02      e2
e4m1             49.84          22.72          50.10           1.03      e4
e1m5             75.30          33.72          49.59           1.02      e1
start           102.52          52.23          49.58           1.02   other
e3m4            121.40          63.10          49.44           1.02      e3
e2m2             83.15          41.51          48.92           1.02      e2
e4m8             69.52          32.43          48.40           1.02      e4
e1m6             53.56          24.37          48.25           1.02      e1
e3m7             90.62          45.16          48.24           1.02      e3
e1m8             94.82          47.51          47.33           1.02      e1
e2m6             72.74          38.42          46.33           1.02      e2
e4m4            136.68          76.59          45.25           1.02      e4
e2m3            118.56          66.35          45.10           1.02      e2
e1m1             95.97          51.20          44.20           1.02      e1
dm4              32.78          15.30          43.28           1.02      dm
e1m2            102.04          58.45          43.07           1.02      e1
e3m3            109.94          62.97          42.98           1.02      e3
e1m4             68.85          36.97          42.85           1.01      e1
end              34.19          18.35          41.62           1.02   other
e4m3            128.36          75.73          40.86           1.02      e4
e4m7            214.77         139.76          40.52           1.02      e4
e4m5            201.30         133.26          39.76           1.02      e4
dm1              24.08          12.01          39.34           1.01      dm
dm3              73.21          43.61          39.12           1.02      dm
e3m1            158.12         108.51          39.02           1.02      e3
e2m5            136.89          84.92          38.99           1.02      e2
e2m7            126.71          79.65          38.74           1.02      e2
dm2              80.05          48.72          38.48           1.02      dm
dm5              51.40          32.19          38.38           1.02      dm
e4m6             65.66          39.30          38.10           1.02      e4
e1m7             25.91          13.63          37.86           1.02      e1
e3m6            153.96         102.47          36.79           1.02      e3
e3m2             90.58          61.61          34.97           1.02      e3
e3m5            134.34          96.64          34.80           1.02      e3
dm6              55.31          36.08          32.63           1.01      dm
e4m2             78.53          56.15          32.39           1.01      e4
e1m3            157.85         126.36          27.19           1.01      e1

overall mean reduction 42.4%
overall suboptimality:  1.0195
         euclid_nodes  learned_nodes  reduction_pct  suboptimality
episode
dm              52.80          31.32          38.54           1.02
e1              84.29          49.03          42.54           1.02
e2             103.58          55.59          46.52           1.02
e3             122.71          77.21          40.89           1.02
e4             118.08          71.99          41.92           1.02
other           68.35          35.29          45.60           1.02

Process finished with exit code 0
"""