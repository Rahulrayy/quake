import sys
import random
import time
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from bsp_parser import BSPParser
from nav_graph import build_nav_graph
from astar import astar
from learned_heuristics import LearnedHeuristic
from xg_heuristic import XGBoostHeuristic

# tunable parameters
N_QUERIES = 1000
MIN_DIST  = 300.0
SEED      = 50


def benchmark_map(map_name, bsp_path):
    try:
        data  = Path(bsp_path).read_bytes()
        bsp   = BSPParser(data, map_name=map_name).parse()
        graph = build_nav_graph(bsp)

        main  = max(nx.weakly_connected_components(graph), key=len)
        graph = graph.subgraph(main).copy()

        node_list      = list(graph.nodes)
        node_positions = {n: graph.nodes[n]['pos'] for n in graph.nodes}

        if len(node_list) < 10:
            print(f"  {map_name}: too few nodes, skipping")
            return []

        def euclid_h(n, g):
            return float(np.linalg.norm(node_positions[n] - node_positions[g]))

        def zero_h(n, g):
            return 0.0

        mlp_h = LearnedHeuristic(graph, node_positions)
        xgb_h = XGBoostHeuristic(graph, node_positions)

        # sample pairs far enough apart
        rng      = random.Random(SEED)
        pairs    = []
        attempts = 0

        while len(pairs) < N_QUERIES and attempts < N_QUERIES * 20:
            attempts += 1
            src  = rng.choice(node_list)
            goal = rng.choice(node_list)
            if src == goal:
                continue
            dist = np.linalg.norm(node_positions[src] - node_positions[goal])
            if dist >= MIN_DIST:
                pairs.append((src, goal))

        if not pairs:
            print(f"  {map_name}: no valid pairs found")
            return []

        results = []
        for src, goal in pairs:

            # dijkstra  optimal reference
            _, cost_d, nodes_d = astar(graph, src, goal, zero_h)
            if cost_d == float('inf'):
                continue

            # euclidean a*
            t0 = time.perf_counter()
            _, cost_e, nodes_e = astar(graph, src, goal, euclid_h)
            time_e = (time.perf_counter() - t0) * 1000

            # mlp learned a*
            mlp_h.clear_cache()
            t0 = time.perf_counter()
            _, cost_m, nodes_m = astar(graph, src, goal, mlp_h)
            time_m = (time.perf_counter() - t0) * 1000

            # xgboost a*
            xgb_h.clear_cache()
            t0 = time.perf_counter()
            _, cost_x, nodes_x = astar(graph, src, goal, xgb_h)
            time_x = (time.perf_counter() - t0) * 1000

            results.append({
                'map_name':       map_name,
                'src':            src,
                'goal':           goal,
                'optimal_cost':   cost_d,
                'euclid_nodes':   nodes_e,
                'mlp_nodes':      nodes_m,
                'xgb_nodes':      nodes_x,
                'dijkstra_nodes': nodes_d,
                'euclid_cost':    cost_e,
                'mlp_cost':       cost_m,
                'xgb_cost':       cost_x,
                'euclid_time_ms': time_e,
                'mlp_time_ms':    time_m,
                'xgb_time_ms':    time_x,
                'euclid_subopt':  cost_e / cost_d if cost_d > 0 else 1.0,
                'mlp_subopt':     cost_m / cost_d if cost_d > 0 else 1.0,
                'xgb_subopt':     cost_x / cost_d if cost_d > 0 else 1.0,
            })

        print(f"  {map_name}: {len(results)} queries done")
        return results

    except Exception as e:
        print(f"  {map_name}: failed — {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    bsp_dir  = BASE_DIR / "data" / "maps"
    out_path = BASE_DIR / "data" / "xgboost_benchmark_results.parquet"

    bsp_files = sorted([p for p in bsp_dir.glob("*.bsp")
                        if not p.stem.startswith("b_")])

    print(f"benchmarking {len(bsp_files)} maps, {N_QUERIES} queries each")
    print(f"comparing: dijkstra | euclidean | mlp | xgboost\n")

    all_results = []
    for p in bsp_files:
        results = benchmark_map(p.stem, str(p))
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    # compute reduction percentages
    df['mlp_reduction_pct'] = (df['euclid_nodes'] - df['mlp_nodes']) / df['euclid_nodes'] * 100
    df['xgb_reduction_pct'] = (df['euclid_nodes'] - df['xgb_nodes']) / df['euclid_nodes'] * 100

    df.to_parquet(out_path, index=False)
    print(f"\nsaved: {out_path}")

    # summary table
    summary = df.groupby('map_name').agg(
        euclid_nodes  = ('euclid_nodes',     'mean'),
        mlp_nodes     = ('mlp_nodes',         'mean'),
        xgb_nodes     = ('xgb_nodes',         'mean'),
        mlp_reduction = ('mlp_reduction_pct', 'mean'),
        xgb_reduction = ('xgb_reduction_pct', 'mean'),
        mlp_subopt    = ('mlp_subopt',         'mean'),
        xgb_subopt    = ('xgb_subopt',         'mean'),
    ).round(2)

    print("\n--- benchmark summary ---")
    print(summary.sort_values('mlp_reduction', ascending=False).to_string())
    print(f"\noverall mlp reduction: {df['mlp_reduction_pct'].mean():.1f}%")
    print(f"overall xgb reduction: {df['xgb_reduction_pct'].mean():.1f}%")
    print(f"overall mlp subopt:    {df['mlp_subopt'].mean():.4f}")
    print(f"overall xgb subopt:    {df['xgb_subopt'].mean():.4f}")

    # plot 1 - side by side reduction per map
    fig, ax = plt.subplots(figsize=(12, 12))
    x       = np.arange(len(summary))
    width   = 0.35
    ax.barh(x - width/2, summary['mlp_reduction'], width,
            color='steelblue', label='mlp')
    ax.barh(x + width/2, summary['xgb_reduction'], width,
            color='coral', label='xgboost')
    ax.set_yticks(x)
    ax.set_yticklabels(summary.index)
    ax.set_xlabel('nodes expanded reduction vs euclidean (%)')
    ax.set_title('mlp vs xgboost\nnodes expanded reduction per map')
    ax.axvline(0, color='black', linewidth=1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "xgboost_vs_mlp_reduction.png", dpi=150)
    print("\nsave xgboost_vs_mlp_reduction.png")

    # plot 2 - overall comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    methods       = ['euclidean', 'mlp', 'xgboost']
    mean_nodes    = [df['euclid_nodes'].mean(), df['mlp_nodes'].mean(), df['xgb_nodes'].mean()]
    mean_subopt   = [df['euclid_subopt'].mean(), df['mlp_subopt'].mean(), df['xgb_subopt'].mean()]
    colors        = ['gray', 'steelblue', 'coral']

    axes[0].bar(methods, mean_nodes, color=colors, edgecolor='black')
    axes[0].set_ylabel('mean nodes expanded')
    axes[0].set_title('mean nodes expanded\nall maps combined')
    for i, v in enumerate(mean_nodes):
        axes[0].text(i, v + 1, f'{v:.1f}', ha='center', fontsize=10)

    axes[1].bar(methods, mean_subopt, color=colors, edgecolor='black')
    axes[1].set_ylabel('mean suboptimality ratio')
    axes[1].set_title('mean suboptimality\n(1.0 = optimal)')
    axes[1].set_ylim(0.99, max(mean_subopt) * 1.02)
    for i, v in enumerate(mean_subopt):
        axes[1].text(i, v + 0.0002, f'{v:.4f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "xgboost_overall_comparison.png", dpi=150)
    print("saved xgboost_overall_comparison.png")

    plt.show()