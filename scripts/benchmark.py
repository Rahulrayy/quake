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
# tunable parameters
N_QUERIES = 1000   # queries per map
MIN_DIST  = 300.0 # only test pairs at least this far apart
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

        def euclid_h(node_idx, goal_idx):
            return float(np.linalg.norm(node_positions[node_idx] - node_positions[goal_idx]))

        def zero_h(node_idx, goal_idx):
            return 0.0

        learned_h = LearnedHeuristic(graph, node_positions)

        # sample pairs far enough apart to be interesting
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

            # dijkstra for optimal reference
            _, cost_d, nodes_d = astar(graph, src, goal, zero_h)
            if cost_d == float('inf'):
                continue

            # euclidean a*
            t0 = time.perf_counter()
            _, cost_e, nodes_e = astar(graph, src, goal, euclid_h)
            time_e = (time.perf_counter() - t0) * 1000

            # learned a*
            learned_h.clear_cache()
            t0 = time.perf_counter()
            _, cost_l, nodes_l = astar(graph, src, goal, learned_h)
            time_l = (time.perf_counter() - t0) * 1000

            results.append({
                'map_name':        map_name,
                'src':             src,
                'goal':            goal,
                'optimal_cost':    cost_d,
                'euclid_cost':     cost_e,
                'learned_cost':    cost_l,
                'euclid_nodes':    nodes_e,
                'learned_nodes':   nodes_l,
                'dijkstra_nodes':  nodes_d,
                'euclid_time_ms':  time_e,
                'learned_time_ms': time_l,
                'euclid_subopt':   cost_e / cost_d if cost_d > 0 else 1.0,
                'learned_subopt':  cost_l / cost_d if cost_d > 0 else 1.0,
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
    out_path = BASE_DIR / "data" / "benchmark_results.parquet"

    bsp_files = sorted([p for p in bsp_dir.glob("*.bsp")
                        if not p.stem.startswith("b_")])

    print(f"benchmarking {len(bsp_files)} maps, {N_QUERIES} queries each")
    print(f"minimum pair distance {MIN_DIST} quake units\n")

    all_results = []
    for p in bsp_files:
        results = benchmark_map(p.stem, str(p))
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df['nodes_reduction_pct'] = (
        (df['euclid_nodes'] - df['learned_nodes']) / df['euclid_nodes'] * 100
    )

    df.to_parquet(out_path, index=False)
    print(f"\nsaved: {out_path}")

    # summary
    summary = df.groupby('map_name').agg(
        euclid_nodes_mean  = ('euclid_nodes',       'mean'),
        learned_nodes_mean = ('learned_nodes',       'mean'),
        reduction_pct_mean = ('nodes_reduction_pct', 'mean'),
        learned_subopt     = ('learned_subopt',      'mean'),
        n_queries          = ('map_name',            'count'),
    ).round(2)

    print(summary.sort_values('reduction_pct_mean', ascending=False).to_string())
    print(f"\noverall mean reduction: {df['nodes_reduction_pct'].mean():.1f}%")
    print(f"overall suboptimality  {df['learned_subopt'].mean():.4f}")

    # plot 1 - reduction per map
    fig, ax = plt.subplots(figsize=(10, 12))
    per_map = df.groupby('map_name')['nodes_reduction_pct'].mean().sort_values()
    colors  = ['red' if x < 0 else 'steelblue' for x in per_map]
    per_map.plot(kind='barh', ax=ax, color=colors)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('nodes expanded reduction (%)')
    ax.set_title('learned vs euclidean a*\nnodes expanded reduction per map')
    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "benchmark_reduction.png", dpi=150)
    print(f"\nsaved plots/benchmark_reduction.png")

    # plot 2 - scatter euclid vs learned nodes
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df['euclid_nodes'], df['learned_nodes'],
               alpha=0.1, s=5, c='steelblue')
    max_val = max(df['euclid_nodes'].max(), df['learned_nodes'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='no improvement line')
    ax.set_xlabel('euclidean a* nodes expanded')
    ax.set_ylabel('learned a* nodes expanded')
    ax.set_title('nodes expanded euclidean vs learned')
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "benchmark_scatter.png", dpi=150)
    print(f"saved plots/benchmark_scatter.png")

    plt.show()