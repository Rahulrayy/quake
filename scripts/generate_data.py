import sys
import random
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from multiprocessing import Pool

# add src to path to import  modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bsp_parser import BSPParser
from nav_graph import build_nav_graph

# tunable parameters
PAIRS_PER_MAP = 50000  # raise for more training data, lower to run faster
N_SOURCES     =1000   # how many source nodes to run dikhstra from per map
N_GOALS       =30    # how many goals to sample per source node
RANDOM_SEED   = 50


def sample_pairs(graph, node_positions, n_sources, n_goals, seed):
    rng = random.Random(seed)
    nodes = list(graph.nodes)

    if len(nodes) < 2:
        return []

    # cap sources to available nodes
    n_sources = min(n_sources, len(nodes))
    sources = rng.sample(nodes, n_sources)
    records = []

    for src in sources:
        # dijkstra from this source to all reachable nodes
        lengths = nx.single_source_dijkstra_path_length(graph, src, weight='weight')

        # only keep reachable nodes that arent the source itself
        reachable = [(n, d) for n, d in lengths.items() if n != src and d < float('inf')]
        if not reachable:
            continue

        # sample random goals from reachable nodes
        goals = rng.sample(reachable, min(n_goals, len(reachable)))

        for goal, true_cost in goals:
            pos_s = node_positions[src]
            pos_g = node_positions[goal]
            euclid = float(np.linalg.norm(pos_s - pos_g))

            # skip pairs that are almost on top of each other
            if euclid < 1.0:
                continue

            cf = true_cost / euclid  # correction factor training

            records.append({
                'src':              src,
                'goal':             goal,
                'true_cost':        true_cost,
                'euclidean_dist':   euclid,
                'correction_factor': cf,
                'src_x':  float(pos_s[0]),
                'src_y':  float(pos_s[1]),
                'src_z':  float(pos_s[2]),
                'goal_x': float(pos_g[0]),
                'goal_y': float(pos_g[1]),
                'goal_z': float(pos_g[2]),
            })

        if len(records) >= PAIRS_PER_MAP:
            break

    return records[:PAIRS_PER_MAP]


def process_map(args):
    map_name, bsp_path = args

    # need to reimport otherwise stuck on single thread
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from bsp_parser import BSPParser
    from nav_graph import build_nav_graph

    try:
        data = Path(bsp_path).read_bytes()
        bsp  = BSPParser(data, map_name=map_name).parse()
        graph = build_nav_graph(bsp)

        components = list(nx.weakly_connected_components(graph))
        main = max(components, key=len)
        graph = graph.subgraph(main).copy()

        # build position lookup keyed by node index
        node_positions = {n: graph.nodes[n]['pos'] for n in graph.nodes}

        records = sample_pairs(graph, node_positions, N_SOURCES, N_GOALS, RANDOM_SEED)

        for r in records:
            r['map_name'] = map_name

        print(f"  {map_name}: {len(records)} pairs")
        return records

    except Exception as e:
        print(f"  {map_name}: failed - {e}")
        return []


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    bsp_dir  = BASE_DIR / "data" / "maps"
    out_path = BASE_DIR / "data" / "ground_truth.parquet"

    # only process actual level maps, skip item models (b_*.bsp)
    bsp_files = [p for p in bsp_dir.glob("*.bsp")
                 if not p.stem.startswith("b_")]

    print(f"found {len(bsp_files)} maps to process")
    args = [(p.stem, str(p)) for p in sorted(bsp_files)]


    n_workers = 4
    print(f"running with {n_workers} worker\n")

    with Pool(n_workers) as pool:
        results = pool.map(process_map, args)

    # flattenand save
    all_records = [r for result in results for r in result]
    df = pd.DataFrame(all_records)

    print(f"\ntotal pairs generated- {len(df)}")
    print(f"maps covered          {df['map_name'].nunique()}")
    print(f"\ncorrection factor stats")
    print(df['correction_factor'].describe().round(3))

    df.to_parquet(out_path, index=False)
    print(f"\nsave {out_path}")