import numpy as np
import pandas as pd
from pathlib import Path


def spatial_features(src_pos, goal_pos):
    # basic geometric relationship between source and goal
    diff = goal_pos - src_pos
    euclid = np.linalg.norm(diff)

    return {
        'euclid_dist': euclid,
        'height_diff': abs(diff[2]),
        'horiz_dist': np.linalg.norm(diff[:2]),
        'height_ratio': abs(diff[2]) / (euclid + 1e-6),  # how much of the path is vertical
        'dx': abs(diff[0]),
        'dy': abs(diff[1]),
        'dz': diff[2],  # signed ositive goal is higher
    }


def node_context_features(node_idx, node_positions, graph, radius=150.0):
    # captures how open or enclosed the space is around a node
    pos = node_positions[node_idx]

    # count nearby nodeshigh density = complex geometry, low = open space
    nearby = sum(
        1 for n in graph.nodes
        if n != node_idx and np.linalg.norm(node_positions[n] - pos) < radius
    )

    edges = list(graph.edges(node_idx, data=True))
    if edges:
        weights = [e[2]['weight'] for e in edges]
        avg_edge = float(np.mean(weights))
        max_edge = float(np.max(weights))
        degree = len(edges)
    else:
        avg_edge = max_edge = degree = 0.0

    return {
        'density': nearby,
        'degree': degree,
        'avg_edge': avg_edge,
        'max_edge': max_edge,
    }


def extract_features(src_idx, goal_idx, node_positions, graph):
    src_pos = node_positions[src_idx]
    goal_pos = node_positions[goal_idx]

    feats = {}
    feats.update(spatial_features(src_pos, goal_pos))

    # context features for both source and goal nodes
    src_ctx = node_context_features(src_idx, node_positions, graph)
    goal_ctx = node_context_features(goal_idx, node_positions, graph)

    feats.update({f'src_{k}': v for k, v in src_ctx.items()})
    feats.update({f'goal_{k}': v for k, v in goal_ctx.items()})

    # ratio features,relative difference between src and goal context
    feats['density_ratio'] = (src_ctx['density'] + 1) / (goal_ctx['density'] + 1)
    feats['degree_ratio'] = (src_ctx['degree'] + 1) / (goal_ctx['degree'] + 1)

    return feats


def build_feature_matrix(df, graph_lookup, position_lookup):
    all_features = []

    for map_name, group in df.groupby('map_name'):
        if map_name not in graph_lookup:
            print(f"  skipping {map_name} - no graph loaded")
            continue

        graph = graph_lookup[map_name]
        positions = position_lookup[map_name]

        print(f"  extracting features for {map_name} ({len(group)} pairs)...")

        for _, row in group.iterrows():
            src_idx = int(row['src'])
            goal_idx = int(row['goal'])

            # skip if nodes not in this graph (can happen after subgraph trim)
            if src_idx not in positions or goal_idx not in positions:
                continue

            feats = extract_features(src_idx, goal_idx, positions, graph)
            feats['map_name'] = map_name
            feats['src'] = src_idx
            feats['goal'] = goal_idx
            feats['correction_factor'] = row['correction_factor']
            feats['log_cf'] = np.log(max(row['correction_factor'], 1.0))
            all_features.append(feats)

    return pd.DataFrame(all_features)


if __name__ == "__main__":
    import sys
    import networkx as nx

    sys.path.append(str(Path(__file__).parent))

    from bsp_parser import BSPParser
    from nav_graph import build_nav_graph

    BASE_DIR = Path(__file__).parent.parent
    data_path = BASE_DIR / "data" / "ground_truth.parquet"
    out_path = BASE_DIR / "data" / "features.parquet"
    bsp_dir = BASE_DIR / "data" / "maps"

    print("load ground truth data")
    df = pd.read_parquet(data_path)
    print(f"  {len(df)} pairs across {df['map_name'].nunique()} maps")

    # build graphs for all maps
    print("\nbuilding nav graphs")
    graph_lookup = {}
    position_lookup = {}

    for map_name in sorted(df['map_name'].unique()):
        bsp_path = bsp_dir / f"{map_name}.bsp"
        if not bsp_path.exists():
            continue

        data = bsp_path.read_bytes()
        bsp = BSPParser(data, map_name=map_name).parse()
        G = build_nav_graph(bsp)

        # trim to main component
        main = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(main).copy()

        graph_lookup[map_name] = G
        position_lookup[map_name] = {n: G.nodes[n]['pos'] for n in G.nodes}

    # extract features
    print("\nextracting features")
    features_df = build_feature_matrix(df, graph_lookup, position_lookup)

    print(f"\nfeature matrix shape- {features_df.shape}")
    print(
        f"features are {[c for c in features_df.columns if c not in ['map_name', 'src', 'goal', 'correction_factor', 'log_cf']]}")

    features_df.to_parquet(out_path, index=False)
    print(f"\nsaved {out_path}")