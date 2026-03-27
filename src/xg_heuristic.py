import numpy as np
import pickle
import xgboost as xgb
from pathlib import Path
from features import extract_features

BASE_DIR = Path(__file__).parent.parent

FEATURE_COLS = [
    'euclid_dist', 'height_diff', 'horiz_dist', 'height_ratio',
    'dx', 'dy', 'dz',
    'src_density', 'src_degree', 'src_avg_edge', 'src_max_edge',
    'goal_density', 'goal_degree', 'goal_avg_edge', 'goal_max_edge',
    'density_ratio', 'degree_ratio'
]


class XGBoostHeuristic:
    def __init__(self, graph, node_positions):
        self.graph          = graph
        self.node_positions = node_positions
        self._cache         = {}

        checkpoints = BASE_DIR / 'checkpoints'

        with open(checkpoints / 'xg_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        self.model = xgb.XGBRegressor()
        self.model.load_model(str(checkpoints / 'xg_model.json'))

    def __call__(self, node_idx, goal_idx):
        key = (node_idx, goal_idx)
        if key in self._cache:
            return self._cache[key]

        pos_n  = self.node_positions[node_idx]
        pos_g  = self.node_positions[goal_idx]
        euclid = float(np.linalg.norm(pos_n - pos_g))

        if euclid < 1e-6:
            return 0.0

        raw_feats = extract_features(node_idx, goal_idx, self.node_positions, self.graph)
        feat_vec  = np.array([raw_feats[k] for k in FEATURE_COLS], dtype=np.float32)
        feat_vec  = self.scaler.transform(feat_vec.reshape(1, -1))[0]

        # predict log_cf then convert back to cf
        log_cf = self.model.predict(feat_vec.reshape(1, -1))[0]
        cf     = max(1.0, float(np.exp(log_cf)))
        h      = euclid * cf

        self._cache[key] = h
        return h

    def clear_cache(self):
        self._cache = {}


if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt
    from bsp_parser import BSPParser
    from nav_graph import build_nav_graph
    from astar import astar

    bsp_path = BASE_DIR / "data" / "maps" / "e1m1.bsp"
    data     = bsp_path.read_bytes()
    bsp      = BSPParser(data, map_name="e1m1").parse()
    graph    = build_nav_graph(bsp)

    main  = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(main).copy()

    node_list      = list(graph.nodes)
    node_positions = {n: graph.nodes[n]['pos'] for n in graph.nodes}

    src  = node_list[0]
    goal = node_list[len(node_list) // 2]

    def euclid_h(n, g):
        return float(np.linalg.norm(node_positions[n] - node_positions[g]))

    def zero_h(n, g):
        return 0.0

    xgb_h = XGBoostHeuristic(graph, node_positions)

    _, cost_d, nodes_d = astar(graph, src, goal, zero_h)
    _, cost_e, nodes_e = astar(graph, src, goal, euclid_h)
    _, cost_x, nodes_x = astar(graph, src, goal, xgb_h)

    print(f"src: {src}  goal: {goal}")
    print(f"\ndijkstra  {nodes_d} nodes, cost {cost_d:.1f}")
    print(f"euclidean {nodes_e} nodes, cost {cost_e:.1f}")
    print(f"xgboost   {nodes_x} nodes, cost {cost_x:.1f}")

    if nodes_e > 0:
        print(f"\nxgboost reduction vs euclidean {(nodes_e - nodes_x) / nodes_e * 100:.1f}%")
        print(f"xgboost suboptimality        {cost_x / cost_d:.4f}")