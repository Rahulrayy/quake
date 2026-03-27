import torch
import numpy as np
import pickle
from pathlib import Path
from model import HeuristicNet, FEATURE_COLS
from features import extract_features

BASE_DIR = Path(__file__).parent.parent


class LearnedHeuristic:
    def __init__(self, graph, node_positions):
        self.graph          = graph
        self.node_positions = node_positions
        self._cache         = {}  # cache predictions to avoid recomputing

        # load scaler and model from checkpoints
        checkpoints = BASE_DIR / 'checkpoints'

        with open(checkpoints / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        self.model = HeuristicNet(input_dim=len(FEATURE_COLS))
        self.model.load_state_dict(torch.load(checkpoints / 'best_model.pt',
                                              map_location='cpu'))
        self.model.eval()

    def __call__(self, node_idx, goal_idx):
        # check cache first
        key = (node_idx, goal_idx)
        if key in self._cache:
            return self._cache[key]

        pos_n  = self.node_positions[node_idx]
        pos_g  = self.node_positions[goal_idx]
        euclid = float(np.linalg.norm(pos_n - pos_g))

        if euclid < 1e-6:
            return 0.0

        # extract features, scale them, run through model
        raw_feats = extract_features(node_idx, goal_idx, self.node_positions, self.graph)
        feat_vec  = np.array([raw_feats[k] for k in FEATURE_COLS], dtype=np.float32)
        feat_vec  = self.scaler.transform(feat_vec.reshape(1, -1))[0]

        cf = self.model.predict_single(feat_vec)
        h  = euclid * cf

        self._cache[key] = h
        return h

    def clear_cache(self):
        self._cache = {}


if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt
    from bsp_parser import BSPParser
    from nav_graph import build_nav_graph
    from astar import astar, euclidean_heuristic

    bsp_path = BASE_DIR / "data" / "maps" / "e1m1.bsp"

    data  = bsp_path.read_bytes()
    bsp   = BSPParser(data, map_name="e1m1").parse()
    graph = build_nav_graph(bsp)

    # trim to main component
    main  = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(main).copy()

    node_list      = list(graph.nodes)
    node_positions = {n: graph.nodes[n]['pos'] for n in graph.nodes}

    # pick a src and goal that are far apart for a meaningful comparison
    src  = node_list[0]
    goal = node_list[len(node_list) // 2]

    print(f"src  {src}")
    print(f"goal {goal}")


    # euclidean heuristic using dict to handle non-contiguous node indices
    def euclid_h(node_idx, goal_idx):
        return np.linalg.norm(node_positions[node_idx] - node_positions[goal_idx])

    path_e, cost_e, expanded_e = astar(graph, src, goal, euclid_h)
    print(f"\neuclidean a*")
    print(f"  nodes expanded {expanded_e}")
    print(f"  path cost      {cost_e:.1f}")

    # learned heuristic
    learned_h = LearnedHeuristic(graph, node_positions)
    path_l, cost_l, expanded_l = astar(graph, src, goal, learned_h)
    print(f"\nlearned a*")
    print(f"  nodes expanded {expanded_l}")
    print(f"  path cost      {cost_l:.1f}")

    # summary
    if expanded_e > 0:
        reduction = (expanded_e - expanded_l) / expanded_e * 100
        print(f"\nnodes expanded reduction: {reduction:.1f}%")

    # plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor('#1a1a1a')

    all_pos = np.array([graph.nodes[n]['pos'] for n in graph.nodes])

    for ax, (path, expanded, title) in zip(axes, [
        (path_e, expanded_e, f"euclidean a*\n{expanded_e} nodes expanded"),
        (path_l, expanded_l, f"learned a*\n{expanded_l} nodes expanded"),
    ]):
        ax.set_facecolor('#1a1a1a')

        # all edges
        for u, v in graph.edges():
            pu = graph.nodes[u]['pos']
            pv = graph.nodes[v]['pos']
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]],
                    color='gray', alpha=0.15, linewidth=0.4)

        # all nodes
        ax.scatter(all_pos[:, 0], all_pos[:, 1], s=5, c='steelblue', zorder=2)

        # found path
        if path:
            px = [graph.nodes[n]['pos'][0] for n in path]
            py = [graph.nodes[n]['pos'][1] for n in path]
            ax.plot(px, py, color='orange', linewidth=2, zorder=4, label='path')

        # start and goal
        ax.scatter(*graph.nodes[src]['pos'][:2],  c='lime', s=150, zorder=5, marker='*', label='start')
        ax.scatter(*graph.nodes[goal]['pos'][:2], c='red',  s=150, zorder=5, marker='X', label='goal')

        ax.set_title(title, color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#333333', labelcolor='white')
        ax.set_aspect('equal')

    plt.suptitle("euclidean vs learned heuristic", color='white', fontsize=14)
    plt.tight_layout()
    out = BASE_DIR / "plots" / "heuristic_comparison.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nsaved {out}")
    plt.show()