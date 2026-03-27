import heapq
import numpy as np
from pathlib import Path


# tunable parameters
WEIGHTED_A_STAR_W = 1.5

def astar(graph, start, goal, heuristic):
    # heap entries - (f_score, g_score, node)
    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, start))

    came_from = {}
    g_score   = {start: 0.0}
    closed    = set()
    nodes_expanded = 0

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            return reconstruct_path(came_from, current), g_score[goal], nodes_expanded

        for neighbor in graph.successors(current):
            if neighbor in closed:
                continue

            tentative_g = g_score[current] + graph[current][neighbor]['weight']

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                h     = heuristic(neighbor, goal)
                f_new = tentative_g + h
                heapq.heappush(open_heap, (f_new, tentative_g, neighbor))

    return None, float('inf'), nodes_expanded


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))


def euclidean_heuristic(node_positions):
    # straight line distance
    def h(node_idx, goal_idx):
        return np.linalg.norm(node_positions[node_idx] - node_positions[goal_idx])
    return h


def zero_heuristic():
    # no heuristic = dijkstra, use this to get the true optimal cost
    def h(node_idx, goal_idx):
        return 0.0
    return h


def weighted_astar(graph, start, goal, heuristic, w=WEIGHTED_A_STAR_W):
    # inflates heuristic by w, expands fewer nodes but path may be suboptimal
    def inflated_h(node_idx, goal_idx):
        return w * heuristic(node_idx, goal_idx)
    return astar(graph, start, goal, inflated_h)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx
    from bsp_parser import BSPParser
    from nav_graph import build_nav_graph

    BASE_DIR = Path(__file__).parent.parent
    bsp_path = BASE_DIR / "data" / "maps" / "e1m1.bsp"

    data  = bsp_path.read_bytes()
    bsp   = BSPParser(data, map_name="e1m1").parse()
    graph = build_nav_graph(bsp)

    # work with main component only
    main  = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(main).copy()

    node_list = list(graph.nodes)
    src  = node_list[0]
    goal = node_list[len(node_list) // 2]

    # build positions dict keyed by actual node index
    positions = {n: graph.nodes[n]['pos'] for n in graph.nodes}

    def h(node_idx, goal_idx):
        return np.linalg.norm(positions[node_idx] - positions[goal_idx])

    path, cost, expanded = astar(graph, src, goal, h)

    print(f"start         {src}")
    print(f"goal          {goal}")
    print(f"path length-   {len(path)} nodes")
    print(f"path cost      {cost:.1f} quake units")
    print(f"nodes expanded {expanded}")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')

    for u, v in graph.edges():
        pu = graph.nodes[u]['pos']
        pv = graph.nodes[v]['pos']
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color='gray', alpha=0.2, linewidth=0.5)

    all_pos = np.array([graph.nodes[n]['pos'] for n in graph.nodes])
    ax.scatter(all_pos[:, 0], all_pos[:, 1], s=6, c='steelblue', zorder=3)

    if path:
        px = [graph.nodes[n]['pos'][0] for n in path]
        py = [graph.nodes[n]['pos'][1] for n in path]
        ax.plot(px, py, color='orange', linewidth=2, zorder=4, label='path')

    ax.scatter(*graph.nodes[src]['pos'][:2],  c='lime', s=150, zorder=5, marker='*', label='start')
    ax.scatter(*graph.nodes[goal]['pos'][:2], c='red',  s=150, zorder=5, marker='X', label='goal')

    ax.set_title(f"a* {expanded} nodes expanded cost {cost:.0f} units", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#333333', labelcolor='white')
    ax.set_aspect('equal')

    plt.tight_layout()
    out = BASE_DIR / "plots" / "e1m1_astar.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nsaved {out}")
    plt.show()