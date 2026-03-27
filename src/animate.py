import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from bsp_parser import BSPParser
from nav_graph import build_nav_graph
from astar import astar
from learned_heuristics import LearnedHeuristic
from xg_heuristic import XGBoostHeuristic

# tunable parameters
MAP_NAME   = "e1m1"
RNG_SEED   = 99       # changeto get different start/goal pairs
INTERVAL   = 50       # milliseconds between frames
MIN_DIST   = 800.0    # minimum distance between src and goal t


def astar_with_history(graph, start, goal, heuristic):
    # same as astar but records expansion order for animation
    import heapq

    open_heap  = []
    heapq.heappush(open_heap, (0.0, 0.0, start))
    came_from  = {}
    g_score    = {start: 0.0}
    closed     = set()
    expansion_order = []  # track order nodes were expanded

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expansion_order.append(current)

        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path)), g_score[goal], expansion_order

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

    return None, float('inf'), expansion_order


def find_dramatic_pair(node_list, node_positions, min_dist, seed):
    # find the pair of nodes furthest apart for a dramatic animation
    rng       = np.random.default_rng(seed)
    best_pair = None
    best_dist = 0
    for _ in range(500):
        s = node_list[rng.integers(len(node_list))]
        g = node_list[rng.integers(len(node_list))]
        if s == g:
            continue
        d = np.linalg.norm(node_positions[s] - node_positions[g])
        if d > best_dist and d >= min_dist:
            best_dist = d
            best_pair = (s, g)
    return best_pair


if __name__ == "__main__":
    print(f"loading {MAP_NAME}...")
    bsp_path = BASE_DIR / "data" / "maps" / f"{MAP_NAME}.bsp"
    data     = bsp_path.read_bytes()
    bsp      = BSPParser(data, map_name=MAP_NAME).parse()
    graph    = build_nav_graph(bsp)

    main  = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(main).copy()

    node_list      = list(graph.nodes)
    node_positions = {n: graph.nodes[n]['pos'] for n in graph.nodes}
    all_pos        = np.array([node_positions[n] for n in graph.nodes])
    all_nodes      = list(graph.nodes)

    # find a dramatic src/goal pair
    pair = find_dramatic_pair(node_list, node_positions, MIN_DIST, RNG_SEED)
    if pair is None:
        print("no pair found with min dist, using fallback")
        pair = (node_list[0], node_list[len(node_list) // 2])
    src, goal = pair

    print(f"src: {src}  goal: {goal}")
    print(f"distance {np.linalg.norm(node_positions[src] - node_positions[goal]):.0f} units")

    # set up heuristics
    def euclid_h(n, g):
        return float(np.linalg.norm(node_positions[n] - node_positions[g]))

    def zero_h(n, g):
        return 0.0

    mlp_h = LearnedHeuristic(graph, node_positions)
    xgb_h = XGBoostHeuristic(graph, node_positions)

    # run all 4 methods and collect expansion histories
    print("running dijkstra.")
    path_d, cost_d, history_d = astar_with_history(graph, src, goal, zero_h)
    print(f"  expanded: {len(history_d)}, cost: {cost_d:.0f}")

    print("running euclidean a*")
    path_e, cost_e, history_e = astar_with_history(graph, src, goal, euclid_h)
    print(f"  expanded: {len(history_e)}, cost: {cost_e:.0f}")

    print("running mlp a*")
    path_m, cost_m, history_m = astar_with_history(graph, src, goal, mlp_h)
    print(f"  expanded: {len(history_m)}, cost: {cost_m:.0f}")

    print("running xgboost a*")
    xgb_h.clear_cache()
    path_x, cost_x, history_x = astar_with_history(graph, src, goal, xgb_h)
    print(f"  expanded: {len(history_x)}, cost: {cost_x:.0f}")

    # pad all histories to same length for synchronized animation
    max_frames = max(len(history_d), len(history_e), len(history_m), len(history_x))
    print(f"\nmax frames: {max_frames}")

    def pad(h):
        return h + [h[-1]] * (max_frames - len(h))

    history_d = pad(history_d)
    history_e = pad(history_e)
    history_m = pad(history_m)
    history_x = pad(history_x)

    # set up the figure - 4 panels side by side
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(f"a* search comparison on {MAP_NAME}  |  src to goal  ({np.linalg.norm(node_positions[src] - node_positions[goal]):.0f} units apart)",
                 color='white', fontsize=12)

    titles  = ['dijkstra\n(optimal reference)', 'euclidean a*', 'mlp a*', 'xgboost a*']
    paths   = [path_d, path_e, path_m, path_x]
    histories = [history_d, history_e, history_m, history_x]
    costs   = [cost_d, cost_e, cost_m, cost_x]
    colors  = ['mediumpurple', 'gray', 'steelblue', 'coral']

    # draw static background for each panel
    for ax, title, path, cost in zip(axes, titles, paths, costs):
        ax.set_facecolor('#1a1a1a')

        # draw all edges faintly
        for u, v in graph.edges():
            pu = node_positions[u]
            pv = node_positions[v]
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]],
                    color='#333333', alpha=0.4, linewidth=0.3)

        # draw all nodes
        ax.scatter(all_pos[:, 0], all_pos[:, 1],
                   s=4, c='#555555', zorder=2)

        # start and goal markers
        ax.scatter(*node_positions[src][:2],
                   c='lime', s=200, zorder=6, marker='*')
        ax.scatter(*node_positions[goal][:2],
                   c='red', s=200, zorder=6, marker='X')

        ax.set_aspect('equal')
        ax.tick_params(colors='white')
        ax.set_title(title, color='white', fontsize=9)

    # animated elements  expanded nodes and path line
    scat_list = []
    path_list = []
    text_list = []

    for ax, color, path, history, cost in zip(axes, colors, paths, histories, costs):
        scat = ax.scatter([], [], s=12, c=color, alpha=0.6, zorder=3)
        line, = ax.plot([], [], color='orange', linewidth=2, zorder=5)
        text  = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        color='white', fontsize=8, va='top',
                        bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
        scat_list.append(scat)
        path_list.append(line)
        text_list.append(text)

    def update(frame):
        artists = []
        for i, (scat, line, text, history, path, cost, color) in enumerate(
                zip(scat_list, path_list, text_list, histories, paths, costs, colors)):

            # nodes expanded so far
            expanded_so_far = history[:frame + 1]
            unique_expanded = list(dict.fromkeys(expanded_so_far))  # preserve order, remove dups

            if unique_expanded:
                exp_pos = np.array([node_positions[n] for n in unique_expanded])
                scat.set_offsets(exp_pos[:, :2])

            # show path once search is done
            done = (frame >= len(set(history)) - 1) or (history[frame] == goal)
            if done and path:
                px = [node_positions[n][0] for n in path]
                py = [node_positions[n][1] for n in path]
                line.set_data(px, py)

            # update counter text
            n_expanded = len(unique_expanded)
            text.set_text(f"expanded: {n_expanded}\ncost: {cost:.0f}")

            artists.extend([scat, line, text])

        return artists

    print("building animation")
    ani = animation.FuncAnimation(
        fig, update,
        frames=max_frames,
        interval=INTERVAL,
        blit=True
    )

    # save as gif
    out_gif = BASE_DIR / "plots" / f"{MAP_NAME}_search_animation.gif"
    print(f"saving gif to {out_gif} ")
    ani.save(str(out_gif), writer='pillow', fps=30,
             savefig_kwargs={'facecolor': '#1a1a1a'})
    print(f"saved {out_gif}")

    plt.show()