import numpy as np
import networkx as nx
from pathlib import Path
from bsp_parser import BSPParser

# tunable parameters
WALKABLE_DOT_THRESHOLD = 0.7   # cos(45°), lower = allow steeper ramps
MIN_FACE_AREA          = 8.0   # raise to filter more junk faces
MERGE_RADIUS           = 48.0  # raise = sparser graph, lower = denser
MAX_EDGE_DISTANCE      = 200.0 # max distance between connected nodes
MAX_STEP_HEIGHT        = 200.0 # max upward height per edge, down is always free


def compute_polygon_area(vertices):
    # fan triangulation from v0, sum cross product areas
    if len(vertices) < 3:
        return 0.0
    total = np.zeros(3)
    v0 = vertices[0]
    for i in range(1, len(vertices) - 1):
        total += np.cross(vertices[i] - v0, vertices[i + 1] - v0)
    return np.linalg.norm(total) * 0.5


def is_walkable(face):
    if face.is_special:
        return False

    # dot with up vector: 1.0 = flat floor 0.0 = wall
    up = np.array([0.0, 0.0, 1.0])
    norm = np.linalg.norm(face.normal)
    if norm < 1e-6:
        return False
    if np.dot(face.normal / norm, up) < WALKABLE_DOT_THRESHOLD:
        return False

    if face.vertices is None or len(face.vertices) < 3:
        return False
    if compute_polygon_area(face.vertices) < MIN_FACE_AREA:
        return False

    return True


def place_nodes(walkable_faces):
    if not walkable_faces:
        return np.array([]).reshape(0, 3)

    raw_positions = np.array([f.centroid for f in walkable_faces])
    merged = []
    used = set()

    for i, pos in enumerate(raw_positions):
        if i in used:
            continue

        cluster_positions = [pos]
        cluster_z = [pos[2]]

        for j in range(i + 1, len(raw_positions)):
            if j not in used:
                if np.linalg.norm(pos - raw_positions[j]) < MERGE_RADIUS:
                    cluster_positions.append(raw_positions[j])
                    cluster_z.append(raw_positions[j][2])
                    used.add(j)
        used.add(i)

        # mean XY but min Z so nodes sit on floor not floating mid-cluster
        avg = np.mean(cluster_positions, axis=0)
        avg[2] = min(cluster_z) + 32.0  # 32 = half player height
        merged.append(avg)

    return np.array(merged)


def should_add_edge(node_a, node_b):
    dz = node_b[2] - node_a[2]
    if dz > MAX_STEP_HEIGHT:
        return False  # too steep to walk up
    return True  # falling down always allowed


def build_edges(graph, node_positions):
    n = len(node_positions)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = node_positions[i]
            b = node_positions[j]
            dist = np.linalg.norm(a - b)
            if dist > MAX_EDGE_DISTANCE:
                continue
            if should_add_edge(a, b):
                graph.add_edge(i, j, weight=dist)


def parse_origin(origin_str):
    try:
        parts = origin_str.strip().split()
        if len(parts) != 3:
            return None
        return np.array([float(x) for x in parts])
    except:
        return None


def nearest_node(positions, target, max_dist=256.0):
    if len(positions) == 0:
        return None
    dists = np.linalg.norm(positions - target, axis=1)
    idx = int(np.argmin(dists))
    return idx if dists[idx] <= max_dist else None


def add_teleporter_edges(graph, node_positions, entities):
    triggers     = {e['target']: e for e in entities
                    if e.get('classname') == 'trigger_teleport' and 'target' in e}
    destinations = {e['targetname']: e for e in entities
                    if e.get('classname') == 'info_teleport_destination' and 'targetname' in e}

    added = 0
    for target, trigger in triggers.items():
        if target not in destinations:
            continue
        src_origin = parse_origin(trigger.get('origin', ''))
        dst_origin = parse_origin(destinations[target].get('origin', ''))
        if src_origin is None or dst_origin is None:
            continue
        src_node = nearest_node(node_positions, src_origin)
        dst_node = nearest_node(node_positions, dst_origin)
        if src_node is not None and dst_node is not None:
            graph.add_edge(src_node, dst_node, weight=0.0, teleporter=True)
            added += 1

    if added > 0:
        print(f"  teleporter edges added{added}")


def build_nav_graph(bsp):
    print(f"\nbuilding nav graph for {bsp.map_name}")

    walkable = [f for f in bsp.faces if is_walkable(f)]
    print(f"  walkable faces- {len(walkable)} / {len(bsp.faces)}")

    node_positions = place_nodes(walkable)
    print(f"  nav nodes after merge {len(node_positions)}")

    if len(node_positions) == 0:
        print("   no nodes placed")
        return nx.DiGraph()

    G = nx.DiGraph()
    for i, pos in enumerate(node_positions):
        G.add_node(i, pos=pos)

    build_edges(G, node_positions)
    print(f"  edges: {G.number_of_edges()}")

    add_teleporter_edges(G, node_positions, bsp.entities)

    components = list(nx.weakly_connected_components(G))
    print(f"  connected componentss {len(components)}")
    main = max(components, key=len)
    print(f"  main component {len(main)} / {len(G.nodes)} nodes")

    return G


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    BASE_DIR = Path(__file__).parent.parent
    bsp_path = BASE_DIR / "data" / "maps" / "e1m1.bsp"

    data  = bsp_path.read_bytes()
    bsp   = BSPParser(data, map_name="e1m1").parse()
    graph = build_nav_graph(bsp)

    nodes     = sorted(graph.nodes)
    positions = np.array([graph.nodes[n]['pos'] for n in nodes])

    fig, ax = plt.subplots(figsize=(12, 12))

    for u, v, data in graph.edges(data=True):
        pu = graph.nodes[u]['pos']
        pv = graph.nodes[v]['pos']
        color = 'lime' if data.get('teleporter') else 'gray'
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color=color, alpha=0.3, linewidth=0.5)

    ax.scatter(positions[:, 0], positions[:, 1], s=8, c='steelblue', zorder=5)
    ax.set_title(f"nav graph - e1m1\n{len(nodes)} nodes, {graph.number_of_edges()} edges")
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    ax.title.set_color('white')

    plt.tight_layout()
    out = BASE_DIR / "plots" / "e1m1_nav_graph.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nsaved {out}")
    plt.show()