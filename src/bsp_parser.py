import struct
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# the BSP version we expect - quake 1 is always 29
BSP_VERSION = 29

# lump indices  these tell where each chunk of data lives in the file
LUMP_ENTITIES     = 0
LUMP_PLANES       = 1
LUMP_TEXTURES     = 2
LUMP_VERTICES     = 3
LUMP_VISIBILITY   = 4
LUMP_NODES        = 5
LUMP_TEXINFO      = 6
LUMP_FACES        = 7
LUMP_LIGHTING     = 8
LUMP_CLIPNODES    = 9
LUMP_LEAVES       = 10
LUMP_MARKSURFACES = 11
LUMP_EDGES        = 12
LUMP_SURFEDGES    = 13
LUMP_MODELS       = 14

# what each leaf contains tells  if it's air, water, lava etc
CONTENTS_EMPTY = -1
CONTENTS_SOLID = -2
CONTENTS_WATER = -3
CONTENTS_SLIME = -4
CONTENTS_LAVA  = -5
CONTENTS_SKY   = -6

# texinfo flag - sky and liquid surfaces are marked with this
TEXINFO_SPECIAL = 1


@dataclass
class Face:
    plane_idx: int
    side: int
    surfedge_start: int
    num_edges: int
    texinfo_idx: int
    vertices: np.ndarray = field(default=None)
    normal: np.ndarray = field(default=None)
    centroid: np.ndarray = field(default=None)
    is_special: bool = False   # True = sky or liquid, not walkable


@dataclass
class Leaf:
    contents: int
    visofs: int
    mins: Tuple
    maxs: Tuple
    first_mark: int
    num_marks: int
    face_indices: List[int] = field(default_factory=list)
    centroid: np.ndarray = None


@dataclass
class BSPData:
    vertices: np.ndarray
    planes: List
    faces: List[Face]
    leaves: List[Leaf]
    entities: List[Dict]
    map_name: str = ""


class BSPParser:
    def __init__(self, data: bytes, map_name: str = ""):
        self.data = data
        self.map_name = map_name

    def parse(self) -> BSPData:
        # first 4 bytes are the version number
        version = struct.unpack_from('<i', self.data, 0)[0]
        if version != BSP_VERSION:
            raise ValueError(f"{self.map_name}: expected BSP v{BSP_VERSION}, got v{version}")

        # read the lump directory - 15 lumps, each with offset + size
        lumps = []
        for i in range(15):
            offset = 4 + i * 8
            lump_offset, lump_size = struct.unpack_from('<ii', self.data, offset)
            lumps.append((lump_offset, lump_size))

        # parse each section we actually need
        vertices     = self._parse_vertices(lumps[LUMP_VERTICES])
        planes       = self._parse_planes(lumps[LUMP_PLANES])
        texinfo      = self._parse_texinfo(lumps[LUMP_TEXINFO])
        edges        = self._parse_edges(lumps[LUMP_EDGES])
        surfedges    = self._parse_surfedges(lumps[LUMP_SURFEDGES])
        raw_faces    = self._parse_faces_raw(lumps[LUMP_FACES])
        marksurfaces = self._parse_marksurfaces(lumps[LUMP_MARKSURFACES])
        leaves       = self._parse_leaves(lumps[LUMP_LEAVES], marksurfaces)
        entities     = self._parse_entities(lumps[LUMP_ENTITIES])

        # now build the actual face objects with geometry attached
        faces = self._build_faces(raw_faces, vertices, edges, surfedges, planes, texinfo)

        # attach face lists and centroids to leaves
        for leaf in leaves:
            leaf.face_indices = list(range(leaf.first_mark, leaf.first_mark + leaf.num_marks))
            leaf.centroid = np.array([
                (leaf.mins[0] + leaf.maxs[0]) / 2,
                (leaf.mins[1] + leaf.maxs[1]) / 2,
                (leaf.mins[2] + leaf.maxs[2]) / 2,
            ], dtype=float)

        return BSPData(
            vertices=vertices,
            planes=planes,
            faces=faces,
            leaves=leaves,
            entities=entities,
            map_name=self.map_name
        )

    def _parse_vertices(self, lump):
        offset, size = lump
        n = size // 12  # each vertex is 3 floats = 12 bytes
        verts = np.frombuffer(self.data[offset:offset + size], dtype='<f4').reshape(n, 3)
        return verts.astype(np.float64)

    def _parse_planes(self, lump):
        offset, size = lump
        n = size // 20  # each plane is 20 bytes
        planes = []
        for i in range(n):
            pos = offset + i * 20
            nx, ny, nz, dist, ptype = struct.unpack_from('<ffffi', self.data, pos)
            planes.append((np.array([nx, ny, nz]), dist, ptype))
        return planes

    def _parse_texinfo(self, lump):
        offset, size = lump
        n = size // 40  # each texinfo is 40 bytes
        texinfos = []
        for i in range(n):
            pos = offset + i * 40
            vals = struct.unpack_from('<8f2i', self.data, pos)
            flags = vals[9]
            texinfos.append({
                'flags': flags,
                'is_special': bool(flags & TEXINFO_SPECIAL)
            })
        return texinfos

    def _parse_edges(self, lump):
        offset, size = lump
        n = size // 4  # each edge is 2 uint16s = 4 bytes
        return np.frombuffer(self.data[offset:offset + size], dtype='<u2').reshape(n, 2)

    def _parse_surfedges(self, lump):
        offset, size = lump
        n = size // 4
        return np.frombuffer(self.data[offset:offset + size], dtype='<i4')

    def _parse_faces_raw(self, lump):
        offset, size = lump
        n = size // 20
        faces = []
        for i in range(n):
            pos = offset + i * 20
            plane_idx, side, surfedge_start, num_edges, texinfo_idx = struct.unpack_from('<HHiHH', self.data, pos)
            faces.append((plane_idx, side, surfedge_start, num_edges, texinfo_idx))
        return faces

    def _parse_marksurfaces(self, lump):
        offset, size = lump
        n = size // 2
        return np.frombuffer(self.data[offset:offset + size], dtype='<u2')

    def _parse_leaves(self, lump, marksurfaces):
        offset, size = lump
        n = size // 28  # each leaf is 28 bytes
        leaves = []
        for i in range(n):
            pos = offset + i * 28
            contents, visofs = struct.unpack_from('<ii', self.data, pos)
            mins = struct.unpack_from('<3h', self.data, pos + 8)
            maxs = struct.unpack_from('<3h', self.data, pos + 14)
            first_mark, num_marks = struct.unpack_from('<HH', self.data, pos + 20)
            leaves.append(Leaf(
                contents=contents,
                visofs=visofs,
                mins=mins,
                maxs=maxs,
                first_mark=first_mark,
                num_marks=num_marks
            ))
        return leaves

    def _parse_entities(self, lump):
        offset, size = lump
        raw = self.data[offset:offset + size].decode('ascii', errors='replace')
        entities = []
        current = {}
        for line in raw.splitlines():
            line = line.strip()
            if line == '{':
                current = {}
            elif line == '}':
                if current:
                    entities.append(current)
                current = {}
            elif line.startswith('"'):
                parts = line.split('"')
                if len(parts) >= 4:
                    key = parts[1]
                    val = parts[3]
                    current[key] = val
        return entities

    def _build_faces(self, raw_faces, vertices, edges, surfedges, planes, texinfo):
        faces = []
        for plane_idx, side, surfedge_start, num_edges, texinfo_idx in raw_faces:

            # reconstruct the vertices for this face using the surfedge list
            verts = []
            for i in range(num_edges):
                se = surfedges[surfedge_start + i]
                if se >= 0:
                    v0 = edges[se][0]
                else:
                    v0 = edges[-se][1]
                verts.append(vertices[v0])
            verts = np.array(verts)

            # get the plane normal and flip it if the face is on the back side
            plane_normal = planes[plane_idx][0]
            normal = np.array(plane_normal)
            if side:
                normal = -normal

            centroid = verts.mean(axis=0)
            is_special = texinfo[texinfo_idx]['is_special']

            faces.append(Face(
                plane_idx=plane_idx,
                side=side,
                surfedge_start=surfedge_start,
                num_edges=num_edges,
                texinfo_idx=texinfo_idx,
                vertices=verts,
                normal=normal,
                centroid=centroid,
                is_special=is_special
            ))
        return faces


# check on one file if jusr this is run
if __name__ == "__main__":
    from pathlib import Path


    BASE_DIR = Path(__file__).parent.parent
    bsp_path = BASE_DIR / "data" / "maps" / "e1m1.bsp"

    if not bsp_path.exists():
        print(f"e1m1.bsp not found at {bsp_path}")
    else:
        data = bsp_path.read_bytes()
        parser = BSPParser(data, map_name="e1m1")
        bsp = parser.parse()

        print(f"map:      {bsp.map_name}")
        print(f"vertices: {len(bsp.vertices)}")
        print(f"faces:    {len(bsp.faces)}")
        print(f"leaves:   {len(bsp.leaves)}")
        print(f"entities: {len(bsp.entities)}")

        print("\nfirst 5 entities ")
        for e in bsp.entities[:5]:
            print(f"  {e.get('classname', '???')}")