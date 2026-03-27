import struct
from pathlib import Path


class PAKReader:
    def __init__(self, pak_path):
        self.pak_path = Path(pak_path)
        self.files = {}
        self._parse()

    def _parse(self):
        with open(self.pak_path, 'rb') as f:

            # read the header
            magic = f.read(4)
            if magic != b'PACK':
                raise ValueError(f"not a valid PAK file {self.pak_path}")

            dir_offset, dir_size = struct.unpack('<ii', f.read(8))
            num_files = dir_size // 64

            # read the file directory
            f.seek(dir_offset)
            for _ in range(num_files):
                raw = f.read(64)
                name = raw[:56].rstrip(b'\x00').decode('latin-1')
                offset, size = struct.unpack('<ii', raw[56:64])
                self.files[name] = (offset, size)

    def list_maps(self):
        return [k for k in self.files if k.startswith('maps/') and k.endswith('.bsp')]

    def extract(self, filename):
        if filename not in self.files:
            raise KeyError(f"{filename} not in PAK")
        offset, size = self.files[filename]
        with open(self.pak_path, 'rb') as f:
            f.seek(offset)
            return f.read(size)

    def extract_all_maps(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        maps = self.list_maps()
        if not maps:
            print("no maps found in this PAK ")
            return

        for map_path in maps:
            data = self.extract(map_path)
            out_path = output_dir / Path(map_path).name
            out_path.write_bytes(data)
            print(f"extracted: {map_path}  ({len(data):,} bytes)")

        print(f"\n extracted {len(maps)} maps to {output_dir}")


if __name__ == "__main__":
    import sys


    PAK0 = r"C:\Users\rahul\OneDrive\Desktop\LEIDEN SEM 2\games\assignment 3\game\id1\PAK0.PAK"
    PAK1 = r"C:\Users\rahul\OneDrive\Desktop\LEIDEN SEM 2\games\assignment 3\game\id1\PAK1.PAK"
    OUTPUT = r"C:\Users\rahul\OneDrive\Desktop\LEIDEN SEM 2\games\assignment 3\data\maps"

    for pak_path in [PAK0, PAK1]:
        if Path(pak_path).exists():
            print(f"\nreading {pak_path}")
            reader = PAKReader(pak_path)
            print(f"total files in PAK, {len(reader.files)}")
            print(f"maps found- {len(reader.list_maps())}")
            reader.extract_all_maps(OUTPUT)
        else:
            print(f"skipping {pak_path} - not found")

