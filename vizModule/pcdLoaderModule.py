"""
Load PCD (Point Cloud Data) files and render them as GL points.
Supports ASCII and binary PCD with x y z or x y z intensity.
"""
from pathlib import Path
import numpy as np
from OpenGL.GL import (
    glGenVertexArrays, glGenBuffers, glDeleteVertexArrays, glDeleteBuffers,
    glBindVertexArray, glBindBuffer, glBufferData, glVertexAttribPointer,
    glEnableVertexAttribArray, glDrawArrays,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_FALSE,
    GL_POINTS,
)


def _field_index(fields, name_candidates):
    """Return index of first matching field (case-insensitive). Default 0 if not found."""
    fields_lower = [f.lower() for f in fields]
    for n in name_candidates:
        if n in fields_lower:
            return fields_lower.index(n)
    return -1


def load_pcd(filepath: str, max_points: int = 2_000_000):
    """
    Load a PCD file. Returns (xyz, colors) as float32 arrays.
    colors are derived from intensity if present, else gray.
    If point count > max_points, downsample by taking every nth point.
    """
    path = Path(filepath)
    print(f"[PCD] Loading: {path.resolve()}")
    if not path.exists():
        raise FileNotFoundError(f"PCD file not found: {filepath}")
    print(f"[PCD] File exists, size={path.stat().st_size} bytes")

    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip().replace("\r", "")
            header_lines.append(line)
            if line.startswith("DATA"):
                break
        data_start_pos = f.tell()

    print(f"[PCD] Header: {len(header_lines)} lines, data starts at byte {data_start_pos}")
    for i, h in enumerate(header_lines):
        print(f"  {i}: {h!r}")

    # Parse header
    fields = []
    sizes = []
    types = []
    counts = []
    width = height = points = 0
    data_format = "ascii"

    for line in header_lines:
        if line.startswith("FIELDS"):
            fields = line.split()[1:]
        elif line.startswith("SIZE"):
            sizes = [int(x) for x in line.split()[1:]]
        elif line.startswith("TYPE"):
            types = line.split()[1:]
        elif line.startswith("COUNT"):
            counts = [int(x) for x in line.split()[1:]]
        elif line.startswith("WIDTH"):
            width = int(line.split()[1])
        elif line.startswith("HEIGHT"):
            height = int(line.split()[1])
        elif line.startswith("POINTS"):
            points = int(line.split()[1])
        elif line.startswith("DATA"):
            data_format = line.split()[1].lower().strip()

    print(f"[PCD] Parsed: fields={fields}, sizes={sizes}, types={types}, counts={counts}")
    print(f"[PCD] width={width}, height={height}, points={points}, data_format={data_format!r}")

    if "compressed" in data_format:
        raise ValueError("binary_compressed PCD is not supported; save as ASCII or binary (uncompressed)")

    if points == 0 and width > 0 and height > 0:
        points = width * height
        print(f"[PCD] Inferred points from WIDTH*HEIGHT: {points}")

    # Case-insensitive field indices; default to first 3 columns
    x_idx = _field_index(fields, ["x"])
    y_idx = _field_index(fields, ["y"])
    z_idx = _field_index(fields, ["z"])
    if x_idx < 0:
        x_idx = 0
    if y_idx < 0:
        y_idx = 1
    if z_idx < 0:
        z_idx = 2
    i_idx = _field_index(fields, ["intensity", "i"])
    has_intensity = i_idx >= 0
    print(f"[PCD] Indices: x={x_idx}, y={y_idx}, z={z_idx}, intensity={i_idx}, has_intensity={has_intensity}")

    # Stride per point (bytes)
    if counts:
        stride = sum(s * (c if c > 0 else 1) for s, c in zip(sizes, counts))
    else:
        stride = sum(sizes) if sizes else 12
    if stride <= 0:
        stride = 12
    print(f"[PCD] Stride={stride} bytes per point")

    if data_format == "ascii":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in header_lines:
                f.readline()
            lines = f.readlines()
        print(f"[PCD] ASCII: {len(lines)} data lines after header")
        if lines:
            print(f"[PCD] First data line sample: {lines[0][:80]!r}...")
        rows = []
        for line in lines:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])
            except (ValueError, IndexError):
                continue
            if has_intensity and i_idx >= 0 and len(parts) > i_idx:
                try:
                    i = float(parts[i_idx])
                except (ValueError, IndexError):
                    i = 0.5
            else:
                i = 0.5
            rows.append((x, y, z, i))
        data = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)
        print(f"[PCD] ASCII: parsed {len(rows)} rows -> data.shape={data.shape}")
    else:
        with open(path, "rb") as f:
            f.seek(data_start_pos)
            raw = f.read()
        n = len(raw) // stride
        print(f"[PCD] Binary: raw bytes={len(raw)}, stride={stride}, n_points={n}")
        data = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            off = i * stride
            data[i, 0] = np.frombuffer(raw[off:off+4], dtype=np.float32)[0]
            data[i, 1] = np.frombuffer(raw[off+4:off+8], dtype=np.float32)[0]
            data[i, 2] = np.frombuffer(raw[off+8:off+12], dtype=np.float32)[0]
            if has_intensity and stride >= 16:
                data[i, 3] = np.frombuffer(raw[off+12:off+16], dtype=np.float32)[0]
            else:
                data[i, 3] = 0.5
        if n > 0:
            print(f"[PCD] Binary: first point xyz=({data[0,0]:.3f}, {data[0,1]:.3f}, {data[0,2]:.3f})")

    if len(data) == 0:
        raise ValueError(
            "PCD file has 0 points. Check that DATA is 'ascii' or 'binary' (not binary_compressed), "
            "and that the file has x y z fields."
        )

    xyz = data[:, :3].copy()
    intensity = data[:, 3]

    # Color from intensity (gray or white-to-blue like the C++ viewer)
    i_min, i_max = float(np.min(intensity)), float(np.max(intensity))
    if i_max > i_min:
        t = (intensity - i_min) / (i_max - i_min)
    else:
        t = np.ones(len(intensity), dtype=np.float32) * 0.5
    r = 0.3 + t * 0.7
    g = 0.3 + t * 0.7
    b = 0.5 + t * 0.5
    colors = np.column_stack((r, g, b)).astype(np.float32)

    # Downsample
    if len(xyz) > max_points:
        step = (len(xyz) + max_points - 1) // max_points
        idx = np.arange(0, len(xyz), step, dtype=np.intp)[:max_points]
        xyz = xyz[idx]
        colors = colors[idx]
        print(f"[PCD] Downsampled to {len(xyz)} points (max_points={max_points})")

    print(f"[PCD] Done: returning {len(xyz)} points")
    return xyz, colors


class PcdMapLoader:
    """OpenGL renderer for a loaded PCD map (VAO/VBO, draw with GL_POINTS)."""

    def __init__(self):
        self._vao = 0
        self._vbo = 0
        self._cbo = 0
        self._count = 0
        self._loaded = False

    def is_loaded(self):
        return self._loaded and self._count > 0

    def get_point_count(self):
        return self._count

    def load(self, filepath: str, enable_downsampling: bool = True, max_points: int = 2_000_000):
        """Load PCD from file and upload to GPU. Optional downsampling."""
        print(f"[PCD] PcdMapLoader.load: filepath={filepath!r}, downsample={enable_downsampling}, max_points={max_points}")
        xyz, colors = load_pcd(filepath, max_points=max_points if enable_downsampling else 100_000_000)
        n_points = len(xyz)
        print(f"[PCD] PcdMapLoader: got {n_points} points from load_pcd")
        if n_points == 0:
            self._loaded = False
            print("[PCD] PcdMapLoader: 0 points, not uploading to GPU")
            return

        # Clean up existing buffers (_release() sets self._count = 0, so we restore n_points after)
        self._release()
        self._count = n_points

        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        self._cbo = glGenBuffers(1)

        glBindVertexArray(self._vao)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, xyz.nbytes, xyz, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, self._cbo)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._loaded = True
        print(f"[PCD] PcdMapLoader: uploaded {self._count} points to GPU")

    def _release(self):
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
            self._vao = 0
        if self._vbo:
            glDeleteBuffers(1, [self._vbo])
            self._vbo = 0
        if self._cbo:
            glDeleteBuffers(1, [self._cbo])
            self._cbo = 0
        self._count = 0

    def draw(self):
        """Draw the point cloud. Call with point shader already bound and uniforms set."""
        if not self.is_loaded():
            return
        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self._count)
        glBindVertexArray(0)
