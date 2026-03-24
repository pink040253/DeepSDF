#!/usr/bin/env python3
"""
Browse a sequence of OBJ meshes saved under subfolders, one by one.

Example:
    python inspect_mesh_sequence.py \
        --root_dir /path/to/test_dir/infer_from_points_sdf_cumulative \
        --mesh_name output_mesh.obj

Optional reference overlay:
    python inspect_mesh_sequence.py \
        --root_dir /path/to/test_dir/infer_from_points_sdf_cumulative \
        --mesh_name output_mesh.obj \
        --reference /path/to/mesh_deepsdf.obj

Controls:
    Right / n / space : next mesh
    Left  / p         : previous mesh
    q / Esc           : quit
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def numeric_key(path: Path):
    try:
        return int(path.name)
    except ValueError:
        return path.name


def set_axes_equal(ax, vertices: np.ndarray):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if radius <= 0:
        radius = 1e-3
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected mesh at {path}, got {type(mesh)}")
    return mesh


def mesh_stats(mesh: trimesh.Trimesh) -> str:
    v = np.asarray(mesh.vertices)
    mins = v.min(axis=0)
    maxs = v.max(axis=0)
    return (
        f"verts={len(mesh.vertices)}, faces={len(mesh.faces)} | "
        f"min=({mins[0]:.3f}, {mins[1]:.3f}, {mins[2]:.3f}) "
        f"max=({maxs[0]:.3f}, {maxs[1]:.3f}, {maxs[2]:.3f})"
    )


def add_mesh(ax, mesh: trimesh.Trimesh, alpha: float, linewidth: float):
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    tris = verts[faces]
    coll = Poly3DCollection(
        tris,
        alpha=alpha,
        linewidths=linewidth,
        edgecolor="k" if linewidth > 0 else None,
    )
    ax.add_collection3d(coll)
    return coll


def main():
    parser = argparse.ArgumentParser(description="Browse OBJ meshes in subfolders one by one.")
    parser.add_argument("--root_dir", required=True, help="Folder containing subfolders such as 0,1,2,...")
    parser.add_argument("--mesh_name", default="output_mesh.obj", help="Mesh filename inside each subfolder")
    parser.add_argument("--reference", default=None, help="Optional reference mesh to overlay")
    parser.add_argument("--start", type=int, default=0, help="Start index in sorted mesh list")
    parser.add_argument("--mesh_alpha", type=float, default=0.65, help="Alpha for current mesh")
    parser.add_argument("--ref_alpha", type=float, default=0.18, help="Alpha for reference mesh")
    parser.add_argument("--show_edges", action="store_true", help="Draw mesh edges")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Missing root_dir: {root_dir}")

    mesh_entries = []
    for sub in sorted([p for p in root_dir.iterdir() if p.is_dir()], key=numeric_key):
        mesh_path = sub / args.mesh_name
        if mesh_path.exists():
            mesh_entries.append((sub.name, mesh_path))

    if not mesh_entries:
        raise RuntimeError(f"No meshes named '{args.mesh_name}' found under {root_dir}")

    ref_mesh = load_mesh(Path(args.reference)) if args.reference else None
    idx = max(0, min(args.start, len(mesh_entries) - 1))

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    def draw():
        ax.cla()

        folder_name, mesh_path = mesh_entries[idx]
        mesh = load_mesh(mesh_path)

        linewidth = 0.15 if args.show_edges else 0.0

        verts_all = [np.asarray(mesh.vertices)]
        if ref_mesh is not None:
            add_mesh(ax, ref_mesh, alpha=args.ref_alpha, linewidth=0.0)
            verts_all.append(np.asarray(ref_mesh.vertices))

        add_mesh(ax, mesh, alpha=args.mesh_alpha, linewidth=linewidth)

        stacked = np.vstack(verts_all)
        set_axes_equal(ax, stacked)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"[{idx+1}/{len(mesh_entries)}] folder={folder_name}\n"
            f"{mesh_stats(mesh)}"
        )
        fig.canvas.draw_idle()

        print("=" * 80)
        print(f"[{idx+1}/{len(mesh_entries)}] folder={folder_name}")
        print("mesh:", mesh_path)
        print(mesh_stats(mesh))

    def on_key(event):
        nonlocal idx
        if event.key in ["right", "n", " "]:
            idx = min(idx + 1, len(mesh_entries) - 1)
            draw()
        elif event.key in ["left", "p"]:
            idx = max(idx - 1, 0)
            draw()
        elif event.key in ["q", "escape"]:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
