#!/usr/bin/env python3
"""
Cut the middle band of a normalized bottle mesh and save it in the same format as
completion_data_stage1_gt.pt:

    {
        "coords": FloatTensor [N, 3],
        "sdf_targets": FloatTensor [N]
    }

Default behavior:
- Loads a mesh such as model_normalized.obj
- Detects the longest axis (usually bottle height)
- Uniformly samples surface points
- Keeps only the middle band along that axis
- Saves a .pt file with sdf_targets = 0 for all kept points

Optional:
- Also saves .npy / .ply debug outputs for quick visualization
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def choose_axis(vertices: np.ndarray, axis: str) -> int:
    if axis in AXIS_TO_INDEX:
        return AXIS_TO_INDEX[axis]

    extents = vertices.max(axis=0) - vertices.min(axis=0)
    return int(np.argmax(extents))


def sample_middle_band(
    mesh: trimesh.Trimesh,
    axis_idx: int,
    band_center_frac: float,
    band_width_frac: float,
    target_points: int,
    seed: int,
    max_rounds: int = 8,
) -> np.ndarray:
    """
    Sample mesh surface points uniformly, then keep only points within a middle band.

    band_center_frac:
        0.0 means min along axis, 1.0 means max along axis
    band_width_frac:
        fraction of full axis extent to keep
    """
    rng = np.random.default_rng(seed)
    vmin = mesh.vertices.min(axis=0)
    vmax = mesh.vertices.max(axis=0)
    extent = vmax - vmin

    axis_min = float(vmin[axis_idx])
    axis_max = float(vmax[axis_idx])
    axis_extent = float(extent[axis_idx])

    center_value = axis_min + band_center_frac * axis_extent
    half_width = 0.5 * band_width_frac * axis_extent
    lower = center_value - half_width
    upper = center_value + half_width

    kept = []
    current_sample_count = max(target_points * 8, 20000)

    for _ in range(max_rounds):
        # trimesh uses NumPy's global RNG internally in some paths, so seed globally here
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        points, _ = trimesh.sample.sample_surface(mesh, current_sample_count)

        mask = (points[:, axis_idx] >= lower) & (points[:, axis_idx] <= upper)
        band_points = points[mask]
        if len(band_points) > 0:
            kept.append(band_points)

        total = sum(len(x) for x in kept)
        if total >= target_points:
            break

        current_sample_count *= 2

    if not kept:
        raise RuntimeError(
            "No points were found in the requested middle band. "
            "Try increasing --band_width_frac or checking the axis."
        )

    pts = np.concatenate(kept, axis=0)

    # Deduplicate a little so repeated oversampling does not create too many near-identical points
    pts = np.unique(np.round(pts, decimals=6), axis=0).astype(np.float32)

    if len(pts) > target_points:
        idx = rng.choice(len(pts), size=target_points, replace=False)
        pts = pts[idx]

    return pts


def save_ply(points: np.ndarray, path: Path) -> None:
    cloud = trimesh.points.PointCloud(points)
    cloud.export(path)


def main():
    parser = argparse.ArgumentParser(
        description="Cut the middle region of a bottle mesh and save it as a DeepSDF-style .pt observation file."
    )
    parser.add_argument("--mesh", required=True, help="Input mesh file, e.g. model_normalized.obj")
    parser.add_argument("--output_pt", required=True, help="Output .pt path")
    parser.add_argument(
        "--axis",
        default="auto",
        choices=["auto", "x", "y", "z"],
        help="Axis used to define the middle band. Default: auto = longest axis.",
    )
    parser.add_argument(
        "--band_center_frac",
        type=float,
        default=0.5,
        help="Band center as fraction along axis extent. 0=min end, 1=max end. Default: 0.5",
    )
    parser.add_argument(
        "--band_width_frac",
        type=float,
        default=0.25,
        help="Band width as fraction of full axis extent. Default: 0.25",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=4000,
        help="Number of kept surface points in the output. Default: 4000",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_npy", action="store_true", help="Also save coords as .npy")
    parser.add_argument("--save_ply", action="store_true", help="Also save point cloud as .ply")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    output_pt = Path(args.output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a mesh, got {type(mesh)}")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    axis_idx = choose_axis(vertices, args.axis)
    axis_name = ["x", "y", "z"][axis_idx]

    points = sample_middle_band(
        mesh=mesh,
        axis_idx=axis_idx,
        band_center_frac=args.band_center_frac,
        band_width_frac=args.band_width_frac,
        target_points=args.num_points,
        seed=args.seed,
    )

    sdf_targets = np.zeros((len(points),), dtype=np.float32)

    torch.save(
        {
            "coords": torch.from_numpy(points).float(),
            "sdf_targets": torch.from_numpy(sdf_targets).float(),
        },
        output_pt,
    )

    if args.save_npy:
        np.save(output_pt.with_suffix(".npy"), points)

    if args.save_ply:
        save_ply(points, output_pt.with_suffix(".ply"))

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = vmax - vmin
    axis_min = float(vmin[axis_idx])
    axis_max = float(vmax[axis_idx])
    axis_extent = float(extent[axis_idx])
    center_value = axis_min + args.band_center_frac * axis_extent
    half_width = 0.5 * args.band_width_frac * axis_extent
    lower = center_value - half_width
    upper = center_value + half_width

    print("Saved:", output_pt)
    print("Mesh bounds min:", vmin)
    print("Mesh bounds max:", vmax)
    print("Chosen axis:", axis_name)
    print(f"Band along {axis_name}: [{lower:.6f}, {upper:.6f}]")
    print("Output coords shape:", points.shape)
    print("Output coords min:", points.min(axis=0))
    print("Output coords max:", points.max(axis=0))
    print("Output centroid:", points.mean(axis=0))


if __name__ == "__main__":
    main()
