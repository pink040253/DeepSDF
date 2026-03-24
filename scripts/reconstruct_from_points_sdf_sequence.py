#!/usr/bin/env python3
"""
Reconstruct meshes from per-touch DeepSDF observations stored under:

    <test_dir>/data/<k>/points_sdf.pt

This script is designed to match the user's current workflow:
    stage1_tactile_collect_mesh_filter.py
    -> prepare_points_sdf_per_touch.py
    -> reconstruct_from_points_sdf_sequence.py

Supported per-sample .pt formats:
1) dict:
    {
        "coords": FloatTensor [N, 3],
        "sdf_targets": FloatTensor [N] or [N, 1]
    }
2) list/tuple:
    [coords, sdf_targets]

Modes:
- single: reconstruct each touch independently
- cumulative: reconstruct using touches [0..k] for step k

Optional:
- warm-start latent code from previous step
- optional finetuning if the loaded model supports it
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
import trimesh
from torch.utils.tensorboard import SummaryWriter

import config_files
from results import runs_sdf
import model.model_sdf as sdf_model
from utils import utils_deepsdf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def numeric_key(path: Path):
    try:
        return int(path.name)
    except ValueError:
        return path.name


def read_shape_completion_cfg():
    cfg_path = os.path.join(os.path.dirname(config_files.__file__), "shape_completion.yaml")
    with open(cfg_path, "rb") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def read_training_settings(folder_sdf: str):
    settings_path = os.path.join(os.path.dirname(runs_sdf.__file__), folder_sdf, "settings.yaml")
    with open(settings_path, "rb") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_points_sdf(pt_path: Path):
    obj = torch.load(pt_path, map_location="cpu")

    if isinstance(obj, dict):
        if "coords" not in obj or "sdf_targets" not in obj:
            raise ValueError(f"{pt_path} dict must contain 'coords' and 'sdf_targets'")
        coords = obj["coords"]
        sdf_targets = obj["sdf_targets"]
    elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
        coords = obj[0]
        sdf_targets = obj[1]
    else:
        raise ValueError(
            f"Unsupported points_sdf format in {pt_path}. "
            f"Expected dict{{coords,sdf_targets}} or [coords, sdf_targets]."
        )

    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.float32)
    if not isinstance(sdf_targets, torch.Tensor):
        sdf_targets = torch.tensor(sdf_targets, dtype=torch.float32)

    coords = coords.float()
    sdf_targets = sdf_targets.float().reshape(-1, 1)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{pt_path}: coords must have shape [N, 3], got {tuple(coords.shape)}")
    if sdf_targets.ndim != 2 or sdf_targets.shape[1] != 1:
        raise ValueError(f"{pt_path}: sdf_targets must have shape [N, 1], got {tuple(sdf_targets.shape)}")
    if coords.shape[0] != sdf_targets.shape[0]:
        raise ValueError(
            f"{pt_path}: coords and sdf_targets length mismatch: "
            f"{coords.shape[0]} vs {sdf_targets.shape[0]}"
        )

    return coords, sdf_targets


def build_sdf_model(model_settings, weights_path: str):
    model = sdf_model.SDFModel(
        num_layers=model_settings["num_layers"],
        skip_connections=model_settings["latent_size"],
        latent_size=model_settings["latent_size"],
        inner_dim=model_settings["inner_dim"],
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def get_average_latent(folder_sdf: str):
    results_path = os.path.join(os.path.dirname(runs_sdf.__file__), folder_sdf, "results.npy")
    results = np.load(results_path, allow_pickle=True).item()
    latent_code = results["best_latent_codes"]
    latent_code = torch.mean(torch.tensor(latent_code, dtype=torch.float32), dim=0).to(device)
    latent_code.requires_grad = True
    return latent_code


def save_mesh(vertices, faces, out_path: Path):
    mesh = trimesh.Trimesh(vertices, faces)
    trimesh.exchange.export.export_mesh(mesh, str(out_path), file_type="obj")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct from per-touch points_sdf observations.")
    parser.add_argument("--test_dir", type=str, required=True, help="Stage1/prepare directory containing data/<k>/points_sdf.pt")
    parser.add_argument("--points_name", type=str, default="points_sdf.pt", help="Per-sample file name. Default: points_sdf.pt")
    parser.add_argument("--mode", choices=["single", "cumulative"], default="cumulative")
    parser.add_argument("--warm_start", action="store_true", help="For cumulative mode, initialize step k from step k-1 best latent")
    parser.add_argument("--finetuning", action="store_true", help="If model has finetune(...), run it after latent inference")
    parser.add_argument("--save_observation_npy", action="store_true", help="Save observation coords/sdf used at each step")
    args = parser.parse_args()

    base_cfg = read_shape_completion_cfg()
    model_settings = read_training_settings(base_cfg["folder_sdf"])

    model_dir = os.path.join(os.path.dirname(runs_sdf.__file__), base_cfg["folder_sdf"])
    weights_path = os.path.join(model_dir, "weights.pt")

    model = build_sdf_model(model_settings, weights_path)

    coords_grid, grad_size_axis = utils_deepsdf.get_volume_coords(base_cfg["resolution"])
    coords_grid = coords_grid.to(device)
    coords_batches = torch.split(coords_grid, 100000)

    test_dir = Path(args.test_dir)
    data_dir = test_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    sample_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()], key=numeric_key)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found in {data_dir}")

    infer_dir = test_dir / f"infer_from_{args.points_name.replace('.pt','')}_{args.mode}"
    infer_dir.mkdir(parents=True, exist_ok=True)

    prev_best_latent = None
    avg_latent_template = get_average_latent(base_cfg["folder_sdf"])

    for idx, sample_dir in enumerate(sample_dirs):
        sample_name = sample_dir.name

        if args.mode == "single":
            selected_dirs = [sample_dir]
        else:
            selected_dirs = sample_dirs[: idx + 1]

        coords_list = []
        sdf_list = []

        for d in selected_dirs:
            pt_path = d / args.points_name
            if not pt_path.exists():
                raise FileNotFoundError(f"Missing {pt_path}")
            coords_i, sdf_i = load_points_sdf(pt_path)
            coords_list.append(coords_i)
            sdf_list.append(sdf_i)

        pointcloud = torch.cat(coords_list, dim=0).float().to(device)
        sdf_gt = torch.cat(sdf_list, dim=0).float().reshape(-1, 1).to(device)

        step_dir = infer_dir / sample_name
        step_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(log_dir=str(step_dir))

        if args.warm_start and args.mode == "cumulative" and prev_best_latent is not None:
            latent_code = prev_best_latent.detach().clone().to(device)
            latent_code.requires_grad = True
        else:
            latent_code = avg_latent_template.detach().clone().to(device)
            latent_code.requires_grad = True

        print("=" * 80)
        print(f"step={sample_name} mode={args.mode}")
        print(f"observation points: {pointcloud.shape[0]}")
        print(f"sdf unique: {torch.unique(sdf_gt.detach().cpu().view(-1))}")

        best_latent_code = model.infer_latent_code(base_cfg, pointcloud, sdf_gt, writer, latent_code)

        if args.finetuning:
            if hasattr(model, "finetune"):
                best_weights = model.finetune(base_cfg, best_latent_code, pointcloud, sdf_gt, writer)
                model.load_state_dict(best_weights)
            else:
                print("Warning: finetuning requested, but model has no finetune(...) method. Skipping.")

        sdf = utils_deepsdf.predict_sdf(best_latent_code, coords_batches, model)
        vertices, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf)

        mesh_path = step_dir / "output_mesh.obj"
        save_mesh(vertices, faces, mesh_path)

        latent_path = step_dir / "latent_code.pt"
        torch.save(best_latent_code.detach().cpu(), latent_path)

        if args.save_observation_npy:
            np.save(step_dir / "observation_coords.npy", pointcloud.detach().cpu().numpy())
            np.save(step_dir / "observation_sdf.npy", sdf_gt.detach().cpu().numpy())

        with open(step_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(f"sample_name: {sample_name}\n")
            f.write(f"mode: {args.mode}\n")
            f.write(f"warm_start: {args.warm_start}\n")
            f.write(f"finetuning: {args.finetuning}\n")
            f.write(f"num_observation_points: {pointcloud.shape[0]}\n")
            f.write(f"sdf_unique: {torch.unique(sdf_gt.detach().cpu().view(-1)).tolist()}\n")
            f.write(f"mesh_path: {mesh_path}\n")
            f.write(f"latent_path: {latent_path}\n")

        prev_best_latent = best_latent_code.detach().clone()

    print(f"Saved reconstructions to: {infer_dir}")


if __name__ == "__main__":
    main()
