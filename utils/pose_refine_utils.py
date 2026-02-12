import os
import shutil
from typing import Optional

import numpy as np
import torch

from utils.read_write_model import read_model, write_model, qvec2rotmat, rotmat2qvec
from utils.system_utils import mkdir_p


def _detect_ext(source_sparse_dir: str) -> Optional[str]:
    if os.path.isfile(os.path.join(source_sparse_dir, "images.bin")):
        return ".bin"
    if os.path.isfile(os.path.join(source_sparse_dir, "images.txt")):
        return ".txt"
    return None


def export_refined_colmap_model(
    source_sparse_dir: str, out_dir: str, global_transform: torch.Tensor
) -> None:
    """
    Export a COLMAP sparse model with refined extrinsics into `out_dir`.

    This function reads the source model under `source_sparse_dir` (binary or text),
    left-multiplies every image's W2C by `global_transform`, and writes out the
    updated model (preserving tracks and other metadata).
    """
    if not os.path.isdir(source_sparse_dir):
        raise FileNotFoundError(f"COLMAP sparse dir not found: {source_sparse_dir}")

    ext = _detect_ext(source_sparse_dir)
    if ext is None:
        raise FileNotFoundError(
            f"Cannot detect COLMAP model format under: {source_sparse_dir} (missing images.bin/images.txt)"
        )

    mkdir_p(out_dir)

    cameras, images, points3D = read_model(source_sparse_dir, ext=ext)

    if isinstance(global_transform, torch.Tensor):
        g_np = global_transform.detach().float().cpu().numpy()
    else:
        g_np = np.asarray(global_transform, dtype=np.float32)

    if g_np.shape != (4, 4):
        raise ValueError(f"global_transform must be 4x4, got {tuple(g_np.shape)}")

    # Update extrinsics for all images (train/test) with the same global transform.
    new_images = {}
    for image_id, im in images.items():
        r = qvec2rotmat(im.qvec)
        t = np.asarray(im.tvec, dtype=np.float64).reshape(3)

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = r
        w2c[:3, 3] = t

        new_w2c = g_np.astype(np.float64) @ w2c

        qvec_new = rotmat2qvec(new_w2c[:3, :3])
        tvec_new = new_w2c[:3, 3]

        new_images[image_id] = im._replace(qvec=qvec_new, tvec=tvec_new)

    write_model(cameras, new_images, points3D, out_dir, ext=ext)

    # Copy other sidecar files (e.g. project.ini, points3D.ply) without modification.
    reserved = {f"cameras{ext}", f"images{ext}", f"points3D{ext}"}
    for name in os.listdir(source_sparse_dir):
        if name in reserved:
            continue
        src = os.path.join(source_sparse_dir, name)
        dst = os.path.join(out_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # Record the global transform used for exporting.
    gt_path = os.path.join(out_dir, "global_transform.txt")
    with open(gt_path, "w", encoding="utf-8") as f:
        for row in g_np:
            f.write(" ".join([f"{float(x):.8f}" for x in row]) + "\n")

