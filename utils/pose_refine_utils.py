import os
import shutil
from typing import Optional

import numpy as np

from utils.read_write_model import read_model, rotmat2qvec, write_model
from utils.system_utils import mkdir_p


def _detect_ext(source_sparse_dir: str) -> Optional[str]:
    if os.path.isfile(os.path.join(source_sparse_dir, "images.bin")):
        return ".bin"
    if os.path.isfile(os.path.join(source_sparse_dir, "images.txt")):
        return ".txt"
    return None


def _cam_to_colmap_w2c(camera):
    # camera.R stores C2W rotation in this codebase, so W2C rotation is R^T.
    r_w2c = camera.R.detach().float().cpu().numpy().T.astype(np.float64)
    t_w2c = camera.T.detach().float().cpu().numpy().astype(np.float64).reshape(3)
    return r_w2c, t_w2c


def _reset_output_dir(out_dir: str) -> None:
    mkdir_p(out_dir)
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def export_refined_colmap_model(
    source_sparse_dir: str, out_dir: str, refined_train_cameras
) -> None:
    """
    Export a refined COLMAP sparse model to `out_dir` with fixed `.bin` outputs.
    Only images matched by `image_name` in `refined_train_cameras` are updated.
    """
    if not os.path.isdir(source_sparse_dir):
        raise FileNotFoundError(f"COLMAP sparse dir not found: {source_sparse_dir}")

    ext = _detect_ext(source_sparse_dir)
    if ext is None:
        raise FileNotFoundError(
            f"Cannot detect COLMAP model format under: {source_sparse_dir} (missing images.bin/images.txt)"
        )

    _reset_output_dir(out_dir)
    cameras, images, points3D = read_model(source_sparse_dir, ext=ext)

    cam_by_name = {cam.image_name: cam for cam in refined_train_cameras}
    new_images = {}
    for image_id, im in images.items():
        matched_cam = cam_by_name.get(im.name, None)
        if matched_cam is None:
            new_images[image_id] = im
            continue

        r_w2c, t_w2c = _cam_to_colmap_w2c(matched_cam)
        qvec_new = rotmat2qvec(r_w2c).astype(np.float64)
        tvec_new = np.asarray(t_w2c, dtype=np.float64).reshape(3)
        new_images[image_id] = im._replace(qvec=qvec_new, tvec=tvec_new)

    # Always export as COLMAP binary files.
    write_model(cameras, new_images, points3D, out_dir, ext=".bin")
