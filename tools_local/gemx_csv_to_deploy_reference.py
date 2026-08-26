import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


# MJ_TO_IL[mj] = il
MJ_TO_IL = np.array(
    [
        0, 3, 6, 9, 13, 17,
        1, 4, 7, 10, 14, 18,
        2, 5, 8, 11, 15, 19,
        21, 23, 25, 27,
        12, 16, 20, 22, 24, 26, 28,
    ],
    dtype=np.int32,
)


def resample_array(x, src_fps, dst_fps):
    if src_fps == dst_fps:
        return x.astype(np.float32)

    t_src = np.arange(len(x), dtype=np.float64) / float(src_fps)
    duration = t_src[-1]
    n_dst = int(np.floor(duration * dst_fps)) + 1
    t_dst = np.arange(n_dst, dtype=np.float64) / float(dst_fps)

    f = interp1d(t_src, x, axis=0, kind="linear", fill_value="extrapolate")
    return f(t_dst).astype(np.float32)


def resample_quat_xyzw(q_xyzw, src_fps, dst_fps):
    if src_fps == dst_fps:
        return q_xyzw.astype(np.float32)

    t_src = np.arange(len(q_xyzw), dtype=np.float64) / float(src_fps)
    duration = t_src[-1]
    n_dst = int(np.floor(duration * dst_fps)) + 1
    t_dst = np.arange(n_dst, dtype=np.float64) / float(dst_fps)

    rots = Rotation.from_quat(q_xyzw)
    slerp = Slerp(t_src, rots)
    return slerp(t_dst).as_quat().astype(np.float32)


def save_csv(path, arr, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in arr:
            f.write(",".join(f"{float(v):.8f}" for v in row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--src_fps", type=float, default=30.0)
    parser.add_argument("--dst_fps", type=float, default=50.0)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(input_csv)
    T = len(df)

    # Root position: GEM-X/SOMA Retargeter CSV is cm, deploy reference wants meters.
    root_pos = np.stack(
        [
            df["root_translateX"].values,
            df["root_translateY"].values,
            df["root_translateZ"].values,
        ],
        axis=1,
    ).astype(np.float32) / 100.0

    # Root rotation: GEM-X CSV uses Euler xyz degrees.
    euler_deg = np.stack(
        [
            df["root_rotateX"].values,
            df["root_rotateY"].values,
            df["root_rotateZ"].values,
        ],
        axis=1,
    ).astype(np.float64)
    root_quat_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)

    # Joint position: CSV joint columns are in MuJoCo/MJCF order, degrees.
    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    if len(joint_cols) != 29:
        raise RuntimeError(f"Expected 29 joint columns, got {len(joint_cols)}")

    joint_mj = np.deg2rad(df[joint_cols].values).astype(np.float32)

    # Convert MJ order → IsaacLab order for deploy reference.
    joint_il = np.zeros_like(joint_mj)
    joint_il[:, MJ_TO_IL] = joint_mj

    # Resample all data to 50 Hz, as C++ reference docs expect 50 Hz.
    root_pos = resample_array(root_pos, args.src_fps, args.dst_fps)
    root_quat_xyzw = resample_quat_xyzw(root_quat_xyzw, args.src_fps, args.dst_fps)
    joint_il = resample_array(joint_il, args.src_fps, args.dst_fps)

    # Joint velocity rad/s.
    dt = 1.0 / float(args.dst_fps)
    joint_vel = np.gradient(joint_il, dt, axis=0).astype(np.float32)

    # body_pos.csv root only.
    body_pos = root_pos.astype(np.float32)

    # body_quat.csv root only, wxyz.
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)

    save_csv(
        output_dir / "joint_pos.csv",
        joint_il,
        [f"joint_{i}" for i in range(29)],
    )
    save_csv(
        output_dir / "joint_vel.csv",
        joint_vel,
        [f"joint_vel_{i}" for i in range(29)],
    )
    save_csv(
        output_dir / "body_pos.csv",
        body_pos,
        ["body_0_x", "body_0_y", "body_0_z"],
    )
    save_csv(
        output_dir / "body_quat.csv",
        root_quat_wxyz,
        ["body_0_w", "body_0_x", "body_0_y", "body_0_z"],
    )

    with open(output_dir / "metadata.txt", "w") as f:
        f.write(f"Metadata for: {output_dir.name}\n")
        f.write("==============================\n\n")
        f.write("Body part indexes:\n")
        f.write("[0]\n\n")
        f.write(f"Total timesteps: {len(joint_il)}\n")

    with open(output_dir / "info.txt", "w") as f:
        f.write(f"source: {input_csv}\n")
        f.write(f"src_fps: {args.src_fps}\n")
        f.write(f"dst_fps: {args.dst_fps}\n")
        f.write(f"frames: {len(joint_il)}\n")

    print("Saved deploy reference motion to:", output_dir)


if __name__ == "__main__":
    main()
