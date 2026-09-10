"""读取 BUMI3 sim2sim 的多轨迹数据集清单。

数据集使用 JSON 或 YAML 保存有序的 ``motions`` 列表，每项提供唯一 ``name``、
机器人动作路径 ``robot``，以及可选的配对 ``smpl`` 和容器内 ``motion_key``。
相对路径以清单所在目录为基准，不依赖启动程序时的工作目录。SMPL 供 Isaac Lab
评估使用，MuJoCo Robot Encoder 只读取 robot；这里仍检查声明的配对文件存在，
避免清单引用丢失文件。动作数值、帧率和关节顺序由既有 BUMI3 加载器统一校验。
本模块只读原始数据，不执行仿真、不重采样、不改写 root 高度或源文件。
"""

from dataclasses import replace
from pathlib import Path

import yaml

from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    ReferenceMotion,
    load_reference_motion,
)


def load_motion_dataset(path: str | Path, contract: Bumi3Contract) -> list[ReferenceMotion]:
    """按清单顺序加载所有轨迹；格式、重复名称或缺失路径均在打开窗口前报错。"""

    path = Path(path).expanduser().resolve()
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or content.get("version") != 1:
        raise ValueError("数据集清单必须是含 version: 1 的 JSON/YAML 对象")
    if content.get("robot_type") != "bumi3":
        raise ValueError("数据集 robot_type 必须为 bumi3")
    entries = content.get("motions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("数据集 motions 必须为非空列表")
    names = set()
    motions = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"motions[{index}] 必须为对象")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError(f"motions[{index}] 的 name 必须非空且不能重复")
        names.add(name)
        paths = {}
        for field in ("robot", "smpl"):
            value = entry.get(field)
            if field == "smpl" and value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"轨迹 {name} 缺少有效的 {field} 路径")
            target = Path(value).expanduser()
            target = (path.parent / target).resolve() if not target.is_absolute() else target.resolve()
            if not target.exists():
                raise FileNotFoundError(f"轨迹 {name} 的 {field} 不存在: {target}")
            paths[field] = target
        motion = load_reference_motion(
            paths["robot"], contract,
            motion_key=entry.get("motion_key"),
            joint_order=entry.get("joint_order", "auto"),
            quaternion_order=entry.get("quaternion_order", "auto"),
        )
        motions.append(replace(motion, name=name))
    return motions
