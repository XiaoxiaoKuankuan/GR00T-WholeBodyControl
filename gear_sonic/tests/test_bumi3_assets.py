"""验证 4340 资产迁移的边界及训练/部署肩限位的一致性。

通过历史 checkpoint 的最小配置模拟评估入口，保证旧参考路径被显式迁移、新路径
保持幂等，G1/H2 不受影响，未知自定义路径不会静默混用。直接解析交付资产并加载
MuJoCo 验证真实关节限位和总质量，覆盖只改 YAML 却没有改资产的回归风险。
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from omegaconf import OmegaConf
import pytest

from gear_sonic.utils.bumi3_assets import (
    BUMI3_ASSET_ROOT, BUMI3_MJCF_NAME, BUMI3_URDF_PATH, configure_bumi3_assets,
)


def _config(robot="bumi3", model="bumi3.xml"):
    """构造与旧 checkpoint 相同的路径层级，省略不相关的算法字段。"""
    return OmegaConf.create({"manager_env": {
        "config": {"robot": {"type": robot}},
        "commands": {"motion": {"motion_lib_cfg": {"asset": {
            "assetRoot": BUMI3_ASSET_ROOT, "assetFileName": model,
        }}}},
    }})


def test_legacy_checkpoint_migration_is_explicit_and_idempotent(capsys):
    cfg = _config()
    configure_bumi3_assets(cfg)
    assert cfg.manager_env.commands.motion.motion_lib_cfg.asset.assetFileName == BUMI3_MJCF_NAME
    assert "bumi3.xml -> bumi3_4340.xml" in capsys.readouterr().out
    snapshot = OmegaConf.to_container(cfg)
    configure_bumi3_assets(cfg)
    assert OmegaConf.to_container(cfg) == snapshot
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("robot", ["g1", "h2"])
def test_other_robots_keep_original_asset_config(robot):
    cfg = _config(robot, model=f"{robot}.xml")
    snapshot = OmegaConf.to_container(cfg)
    configure_bumi3_assets(cfg)
    assert OmegaConf.to_container(cfg) == snapshot


def test_old_config_without_robot_type_keeps_g1_default():
    cfg = OmegaConf.create({"manager_env": {"config": {}}})
    configure_bumi3_assets(cfg)
    assert OmegaConf.to_container(cfg) == {"manager_env": {"config": {}}}


@pytest.mark.parametrize("field,value", [
    ("assetFileName", "custom.xml"), ("assetRoot", "/tmp/custom-robot/"),
])
def test_unknown_bumi_asset_is_rejected(field, value):
    cfg = _config()
    cfg.manager_env.commands.motion.motion_lib_cfg.asset[field] = value
    with pytest.raises(ValueError):
        configure_bumi3_assets(cfg)


def test_4340_asset_limits_and_mass():
    root = Path(__file__).resolve().parents[2]
    urdf = ET.parse(root / BUMI3_URDF_PATH)
    model = mujoco.MjModel.from_xml_path(str(root / BUMI3_ASSET_ROOT / BUMI3_MJCF_NAME))
    for joint, expected in {
        "l_arm_roll_joint": [-0.14, 1.94], "r_arm_roll_joint": [-1.94, 0.14],
    }.items():
        limit = urdf.find(f"./joint[@name='{joint}']/limit")
        np.testing.assert_allclose([float(limit.get(k)) for k in ("lower", "upper")], expected)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        np.testing.assert_allclose(model.jnt_range[joint_id], expected)
    assert sum(model.body_mass) == pytest.approx(20.7073094)
