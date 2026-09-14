"""统一 BUMI3 当前训练与参考运动学使用的 4340 资产。

机器人配置负责加载 4340 URDF，本模块在保存训练配置及创建评估环境前同步参考
运动学的 MJCF 路径。旧 checkpoint 保存的 bumi3.xml 会明确迁移并打印说明，防止
新物理资产搭配旧参考模型；其他机器人保持原配置。未知的 BUMI3 自定义模型路径
拒绝静默替换，便于调用者发现资产不匹配。路径均相对 SONIC 仓库，服务器无需
安装 legged_lab；历史资产文件仍保留，已启动的进程不受默认配置修改影响。
"""

from omegaconf import OmegaConf


BUMI3_ASSET_ROOT = "gear_sonic/data/assets/robot_description/mjcf/"
BUMI3_MJCF_NAME = "bumi3_4340.xml"
BUMI3_URDF_PATH = "gear_sonic/data/assets/robot_description/urdf/bumi3_4340/bumi3_4340.urdf"


def configure_bumi3_assets(config) -> None:
    """在内存配置中迁移已知旧路径；不写回 checkpoint 的历史配置文件。"""
    if OmegaConf.select(config, "manager_env.config.robot.type", default="g1") != "bumi3":
        return
    key = "manager_env.commands.motion.motion_lib_cfg.asset"
    asset = OmegaConf.select(config, key)
    if asset is None:
        raise ValueError("BUMI3 缺少参考运动学资产配置，无法核对 4340 URDF/MJCF 配对")
    if asset.assetFileName not in ("bumi3.xml", BUMI3_MJCF_NAME):
        raise ValueError(f"BUMI3 训练使用 4340 URDF，但参考 MJCF 未识别: {asset.assetFileName}")
    if str(asset.assetRoot).rstrip("/") != BUMI3_ASSET_ROOT.rstrip("/"):
        raise ValueError(f"BUMI3 参考模型目录与仓库 4340 资产不一致: {asset.assetRoot}")
    if asset.assetFileName != BUMI3_MJCF_NAME:
        print(f"[BUMI3 资产迁移] {asset.assetFileName} -> {BUMI3_MJCF_NAME}; URDF={BUMI3_URDF_PATH}")
        OmegaConf.update(config, f"{key}.assetFileName", BUMI3_MJCF_NAME)
