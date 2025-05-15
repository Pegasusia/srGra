import os
import yaml


class ConfigManager:
    """加载和保存用户数据"""
    CONFIG_FILE = "./userdata/config.yaml"

    @staticmethod
    def ensure_userdata_folder():
        userdata_folder = os.path.dirname(ConfigManager.CONFIG_FILE)
        if not os.path.exists(userdata_folder):
            os.makedirs(userdata_folder)

    @staticmethod
    def load():
        if os.path.exists(ConfigManager.CONFIG_FILE):
            with open(ConfigManager.CONFIG_FILE, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def save(config):
        with open(ConfigManager.CONFIG_FILE, "w") as f:
            yaml.safe_dump(config, f)
