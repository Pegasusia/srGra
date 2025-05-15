import os
import yaml
from PyQt5.QtWidgets import QMessageBox


class ConfigManager:
    """配置管理类 负责加载和保存配置文件"""
    CONFIG_FILE = "./userdata/config.yaml"

    def __init__(self, log_callback):
        self.log = log_callback

    def ensure_userdata_folder(self):
        """确保userdata文件夹存在"""
        userdata_folder = os.path.dirname(self.CONFIG_FILE)
        if not os.path.exists(userdata_folder):
            os.makedirs(userdata_folder)
            self.log("INFO", f"Created folder: {userdata_folder}")

    def load_config(self, ui_fields):
        """加载配置到UI组件"""
        self.ensure_userdata_folder()
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, "r") as f:
                    config = yaml.safe_load(f)
                    ui_fields.input_path.setText(config.get("input_path", ""))
                    ui_fields.model_path.setText(config.get("model_path", ""))
                    ui_fields.output_path.setText(config.get("output_folder", ""))
                    selected_model = config.get("selected_model", "ESRGAN")
                    index = ui_fields.model_selection_combo.findText(selected_model)
                    if index != -1:
                        ui_fields.model_selection_combo.setCurrentIndex(index)
                self.log("INFO", f"Loaded config: {self.CONFIG_FILE}")
            except Exception as e:
                self.log("ERROR", f"Config load failed: {str(e)}")
                QMessageBox.warning(None, "Config Error", "Failed to load config file!")

    def save_config(self, ui_fields):
        """从UI组件保存配置"""
        config = {
            "input_path": ui_fields.input_path.text(),
            "model_path": ui_fields.model_path.text(),
            "output_folder": ui_fields.output_path.text(),
            "selected_model": ui_fields.model_selection_combo.currentText()
        }
        try:
            with open(self.CONFIG_FILE, "w") as f:
                yaml.safe_dump(config, f)
            self.log("INFO", f"Saved config: {self.CONFIG_FILE}")
        except Exception as e:
            self.log("ERROR", f"Config save failed: {str(e)}")
            QMessageBox.warning(None, "Config Error", "Failed to save config file!")