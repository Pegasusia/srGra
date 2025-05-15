# 自动根据输入类型和放大倍数选择模型
def get_model_info(input_type: str, scale: int):
    if input_type == 'video':
        model_type = 'DUF'
        script = f"inference/inference_duf_x{scale}.py"
    else:
        model_type = 'EDSR'
        script = f"inference/inference_edsr_x{scale}.py"
    return model_type, script
