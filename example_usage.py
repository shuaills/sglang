#!/usr/bin/env python3
"""
示例：如何在不使用 EAGLE3 的情况下获取 aux_hidden_states

SGLANG_CAPTURE_AUX_HIDDEN_STATES=true SGLANG_CAPTURE_LAYER_INDICES='2,16,29' python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct
"""

import torch
from transformers import LlamaConfig

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.model_runner import ModelRunner


def setup_model_with_aux_capture():
    """设置模型以捕获辅助隐藏状态"""

    # 假设您已经有了模型实例
    # 这里展示如何在模型加载后启用 aux_hidden_states 捕获

    # 方法1：捕获默认层（第2层、中间层、倒数第3层）
    # model.set_custom_layers_to_capture()

    # 方法2：捕获指定层
    # model.set_custom_layers_to_capture([0, 5, 10, 15])  # 捕获第0、5、10、15层

    # 方法3：捕获所有层（如果您想要）
    # num_layers = model.config.num_hidden_layers
    # model.set_custom_layers_to_capture(list(range(num_layers)))

    print("模型已设置为捕获辅助隐藏状态")


def example_inference_with_aux_states():
    """演示如何在推理中获取 aux_hidden_states"""

    # 注意：这是一个概念性示例，实际使用时需要根据您的具体情况调整

    # 1. 首先设置模型捕获层
    # model.set_custom_layers_to_capture([2, 16, 29])  # 示例层索引

    # 2. 进行推理
    # output = model.forward(...)

    # 3. 检查输出
    # if isinstance(output, tuple) and len(output) == 2:
    #     logits, aux_hidden_states = output
    #     print(f"获取到 {len(aux_hidden_states)} 个辅助隐藏状态")
    #     for i, aux_state in enumerate(aux_hidden_states):
    #         print(f"辅助状态 {i}: shape = {aux_state.shape}")
    # else:
    #     print("未获取到辅助隐藏状态")

    pass


if __name__ == "__main__":
    print("使用示例：")
    print("1. 启动普通的 SGLang 服务器（不使用 EAGLE3）")
    print("2. 在模型加载后调用 model.set_custom_layers_to_capture()")
    print("3. 进行推理时会自动返回 aux_hidden_states")

    setup_model_with_aux_capture()
    example_inference_with_aux_states()
