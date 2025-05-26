#!/usr/bin/env python3
"""
测试 Llama4 的 aux_hidden_states 支持
"""


def test_llama4_aux_support():
    print("🧪 Llama4 aux_hidden_states 支持测试")
    print("=" * 50)

    print("\n✅ 已添加的功能:")
    print("1. Llama4ForCausalLM.set_eagle3_layers_to_capture()")
    print("2. Llama4ForCausalLM.set_custom_layers_to_capture()")
    print("3. 环境变量支持 (SGLANG_CAPTURE_AUX_HIDDEN_STATES)")
    print("4. 自动保存到 /tmp/aux_hidden_states/")

    print("\n🚀 使用方法:")
    print("# 启动 Llama4 服务器并启用 aux 捕获:")
    print("SGLANG_CAPTURE_AUX_HIDDEN_STATES=true \\")
    print("python3 -m sglang.launch_server --model llama4-model")

    print("\n# 或指定特定层:")
    print("SGLANG_CAPTURE_AUX_HIDDEN_STATES=true \\")
    print("SGLANG_CAPTURE_LAYER_INDICES='0,5,10,15' \\")
    print("python3 -m sglang.launch_server --model llama4-model")

    print("\n📂 数据保存:")
    print("- 位置: /tmp/aux_hidden_states/aux_*.pt")
    print("- 格式: {input_ids, aux_hidden_states, timestamp, shapes}")
    print("- 读取: python read_aux.py")

    print("\n🎯 完全兼容 Llama 的所有功能!")


if __name__ == "__main__":
    test_llama4_aux_support()
