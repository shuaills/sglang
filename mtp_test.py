import time
from types import SimpleNamespace

import requests


def test_acceptance_rate(base_url="http://127.0.0.1:30000", num_requests=10):
    """测试 EAGLE3 投机解码的接受率"""

    # 清空缓存
    requests.get(f"{base_url}/flush_cache")

    # 发送一些测试请求来生成统计数据
    test_prompts = [
        "Today is a sunny day and I like",
        "The capital of France is",
        "Explain quantum computing in simple terms",
        "Write a Python function to sort a list",
        "What is the meaning of life?",
    ]

    for i in range(num_requests):
        prompt = test_prompts[i % len(test_prompts)]
        data = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 100,
            },
        }

        response = requests.post(f"{base_url}/generate", json=data)
        if response.status_code != 200:
            print(f"Request {i+1} failed: {response.status_code}")
            continue

        print(f"Request {i+1} completed")

    # 获取服务器信息和接受率统计
    server_info = requests.get(f"{base_url}/get_server_info").json()
    avg_spec_accept_length = server_info.get("avg_spec_accept_length", 0)

    print(f"\n=== 接受率统计 ===")
    print(f"平均投机接受长度: {avg_spec_accept_length:.3f}")
    print(f"投机算法: {server_info.get('speculative_algorithm', 'None')}")
    print(f"Draft 模型: {server_info.get('speculative_draft_model_path', 'None')}")

    if avg_spec_accept_length > 1.0:
        print("✅ 投机解码正常工作")
    else:
        print("❌ 投机解码可能有问题")

    return avg_spec_accept_length


if __name__ == "__main__":
    acceptance_rate = test_acceptance_rate()
