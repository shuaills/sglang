#!/usr/bin/env python3
import glob
import os

import torch


def read_latest_aux():
    """读取最新的 aux_hidden_states"""
    files = glob.glob("/tmp/aux_hidden_states/aux_*.pt")
    if not files:
        print("❌ 未找到保存的文件")
        return

    latest = max(files, key=os.path.getctime)
    print(f"📂 读取: {latest}")

    data = torch.load(latest, map_location="cpu")
    print(f"🔢 input_ids: {data['input_ids']}")
    print(f"📊 aux层数: {len(data['aux_hidden_states'])}")
    for i, shape in enumerate(data["shapes"]):
        print(f"   层{i}: {shape}")

    return data


if __name__ == "__main__":
    read_latest_aux()
