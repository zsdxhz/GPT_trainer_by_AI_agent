# -*- coding: utf-8 -*-
"""
将任意 PyTorch 模型文件转换为 C++ 训练器可直接读取的格式。
- 旧版 torch (<2.14) 保存的 state_dict 使用了复杂的 OrderedDict 嵌套结构，
  libtorch 的 C++ 解析器无法读取，需要一次性转换。
- torch >= 2.14 保存的文件无需转换（C++ 可直接读取）。
用法: python convert_model.py <输入.pt> [输出.pt]
"""
import os
import sys
import shutil

import torch


def main():
    if len(sys.argv) < 2:
        print("用法: python convert_model.py <输入.pt> [输出.pt]")
        return 1
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src

    print(f"加载: {src}")
    obj = torch.load(src, map_location="cpu", weights_only=False)

    if hasattr(obj, "state_dict"):
        # nn.Module
        data = dict(obj.state_dict())
    elif hasattr(obj, "keys"):
        data = dict(obj)
    else:
        print(f"不支持的类型: {type(obj)}")
        return 2

    if dst == src:
        shutil.copy2(src, src + ".bak")
        print(f"已备份原文件: {src}.bak")

    # dict() 会把 OrderedDict 变成普通 dict → 生成 C++ 可读的简洁 pickle
    torch.save(data, dst)
    print(f"转换完成: {dst} (共 {len(data)} 个键)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
