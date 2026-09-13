# -*- coding: utf-8 -*-
"""
验证 C++ 训练器保存的模型可以被 Python (py训练器) 直接加载。
用法: python verify_interop.py <cpp saves 目录, 如 E:/project/cpp训练器/saves/model2>
"""
import sys
import json
import os
import torch

sys.path.insert(0, r"E:\project\py训练器\python")
from model import GPT, GPTConfig  # noqa: E402

def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else r"E:\project\cpp训练器\saves\model2"
    cfg_path = os.path.join(model_dir, "config.json")
    pt_path = os.path.join(model_dir, "model.pt")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfgd = json.load(f)

    cfg = GPTConfig(cfgd["vocab_size"], cfgd["embed_dim"], cfgd["num_heads"],
                    cfgd["num_layers"], cfgd["max_seq_len"], cfgd.get("dropout", 0.1))
    model = GPT(cfg)
    sd = torch.load(pt_path, map_location="cpu")
    # strict=True: C++ 与 Python 键必须完全一致
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.load_state_dict(sd, strict=True)
    print(f"[OK] state_dict 键数: {len(sd)}, 缺失: {missing}, 多余: {unexpected}")

    model.eval()
    x = torch.randint(0, cfgd["vocab_size"], (2, 16), dtype=torch.long)
    with torch.no_grad():
        y = model(x)
    print(f"[OK] 前向传播输出形状: {tuple(y.shape)}")
    print("INTEROP_OK")

if __name__ == "__main__":
    main()
