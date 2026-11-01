import json
import os
import shutil
import sys
import zipfile

import sentencepiece as spm

ROOT = r"F:\project\cpp训练器"
DATA = os.path.join(ROOT, "data")
SPM = os.path.join(DATA, "models", "spm.model")
TIDS = os.path.join(DATA, "data", "processed", "token_ids.txt")
CFG = os.path.join(DATA, "data", "config.json")
ZIP = os.path.join(DATA, "dataset.zip")
# sentencepiece 的 Windows 实现用窄字符 fopen，中文路径打不开：复制到 ASCII 路径再加载
ASCII_DIR = r"F:\tmp_spm"
ASCII_SPM = os.path.join(ASCII_DIR, "spm.model")

ok = True


def check(cond, msg):
    global ok
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        ok = False


print("=== 1. config.json ===")
with open(CFG, encoding="utf-8") as f:
    cfg = json.load(f)
print(f"  {cfg}")
check(cfg.get("vocab_size") == 32000, f"vocab_size == 32000（实际 {cfg.get('vocab_size')}）")

print("=== 2. spm.model ===")
os.makedirs(ASCII_DIR, exist_ok=True)
shutil.copyfile(SPM, ASCII_SPM)
sp = spm.SentencePieceProcessor()
sp.Load(ASCII_SPM)
check(sp.GetPieceSize() == 32000, f"词表大小 {sp.GetPieceSize()}")
print(f"  unk={sp.unk_id()} bos={sp.bos_id()} eos={sp.eos_id()}")

samples = [
    "数学是研究数量、结构以及空间等概念的一门学科。",
    "他也带领香港年轻人拍摄大量英语短视频，展示香港多姿多彩的一面。",
    "乾隆皇帝与乾坤之道，藉口与慰藉，瞭解与瞭望。",
    "數學是研究數量、結構以及空間等概念的一門學科。",
]
for s in samples:
    ids = sp.Encode(s)
    back = sp.Decode(ids)
    print(f"  {len(s):3d} 字 -> {len(ids):3d} token ({len(s)/max(1,len(ids)):.2f} 字/token)")
    print(f"        回译: {back}")
    check(len(ids) > 0, "编码/解码可用")

print("=== 3. token_ids.txt ===")
size = os.path.getsize(TIDS)
print(f"  大小 {size/1024/1024/1024:.2f} GB")
sampled_tokens = 0
sampled_bytes = 0
max_id = -1
bad = 0
offsets = [0, int(size * 0.25), int(size * 0.5), int(size * 0.75), int(size * 0.98)]
BLOCK = 32 * 1024 * 1024
with open(TIDS, "rb") as f:
    for idx, off in enumerate(offsets):
        f.seek(off)
        chunk = f.read(BLOCK)
        if off > 0:
            nl = chunk.find(b"\n")
            chunk = chunk[nl + 1:] if nl >= 0 else chunk
        nl = chunk.rfind(b"\n")
        if nl > 0:
            chunk = chunk[:nl]
        parts = chunk.split()
        ids = []
        for p in parts:
            try:
                ids.append(int(p))
            except ValueError:
                bad += 1
        toks = len(ids)
        sampled_tokens += toks
        sampled_bytes += len(chunk)
        if ids:
            max_id = max(max_id, max(ids))
            if idx == 0:
                print(f"  首块: {toks} token, 前 16 个 {ids[:16]}")
                print(f"        解码: {sp.Decode(ids[:80])[:100]}")
            if idx == len(offsets) // 2:
                print(f"  中间块解码: {sp.Decode(ids[:80])[:100]}")
        print(f"  偏移 {off/1024/1024/1024:5.2f} GB: {toks} token, 块内最大 id {max(ids) if ids else -1}")

check(bad == 0, f"无非法数值（{bad} 个）")
check(max_id < 32000, f"抽样最大 id < 32000（实际 {max_id}）")
est_tokens = int(sampled_tokens / sampled_bytes * size) if sampled_bytes else 0
print(f"  抽样 {sampled_bytes/1024/1024:.0f} MB / {sampled_tokens:,} token")
print(f"  推算总 token ≈ {est_tokens:,}（约 {size/max(1,est_tokens):.2f} 字节/token）")

print("=== 4. dataset.zip ===")
with zipfile.ZipFile(ZIP) as z:
    infos = z.infolist()
    for i in infos:
        print(f"  {i.filename:40s} {i.file_size/1024/1024:9.1f} MB")
    check(any(i.filename.endswith("token_ids.txt") for i in infos), "包含 token_ids.txt")
    check(any(i.filename.endswith("config.json") for i in infos), "包含 config.json")
    c = z.testzip()
    check(c is None, f"CRC 校验通过（坏条目 {c}）")
print(f"  压缩包 {os.path.getsize(ZIP)/1024/1024:.0f} MB")

shutil.rmtree(ASCII_DIR, ignore_errors=True)
print("\n结果:", "全部通过" if ok else "有失败项")
sys.exit(0 if ok else 1)
