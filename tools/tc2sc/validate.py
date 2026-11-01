import opencc
import subprocess
import sys
import os

EXE = r"F:\project\cpp训练器\tools\tc2sc\tc2sc.exe"
DICT = r"F:\project\cpp训练器\tools\tc2sc"
OUT = r"F:\project\cpp训练器\tools\tc2sc\_validate_out"
os.makedirs(OUT, exist_ok=True)

conv = opencc.OpenCC("t2s")


def run_exe(in_path, out_path):
    r = subprocess.run(
        [EXE, os.path.join(DICT, "TSPhrases.txt"), os.path.join(DICT, "TSCharacters.txt"),
         in_path, out_path],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace"))
    with open(out_path, "rb") as f:
        return f.read().decode("utf-8", "replace")


blocks = [
    "頭髮很好看，乾隆皇帝與乾坤之道。",
    "藉口與慰藉，瞭解與瞭望，著名與看著。",
    "一隻小鳥飛過，皇后在後宮。",
    "這裡有臺灣、檯燈、裡面、裏面。",
    "數學是研究數量、結構以及空間等概念的一門學科，屬於形式科學的一種。",
    "電腦、網絡、軟體、硬碟、記憶體、程式設計。",
    "他說：「這樣不對吧？」我們都覺得為難。",
    "後來發現開關壞了，於是買了新的。",
    "傳統與繁體字、簡體字、傳記、傳染。",
    "乾淨、乾燥、干擾、若干、才幹、幹部。",
]

fails = 0
for i, b in enumerate(blocks):
    inp = os.path.join(OUT, f"block{i}.txt")
    outp = os.path.join(OUT, f"block{i}.out.txt")
    with open(inp, "w", encoding="utf-8", newline="") as f:
        f.write(b)
    mine = run_exe(inp, outp).rstrip("\n")
    ref = conv.convert(b)
    ok = mine == ref
    if not ok:
        fails += 1
    print(("  OK  " if ok else "  DIFF") + f" [{i}] {b}")
    if not ok:
        print(f"        mine: {mine}")
        print(f"        ref : {ref}")

print("\n=== 真实语料切片对比 ===")
slices = [
    (r"F:\project\cpp训练器\语料\维基\wiki_pure.txt", 0),
    (r"F:\project\cpp训练器\语料\维基\wiki_pure.txt", 700_000_000),
    (r"F:\project\cpp训练器\语料\维基\wiki_pure.txt", 2_100_000_000),
    (r"F:\project\cpp训练器\语料\小说\91.txt", 1_400_000_000),
    (r"F:\project\cpp训练器\语料\小说\91.txt", 2_700_000_000),
]
for idx, (path, off) in enumerate(slices):
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(400_000)
    s = 0
    while s < len(raw) and (raw[s] & 0xC0) == 0x80:
        s += 1
    e = len(raw)
    while e > s and (raw[e - 1] & 0xC0) == 0x80:
        e -= 1
    raw = raw[s:e]
    text = raw.decode("utf-8", "replace")
    # 截到最后一行完整处
    text = text[: text.rfind("\n")] if "\n" in text else text
    inp = os.path.join(OUT, f"slice{idx}.txt")
    outp = os.path.join(OUT, f"slice{idx}.out.txt")
    with open(inp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    mine = run_exe(inp, outp)
    ref = conv.convert(text)
    ok = mine == ref
    if not ok:
        fails += 1
        # 找第一个差异位置
        m = 0
        while m < min(len(mine), len(ref)) and mine[m] == ref[m]:
            m += 1
        print(f"  DIFF {os.path.basename(path)}@{off}: 位置 {m}")
        print(f"        mine: ...{mine[max(0,m-30):m+30]!r}")
        print(f"        ref : ...{ref[max(0,m-30):m+30]!r}")
    else:
        changed = sum(1 for a, b in zip(text, mine) if a != b)
        print(f"  OK   {os.path.basename(path)}@{off}  {len(text)} 字, 变化 {changed} 处")

print(f"\n结果: {'全部一致' if fails == 0 else str(fails) + ' 处差异'}")
sys.exit(1 if fails else 0)
