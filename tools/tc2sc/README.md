# tc2sc — 繁体中文 → 简体中文 批量转换工具

语料规范化用的小工具：以 OpenCC 的 `t2s` 字典驱动，流式处理，内存占用恒定，速度约 40–50 MB/s。

## 文件

| 文件 | 说明 |
|---|---|
| `tc2sc.cpp` | 源码（C++17，无第三方依赖，仅用 Windows CRT） |
| `tc2sc.exe` | 已编译版本（`cl /O2 /EHsc /utf-8 /std:c++17 tc2sc.cpp`） |
| `TSPhrases.txt` | 短语字典（OpenCC，277 条，最长 14 字，处理「乾隆/乾坤/瞭解/藉口」这类一繁对多简） |
| `TSCharacters.txt` | 单字字典（OpenCC，4113 条，取每行第一个候选，与 OpenCC 默认策略一致） |
| `OPENCC-NOTICE.txt` | OpenCC 的许可与版权声明（Apache-2.0） |
| `validate.py` | 与 Python OpenCC 逐字节对比的验证脚本（需要 `pip install opencc-python-reimplemented`） |

## 用法

```bat
:: 单文件 → 新文件
tc2sc.exe TSPhrases.txt TSCharacters.txt 输入.txt 输出.txt

:: 单文件原地替换（写 .sc.tmp 后替换，失败不会破坏原文件）
tc2sc.exe TSPhrases.txt TSCharacters.txt 语料.txt 语料.txt --inplace

:: 整个目录的 *.txt → 输出目录
tc2sc.exe TSPhrases.txt TSCharacters.txt 语料目录 输出目录
```

转换规则与 OpenCC `t2s` 等价：**最长短语优先 → 单字 → 原样输出**；
字典里未收录的字（如 `妳`、`祂`、简繁同形的 `繁`）保持原样。

## 验证

```bat
python validate.py
```

输出 `全部一致` 表示与 OpenCC 的 Python 实现在易错短语（乾隆/乾坤/藉口/瞭解/一隻/皇后）
和真实语料切片上逐字节相同。

## 本次使用记录（2026-10-17）

对 `cpp训练器\语料` 全量转换：

| 文件 | 规模 | 单字替换 | 短语替换 | 耗时 |
|---|---|---|---|---|
| `维基\wiki_pure.txt` | 2169.6 MB / 8.42 亿码点 | 112,337,234 | 154,398 | 53 s |
| `小说\91.txt` | 2675.3 MB / 10.28 亿码点 | 112,362,642 | 174,977 | 64 s |
| 其余 107 个 txt | 0.51 GB | 少量 | 少量 | 65 s |

转换后全语料繁体标记占比从 7%～9%（维基/合集）降到 0.02% 以下，UTF-8 校验全部合法。
