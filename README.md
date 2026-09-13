# cpp训练器 — GPT 训练与对话系统（C++ / libtorch 移植版）

本项目将 `py训练器`（PyTorch + PyQt5）的全部功能移植为 C++ 实现：

| Python 文件 | C++ 对应 | 说明 |
|---|---|---|
| `config.py` | `src/config.{h,cpp}` | 训练配置、路径、数据集 zip 解压 |
| `dataset.py` | `src/dataset.{h,cpp}` | token_ids 加载、滑动窗口、每 epoch 洗牌、drop_last 批加载 |
| `model.py` | `src/model.{h,cpp}` | GPT（Embedding + 6×TransformerBlock + lm_head），权重键与 Python 版完全一致 |
| `train.py` | `src/train.{h,cpp}` | AdamW + CosineAnnealingLR + 梯度裁剪 + 断点保存/续训 + 检查点/最终模型/saves 保存 |
| `gui.py` | `src/gui.{h,cpp}` | Qt6 三标签页 GUI（训练 + 对话 + **分词器**），MIUIX/HyperOS 卡片风格 · **蓝色主题** · 多轮对话/问答模板/生成参数 |
| `main.py` | `src/main.cpp` | `--cli [--resume] [--init 模型]` / `--selftest` / `--loadtest` / `--tokenize` |
| TensorBoard | `src/tfevents.{h,cpp}` | 手写 tfevents 写出器（无需 protobuf 依赖，格式与 TensorBoard 完全一致） |
| `cpp分词器/main.cpp` | `src/tokenizer_worker.{h,cpp}` | SentencePiece 训练/分词/打包全流程（进程内调用库，无需外部 exe） |
| zip 解压/打包 | `src/zip_util.{h,cpp}` | 极简 ZIP（deflate，与 Python zipfile 兼容） |
| Python 模型互读 | `src/legacy_serial.{h,cpp}` | 读写 `torch.save(state_dict)` 兼容格式（双向互读） |

## 从源码构建（其他电脑 / 开源使用）

本仓库只含源码；下列依赖体积巨大，请自行下载/构建后放入对应目录：

```bat
:: 1. libtorch 2.14.0 + cu130（约 3.9GB zip，解压后约 12GB）
::    下载: https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.14.0%2Bcu130.zip
::    解压到  libtorch\

:: 2. Qt 6.9.3（仅 qtbase，约 280MB）
pip install aqtinstall
python -m aqt install-qt windows desktop 6.9.3 win64_msvc2022_64 -O qt --archives qtbase

:: 3. zlib（或从任意 vcpkg/官方 zlib 获取，放入 third_party\zlib\ 三个文件）
::    third_party\zlib\z.h  zconf.h  zlib.h  z.lib  z.dll（后两个为构建产物，不入库）

:: 4. sentencepiece 0.2.2 静态库（或自行用 vcpkg: vcpkg install sentencepiece:x64-windows）
::    构建后修改 CMakeLists.txt 中的 SP_ROOT / SP_BUILD 路径
::    （本机开发时通过 ASCII junction E:\cpptrn\sp 指向既有构建产物）

:: 5. 编译器：VS2026 内含 MSVC 14.44 + CMake/Ninja
::    修改 build.ps1 顶部的 $vsCmake / $vsNinja / $vcvars / $msvcVer 为本机路径
```

环境适配清单（本仓库当前为作者机器专用绝对路径，需按实际环境修改）：

- `build.ps1`：VS 安装路径、MSVC 工具集版本
- `CMakeLists.txt`：sentencepiece 路径（SP_ROOT/SP_BUILD）、zlib 路径
- `build.ps1` 中的 junction 逻辑（`E:\cpptrn\*`）：仅为规避"MSVC link.exe 按 ANSI 解析 Ninja 响应文件导致中文路径乱码"；纯英文路径的项目可直接删除该段并把 CMakeLists 里的 `E:/cpptrn/*` 改成真实路径
- `src/config.h`：默认数据路径 `E:/data/processed/token_ids.txt`
- 运行期目录 `models/`、`logs/`、`saves/`、`data/` 由程序自动创建（亦可用环境变量 `GPT_ROOT` 指定根目录）

## 依赖与工具链（作者本机环境参考）

- **libtorch 2.14.0 + cu130**：内置 CUDA 13.0 运行时 DLL，**不需要也不调用 nvcc**
- **Qt 6.9.3**（aqtinstall 下载，仅 qtbase）
- **sentencepiece 0.2.2** 静态库（含 internal protobuf）
- **zlib**（vcpkg 产物）
- **MSVC 14.44**（VS2026 内置；14.44 是 CUDA 13.0 支持的最新工具集，且与 sentencepiece 所用默认工具集的 CRT ABI 兼容）

## 构建

```bat
powershell -ExecutionPolicy Bypass -File build.ps1
```

- 脚本内部使用绝对路径调用 `vcvars64.bat -vcvars_ver=14.44`、`cmake.exe`、`ninja.exe`，不依赖系统 PATH。
- 每次构建默认全量重编（中文 locale 下 cl.exe 的 /showIncludes 前缀会导致 Ninja 依赖跟踪失效）；增量调试可用 `-Fast`。
- 构建完成后 Qt/zlib 运行时 DLL 与插件自动复制到 `build/`。

## 打包（自包含 exe，双击即用、无控制台窗口）

```bat
powershell -ExecutionPolicy Bypass -File package.ps1
```

- 产出 `Release\`（约 2.7GB）：`gpt_trainer.exe` + 全部 libtorch cu130 运行时 DLL + Qt DLL/插件 + `使用说明.txt`。
- 主程序为 WIN32 子系统：**双击直接运行，不会弹出任何 cmd/控制台窗口**（全部 DLL 与 exe 同目录，无需 PATH）。
- 模型/日志/存档仍写入项目根目录（`models/`、`logs/`、`saves/`），可用环境变量 `GPT_ROOT` 覆盖。
- 换机器运行需 VC++ 2015-2022 x64 运行库 + 支持 CUDA 13 的 NVIDIA 驱动。

## 运行

```bat
run.bat                 :: GUI 模式
run.bat --cli           :: 命令行模式（默认配置直接训练）
run.bat --selftest      :: 互读自检（加载 py训练器 的模型并生成文本）
run.bat --loadtest <x>  :: 调试：加载任意 state_dict 文件并打印张量信息
```

`run.bat` 会把 `libtorch\lib`（含 cudart64_13.dll 等 CUDA 13 运行时）与 Qt bin 置于 PATH 最前。
GPU：RTX 3060（sm_86），驱动 616.64（CUDA UMD 13.4）满足 cu130 要求。

## 功能清单（与 Python 版一一对应）

1. 训练页：数据路径浏览、学习率/批次/序列长度/训练步数/Epoch 数/滑动步长/嵌入维度/头数/层数参数面板（含整除校验）、Loss/Epoch/进度条状态区、训练日志、开始/停止/退出。
2. 对话页：saves 模型列表（加载/删除/刷新）、打开数据集 zip（自动解压并加载 spm.model + token_ids.txt + config.json）、分词器选择、模型信息、对话历史/系统日志、中文输入与生成回复（temperature=0.8, top_k=40, 50 token）、清空按钮。
3. **分词器页（集成自 cpp分词器）**：多选/目录选择语料 txt、语料统计（总大小/档位/唯一字符数）、按语料大小自动分档参数（线程数/词表大小/最大句子数，词表下限=字符集+余量）、手动覆盖参数、进程内 SentencePiece unigram 训练 → 生成 token_ids.txt → config.json → 打包 dataset.zip、训练后可直接测试分词（词片段 + Token ID）。命令行可用 `--tokenize <输出目录> <txt文件...>` 脚本化调用。
4. 训练核心：滑动窗口采样、AdamW、CosineAnnealingLR（T_max=总步数, eta_min=1e-5）、梯度裁剪 1.0、每 10 步 GUI 回调、每 20 步 TensorBoard 记录（loss/train、lr、loss/epoch）、每 epoch 检查点、final_model.pt、saves/model{N} 自动编号（model.pt + config.json + info.txt）。
5. 训练中可随时停止；关闭窗口时确认并安全停止。
6. 修复了 Python 版 `update_progress` 引用不存在的 `epoch_label` 导致的运行时 AttributeError。

## 模型互读

C++ 模型的子模块命名与 Python 版完全一致
（`token_embedding` / `position_embedding` / `blocks.N.{ln1,attn.qkv,attn.proj,ln2,ff.net}` / `ln_f` / `lm_head` / `blocks.N.attn.mask`），
且 `saves/modelN/model.pt` 与 `models/final_model.pt` 采用与 `torch.save(state_dict)` 字节兼容的格式：

- Python 可以直接 `torch.load("saves/modelN/model.pt")` 加载 C++ 训练出的权重（`verify_interop.py` 已验证，strict 模式 84 键完全匹配）；
- C++ 可以直接加载 Python 训练的模型：
  - torch >= 2.14 保存的 state_dict 直接可读；
  - 旧版 torch（如本项目早期 `py训练器/saves/model1`）保存的文件使用了复杂嵌套 pickle 结构，需先用 `convert_model.py` 一次性转换（已自动备份 `.bak`）；
- `--selftest` 命令行开关可自动验证上述互读（加载 Python 模型 → CUDA 前向 → sentencepiece 生成 → C++ 保存/加载回环）；
- `--loadtest <文件.pt>` 可单独调试加载任意 state_dict 文件。

检查点 `models/checkpoint_epoch_N.pt` 使用 libtorch 2.14 的 JIT 序列化格式（含 epoch/loss/model_state_dict/optimizer_state_dict），仅供 C++ 侧留存训练记录（Python 版本身也不会从检查点续训）。

## 训练逻辑优化（相对 Python 版）

1. **CUDA bf16 AMP 混合精度**：前向自动混合精度（bf16），配合手动实现的 GradScaler（与 Python `torch.cuda.amp.GradScaler` 等价的缩放/反缩放/动态因子策略），CUDA 上自动启用，CPU 自动回退 fp32。
2. **Fused 因果注意力**：CUDA 上使用 `scaled_dot_product_attention(is_causal=true)`（flash / memory-efficient kernel），数学上与原手工实现完全等价；CPU 或不可用时回退原实现（保留 mask buffer 以保持 state_dict 兼容）。
3. **零 Python 开销的数据管线**：原生 C++ 解析 + 滑动窗口 + 每 epoch 洗牌，批张量直接从连续缓冲构造。
4. **手写 tfevents**：无 protobuf 运行时依赖，输出格式与新版 TensorBoard `record_writer.py` 完全一致（`tensorboard --logdir logs` 已验证可解析 loss/train、lr、loss/epoch）。

## 目录结构

```
cpp训练器/
├── build.ps1            # 一键构建脚本（绝对路径工具链）
├── run.bat              # 运行脚本（GUI / --cli）
├── CMakeLists.txt
├── libtorch/            # libtorch 2.14.0+cu130（解压后）
├── qt/6.9.3/            # Qt 6.9.3 msvc2022_64（aqtinstall 下载）
├── third_party/zlib/    # zlib 头文件/库/DLL（复用 RTX vcpkg 产物）
├── src/                 # 全部源码
├── build/               # 构建输出（gpt_trainer.exe + 运行时 DLL）
├── models/              # 检查点与最终模型（运行期创建）
├── logs/                # TensorBoard 事件（运行期创建）
└── saves/               # model{N}/（运行期创建，自动编号）
```

## 已知说明

- 数据集 zip 需包含 `spm.model`、`token_ids.txt`、`config.json`（与 Python 版要求一致）。
- `E:\data\processed\token_ids.txt` 为默认数据路径，可在 GUI 中改选。
- CUDA 12.4 与 13.0 并存的环境下，本项目不使用系统 CUDA 路径；所有 CUDA 运行时 DLL 均来自 libtorch 包内。
- 旧版 torch（<2.14）训练的模型文件首次在 C++ 中加载前，先运行 `python convert_model.py <模型.pt>` 转换（原文件自动备份为 `.bak`）。
- 构建依赖 4 个 ASCII junction（`E:\cpptrn\*`，见 `build.ps1`），用于规避 MSVC link.exe 对 Ninja 响应文件中中文路径的 ANSI 误读。
