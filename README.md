# GPT 训练与对话系统 · GPT Trainer

用 C++ 从零训练自己的 GPT，并用 llama.cpp 直接聊任意 GGUF 模型。

**Windows 11 原生风格 GUI · libtorch(CUDA) 训练 · llama.cpp(CUDA/CPU) 推理 · 全自包含单包**

> 项目定位：把「中文分词 → 预训练 → 断点续训 → 导出 GGUF → 对话」整条链路做成一个双击即用的 Windows 桌面程序，
> 不需要装 Python、不需要配 CUDA 环境、不需要单独装 llama.cpp 或 Ollama。

---

## ✨ v1.1.0 亮点

* **全新对话界面**：无边框聊天页 + 底部悬浮圆角输入框，逐字打印动画、每轮 tok 性能、圆形上下文用量环
* **模型一键切换**：下拉选定即自动加载（无需点「加载」），启动自动加载上次模型
* **GPU 推理**：内置 llama.cpp **CUDA 版**（静态，60MB）与 **CPU 版**（静态，9.3MB），统统零依赖
* **任意 GGUF**：本项目训练的模型、Qwen2.5-Instruct、Llama、Gemma… 都能直接聊
* **自动套用对话模板**：Qwen 等 Instruct 模型走 `/v1/chat/completions`，按模型自带 chat 模板渲染（不再"续写"）
* **全自包含**：Qt / VC++ 运行时 / libtorch+cuDNN+cuBLAS / llama.cpp 引擎 / 中文分词器全部内置，解压双击运行

## 🚀 快速开始

1. 从 [Releases](../../releases) 下载 `gpt_trainer_v1.1.0_win64_cu130.zip`（约 1.88 GB）
2. 解压到任意目录（中英文路径均可）
3. 双击 `gpt_trainer.exe`

无需安装任何运行库。程序在自身目录创建 `models/ logs/ saves/ data/`（便携模式，见 `gpt_root.txt`）。

**系统要求**

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10 1809+ / Windows 11（x64） |
| 显卡 | 训练与 GPU 推理需要 NVIDIA 显卡（驱动支持 CUDA 13）；无显卡时自动回退 CPU |
| 显存 | 建议 6 GB+（本机 RTX 3060 Laptop 6GB，训练 58M 参数模型约 3.7GB 占用） |
| 内存 | 8 GB+（加载 2.8GB 语料约占用 6GB 内存） |

## 🎮 三种玩法

### 1) 训练自己的 GPT

1. **一条龙分词**：「高级 → 分词器」选择 txt 语料（或留空用 `data/raw`）→ 点「开始训练分词器」
   → 自动完成 `SentencePiece 训练 → token_ids.txt 编码 → 打包 dataset.zip`
   → **自动载入「训练」页**（词表 / 序列长度 / 数据路径全部接管）→ 直接点「开始训练」
2. 也可以「训练」页手动打开已有的 `dataset.zip`
3. 设置学习率/批次/序列长度/训练步数 → 开始训练
4. 训练中断也不怕：每 N 步自动存断点（模型 + 优化器动量 + 学习率进度），下次勾「断点续训」继续

### 2) 用它聊任意 GGUF

1. 把 `.gguf` 放进 `models/gguf/`（或在「高级」页浏览任意路径）
2. 对话页下拉选中模型 → 自动加载
3. 输入问题，回车/点「↑」发送

### 3) 命令行（可脚本化）

```bat
gpt_trainer.exe train  --data data/processed/token_ids.txt --steps 40000 --batch 16 --seq-len 256 --embed 512 --heads 8 --layers 8
gpt_trainer.exe chat   --model "models\gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf" --prompt "你是谁？" --engine llama
gpt_trainer.exe chat   --model models\gguf\model3.gguf --interactive --chat-mode auto
gpt_trainer.exe export-gguf --model saves\model3 --out models\gguf\model3.gguf
gpt_trainer.exe tokenize --out data --vocab 32000 语料1.txt 语料2.txt
gpt_trainer.exe list   --json
gpt_trainer.exe info   --probe-server --json
```

全部命令：`train` / `chat` / `tokenize` / `export-gguf` / `list` / `info` / `serve` / `selftest` / `loadtest`，
`<命令> --help` 查看参数，`--json` 输出机器可读结果。

## 🧠 推理：提示词通道（自动选择）

| 通道 | 触发条件 | 接口 | 适用 |
|---|---|---|---|
| **spm token** | 模型词表能被本项目 `spm.model` 按大小匹配 | `/completion` + token 数组 | 本项目训练出的中文模型（最稳） |
| **对话模板** | 匹配不到 spm，且模型自带 `chat_template` | `/v1/chat/completions` | **Qwen / Llama-Instruct / Gemma 等** |
| **文本补全** | 以上都不满足 | `/completion` + 文本 | 普通 base 模型 |

GUI：「高级 → 推理引擎 → 提示词通道」可选 自动/对话模板/文本/spm token；CLI：`--chat-mode auto|chat|text|tokens`。

引擎探测顺序：根目录 `llama-server.exe`（CUDA）→ `third_party/llama.cpp/` → `llama.cpp\`（CPU）→ 本机 Ollama → `PATH`；
都没有时自动回退内置 libtorch 引擎。

## 🏋️ 训练实现要点

* libtorch 2.14（CUDA 13）：`AdamW` + `CosineAnnealingLR`（自实现）+ 梯度裁剪
* **bf16 AMP**：手动 GradScaler（未溢出缩放、动态升降、每 2000 步 ×2）+ `at::autocast`
* **SDPA**：CUDA 下用 `scaled_dot_product_attention(is_causal=true)`，CPU 回退手写注意力
* 滑动窗口数据集（`stride` 可调）、每 epoch 重新洗牌、`drop_last`
* 断点：`models/training_state/{model.pt, optim.pt, meta.json}`，恢复学习率进度、AMP 缩放因子、步数
* 初始权重（warm start）：从 `saves/modelN/model.pt` 继续，优化器重新初始化
* 产出：`models/checkpoint_epoch_N.pt`、`models/final_model.pt`、`saves/modelN/{model.pt, config.json, info.txt}`
* 日志：手写 tfevents（无需 protobuf），TensorBoard 可直接打开 `logs/`

## 🧩 代码结构

| 模块 | 文件 | 说明 |
|---|---|---|
| 入口 | `src/main.cpp` | 无参数=GUI，`<命令>`=CLI，`--selftest` 互读自检 |
| 配置/路径 | `src/config.{h,cpp}` | 配置、便携根目录（`gpt_root.txt`）、数据集 zip |
| 数据集 | `src/dataset.{h,cpp}` | token_ids 加载、滑动窗口、洗牌、批加载 |
| 模型 | `src/model.{h,cpp}` | GPT（Embedding + N×Block + lm_head）、SDPA、generate |
| 训练 | `src/train.{h,cpp}` | AMP、优化器、调度器、断点、检查点、saves |
| GGUF | `src/gguf_writer.{h,cpp}` | GGUF v3 写出（gpt2 架构 + SentencePiece 词表） |
| 推理引擎 | `src/llama_engine.{h,cpp}` | llama-server 子进程 + WinHTTP/SSE 流式 + `/props` 模板探测 |
| 通道选择 | `src/llama_prompt.{h,cpp}` | spm / token / 文本 / 对话模板 自动选择与回退 |
| 序列化 | `src/legacy_serial.{h,cpp}` | 读写 `torch.save(state_dict)` 兼容格式（与 Python 双向互读） |
| 日志 | `src/tfevents.{h,cpp}` | tfevents 记录器 |
| 分词器 | `src/tokenizer_worker.{h,cpp}` | SentencePiece 训练/编码/打包 dataset.zip |
| ZIP | `src/zip_util.{h,cpp}` | 解压/打包（deflate） |
| UI | `src/ui_theme.{h,cpp}` `src/ui_kit.{h,cpp}` `src/gui.{h,cpp}` | Fluent 主题与控件、三页 GUI |
| CLI | `src/cli.{h,cpp}` `src/cli_chat.cpp` `src/cli_main.cpp` | 命令解析与实现 |

## 🔧 从源码构建

依赖（体积大，不随仓库提供）：

```bat
:: 1) libtorch 2.14.0 + cu130 → 解压到 libtorch\
::    https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.14.0%2Bcu130.zip
:: 2) Qt 6.9.3（仅 qtbase）
pip install aqtinstall
python -m aqt install-qt windows desktop 6.9.3 win64_msvc2022_64 -O qt --archives qtbase
:: 3) zlib → third_party\zlib\{z.h,zconf.h,zlib.h,z.lib,z.dll}
:: 4) sentencepiece 0.2.2 静态库（或 vcpkg install sentencepiece:x64-windows），改 CMakeLists 里 SP_ROOT/SP_BUILD
```

```bat
powershell -ExecutionPolicy Bypass -File build.ps1        :: 全量构建（默认）
powershell -ExecutionPolicy Bypass -File build.ps1 -Fast  :: 增量
powershell -ExecutionPolicy Bypass -File package.ps1      :: 生成 Release\ 自包含目录
```

* 工具链：MSVC 14.44（VS2026 内置，CUDA 13 支持的最新工具集）+ CMake + Ninja，脚本内全部绝对路径
* 中文路径下 MSVC `link.exe` 会把 Ninja 响应文件按 ANSI 读取导致乱码，`build.ps1` 会自动建立 ASCII junction（`E:\cpptrn\*`）
* llama.cpp 引擎（可选，与包内一致）：

```bat
cmake -S llama.cpp -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DGGML_CUDA=ON ^
      -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0" ^
      -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --target llama-server
```

## ⚙️ 参数与限制

| 参数 | 范围 | 说明 |
|---|---|---|
| 上下文长度 | 512 ~ 131072 | 超过模型训练长度时 llama.cpp 自动按模型能力截断 |
| 回复长度 | 1 ~ 4096 token | 越大越慢；日常 100~300 足够 |
| 温度 | 0.01 ~ 2.0 | 越低越确定 |
| Top-K / Top-P / 重复惩罚 | 0~100000 / 0~1 / 0.5~3 | 高级页可调 |

## ❓ 常见问题

**Q：为什么我的模型答非所问、像在续写小说？**
A：它是在你的语料上做「下一个词预测」的**基座模型**，没有经过指令微调；换 Qwen 等 Instruct GGUF，或往语料里混入问答格式再训练。

**Q：对话很慢 / 显卡没吃上？**
A：「高级 → 推理引擎」查看引擎路径：根目录的 `llama-server.exe` 是 CUDA 版（`--list-devices` 应显示你的显卡）；若用的是 `llama.cpp\` 下那个，就是 CPU 版。

**Q：模型切换没反应？**
A：下拉里**选定**（点击或回车）即自动加载，状态栏会显示「已加载: xxx」；列表里用方向键浏览不会触发加载。

**Q：加载自己训练的模型时报 GGUF 导出失败？**
A：需要匹配词表的 `spm.model`（程序会自动在 `models\spm.model`、`data\models\spm.model` 等位置按词表大小查找）。

**Q：训练步数很小（如 100）时再点开始会闪退吗？**
A：v1.1.0 已修复（线程 join 问题），并加了全局异常兜底：出错会弹窗 + 写 `%TEMP%\gpt_trainer_terminate.log`。

## 🗺️ 版本历史

* **v1.1.0** — 全新对话界面（无边框 + 悬浮输入框 + 打字动画 + tok 指标 + 上下文环）、llama.cpp CUDA 引擎、任意 GGUF + 对话模板、模型一键切换、断点续训、闪退修复、单包自包含
* **v1.0.0** — 首版：PyTorch 项目完整 C++ 移植（训练 / 对话 / 分词器三页、GGUF 导出、Python 模型互读）

## 📄 License

[MIT](LICENSE)
