# GPT 训练与对话系统 v1.1.0

> 一个 Windows 桌面版「从零训练 GPT + 聊任意 GGUF」的全自包含工具。
> 解压双击即用：训练用 libtorch(CUDA)，推理用 llama.cpp(CUDA/CPU)，无需安装 Python / CUDA / Ollama。

---

## 🎉 本版重点

### 🖥 全新对话界面

* **无边框聊天页**：消息区铺满整页，AI 头像 + 纯文本回复，用户右侧圆角气泡
* **底部悬浮圆角输入框**：工具条含「新建对话 / 多轮 / 长度 / 温度 / 上下文 / 模型选择 / 刷新 / 停止 / 圆形 ↑ 发送」
* **逐字打印动画**：回复逐字吐出，可随时「停止」
* **tok 性能指标**：每条回复下方 `⚡ tok · 秒 · tok/s`，输入框下方累计 `本轮/累计/轮次`
* **圆形上下文用量环**：右下角实时百分比（>60% 变黄、>85% 变红、满则「满」），悬停显示已用/上限
* 侧栏切换页面时**强调色指示条滑动动画**；所有下拉列表改为**圆角卡片样式**

### 🚀 推理：llama.cpp（GPU + CPU 双引擎）

* 内置 **CUDA 版** `llama-server.exe`（静态 60MB，自动卸载全部层到显卡；`--list-devices` 可验证）
* 内置 **CPU 版**（静态 9.3MB）作为无 N 卡 / 省显存的后备
* **任意 GGUF**：本项目训练出的模型、Qwen2.5-Instruct、Llama、Gemma… 均可直接加载
* **自动对话模板**：Instruct 模型自动走 `/v1/chat/completions`，按模型自带 `chat_template`（ChatML 等）渲染，是真问答而不是续写
* **模型一键切换**：下拉选定即自动加载（无需「加载」按钮），启动自动加载上次模型；切换后台异步，不卡界面
* 无 llama.cpp 时自动回退内置 libtorch 引擎

### 🏋️ 训练

* **分词 → 预训练一条龙**：分词器训练完成后自动打包 `dataset.zip`（spm.model / token_ids.txt / config.json）并**自动载入「训练」页**，无需再手动开数据集
* libtorch 2.14 + CUDA 13：AdamW、CosineAnnealingLR、梯度裁剪、bf16 AMP（手动 GradScaler）、SDPA（`is_causal`）
* **断点续训**：每 N 步保存 模型 + 优化器动量 + 学习率进度 + AMP 缩放因子，断电/中断后可无缝续训
* **初始权重（warm start）**：从已有 `saves/modelN` 继续训练
* 集成 SentencePiece 分词器训练 → `token_ids.txt` → `dataset.zip` 一条龙
* 手写 tfevents 日志，TensorBoard 直接可看

### 🐞 修复

* 训练步数很小（如 100 步）训练结束后**再次点开始会闪退**（`std::thread` join 问题）
* 模型**切换时界面卡死**、下拉里方向键浏览会反复触发加载（改为 `activated` + 异步合并请求）
* 上下文/回复长度输入极端值（如 2048000 / 11100）导致异常 → 现在自动钳制（512~131072 / 1~4096 / 温度 0.01~2.0）
* 多行回复只显示第一行（QLabel wordWrap 高度未按宽度计算）
* 高 DPI 缩放（125%/150%）下窗口超出屏幕 → 默认最大化
* 增加全局异常兜底：未捕获异常弹窗提示并写入 `%TEMP%\gpt_trainer_terminate.log`

---

## 📦 下载资产

| 文件 | 大小 | SHA256 |
|---|---|---|
| **gpt_trainer_v1.1.0_win64_cu130.zip** | 1.88 GB | `8BA5F746D6B59A26530A69451F63C480CE51D806784FCEC99F88F82A2D405DBE` |
| model3.gguf（可选示例模型，本项目训练产物导出） | 222 MB | `E0DDB633066D397CDB7BD48F36D7D15758C9B401A2A46615F348034C52D7792D` |

校验（PowerShell）：

```powershell
Get-FileHash .\gpt_trainer_v1.1.0_win64_cu130.zip -Algorithm SHA256
```

## 💻 运行要求

| | 要求 |
|---|---|
| 系统 | Windows 10 1809+ / 11（x64） |
| 显卡 | 训练 / GPU 推理需 NVIDIA 显卡（驱动支持 CUDA 13）；无显卡自动回退 CPU |
| 显存 | 建议 6 GB+ |
| 内存 | 8 GB+ |

**开箱即用**：Qt 6.9.3、VC++ 运行时、libtorch + cuDNN + cuBLAS、CUDA/CPU 双 llama.cpp 引擎、中文分词器 `spm.model` 全部内置。

## 🚀 使用

1. 解压 zip 到任意目录（中英文路径均可）
2. 双击 `gpt_trainer.exe`
3. 对话：把 `.gguf` 放进 `models\gguf\` → 下拉选中 → 输入发消息
4. 训练：打开数据集 zip → 设置参数 → 开始训练

命令行同样可用：

```bat
gpt_trainer.exe chat --model "models\gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf" --prompt "你是谁？"
gpt_trainer.exe train --data data/processed/token_ids.txt --steps 40000 --batch 16 --seq-len 256
gpt_trainer.exe export-gguf --model saves\model3 --out models\gguf\model3.gguf
gpt_trainer.exe help
```

## ⚠️ 已知问题

* 本项目训练出的基座模型未做指令微调，可能出现「续写」而非问答，属正常现象
* 内置 CUDA 引擎按 sm_86 编译（RTX 30 系），更老的显卡请用 CPU 引擎或自行替换为官方 CUDA 版 `llama-server.exe`
* 单包 1.88 GB，GitHub 单文件上限 2 GB，请勿再打包附加大模型

## 📄 License

[MIT](https://github.com/zsdxhz/gpt_trainer/blob/main/LICENSE)
