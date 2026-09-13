// main.cpp - 启动入口（对应 python/main.py）
//   无参数      : GUI 模式
//   --cli       : 命令行模式
//   --selftest  : 互读自检（加载 Python 训练的模型并生成文本）
#include <torch/torch.h>

#include <sentencepiece_processor.h>

#include <QApplication>
#include <QFont>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

#include <windows.h>

#include "config.h"
#include "gui.h"
#include "legacy_serial.h"
#include "path_util.h"
#include "tokenizer_worker.h"
#include "train.h"

namespace fs = std::filesystem;

// WIN32 子系统默认无控制台：附加到父进程控制台以便输出日志。
// 仅在标准输出句柄无效时重定向到控制台（不破坏管道/文件重定向）。
static void attach_console() {
  if (!AttachConsole(ATTACH_PARENT_PROCESS)) return;
  HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
  if (h == nullptr || h == INVALID_HANDLE_VALUE) {
    FILE* fp = nullptr;
    freopen_s(&fp, "CONOUT$", "w", stdout);
    freopen_s(&fp, "CONOUT$", "w", stderr);
  }
  SetConsoleOutputCP(CP_UTF8);
}

// ---------------------------------------------------------------------------
// 互读自检：加载 Python(py训练器) 训练的模型，前向 + 生成
// ---------------------------------------------------------------------------
static int run_selftest() {
  std::cout << "=== GPT 互读自检 ===" << std::endl;
  gpt::Config cfg = gpt::Config::defaults();
  cfg.init_dirs();
  const torch::Device dev = torch::cuda::is_available()
                                ? torch::Device(torch::kCUDA)
                                : torch::Device(torch::kCPU);
  std::cout << "设备: " << (dev.is_cuda() ? "cuda" : "cpu") << std::endl;

  // 1) 加载 Python(py训练器) 训练的模型
  const fs::path model_dir = cfg.project_root().parent_path() /
                             fs::path(L"py训练器") / fs::path(L"saves") /
                             fs::path(L"model1");
  if (!fs::exists(model_dir / "model.pt")) {
    std::cerr << "未找到 Python 模型: " << gpt::path_to_utf8(model_dir)
              << std::endl;
    return 2;
  }
  auto model = gpt::load_gpt_from_saves(model_dir, dev);
  std::cout << "[OK] 已加载 Python 模型: " << gpt::path_to_utf8(model_dir)
            << std::endl;

  // 2) 前向传播
  {
    torch::NoGradGuard guard;
    auto x = torch::randint(0, model->config.vocab_size, {2, 32},
                            torch::TensorOptions().dtype(torch::kInt64))
                 .to(dev);
    auto logits = model->forward(x);
    std::cout << "[OK] 前向输出形状: " << logits.sizes()
              << ", 均值: " << logits.mean().item<float>() << std::endl;
  }

  // 3) 生成（使用 python 侧训练的 spm.model）
  const fs::path spm_path = cfg.project_root().parent_path() /
                            fs::path(L"py训练器") / fs::path(L"models") /
                            fs::path(L"spm.model");
  sentencepiece::SentencePieceProcessor sp;
  const auto status = sp.Load(spm_path.string());
  if (!status.ok()) {
    std::cerr << "分词器加载失败: " << status.ToString() << std::endl;
    return 3;
  }
  std::vector<int> ids;
  sp.Encode("你好，今天天气不错", &ids);
  std::vector<int64_t> in_ids(ids.begin(), ids.end());
  std::vector<int64_t> out_ids;
  {
    torch::NoGradGuard guard;
    auto input = torch::tensor(in_ids, torch::TensorOptions().dtype(torch::kInt64))
                     .reshape({1, (int64_t)in_ids.size()})
                     .to(dev);
    auto output = model->generate(input, 30, 0.8, 40);
    auto cpu = output[0].to(torch::kCPU).contiguous();
    const int64_t* data = cpu.data_ptr<int64_t>();
    out_ids.assign(data, data + cpu.numel());
  }
  std::vector<int> new_ids(out_ids.begin() + in_ids.size(), out_ids.end());
  std::string text;
  sp.Decode(new_ids, &text);
  std::cout << "[OK] 生成文本: " << text << std::endl;

  // 4) C++ 保存/加载回环（legacy 格式）
  const fs::path tmp = fs::temp_directory_path() / "gpt_roundtrip.pt";
  {
    const auto sd = gpt::collect_state_dict(*model);
    gpt::save_legacy_state_dict(tmp, sd);
  }
  auto m2 = gpt::GPT(model->config);
  {
    const auto sd2 = gpt::load_legacy_state_dict(tmp, torch::kCPU);
    gpt::load_state_dict_into(*m2, sd2);
  }
  fs::remove(tmp);
  const bool same = torch::equal(m2->lm_head->weight,
                                 model->lm_head->weight.to(torch::kCPU));
  std::cout << "[OK] 保存/加载回环通过 (权重一致: " << (same ? "是" : "否") << ")"
            << std::endl;
  if (!same) return 4;
  std::cout << "SELFTEST_OK" << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  attach_console();

  const std::string mode = (argc > 1) ? argv[1] : "";

  if (mode == "--cli") {
    std::cout << "=== GPT 训练 (命令行模式) ===" << std::endl;
    try {
      // 可选参数：--resume（从断点继续）；--init <模型.pt>（用已有权重初始化）
      gpt::TrainerOptions opts;
      for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--resume") {
          opts.resume = true;
        } else if (a == "--init" && i + 1 < argc) {
          opts.init_model = fs::path(argv[++i]);
        }
      }
      gpt::Config cfg = gpt::Config::defaults();
      cfg.init_dirs();
      gpt::Trainer trainer(cfg, {}, opts);
      trainer.train();
      return 0;
    } catch (const std::exception& e) {
      std::cerr << "训练出错: " << e.what() << std::endl;
      return 1;
    }
  }

  if (mode == "--selftest") {
    try {
      return run_selftest();
    } catch (const std::exception& e) {
      std::cerr << "自检出错: " << e.what() << std::endl;
      return 1;
    }
  }

  if (mode == "--loadtest" && argc > 2) {
    try {
      const auto sd = gpt::load_legacy_state_dict(fs::path(argv[2]), torch::kCPU);
      std::cout << "LOADTEST: 读取 " << sd.size() << " 个张量" << std::endl;
      for (const auto& [k, t] : sd) {
        std::cout << "  " << k << " dtype=" << t.scalar_type() << " sizes=" << t.sizes()
                  << " sum=" << t.sum().item<double>() << std::endl;
      }
      std::cout << "LOADTEST_OK" << std::endl;
      return 0;
    } catch (const std::exception& e) {
      std::cerr << "LOADTEST 失败: " << e.what() << std::endl;
      return 1;
    }
  }

  if (mode == "--tokenize" && argc >= 4) {
    // 命令行分词器训练：--tokenize <输出目录> <txt文件1> [txt文件2 ...]
    std::cout << "=== SentencePiece 分词器训练 (命令行模式) ===" << std::endl;
    try {
      gpt::TokenizerTask task;
      task.output_dir = fs::path(argv[2]);
      for (int i = 3; i < argc; ++i) task.input_files.push_back(fs::path(argv[i]));
      task.log = [](const std::string& s) { std::cout << s << std::endl; };
      gpt::TokenizerResult result;
      std::string err;
      if (!gpt::run_tokenizer_task(task, &result, &err)) {
        std::cerr << "分词器训练失败: " << err << std::endl;
        return 1;
      }
      std::cout << "TOKENIZE_OK" << std::endl;
      return 0;
    } catch (const std::exception& e) {
      std::cerr << "分词器训练出错: " << e.what() << std::endl;
      return 1;
    }
  }

  // ---- GUI 模式 ----
  QApplication app(argc, argv);
  app.setStyle(QStringLiteral("Fusion"));
  QFont font(QStringLiteral("Microsoft YaHei UI"), 10);
  app.setFont(font);

  gpt::TrainingGUI gui;
  gui.show();
  return app.exec();
}
