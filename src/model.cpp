#include "model.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "json_util.h"
#include "legacy_serial.h"
#include "path_util.h"

namespace gpt {

using namespace torch::indexing;

bool sdpa_available() {
  static bool checked = false;
  static bool ok = false;
  if (!checked) {
    checked = true;
    try {
      if (!torch::cuda::is_available()) {
        ok = false;
      } else {
        auto q = torch::randn({1, 2, 4, 8},
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        at::scaled_dot_product_attention(q, q, q, {}, 0.0, true);
        torch::cuda::synchronize();
        ok = true;
      }
    } catch (...) {
      ok = false;
    }
  }
  return ok;
}

// ---------------------------------------------------------------------------
// SelfAttention
// ---------------------------------------------------------------------------
SelfAttentionImpl::SelfAttentionImpl(const GPTConfig& cfg)
    : qkv(register_module("qkv", torch::nn::Linear(cfg.embed_dim, 3 * cfg.embed_dim))),
      proj(register_module("proj", torch::nn::Linear(cfg.embed_dim, cfg.embed_dim))),
      attn_dropout(register_module("dropout", torch::nn::Dropout(cfg.dropout))) {
  if (cfg.embed_dim % cfg.num_heads != 0) {
    throw std::runtime_error("嵌入维度必须能被注意力头数整除");
  }
  num_heads = cfg.num_heads;
  head_dim = cfg.embed_dim / cfg.num_heads;

  auto ones = torch::ones({cfg.max_seq_len, cfg.max_seq_len});
  auto tril = torch::tril(ones).view({1, 1, cfg.max_seq_len, cfg.max_seq_len});
  register_buffer("mask", tril);
}

torch::Tensor SelfAttentionImpl::forward(torch::Tensor x) {
  const int64_t B = x.size(0), T = x.size(1), C = x.size(2);
  // [B,T,3C] -> [B,T,3,H,D] -> [3,B,H,T,D]
  auto qkv_t = qkv(x).reshape({B, T, 3, num_heads, head_dim}).permute({2, 0, 3, 1, 4});
  auto q = qkv_t[0], k = qkv_t[1], v = qkv_t[2];

  torch::Tensor y;
  if (use_sdpa && x.is_cuda()) {
    // 优化：fused 因果注意力（flash / mem-efficient kernel），数学上等价于手工实现
    y = at::scaled_dot_product_attention(
        q, k, v, {}, is_training() ? attn_dropout->options.p() : 0.0, true);
  } else {
    auto att = q.matmul(k.transpose(-2, -1)) * (1.0 / std::sqrt((double)head_dim));
    auto m = mask.index({Slice(), Slice(), Slice(0, T), Slice(0, T)});
    att = att.masked_fill(m.eq(0), -std::numeric_limits<double>::infinity());
    att = torch::softmax(att, -1);
    att = attn_dropout(att);
    y = att.matmul(v);
  }
  y = y.transpose(1, 2).contiguous().reshape({B, T, C});
  return proj(y);
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------
FeedForwardImpl::FeedForwardImpl(const GPTConfig& cfg)
    : net(register_module("net", torch::nn::Sequential())) {
  net->push_back(torch::nn::Linear(cfg.embed_dim, 4 * cfg.embed_dim));
  net->push_back(torch::nn::GELU());
  net->push_back(torch::nn::Linear(4 * cfg.embed_dim, cfg.embed_dim));
  net->push_back(torch::nn::Dropout(cfg.dropout));
}

torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {
  return net->forward(x);
}

// ---------------------------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------------------------
TransformerBlockImpl::TransformerBlockImpl(const GPTConfig& cfg)
    : ln1(register_module("ln1", torch::nn::LayerNorm(
                                    torch::nn::LayerNormOptions({cfg.embed_dim})))),
      attn(register_module("attn", SelfAttention(cfg))),
      ln2(register_module("ln2", torch::nn::LayerNorm(
                                    torch::nn::LayerNormOptions({cfg.embed_dim})))),
      ff(register_module("ff", FeedForward(cfg))) {}

torch::Tensor TransformerBlockImpl::forward(torch::Tensor x) {
  x = x + attn->forward(ln1(x));
  x = x + ff->forward(ln2(x));
  return x;
}

// ---------------------------------------------------------------------------
// GPT
// ---------------------------------------------------------------------------
GPTImpl::GPTImpl(const GPTConfig& cfg)
    : token_embedding(register_module(
          "token_embedding", torch::nn::Embedding(cfg.vocab_size, cfg.embed_dim))),
      position_embedding(register_module(
          "position_embedding", torch::nn::Embedding(cfg.max_seq_len, cfg.embed_dim))),
      blocks(register_module("blocks", torch::nn::Sequential())),
      ln_f(register_module(
          "ln_f", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.embed_dim})))),
      lm_head(register_module("lm_head",
                              torch::nn::Linear(cfg.embed_dim, cfg.vocab_size))),
      config(cfg) {
  for (int64_t i = 0; i < cfg.num_layers; ++i) {
    blocks->push_back(TransformerBlock(cfg)); // 名称 "0".."N-1"，与 python Sequential 一致
  }
  // 权重初始化（与 python _init_weights 一致）
  apply([](torch::nn::Module& m) {
    if (auto* lin = m.as<torch::nn::Linear>()) {
      torch::nn::init::normal_(lin->weight, 0.0, 0.02);
      if (lin->options.bias()) torch::nn::init::zeros_(lin->bias);
    } else if (auto* emb = m.as<torch::nn::Embedding>()) {
      torch::nn::init::normal_(emb->weight, 0.0, 0.02);
    }
  });
  // 优化开关：CUDA + SDPA 可用时启用 fused 注意力
  if (sdpa_available()) {
    for (auto& child : blocks->children()) {
      auto block = std::dynamic_pointer_cast<TransformerBlockImpl>(child);
      if (block) block->attn->use_sdpa = true;
    }
  }
}

torch::Tensor GPTImpl::forward(torch::Tensor idx) {
  const int64_t B = idx.size(0), T = idx.size(1);
  if (T > config.max_seq_len) {
    throw std::runtime_error("输入序列长度超过 max_seq_len");
  }
  auto tok_emb = token_embedding(idx);
  auto pos = torch::arange(T, idx.options()).unsqueeze(0).expand({B, T});
  auto pos_emb = position_embedding(pos);

  auto x = tok_emb + pos_emb;
  x = blocks->forward(x);
  x = ln_f(x);
  return lm_head(x);
}

torch::Tensor GPTImpl::generate(torch::Tensor idx, int64_t max_new_tokens,
                                double temperature,
                                std::optional<int64_t> top_k) {
  for (int64_t i = 0; i < max_new_tokens; ++i) {
    auto idx_cond =
        idx.index({Slice(), Slice(-config.max_seq_len, None)});
    auto logits = forward(idx_cond).index({Slice(), -1, Slice()}) / temperature;

    if (top_k.has_value()) {
      int64_t k = std::min(*top_k, logits.size(-1));
      auto values = std::get<0>(torch::topk(logits, k, -1));
      auto thresh = values.index({Slice(), Slice(-1, None)});
      logits = torch::where(
          logits < thresh,
          torch::full_like(logits, -std::numeric_limits<float>::infinity()),
          logits);
    }
    auto probs = torch::softmax(logits, -1);
    auto idx_next = torch::multinomial(probs, 1, true);
    idx = torch::cat({idx, idx_next}, 1);
  }
  return idx;
}

// ---------------------------------------------------------------------------
// 从 saves/modelN 目录加载
// ---------------------------------------------------------------------------
GPT load_gpt_from_saves(const std::filesystem::path& model_dir,
                        torch::Device device) {
  namespace fs = std::filesystem;
  const fs::path config_path = model_dir / "config.json";
  const fs::path model_path = model_dir / "model.pt";
  if (!fs::exists(config_path) || !fs::exists(model_path)) {
    throw std::runtime_error("模型文件不完整，请检查 " +
                             path_to_utf8(model_dir));
  }

  std::ifstream f(config_path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  Json cfg_json = Json::parse(ss.str());

  GPTConfig mcfg;
  mcfg.vocab_size = cfg_json.get_int("vocab_size").value_or(8000);
  mcfg.embed_dim = cfg_json.get_int("embed_dim").value_or(256);
  mcfg.num_heads = cfg_json.get_int("num_heads").value_or(8);
  mcfg.num_layers = cfg_json.get_int("num_layers").value_or(6);
  mcfg.max_seq_len = cfg_json.get_int("max_seq_len").value_or(256);
  mcfg.dropout = cfg_json.get_number("dropout").value_or(0.1);

  auto model = GPT(mcfg);
  model->to(device);
  const auto sd = load_legacy_state_dict(model_path, device);
  load_state_dict_into(*model, sd);
  model->eval();
  return model;
}

} // namespace gpt
