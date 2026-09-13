#pragma once
// model.h - GPT 模型定义（对应 python/model.py）
// 子模块命名与 python 版完全一致，state_dict 键可互读：
//   token_embedding / position_embedding / blocks.N.{ln1,attn.{qkv,proj,dropout},ln2,ff.net}
//   ln_f / lm_head / blocks.N.attn.mask(buffer)
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <optional>

namespace gpt {

struct GPTConfig {
  int64_t vocab_size = 8000;
  int64_t embed_dim = 256;
  int64_t num_heads = 8;
  int64_t num_layers = 6;
  int64_t max_seq_len = 256;
  double dropout = 0.1;
};

// ---------------------------------------------------------------------------
// 多头因果自注意力
// ---------------------------------------------------------------------------
struct SelfAttentionImpl : torch::nn::Module {
  SelfAttentionImpl(const GPTConfig& cfg);

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear qkv{nullptr};
  torch::nn::Linear proj{nullptr};
  torch::nn::Dropout attn_dropout{nullptr}; // python 名为 dropout（无参数，不进 state_dict）
  torch::Tensor mask;                       // [1,1,max,max] tril buffer

  int64_t num_heads = 0;
  int64_t head_dim = 0;
  bool use_sdpa = false; // CUDA 时使用 fused 因果注意力（等价加速）
};
TORCH_MODULE(SelfAttention);

// ---------------------------------------------------------------------------
// 前馈网络 Linear -> GELU -> Linear -> Dropout
// ---------------------------------------------------------------------------
struct FeedForwardImpl : torch::nn::Module {
  FeedForwardImpl(const GPTConfig& cfg);

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Sequential net{nullptr};
};
TORCH_MODULE(FeedForward);

// ---------------------------------------------------------------------------
// Transformer Block（Pre-LN）
// ---------------------------------------------------------------------------
struct TransformerBlockImpl : torch::nn::Module {
  TransformerBlockImpl(const GPTConfig& cfg);

  torch::Tensor forward(torch::Tensor x);

  torch::nn::LayerNorm ln1{nullptr};
  SelfAttention attn{nullptr};
  torch::nn::LayerNorm ln2{nullptr};
  FeedForward ff{nullptr};
};
TORCH_MODULE(TransformerBlock);

// ---------------------------------------------------------------------------
// GPT
// ---------------------------------------------------------------------------
struct GPTImpl : torch::nn::Module {
  explicit GPTImpl(const GPTConfig& cfg);

  torch::Tensor forward(torch::Tensor idx);
  // 自回归生成（temperature / top_k 采样）
  torch::Tensor generate(torch::Tensor idx, int64_t max_new_tokens,
                         double temperature = 1.0,
                         std::optional<int64_t> top_k = std::nullopt);

  torch::nn::Embedding token_embedding{nullptr};
  torch::nn::Embedding position_embedding{nullptr};
  torch::nn::Sequential blocks{nullptr};
  torch::nn::LayerNorm ln_f{nullptr};
  torch::nn::Linear lm_head{nullptr};

  GPTConfig config;
};
TORCH_MODULE(GPT);

// CUDA 下探测 scaled_dot_product_attention 是否可用（用于优化开关）
bool sdpa_available();

// 从 saves/modelN 目录加载模型（读取 config.json + model.pt），
// 与 python gui.py 的 load_selected_model 逻辑一致；可加载 Python 训练的权重
GPT load_gpt_from_saves(const std::filesystem::path& model_dir,
                        torch::Device device);

} // namespace gpt
