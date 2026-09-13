#pragma once
// legacy_serial.h - Python torch.save(state_dict) 兼容的读写
//
// libtorch 2.14 的 InputArchive/OutputArchive 已改为 TorchScript(JIT) 后端，
// 无法直接读取 Python torch.save 的裸 state_dict（zip: data.pkl + data/<n>）。
// 本模块：
//   load : 使用官方 torch::jit::readArchiveAndTensors 读取 Python 保存的 state_dict
//   save : 手写与 Python 字节级一致的 pickle（协议 2）+ PyTorchStreamWriter，
//          产出文件可被 python torch.load 直接加载（双向互读）
#include <torch/torch.h>

#include <filesystem>
#include <string>
#include <vector>

namespace gpt {

// 读取 state_dict zip → [(键, 张量)]（兼容 "archive/data.pkl" 与 "data.pkl" 两种布局）
std::vector<std::pair<std::string, torch::Tensor>> load_legacy_state_dict(
    const std::filesystem::path& path, c10::optional<torch::Device> device);

// 写出与 python torch.save(state_dict) 兼容的 zip
void save_legacy_state_dict(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, torch::Tensor>>& tensors);

// 收集模型的 state_dict（参数 + buffer，与 python model.state_dict() 一致）
std::vector<std::pair<std::string, torch::Tensor>> collect_state_dict(
    const torch::nn::Module& module);

// 严格加载（类似 python load_state_dict(strict=True)：缺失/多余键报错）
void load_state_dict_into(
    torch::nn::Module& module,
    const std::vector<std::pair<std::string, torch::Tensor>>& sd);

} // namespace gpt
