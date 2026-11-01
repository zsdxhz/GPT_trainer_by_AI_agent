#pragma once
// gui.h - Qt6 GUI（训练 + 对话 + 高级），Windows 11 Fluent 风格
//  训练 / 对话：只暴露普通使用者看得懂的功能
//  高级：推理引擎（llama.cpp）、生成调参、训练高级参数、分词器、诊断
// 注意：必须先包含 torch（model.h），再包含 Qt 头文件，
// 并用 QT_NO_KEYWORDS 禁用 Qt 的 slots 宏，避免污染 torch 头文件。
#ifndef QT_NO_KEYWORDS
#define QT_NO_KEYWORDS
#endif

#include "config.h"
#include "llama_engine.h"
#include "model.h"
#include "tokenizer_worker.h"

#include <QMainWindow>

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class QCheckBox;
class QCloseEvent;
class QComboBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QShowEvent;
class QStackedWidget;
class QTextEdit;

namespace fl {
class ChatView;
class NavRail;
class RingGauge;
class ToggleSwitch;
} // namespace fl

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace gpt {

class Trainer;

class TrainingGUI : public QMainWindow {
  Q_OBJECT
 public:
  explicit TrainingGUI(QWidget* parent = nullptr);
  ~TrainingGUI() override;

 protected:
  void closeEvent(QCloseEvent* event) override;
  void showEvent(QShowEvent* event) override;

 private:
  // ==================== 外壳 ====================
  fl::NavRail* nav_ = nullptr;
  QStackedWidget* pages_ = nullptr;
  QLabel* sb_device_ = nullptr;
  QLabel* sb_stats_ = nullptr;
  QLabel* sb_engine_ = nullptr;

  // ==================== 训练页（基础） ====================
  QLineEdit* data_path_edit_ = nullptr;
  QPushButton* data_browse_btn_ = nullptr;
  QPushButton* open_dataset_btn_ = nullptr;
  QLabel* dataset_info_label_ = nullptr;
  QLineEdit* lr_edit_ = nullptr;
  QLineEdit* batch_edit_ = nullptr;
  QLineEdit* seq_len_edit_ = nullptr;
  QLineEdit* epochs_edit_ = nullptr;
  QPushButton* start_btn_ = nullptr;
  QPushButton* stop_btn_ = nullptr;
  QPushButton* quit_btn_ = nullptr;
  QLabel* loss_label_ = nullptr;
  QLabel* epoch_label_ = nullptr;
  QLabel* step_label_ = nullptr;
  QLabel* status_label_ = nullptr;
  QProgressBar* progress_bar_ = nullptr;
  QTextEdit* train_log_text_ = nullptr;

  // ==================== 对话页（基础） ====================
  QComboBox* model_combo_ = nullptr;
  QPushButton* load_model_btn_ = nullptr;
  QPushButton* refresh_btn_ = nullptr;
  QLabel* model_info_label_ = nullptr;
  QLabel* engine_state_label_ = nullptr;
  fl::ChatView* chat_view_ = nullptr;
  QTextEdit* input_text_ = nullptr;
  QPushButton* generate_btn_ = nullptr;
  QPushButton* stop_gen_btn_ = nullptr;
  QPushButton* clear_history_btn_ = nullptr;
  QLineEdit* gen_max_tokens_edit_ = nullptr;
  QLineEdit* gen_temperature_edit_ = nullptr;
  QLineEdit* chat_ctx_edit_ = nullptr;
  QLabel* chat_stats_label_ = nullptr;
  fl::RingGauge* ctx_ring_ = nullptr;
  int chat_rounds_ = 0;
  long long chat_total_tokens_ = 0;
  long long ctx_used_ = 0;
  fl::ToggleSwitch* gen_multi_turn_check_ = nullptr;
  QString chat_context_; // 多轮对话上下文（文本）
  bool model_list_filling_ = false;

  // ==================== 模型加载（切换时合并请求，不阻塞 UI）====================
  struct ModelLoadJob {
    std::filesystem::path gguf;       // 最终交给 llama.cpp 的 GGUF
    std::filesystem::path model_dir;  // 空 = 外部 GGUF（无需导出）
    std::filesystem::path model_path; // saves/modelN/model.pt
  };
  void request_model_load(const ModelLoadJob& job);
  void model_load_worker();
  std::mutex load_mutex_;
  ModelLoadJob pending_job_;
  bool has_pending_job_ = false;
  std::atomic<bool> load_worker_running_{false};

  // ==================== 高级页：推理引擎 ====================
  QComboBox* engine_combo_ = nullptr;
  QComboBox* chat_mode_combo_ = nullptr;
  QLineEdit* server_path_edit_ = nullptr;
  QPushButton* server_browse_btn_ = nullptr;
  QPushButton* server_detect_btn_ = nullptr;
  QLineEdit* gguf_path_edit_ = nullptr;
  QPushButton* gguf_browse_btn_ = nullptr;
  QLineEdit* engine_ctx_edit_ = nullptr;
  QLineEdit* engine_threads_edit_ = nullptr;
  QLineEdit* engine_ngl_edit_ = nullptr;
  QPushButton* engine_restart_btn_ = nullptr;
  QPushButton* engine_stop_btn_ = nullptr;
  QLabel* engine_detail_label_ = nullptr;

  // ==================== 高级页：生成参数 ====================
  QLineEdit* gen_topk_edit_ = nullptr;
  QLineEdit* gen_topp_edit_ = nullptr;
  QLineEdit* gen_repeat_edit_ = nullptr;
  QLineEdit* gen_seed_edit_ = nullptr;
  QLineEdit* system_prompt_edit_ = nullptr;
  fl::ToggleSwitch* gen_template_check_ = nullptr;

  // ==================== 高级页：训练高级参数 ====================
  QLineEdit* total_steps_edit_ = nullptr;
  QLineEdit* stride_edit_ = nullptr;
  QLineEdit* embed_dim_edit_ = nullptr;
  QLineEdit* num_heads_edit_ = nullptr;
  QLineEdit* num_layers_edit_ = nullptr;
  QLineEdit* init_model_edit_ = nullptr;
  QPushButton* init_model_browse_btn_ = nullptr;
  QLineEdit* ckpt_interval_edit_ = nullptr;
  fl::ToggleSwitch* resume_check_ = nullptr;

  // ==================== 高级页：模型 / GGUF ====================
  QComboBox* adv_model_combo_ = nullptr;
  QPushButton* delete_model_btn_ = nullptr;
  QPushButton* export_gguf_btn_ = nullptr;
  QLineEdit* export_gguf_edit_ = nullptr;
  QLabel* export_result_label_ = nullptr;

  // ==================== 高级页：分词器 ====================
  QLabel* tokenizer_corpus_label_ = nullptr;
  QLabel* tokenizer_stats_label_ = nullptr;
  QLineEdit* tokenizer_out_edit_ = nullptr;
  QLineEdit* tok_threads_edit_ = nullptr;
  QLineEdit* tok_vocab_edit_ = nullptr;
  QLineEdit* tok_sentences_edit_ = nullptr;
  QPushButton* tokenizer_start_btn_ = nullptr;
  QTextEdit* tokenizer_log_text_ = nullptr;
  QTextEdit* tokenizer_test_input_ = nullptr;
  QPushButton* tokenizer_test_btn_ = nullptr;
  QTextEdit* tokenizer_test_output_ = nullptr;
  QLabel* tokenizer_path_label_ = nullptr;
  QPushButton* select_tokenizer_btn_ = nullptr;

  // ==================== 高级页：诊断 ====================
  QTextEdit* sys_log_text_ = nullptr;
  QPushButton* clear_syslog_btn_ = nullptr;
  QPushButton* open_log_dir_btn_ = nullptr;
  QLabel* diag_label_ = nullptr;

  std::vector<std::filesystem::path> tokenizer_files_;
  std::filesystem::path tokenizer_output_dir_;
  std::shared_ptr<sentencepiece::SentencePieceProcessor> tokenizer_sp_;
  std::thread tokenizer_thread_;
  std::atomic<bool> tokenizer_running_{false};
  TokenizerResult tokenizer_result_;

  // ==================== 状态 ====================
  Config cfg_;
  std::atomic<Trainer*> trainer_{nullptr};
  std::thread training_thread_;
  std::atomic<bool> training_active_{false};
  std::thread gen_thread_;
  std::atomic<bool> generating_{false};
  std::thread engine_thread_;
  std::thread export_thread_;
  std::atomic<bool> engine_starting_{false};
  bool resume_requested_ = false; // 训练线程启动前读取
  std::filesystem::path init_model_path_;

  GPT inference_model_{nullptr};
  std::shared_ptr<sentencepiece::SentencePieceProcessor> sp_;
  std::string sp_model_path_;
  std::string loaded_model_dir_;
  int64_t vocab_size_ = 8000;
  int64_t max_seq_len_ = 256;

  LlamaEngine llama_;
  // 本项目导出的 GGUF 词表是中文 SentencePiece：有匹配分词器时走 token 通道
  std::shared_ptr<sentencepiece::SentencePieceProcessor> llama_sp_;
  QString loaded_gguf_;      // 当前 llama.cpp 已加载的 GGUF
  QString last_export_gguf_; // 最近导出的 GGUF
  std::atomic<bool> stop_pending_{false};

  // ==================== 构建 ====================
  void setup_shell();
  void setup_train_tab();
  void setup_chat_tab();
  void setup_advanced_tab();
  void on_theme_changed();

  // ==================== 辅助 ====================
  void train_log(const QString& text);
  void sys_log(const QString& text);
  void set_status(const QString& text, const char* tone);
  void update_stats_bar();
  void update_chat_stats(int tokens, double secs, long long prompt_tokens = -1);
  void update_engine_ui();
  void refresh_model_list();
  void refresh_tokenizer_stats();
  bool parse_training_params(QString* err);
  bool load_dataset_zip(const QString& zip_path, bool quiet);
  void set_training_ui_running(bool running);
  void load_settings();
  void save_settings();
  QString current_model_name() const;
  std::filesystem::path models_gguf_dir() const;
  // 引擎/模型
  bool engine_is_llama() const;
  bool start_llama_for_gguf(const std::filesystem::path& gguf, bool interactive);
  void load_gguf_and_chat(const std::filesystem::path& gguf);
  QString build_llama_prompt(const QString& user_text) const;
  LlamaSamplingParams current_sampling() const;
  // 训练参数（高级字段可能是空指针以外的控件，统一读取）
  int read_int(QLineEdit* edit, int fallback) const;
  double read_double(QLineEdit* edit, double fallback) const;

  // 工作线程体
  void run_training();
  void run_generate(std::vector<int> ids, int gen_max, double gen_temp,
                    int gen_topk, bool use_template, bool multi_turn,
                    const QString& prompt);
  void run_llama_generate(const QString& prompt, const QString& full_prompt);
  void run_tokenizer_training(const TokenizerTask& task);

 private Q_SLOTS:
  void browse_data_file();
  void browse_init_model();
  void start_training();
  void stop_training();
  void open_dataset();
  void select_tokenizer();
  void load_selected_model();
  void delete_selected_model();
  void generate_response();
  void stop_generation();
  void clear_chat_history();
  void clear_sys_log();
  void on_progress(float loss, int epoch, long long step, long long total_steps);
  void on_training_done();
  void pick_tokenizer_corpus();
  void use_tokenizer_raw_dir();
  void browse_tokenizer_out();
  void start_tokenizer_training();
  void on_tokenizer_done(bool ok, const QString& err);
  void test_tokenize();
  void browse_server_exe();
  void detect_server_exe();
  void browse_gguf_file();
  void restart_engine();
  void stop_engine();
  void export_current_gguf();
  void open_llama_log();
};

} // namespace gpt
