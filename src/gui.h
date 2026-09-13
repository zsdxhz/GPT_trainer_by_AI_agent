#pragma once
// gui.h - Qt6 GUI（训练 + 对话），对应 python/gui.py，MIUIX(HyperOS) 风格
// 注意：必须先包含 torch（model.h），再包含 Qt 头文件，
// 并用 QT_NO_KEYWORDS 禁用 Qt 的 slots 宏，避免污染 torch 头文件。
#ifndef QT_NO_KEYWORDS
#define QT_NO_KEYWORDS
#endif

#include "config.h"
#include "model.h"
#include "tokenizer_worker.h"

#include <QMainWindow>

#include <atomic>
#include <filesystem>
#include <memory>
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
class QTabWidget;
class QTextEdit;

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

 private:
  // ==================== 训练页 ====================
  QLineEdit* data_path_edit_ = nullptr;
  QLineEdit* lr_edit_ = nullptr;
  QLineEdit* batch_edit_ = nullptr;
  QLineEdit* seq_len_edit_ = nullptr;
  QLineEdit* total_steps_edit_ = nullptr;
  QLineEdit* epochs_edit_ = nullptr;
  QLineEdit* stride_edit_ = nullptr;
  QLineEdit* embed_dim_edit_ = nullptr;
  QLineEdit* num_heads_edit_ = nullptr;
  QLineEdit* num_layers_edit_ = nullptr;
  QLineEdit* init_model_edit_ = nullptr;
  QPushButton* init_model_browse_btn_ = nullptr;
  QLineEdit* ckpt_interval_edit_ = nullptr;
  QCheckBox* resume_check_ = nullptr;
  QPushButton* data_browse_btn_ = nullptr;
  QPushButton* start_btn_ = nullptr;
  QPushButton* stop_btn_ = nullptr;
  QPushButton* quit_btn_ = nullptr;
  QLabel* loss_label_ = nullptr;
  QLabel* epoch_label_ = nullptr;
  QLabel* status_label_ = nullptr;
  QProgressBar* progress_bar_ = nullptr;
  QTextEdit* train_log_text_ = nullptr;

  // ==================== 对话页 ====================
  QComboBox* model_combo_ = nullptr;
  QPushButton* load_model_btn_ = nullptr;
  QPushButton* delete_model_btn_ = nullptr;
  QPushButton* refresh_btn_ = nullptr;
  QPushButton* open_dataset_btn_ = nullptr;
  QLabel* dataset_info_label_ = nullptr;
  QLabel* tokenizer_path_label_ = nullptr;
  QPushButton* select_tokenizer_btn_ = nullptr;
  QLabel* model_info_label_ = nullptr;
  QTextEdit* history_text_ = nullptr;
  QTextEdit* sys_log_text_ = nullptr;
  QTextEdit* input_text_ = nullptr;
  QPushButton* generate_btn_ = nullptr;
  QPushButton* clear_history_btn_ = nullptr;
  QPushButton* clear_syslog_btn_ = nullptr;
  // 生成参数
  QLineEdit* gen_max_tokens_edit_ = nullptr;
  QLineEdit* gen_temperature_edit_ = nullptr;
  QLineEdit* gen_topk_edit_ = nullptr;
  QCheckBox* gen_multi_turn_check_ = nullptr;
  QCheckBox* gen_template_check_ = nullptr;
  QString chat_context_; // 多轮对话上下文（文本）

  // ==================== 分词器页 ====================
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

  std::vector<std::filesystem::path> tokenizer_files_;
  std::filesystem::path tokenizer_output_dir_;
  std::shared_ptr<sentencepiece::SentencePieceProcessor> tokenizer_sp_;
  std::thread tokenizer_thread_;
  std::atomic<bool> tokenizer_running_{false};
  TokenizerResult tokenizer_result_;

  QTabWidget* tabs_ = nullptr;

  // ==================== 状态 ====================
  Config cfg_;
  std::atomic<Trainer*> trainer_{nullptr};
  std::thread training_thread_;
  std::atomic<bool> training_active_{false};
  std::thread gen_thread_;
  std::atomic<bool> generating_{false};
  bool resume_requested_ = false;      // 训练线程启动前读取
  std::filesystem::path init_model_path_;

  GPT inference_model_{nullptr};
  std::shared_ptr<sentencepiece::SentencePieceProcessor> sp_;
  std::string sp_model_path_;
  std::string loaded_model_dir_;
  int64_t vocab_size_ = 8000;
  int64_t max_seq_len_ = 256;
  long long target_epochs_ = 10; // 进度条 Epoch 显示用

  // ==================== 构建 ====================
  void setup_train_tab();
  void setup_chat_tab();
  void setup_tokenizer_tab();
  void apply_miuix_theme();

  // ==================== 辅助 ====================
  void train_log(const QString& text);
  void sys_log(const QString& text);
  void refresh_model_list();
  void refresh_tokenizer_stats();
  bool parse_training_params(QString* err);
  void set_training_ui_running(bool running);

  // 工作线程体
  void run_training();
  void run_generate(std::vector<int> ids, int gen_max, double gen_temp,
                    int gen_topk, bool use_template, bool multi_turn,
                    const QString& prompt);
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
};

} // namespace gpt
