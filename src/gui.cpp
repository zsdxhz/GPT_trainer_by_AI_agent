#include "gui.h"

#include <sentencepiece_processor.h>

#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollBar>
#include <QSplitter>
#include <QTabWidget>
#include <QTextEdit>
#include <QTime>
#include <QVBoxLayout>

#include <torch/torch.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json_util.h"
#include "path_util.h"
#include "train.h"

namespace fs = std::filesystem;

namespace gpt {

namespace {

// 路径 → QString（宽字符，避免编码转换歧义）
QString qpath(const fs::path& p) { return QString::fromStdWString(p.wstring()); }

// QString → fs::path（宽字符）
fs::path fspath(const QString& q) { return fs::path(q.toStdWString()); }

QString read_text_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return QString::fromUtf8(ss.str().c_str());
}

} // namespace

// ===========================================================================
// MIUIX (HyperOS) 主题 - 蓝色系
// ===========================================================================
static const char* kMiuixQss = R"QSS(
QWidget {
  background: #F5F6F8;
  color: #1F2329;
  font-family: "Microsoft YaHei UI";
  font-size: 10pt;
}
QMainWindow, QDialog {
  background: #F5F6F8;
}

/* ---- 卡片分组 ---- */
QGroupBox {
  background: #FFFFFF;
  border: 1px solid #E9EBEF;
  border-radius: 16px;
  margin-top: 16px;
  padding: 16px 14px 14px 14px;
  font-weight: bold;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 16px;
  top: 2px;
  padding: 0 6px;
  color: #1677FF;
}

/* ---- 标签页（胶囊风格） ---- */
QTabWidget::pane {
  border: none;
  background: transparent;
  top: -1px;
}
QTabBar::tab {
  background: transparent;
  color: #5A6270;
  padding: 9px 28px;
  margin: 4px 6px 4px 0px;
  border-radius: 14px;
  font-weight: bold;
  font-size: 11pt;
}
QTabBar::tab:selected {
  background: #FFFFFF;
  color: #1677FF;
}
QTabBar::tab:hover:!selected {
  background: #ECEDF1;
}

/* ---- 输入框 ---- */
QLineEdit {
  background: #FFFFFF;
  border: 1px solid #E0E3E8;
  border-radius: 12px;
  padding: 7px 12px;
  selection-background-color: #BADCFF;
}
QLineEdit:focus {
  border: 1px solid #1677FF;
}
QLineEdit:disabled {
  background: #F2F3F5;
  color: #B0B6C0;
}

/* ---- 按钮 ---- */
QPushButton {
  background: #FFFFFF;
  color: #1677FF;
  border: 1px solid #91CAFF;
  border-radius: 12px;
  padding: 8px 20px;
  font-weight: bold;
}
QPushButton:hover {
  background: #E8F3FF;
  border-color: #1677FF;
}
QPushButton:pressed {
  background: #D6E9FF;
}
QPushButton:disabled {
  background: #F2F3F5;
  color: #B8BEC8;
  border-color: #E5E8ED;
}
/* 主操作（蓝色） */
QPushButton#start_btn {
  background: #1677FF;
  color: #FFFFFF;
  border: none;
  border-radius: 12px;
  padding: 9px 28px;
  font-size: 11pt;
}
QPushButton#start_btn:hover { background: #4096FF; }
QPushButton#start_btn:pressed { background: #0958D9; }
QPushButton#start_btn:disabled { background: #91CAFF; color: #FFFFFF; }
/* 危险操作 */
QPushButton#stop_btn, QPushButton#delete_model_btn {
  background: #E5484D;
  color: #FFFFFF;
  border: none;
}
QPushButton#stop_btn:hover, QPushButton#delete_model_btn:hover { background: #FF6B6E; }
QPushButton#stop_btn:disabled, QPushButton#delete_model_btn:disabled { background: #F2F3F5; color: #B8BEC8; }
/* 蓝色操作 */
QPushButton#open_dataset_btn {
  background: #1677FF;
  color: #FFFFFF;
  border: none;
}
QPushButton#open_dataset_btn:hover { background: #4096FF; }

/* ---- 进度条 ---- */
QProgressBar {
  background: #F0F1F4;
  border: none;
  border-radius: 9px;
  min-height: 18px;
  max-height: 18px;
  text-align: center;
  color: #8A919F;
  font-weight: bold;
  font-size: 9pt;
}
QProgressBar::chunk {
  border-radius: 9px;
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #69B1FF, stop:1 #1677FF);
}

/* ---- 文本区 ---- */
QTextEdit {
  background: #FFFFFF;
  border: 1px solid #E0E3E8;
  border-radius: 12px;
  padding: 6px 8px;
}
QTextEdit:focus {
  border: 1px solid #1677FF;
}

/* ---- 下拉框 ---- */
QComboBox {
  background: #FFFFFF;
  border: 1px solid #E0E3E8;
  border-radius: 12px;
  padding: 7px 12px;
  min-width: 120px;
}
QComboBox:focus { border: 1px solid #1677FF; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox QAbstractItemView {
  background: #FFFFFF;
  border: 1px solid #E9EBEF;
  border-radius: 10px;
  selection-background-color: #E8F3FF;
  selection-color: #1677FF;
  outline: none;
  padding: 4px;
}

/* ---- 滚动条 ---- */
QScrollBar:vertical {
  background: transparent;
  width: 10px;
  margin: 2px;
}
QScrollBar::handle:vertical {
  background: #D8DCE3;
  border-radius: 4px;
  min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #69B1FF; }
QScrollBar:horizontal {
  background: transparent;
  height: 10px;
  margin: 2px;
}
QScrollBar::handle:horizontal {
  background: #D8DCE3;
  border-radius: 4px;
  min-width: 24px;
}
QScrollBar::handle:horizontal:hover { background: #69B1FF; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ---- 分割器 ---- */
QSplitter::handle { background: #EDEFF3; }

/* ---- 标签 ---- */
QLabel { background: transparent; }
QLabel[muted="true"] { color: #8A919F; }
QLabel[status="ok"] { color: #00B578; }
QLabel[status="warn"] { color: #E5484D; }
)QSS";

// ===========================================================================
// 构造
// ===========================================================================
TrainingGUI::TrainingGUI(QWidget* parent) : QMainWindow(parent) {
  cfg_ = Config::defaults();
  cfg_.init_dirs();
  vocab_size_ = cfg_.vocab_size;
  max_seq_len_ = cfg_.max_seq_len;

  setWindowTitle(QStringLiteral("GPT 训练与对话系统"));
  setGeometry(100, 100, 1000, 880);

  auto* central = new QWidget(this);
  setCentralWidget(central);
  auto* main_layout = new QVBoxLayout(central);
  main_layout->setContentsMargins(16, 16, 16, 16);

  tabs_ = new QTabWidget(central);
  main_layout->addWidget(tabs_);

  auto* train_tab = new QWidget();
  tabs_->addTab(train_tab, QStringLiteral("训练"));
  auto* chat_tab = new QWidget();
  tabs_->addTab(chat_tab, QStringLiteral("对话"));
  auto* tokenizer_tab = new QWidget();
  tabs_->addTab(tokenizer_tab, QStringLiteral("分词器"));

  setup_train_tab();
  setup_chat_tab();
  setup_tokenizer_tab();
  apply_miuix_theme();

  if (fs::exists(cfg_.data_path)) {
    data_path_edit_->setText(qpath(fs::path(cfg_.data_path)));
  }

  // 分词器默认输出目录：<项目根>/data
  tokenizer_output_dir_ = fs::path(cfg_.project_root()) / "data";
  tokenizer_out_edit_->setText(qpath(tokenizer_output_dir_));
  refresh_tokenizer_stats();
}

TrainingGUI::~TrainingGUI() {
  if (auto* t = trainer_.load()) t->stop_training.store(true);
  if (training_thread_.joinable()) {
    // 等待当前批处理结束（每批都会检查停止标志）
    training_thread_.join();
  }
  if (gen_thread_.joinable()) gen_thread_.detach();
  if (tokenizer_thread_.joinable()) tokenizer_thread_.detach();
}

// ===========================================================================
// 训练页
// ===========================================================================
void TrainingGUI::setup_train_tab() {
  QWidget* tab = tabs_->widget(0);
  auto* layout = new QVBoxLayout(tab);

  // ---------- 数据路径 ----------
  auto* config_group = new QGroupBox(QStringLiteral("数据路径"), tab);
  auto* config_layout = new QVBoxLayout(config_group);
  auto* data_layout = new QHBoxLayout();
  data_layout->addWidget(new QLabel(QStringLiteral("数据文件:"), config_group));
  data_path_edit_ = new QLineEdit(config_group);
  data_path_edit_->setPlaceholderText(QStringLiteral("选择 token_ids.txt 文件..."));
  data_layout->addWidget(data_path_edit_);
  data_browse_btn_ = new QPushButton(QStringLiteral("浏览"), config_group);
  data_browse_btn_->setObjectName("browse_btn");
  connect(data_browse_btn_, &QPushButton::clicked, this, &TrainingGUI::browse_data_file);
  data_layout->addWidget(data_browse_btn_);
  config_layout->addLayout(data_layout);
  layout->addWidget(config_group);

  // ---------- 参数设置面板 ----------
  auto* param_group =
      new QGroupBox(QStringLiteral("训练参数设置 (调整后点击开始训练生效)"), tab);
  auto* param_layout = new QVBoxLayout(param_group);

  // 第一行：学习率、批次大小、序列长度
  auto* row1 = new QHBoxLayout();
  row1->addWidget(new QLabel(QStringLiteral("学习率:"), param_group));
  lr_edit_ = new QLineEdit(QStringLiteral("0.0003"), param_group);
  lr_edit_->setMaximumWidth(110);
  row1->addWidget(lr_edit_);

  row1->addWidget(new QLabel(QStringLiteral("批次大小:"), param_group));
  batch_edit_ = new QLineEdit(QStringLiteral("16"), param_group);
  batch_edit_->setMaximumWidth(80);
  row1->addWidget(batch_edit_);

  row1->addWidget(new QLabel(QStringLiteral("序列长度:"), param_group));
  seq_len_edit_ = new QLineEdit(QStringLiteral("256"), param_group);
  seq_len_edit_->setMaximumWidth(80);
  row1->addWidget(seq_len_edit_);

  row1->addStretch();
  param_layout->addLayout(row1);

  // 第二行：训练步数、Epoch数、滑动步长
  auto* row2 = new QHBoxLayout();
  row2->addWidget(new QLabel(QStringLiteral("训练步数:"), param_group));
  total_steps_edit_ = new QLineEdit(param_group);
  total_steps_edit_->setPlaceholderText(QStringLiteral("留空则按 Epoch 数训练"));
  total_steps_edit_->setMaximumWidth(170);
  row2->addWidget(total_steps_edit_);

  row2->addWidget(new QLabel(QStringLiteral("Epoch 数:"), param_group));
  epochs_edit_ = new QLineEdit(QStringLiteral("10"), param_group);
  epochs_edit_->setMaximumWidth(80);
  row2->addWidget(epochs_edit_);

  row2->addWidget(new QLabel(QStringLiteral("滑动步长:"), param_group));
  stride_edit_ = new QLineEdit(QStringLiteral("128"), param_group);
  stride_edit_->setMaximumWidth(80);
  row2->addWidget(stride_edit_);

  row2->addStretch();
  param_layout->addLayout(row2);

  // 第三行：模型架构参数
  auto* row3 = new QHBoxLayout();
  row3->addWidget(new QLabel(QStringLiteral("嵌入维度:"), param_group));
  embed_dim_edit_ = new QLineEdit(QStringLiteral("256"), param_group);
  embed_dim_edit_->setMaximumWidth(80);
  row3->addWidget(embed_dim_edit_);

  row3->addWidget(new QLabel(QStringLiteral("注意力头数:"), param_group));
  num_heads_edit_ = new QLineEdit(QStringLiteral("8"), param_group);
  num_heads_edit_->setMaximumWidth(80);
  row3->addWidget(num_heads_edit_);

  row3->addWidget(new QLabel(QStringLiteral("层数:"), param_group));
  num_layers_edit_ = new QLineEdit(QStringLiteral("6"), param_group);
  num_layers_edit_->setMaximumWidth(80);
  row3->addWidget(num_layers_edit_);

  row3->addStretch();
  param_layout->addLayout(row3);

  // 第四行：初始权重（warm start）
  auto* row3b = new QHBoxLayout();
  row3b->addWidget(new QLabel(QStringLiteral("初始权重(可选):"), param_group));
  init_model_edit_ = new QLineEdit(param_group);
  init_model_edit_->setPlaceholderText(QStringLiteral("留空 = 随机初始化，可填 saves/modelN/model.pt 继续训练"));
  row3b->addWidget(init_model_edit_);
  init_model_browse_btn_ = new QPushButton(QStringLiteral("浏览"), param_group);
  connect(init_model_browse_btn_, &QPushButton::clicked, this, &TrainingGUI::browse_init_model);
  row3b->addWidget(init_model_browse_btn_);
  param_layout->addLayout(row3b);

  // 第五行：断点
  auto* row3c = new QHBoxLayout();
  row3c->addWidget(new QLabel(QStringLiteral("断点保存间隔(步):"), param_group));
  ckpt_interval_edit_ = new QLineEdit(QStringLiteral("5000"), param_group);
  ckpt_interval_edit_->setMaximumWidth(100);
  row3c->addWidget(ckpt_interval_edit_);
  resume_check_ = new QCheckBox(QStringLiteral("从上次断点继续 (models/training_state)"), param_group);
  row3c->addWidget(resume_check_);
  row3c->addStretch();
  param_layout->addLayout(row3c);

  // 第六行：提示
  auto* row4 = new QHBoxLayout();
  auto* tip_label = new QLabel(
      QStringLiteral("💡 提示：训练步数优先于 Epoch 数 | 嵌入维度必须能被头数整除 | 每 N 步自动保存断点，断电/停止后可勾选「从断点继续」续训"),
      param_group);
  tip_label->setProperty("muted", true);
  tip_label->setStyleSheet("font-size: 9pt;");
  row4->addWidget(tip_label);
  row4->addStretch();
  param_layout->addLayout(row4);

  layout->addWidget(param_group);

  // ---------- 状态区域 ----------
  auto* status_group = new QGroupBox(QStringLiteral("训练状态"), tab);
  auto* status_layout = new QVBoxLayout(status_group);
  loss_label_ = new QLabel(QStringLiteral("Loss: --"), status_group);
  QFont loss_font(QStringLiteral("Arial"), 12);
  loss_font.setBold(true);
  loss_label_->setFont(loss_font);
  status_layout->addWidget(loss_label_);
  epoch_label_ = new QLabel(QStringLiteral("Epoch: --"), status_group);
  status_layout->addWidget(epoch_label_);
  progress_bar_ = new QProgressBar(status_group);
  status_layout->addWidget(progress_bar_);
  status_label_ = new QLabel(QStringLiteral("就绪"), status_group);
  status_label_->setProperty("status", "ok");
  status_layout->addWidget(status_label_);
  layout->addWidget(status_group);

  // ---------- 训练日志 ----------
  auto* log_group = new QGroupBox(QStringLiteral("训练日志"), tab);
  auto* log_layout = new QVBoxLayout(log_group);
  train_log_text_ = new QTextEdit(log_group);
  train_log_text_->setReadOnly(true);
  train_log_text_->setFont(QFont(QStringLiteral("Consolas"), 9));
  log_layout->addWidget(train_log_text_);
  layout->addWidget(log_group);

  // ---------- 按钮 ----------
  auto* btn_layout = new QHBoxLayout();
  start_btn_ = new QPushButton(QStringLiteral("开始训练"), tab);
  start_btn_->setObjectName("start_btn");
  connect(start_btn_, &QPushButton::clicked, this, &TrainingGUI::start_training);
  btn_layout->addWidget(start_btn_);
  stop_btn_ = new QPushButton(QStringLiteral("停止训练"), tab);
  stop_btn_->setObjectName("stop_btn");
  connect(stop_btn_, &QPushButton::clicked, this, &TrainingGUI::stop_training);
  stop_btn_->setEnabled(false);
  btn_layout->addWidget(stop_btn_);
  quit_btn_ = new QPushButton(QStringLiteral("退出"), tab);
  connect(quit_btn_, &QPushButton::clicked, this, &TrainingGUI::close);
  btn_layout->addWidget(quit_btn_);
  layout->addLayout(btn_layout);
}

// ===========================================================================
// 对话页
// ===========================================================================
void TrainingGUI::setup_chat_tab() {
  QWidget* tab = tabs_->widget(1);
  auto* layout = new QVBoxLayout(tab);

  // ---------- 模型管理 + 数据集加载 ----------
  auto* model_group = new QGroupBox(QStringLiteral("模型与数据集管理"), tab);
  auto* model_layout = new QHBoxLayout(model_group);
  model_layout->addWidget(new QLabel(QStringLiteral("已保存模型:"), model_group));
  model_combo_ = new QComboBox(model_group);
  model_combo_->setMinimumWidth(150);
  model_layout->addWidget(model_combo_);
  load_model_btn_ = new QPushButton(QStringLiteral("加载模型"), model_group);
  connect(load_model_btn_, &QPushButton::clicked, this, &TrainingGUI::load_selected_model);
  model_layout->addWidget(load_model_btn_);
  delete_model_btn_ = new QPushButton(QStringLiteral("删除模型"), model_group);
  delete_model_btn_->setObjectName("delete_model_btn");
  connect(delete_model_btn_, &QPushButton::clicked, this, &TrainingGUI::delete_selected_model);
  model_layout->addWidget(delete_model_btn_);
  refresh_btn_ = new QPushButton(QStringLiteral("刷新列表"), model_group);
  connect(refresh_btn_, &QPushButton::clicked, this, &TrainingGUI::refresh_model_list);
  model_layout->addWidget(refresh_btn_);
  model_layout->addStretch();
  open_dataset_btn_ = new QPushButton(QStringLiteral("📂 打开数据集 (.zip)"), model_group);
  open_dataset_btn_->setObjectName("open_dataset_btn");
  connect(open_dataset_btn_, &QPushButton::clicked, this, &TrainingGUI::open_dataset);
  model_layout->addWidget(open_dataset_btn_);
  layout->addWidget(model_group);

  // 数据集信息
  dataset_info_label_ = new QLabel(QStringLiteral("未加载数据集"), tab);
  dataset_info_label_->setProperty("muted", true);
  layout->addWidget(dataset_info_label_);

  // ---------- 分词器 ----------
  auto* tokenizer_group = new QGroupBox(QStringLiteral("分词器设置 (建议纯英文路径)"), tab);
  auto* tokenizer_layout = new QHBoxLayout(tokenizer_group);
  tokenizer_layout->addWidget(new QLabel(QStringLiteral("spm.model:"), tokenizer_group));
  tokenizer_path_label_ = new QLabel(QStringLiteral("未选择"), tokenizer_group);
  tokenizer_path_label_->setProperty("muted", true);
  tokenizer_layout->addWidget(tokenizer_path_label_);
  select_tokenizer_btn_ = new QPushButton(QStringLiteral("浏览"), tokenizer_group);
  connect(select_tokenizer_btn_, &QPushButton::clicked, this, &TrainingGUI::select_tokenizer);
  tokenizer_layout->addWidget(select_tokenizer_btn_);
  tokenizer_layout->addStretch();
  layout->addWidget(tokenizer_group);

  // 模型信息
  model_info_label_ = new QLabel(QStringLiteral("未加载模型"), tab);
  model_info_label_->setProperty("muted", true);
  layout->addWidget(model_info_label_);

  // ---------- 对话区域 ----------
  auto* chat_splitter = new QSplitter(Qt::Vertical, tab);
  layout->addWidget(chat_splitter);

  auto* history_widget = new QWidget(chat_splitter);
  auto* history_layout = new QVBoxLayout(history_widget);
  history_layout->addWidget(new QLabel(QStringLiteral("对话历史:"), history_widget));
  history_text_ = new QTextEdit(history_widget);
  history_text_->setReadOnly(true);
  history_text_->setFont(QFont(QStringLiteral("Consolas"), 10));
  history_layout->addWidget(history_text_);
  chat_splitter->addWidget(history_widget);

  auto* log_widget = new QWidget(chat_splitter);
  auto* log_layout = new QVBoxLayout(log_widget);
  log_layout->addWidget(new QLabel(QStringLiteral("系统日志 (调试信息):"), log_widget));
  sys_log_text_ = new QTextEdit(log_widget);
  sys_log_text_->setReadOnly(true);
  sys_log_text_->setFont(QFont(QStringLiteral("Consolas"), 9));
  sys_log_text_->setMaximumHeight(150);
  log_layout->addWidget(sys_log_text_);
  chat_splitter->addWidget(log_widget);

  chat_splitter->setSizes({450, 150});

  // ---------- 输入区 ----------
  auto* input_widget = new QWidget(tab);
  auto* input_layout = new QVBoxLayout(input_widget);
  input_layout->addWidget(new QLabel(QStringLiteral("输入文本 (支持中文):"), input_widget));
  input_text_ = new QTextEdit(input_widget);
  input_text_->setPlaceholderText(QStringLiteral("输入一句话或小说风格开头；问答模式请勾选下方「问答模板」"));
  input_text_->setMaximumHeight(100);
  input_layout->addWidget(input_text_);

  // 生成参数行
  auto* gen_param_layout = new QHBoxLayout();
  gen_param_layout->addWidget(new QLabel(QStringLiteral("回复长度:"), input_widget));
  gen_max_tokens_edit_ = new QLineEdit(QStringLiteral("50"), input_widget);
  gen_max_tokens_edit_->setMaximumWidth(60);
  gen_param_layout->addWidget(gen_max_tokens_edit_);
  gen_param_layout->addWidget(new QLabel(QStringLiteral("温度:"), input_widget));
  gen_temperature_edit_ = new QLineEdit(QStringLiteral("0.8"), input_widget);
  gen_temperature_edit_->setMaximumWidth(60);
  gen_param_layout->addWidget(gen_temperature_edit_);
  gen_param_layout->addWidget(new QLabel(QStringLiteral("Top-K:"), input_widget));
  gen_topk_edit_ = new QLineEdit(QStringLiteral("40"), input_widget);
  gen_topk_edit_->setMaximumWidth(60);
  gen_param_layout->addWidget(gen_topk_edit_);
  gen_multi_turn_check_ = new QCheckBox(QStringLiteral("多轮对话"), input_widget);
  gen_multi_turn_check_->setChecked(true);
  gen_param_layout->addWidget(gen_multi_turn_check_);
  gen_template_check_ = new QCheckBox(QStringLiteral("问答模板 (问：…\\n答：)"), input_widget);
  gen_param_layout->addWidget(gen_template_check_);
  gen_param_layout->addStretch();
  input_layout->addLayout(gen_param_layout);

  auto* chat_btn_layout = new QHBoxLayout();
  generate_btn_ = new QPushButton(QStringLiteral("生成回复"), input_widget);
  generate_btn_->setObjectName("start_btn");
  connect(generate_btn_, &QPushButton::clicked, this, &TrainingGUI::generate_response);
  generate_btn_->setEnabled(false);
  chat_btn_layout->addWidget(generate_btn_);
  clear_history_btn_ = new QPushButton(QStringLiteral("清空对话历史"), input_widget);
  connect(clear_history_btn_, &QPushButton::clicked, this, &TrainingGUI::clear_chat_history);
  chat_btn_layout->addWidget(clear_history_btn_);
  clear_syslog_btn_ = new QPushButton(QStringLiteral("清空系统日志"), input_widget);
  connect(clear_syslog_btn_, &QPushButton::clicked, this, &TrainingGUI::clear_sys_log);
  chat_btn_layout->addWidget(clear_syslog_btn_);
  chat_btn_layout->addStretch();
  input_layout->addLayout(chat_btn_layout);
  layout->addWidget(input_widget);

  refresh_model_list();
}

// ===========================================================================
// 主题
// ===========================================================================
void TrainingGUI::apply_miuix_theme() {
  qApp->setStyleSheet(QString::fromUtf8(kMiuixQss));
}

// ===========================================================================
// 日志
// ===========================================================================
void TrainingGUI::train_log(const QString& text) {
  train_log_text_->append(text);
  train_log_text_->verticalScrollBar()->setValue(
      train_log_text_->verticalScrollBar()->maximum());
}

void TrainingGUI::sys_log(const QString& text) {
  const QString ts = QTime::currentTime().toString("HH:mm:ss");
  sys_log_text_->append(QString("[%1] %2").arg(ts, text));
  sys_log_text_->verticalScrollBar()->setValue(
      sys_log_text_->verticalScrollBar()->maximum());
}

// ===========================================================================
// 训练控制
// ===========================================================================
void TrainingGUI::browse_data_file() {
  const QString file =
      QFileDialog::getOpenFileName(this, QStringLiteral("选择数据文件"), QString(),
                                   QStringLiteral("Text Files (*.txt);;All Files (*)"));
  if (!file.isEmpty()) data_path_edit_->setText(file);
}

void TrainingGUI::browse_init_model() {
  const QString file = QFileDialog::getOpenFileName(
      this, QStringLiteral("选择初始权重文件"), QString(),
      QStringLiteral("PyTorch 模型 (*.pt);;All Files (*)"));
  if (!file.isEmpty()) init_model_edit_->setText(file);
}

bool TrainingGUI::parse_training_params(QString* err) {
  bool ok = false;
  auto parse_double = [&](QLineEdit* e, double& out) {
    out = e->text().trimmed().toDouble(&ok);
    if (!ok) { if (err) *err = QStringLiteral("无法解析数值: %1").arg(e->text()); }
    return ok;
  };
  auto parse_int = [&](QLineEdit* e, long long& out) {
    out = e->text().trimmed().toLongLong(&ok);
    if (!ok) { if (err) *err = QStringLiteral("无法解析整数: %1").arg(e->text()); }
    return ok;
  };

  double lr = 0;
  long long batch = 0, seq = 0, stride = 0, epochs = 0;
  long long embed = 0, heads = 0, layers = 0;
  if (!parse_double(lr_edit_, lr)) return false;
  if (!parse_int(batch_edit_, batch)) return false;
  if (!parse_int(seq_len_edit_, seq)) return false;
  if (!parse_int(stride_edit_, stride)) return false;
  if (!parse_int(epochs_edit_, epochs)) return false;
  if (!parse_int(embed_dim_edit_, embed)) return false;
  if (!parse_int(num_heads_edit_, heads)) return false;
  if (!parse_int(num_layers_edit_, layers)) return false;

  cfg_.learning_rate = lr;
  cfg_.batch_size = (int64_t)batch;
  cfg_.max_seq_len = (int64_t)seq;
  cfg_.stride = (int64_t)stride;
  cfg_.num_epochs = (int64_t)epochs;
  cfg_.embed_dim = (int64_t)embed;
  cfg_.num_heads = (int64_t)heads;
  cfg_.num_layers = (int64_t)layers;

  const QString steps_text = total_steps_edit_->text().trimmed();
  if (steps_text.isEmpty()) {
    cfg_.total_steps.reset();
  } else {
    long long v = 0;
    if (!parse_int(total_steps_edit_, v)) return false;
    cfg_.total_steps = v;
  }

  if (cfg_.embed_dim % cfg_.num_heads != 0) {
    if (err) *err = QStringLiteral("嵌入维度 (%1) 必须能被头数 (%2) 整除！")
                        .arg(cfg_.embed_dim)
                        .arg(cfg_.num_heads);
    return false;
  }
  return true;
}

void TrainingGUI::set_training_ui_running(bool running) {
  start_btn_->setEnabled(!running);
  stop_btn_->setEnabled(running);
}

void TrainingGUI::start_training() {
  const QString data_path = data_path_edit_->text().trimmed();
  if (data_path.isEmpty() || !fs::exists(fspath(data_path))) {
    QMessageBox::warning(this, QStringLiteral("提示"),
                         QStringLiteral("请选择有效的数据文件！"));
    return;
  }

  QString err;
  if (!parse_training_params(&err)) {
    QMessageBox::warning(this, QStringLiteral("参数错误"),
                         QStringLiteral("请检查数值输入是否正确：\n%1").arg(err));
    return;
  }
  cfg_.data_path = fspath(data_path);

  // ---- 断点间隔 / 初始权重 / 断点续训 ----
  const QString interval_text = ckpt_interval_edit_->text().trimmed();
  if (interval_text.isEmpty()) {
    cfg_.checkpoint_interval = 5000;
  } else {
    bool ok = false;
    const long long v = interval_text.toLongLong(&ok);
    cfg_.checkpoint_interval = (ok && v >= 0) ? (int64_t)v : 5000;
  }
  resume_requested_ = resume_check_->isChecked();
  const QString init_text = init_model_edit_->text().trimmed();
  init_model_path_.clear();
  if (!init_text.isEmpty()) {
    if (!fs::exists(fspath(init_text))) {
      QMessageBox::warning(this, QStringLiteral("提示"),
                           QStringLiteral("初始权重文件不存在：\n%1").arg(init_text));
      return;
    }
    init_model_path_ = fspath(init_text);
  }
  if (resume_requested_ && !init_model_path_.empty()) {
    train_log(QStringLiteral("提示: 已勾选断点恢复，初始权重将被忽略"));
  }

  if (training_active_.load()) return;
  training_active_.store(true);
  set_training_ui_running(true);
  status_label_->setText(QStringLiteral("训练中..."));
  status_label_->setProperty("status", "");
  status_label_->style()->unpolish(status_label_);
  status_label_->style()->polish(status_label_);

  train_log(QStringLiteral("=================================================="));
  train_log(QStringLiteral("开始训练..."));
  train_log(QStringLiteral("学习率: %1").arg(cfg_.learning_rate));
  train_log(QStringLiteral("批次大小: %1").arg(cfg_.batch_size));
  train_log(QStringLiteral("序列长度: %1").arg(cfg_.max_seq_len));
  train_log(QStringLiteral("滑动步长: %1").arg(cfg_.stride));
  train_log(QStringLiteral("嵌入维度: %1").arg(cfg_.embed_dim));
  train_log(QStringLiteral("注意力头数: %1").arg(cfg_.num_heads));
  train_log(QStringLiteral("层数: %1").arg(cfg_.num_layers));
  train_log(QStringLiteral("断点保存间隔: %1 步%2")
                .arg(cfg_.checkpoint_interval)
                .arg(cfg_.checkpoint_interval > 0 ? QString() : QStringLiteral(" (关闭)")));
  if (resume_requested_) train_log(QStringLiteral("模式: 从断点继续"));
  if (!init_model_path_.empty() && !resume_requested_)
    train_log(QStringLiteral("初始权重: %1").arg(qpath(init_model_path_)));
  if (cfg_.total_steps.has_value()) {
    train_log(QStringLiteral("训练步数: %1").arg(*cfg_.total_steps));
  } else {
    train_log(QStringLiteral("训练 Epoch 数: %1").arg(cfg_.num_epochs));
  }
  progress_bar_->setValue(0);

  training_thread_ = std::thread([this] { run_training(); });
}

void TrainingGUI::run_training() {
  try {
    TrainerOptions opts;
    opts.resume = resume_requested_;
    opts.init_model = init_model_path_;
    auto t = std::make_unique<Trainer>(cfg_, TrainerCallbacks{
      [this](float loss, int epoch, long long step, long long total_steps) {
        QMetaObject::invokeMethod(
            this,
            [this, loss, epoch, step, total_steps] {
              on_progress(loss, epoch, step, total_steps);
            },
            Qt::QueuedConnection);
      },
      [this](const std::string& line) {
        const QString q = QString::fromUtf8(line.data(), (int)line.size());
        QMetaObject::invokeMethod(this, [this, q] { train_log(q); },
                                  Qt::QueuedConnection);
      }});
    trainer_.store(t.get());
    t->train();
    trainer_.store(nullptr);
    QMetaObject::invokeMethod(this, [this] { on_training_done(); },
                              Qt::QueuedConnection);
  } catch (const std::exception& e) {
    const QString msg = QString::fromUtf8(e.what());
    trainer_.store(nullptr);
    QMetaObject::invokeMethod(
        this,
        [this, msg] {
          train_log(QStringLiteral("训练出错: %1").arg(msg));
          on_training_done();
        },
        Qt::QueuedConnection);
  }
}

void TrainingGUI::stop_training() {
  if (auto* t = trainer_.load()) t->stop_training.store(true);
  stop_btn_->setEnabled(false);
  status_label_->setText(QStringLiteral("正在停止..."));
  train_log(QStringLiteral("用户请求停止训练..."));
}

void TrainingGUI::on_progress(float loss, int epoch, long long step,
                              long long total_steps) {
  loss_label_->setText(QStringLiteral("Loss: %1").arg(loss, 0, 'f', 4));
  if (cfg_.total_steps.has_value()) {
    epoch_label_->setText(QStringLiteral("Epoch: %1").arg(epoch));
  } else {
    epoch_label_->setText(
        QStringLiteral("Epoch: %1 / %2").arg(epoch).arg(cfg_.num_epochs));
  }
  progress_bar_->setMaximum((int)std::max<long long>(total_steps, 1));
  progress_bar_->setValue((int)std::min<long long>(step, total_steps));
}

void TrainingGUI::on_training_done() {
  set_training_ui_running(false);
  training_active_.store(false);
  status_label_->setText(QStringLiteral("训练完成"));
  status_label_->setProperty("status", "ok");
  status_label_->style()->unpolish(status_label_);
  status_label_->style()->polish(status_label_);
  train_log(QStringLiteral("训练完成！"));
  train_log(QStringLiteral("=================================================="));
  refresh_model_list();
}

// ===========================================================================
// 模型列表 / 数据集 / 分词器
// ===========================================================================
void TrainingGUI::refresh_model_list() {
  model_combo_->clear();
  std::vector<std::pair<int, QString>> items;
  std::error_code ec;
  if (fs::exists(cfg_.save_root)) {
    for (const auto& entry : fs::directory_iterator(cfg_.save_root, ec)) {
      if (!entry.is_directory()) continue;
      const std::string name = entry.path().filename().string();
      int num = 0;
      if (name.rfind("model", 0) == 0) {
        const std::string suffix = name.substr(5);
        if (!suffix.empty() &&
            std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
          num = std::stoi(suffix);
        }
      }
      items.emplace_back(num, qpath(entry.path().filename()));
    }
  }
  std::sort(items.begin(), items.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& [num, name] : items) model_combo_->addItem(name);
  if (model_combo_->count() == 0) {
    model_combo_->addItem(QStringLiteral("没有已保存的模型"));
  }
  model_combo_->setCurrentIndex(0);
}

void TrainingGUI::open_dataset() {
  const QString zip_path = QFileDialog::getOpenFileName(
      this, QStringLiteral("选择数据集压缩包"), QString(),
      QStringLiteral("Zip Files (*.zip);;All Files (*)"));
  if (zip_path.isEmpty()) return;

  try {
    sys_log(QStringLiteral("正在加载数据集: %1").arg(zip_path));
    const DatasetZipPaths paths = load_dataset_from_zip(fspath(zip_path));

    auto sp = std::make_shared<sentencepiece::SentencePieceProcessor>();
    const auto status = sp->Load(path_to_ansi(paths.spm_model));
    if (!status.ok()) {
      throw std::runtime_error("分词器加载失败: " + status.ToString());
    }
    sp_ = sp;
    sp_model_path_ = paths.spm_model.string();
    tokenizer_path_label_->setText(qpath(paths.spm_model));
    tokenizer_path_label_->setProperty("muted", false);
    tokenizer_path_label_->setProperty("status", "ok");
    tokenizer_path_label_->style()->unpolish(tokenizer_path_label_);
    tokenizer_path_label_->style()->polish(tokenizer_path_label_);

    data_path_edit_->setText(qpath(paths.token_ids));
    cfg_.data_path = paths.token_ids;

    // 读取 config.json
    Json cfg_json = Json::parse(read_text_file(paths.config_json).toStdString());
    vocab_size_ = cfg_json.get_int("vocab_size").value_or(8000);
    max_seq_len_ = cfg_json.get_int("max_seq_len").value_or(256);
    cfg_.vocab_size = vocab_size_;
    cfg_.max_seq_len = max_seq_len_;

    dataset_info_label_->setText(
        QStringLiteral("✅ 已加载: %1").arg(QFileInfo(zip_path).fileName()));
    dataset_info_label_->setProperty("muted", false);
    dataset_info_label_->setProperty("status", "ok");
    dataset_info_label_->style()->unpolish(dataset_info_label_);
    dataset_info_label_->style()->polish(dataset_info_label_);

    sys_log(QStringLiteral("数据集加载成功，词表大小=%1, 最大序列=%2")
                .arg(vocab_size_)
                .arg(max_seq_len_));
    sys_log(QStringLiteral("分词器: %1").arg(qpath(paths.spm_model)));
    sys_log(QStringLiteral("数据: %1").arg(qpath(paths.token_ids)));

    tabs_->setCurrentIndex(0);

    QMessageBox::information(
        this, QStringLiteral("成功"),
        QStringLiteral("数据集加载成功！\n\n词表大小: %1\n最大序列长度: %2\n数据路径: %3")
            .arg(vocab_size_)
            .arg(max_seq_len_)
            .arg(qpath(paths.token_ids)));
  } catch (const std::exception& e) {
    const QString msg = QString::fromUtf8(e.what());
    sys_log(QStringLiteral("加载失败: %1").arg(msg));
    QMessageBox::critical(this, QStringLiteral("加载失败"),
                          QStringLiteral("打开数据集出错:\n%1").arg(msg));
  }
}

void TrainingGUI::select_tokenizer() {
  const QString file = QFileDialog::getOpenFileName(
      this, QStringLiteral("选择 SentencePiece 模型文件"), QString(),
      QStringLiteral("SentencePiece Model (*.model);;All Files (*)"));
  if (file.isEmpty()) return;
  const fs::path p = fspath(file);
  if (!fs::exists(p)) {
    QMessageBox::warning(this, QStringLiteral("错误"),
                         QStringLiteral("文件不存在：\n%1").arg(file));
    return;
  }
  try {
    auto test = std::make_shared<sentencepiece::SentencePieceProcessor>();
    const auto status = test->Load(p.string());
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    sp_ = test;
    sp_model_path_ = p.string();
    tokenizer_path_label_->setText(file);
    tokenizer_path_label_->setProperty("muted", false);
    tokenizer_path_label_->setProperty("status", "ok");
    tokenizer_path_label_->style()->unpolish(tokenizer_path_label_);
    tokenizer_path_label_->style()->polish(tokenizer_path_label_);
    sys_log(QStringLiteral("分词器加载成功: %1").arg(file));
    QMessageBox::information(this, QStringLiteral("成功"),
                             QStringLiteral("分词器加载成功！\n%1").arg(file));
  } catch (const std::exception& e) {
    const QString err_msg = QString::fromUtf8(e.what());
    QMessageBox::critical(
        this, QStringLiteral("加载失败"),
        QStringLiteral("无法加载分词器，可能是因为文件路径包含中文字符或特殊符号。\n"
                       "错误信息：%1\n\n解决方案：\n"
                       "1. 将 spm.model 文件复制到一个纯英文路径（如 C:\\spm.model）\n"
                       "2. 然后重新点击 '浏览' 选择该文件")
            .arg(err_msg));
  }
}

// ===========================================================================
// 模型加载 / 删除
// ===========================================================================
void TrainingGUI::load_selected_model() {
  const QString model_name = model_combo_->currentText();
  if (model_name.isEmpty() || model_name == QStringLiteral("没有已保存的模型")) {
    QMessageBox::warning(this, QStringLiteral("提示"),
                         QStringLiteral("请先训练并保存模型！"));
    return;
  }

  if (!sp_) {
    const auto reply = QMessageBox::question(
        this, QStringLiteral("缺少分词器"), QStringLiteral("还未选择分词器，是否现在选择？"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
    if (reply == QMessageBox::Yes) {
      select_tokenizer();
      if (!sp_) return;
    } else {
      return;
    }
  }

  const fs::path model_dir = cfg_.save_root / fspath(model_name);
  const fs::path config_path = model_dir / "config.json";
  const fs::path model_path = model_dir / "model.pt";
  if (!fs::exists(config_path) || !fs::exists(model_path)) {
    QMessageBox::warning(this, QStringLiteral("提示"),
                         QStringLiteral("模型文件不完整，请检查 %1")
                             .arg(qpath(model_dir)));
    return;
  }

  try {
    Json cfg_json = Json::parse(read_text_file(config_path).toStdString());
    vocab_size_ = cfg_json.get_int("vocab_size").value_or(8000);
    max_seq_len_ = cfg_json.get_int("max_seq_len").value_or(256);

    torch::Device dev = torch::cuda::is_available()
                            ? torch::Device(torch::kCUDA)
                            : torch::Device(torch::kCPU);
    inference_model_ = load_gpt_from_saves(model_dir, dev);
    loaded_model_dir_ = model_dir.string();
    chat_context_.clear(); // 新模型 → 新对话

    model_info_label_->setText(
        QStringLiteral("已加载: %1 (词表=%2, 最大序列=%3)")
            .arg(model_name)
            .arg(vocab_size_)
            .arg(max_seq_len_));
    model_info_label_->setProperty("muted", false);
    model_info_label_->setProperty("status", "ok");
    model_info_label_->style()->unpolish(model_info_label_);
    model_info_label_->style()->polish(model_info_label_);
    generate_btn_->setEnabled(true);
    sys_log(QStringLiteral("模型 %1 加载成功！").arg(model_name));
    QMessageBox::information(this, QStringLiteral("成功"),
                             QStringLiteral("模型 %1 加载成功！").arg(model_name));
  } catch (const std::exception& e) {
    const QString msg = QString::fromUtf8(e.what());
    QMessageBox::critical(this, QStringLiteral("加载失败"),
                          QStringLiteral("加载模型时出错: %1").arg(msg));
    model_info_label_->setText(QStringLiteral("加载失败"));
    model_info_label_->setProperty("muted", false);
    model_info_label_->setProperty("status", "warn");
    model_info_label_->style()->unpolish(model_info_label_);
    model_info_label_->style()->polish(model_info_label_);
    generate_btn_->setEnabled(false);
  }
}

void TrainingGUI::delete_selected_model() {
  const QString model_name = model_combo_->currentText();
  if (model_name.isEmpty() || model_name == QStringLiteral("没有已保存的模型")) {
    QMessageBox::warning(this, QStringLiteral("提示"),
                         QStringLiteral("没有可删除的模型。"));
    return;
  }
  const fs::path model_dir = cfg_.save_root / fspath(model_name);
  if (!fs::exists(model_dir)) {
    QMessageBox::warning(this, QStringLiteral("提示"),
                         QStringLiteral("模型目录不存在: %1").arg(qpath(model_dir)));
    return;
  }
  const auto reply = QMessageBox::question(
      this, QStringLiteral("确认删除"),
      QStringLiteral("确定要删除模型 '%1' 吗？\n此操作不可恢复！").arg(model_name),
      QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
  if (reply != QMessageBox::Yes) return;

  std::error_code ec;
  fs::remove_all(model_dir, ec);
  if (ec) {
    QMessageBox::critical(this, QStringLiteral("删除失败"),
                          QStringLiteral("删除时出错: %1")
                              .arg(QString::fromLocal8Bit(ec.message().c_str())));
    return;
  }
  sys_log(QStringLiteral("模型 %1 已删除。").arg(model_name));
  refresh_model_list();
  if (loaded_model_dir_ == model_dir.string()) {
    inference_model_ = nullptr;
    loaded_model_dir_.clear();
    model_info_label_->setText(QStringLiteral("未加载模型"));
    model_info_label_->setProperty("muted", true);
    model_info_label_->setProperty("status", "");
    model_info_label_->style()->unpolish(model_info_label_);
    model_info_label_->style()->polish(model_info_label_);
    generate_btn_->setEnabled(false);
  }
  QMessageBox::information(this, QStringLiteral("成功"),
                           QStringLiteral("模型 %1 已删除。").arg(model_name));
}

// ===========================================================================
// 对话生成
// ===========================================================================
void TrainingGUI::generate_response() {
  if (!inference_model_) {
    QMessageBox::warning(this, QStringLiteral("提示"), QStringLiteral("请先加载模型！"));
    return;
  }
  if (!sp_) {
    QMessageBox::warning(this, QStringLiteral("提示"), QStringLiteral("请先选择分词器！"));
    return;
  }
  const QString prompt = input_text_->toPlainText().trimmed();
  if (prompt.isEmpty()) {
    QMessageBox::warning(this, QStringLiteral("提示"), QStringLiteral("请输入文本！"));
    return;
  }

  // 解析生成参数（非法输入回退默认值）
  auto parse_dbl = [](QLineEdit* e, double def) {
    bool ok = false;
    const double v = e->text().trimmed().toDouble(&ok);
    return ok ? v : def;
  };
  auto parse_int = [](QLineEdit* e, int def) {
    bool ok = false;
    const int v = e->text().trimmed().toInt(&ok);
    return ok ? v : def;
  };
  const int gen_max = std::clamp(parse_int(gen_max_tokens_edit_, 50), 1, 512);
  const double gen_temp = std::clamp(parse_dbl(gen_temperature_edit_, 0.8), 0.1, 2.0);
  const int gen_topk = std::clamp(parse_int(gen_topk_edit_, 40), 0, 32000);
  const bool use_template = gen_template_check_->isChecked();
  const bool multi_turn = gen_multi_turn_check_->isChecked();

  status_label_->setText(QStringLiteral("生成中..."));
  generate_btn_->setEnabled(false);
  sys_log(QStringLiteral("开始生成..."));

  // 组装提示词：问答模板（问：…\n答：）+ 多轮历史
  QString full;
  if (use_template) {
    full = QStringLiteral("问：%1\n答：").arg(prompt);
    if (multi_turn && !chat_context_.isEmpty()) full = chat_context_ + full;
  } else {
    full = prompt;
    if (multi_turn && !chat_context_.isEmpty()) full = chat_context_ + "\n" + full;
  }

  try {
    std::vector<int> ids;
    const auto status = sp_->Encode(full.toUtf8().toStdString(), &ids);
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    // 超长截断：保留最近 max_seq_len 个 token（适合多轮对话）
    if ((int64_t)ids.size() > max_seq_len_) {
      const size_t drop = ids.size() - (size_t)max_seq_len_;
      ids.erase(ids.begin(), ids.begin() + drop);
      sys_log(QStringLiteral("输入较长，已截断保留最近 %1 个 token").arg(max_seq_len_));
    }
    sys_log(QStringLiteral("编码后 token 数量: %1").arg((int)ids.size()));

    generating_.store(true);
    gen_thread_ = std::thread([this, ids, gen_max, gen_temp, gen_topk,
                               use_template, multi_turn, prompt] {
      run_generate(ids, gen_max, gen_temp, gen_topk, use_template, multi_turn,
                   prompt);
    });
  } catch (const std::exception& e) {
    const QString msg = QString::fromUtf8(e.what());
    sys_log(QStringLiteral("编码错误: %1").arg(msg));
    QMessageBox::critical(this, QStringLiteral("编码错误"), msg);
    status_label_->setText(QStringLiteral("就绪"));
    generate_btn_->setEnabled(true);
  }
}

void TrainingGUI::run_generate(std::vector<int> ids, int gen_max, double gen_temp,
                               int gen_topk, bool use_template, bool multi_turn,
                               const QString& prompt) {
  try {
    QMetaObject::invokeMethod(this, [this] { sys_log(QStringLiteral("正在运行模型生成...")); },
                              Qt::QueuedConnection);
    torch::Device dev = torch::cuda::is_available()
                            ? torch::Device(torch::kCUDA)
                            : torch::Device(torch::kCPU);
    std::vector<int64_t> out_ids;
    {
      torch::NoGradGuard no_grad;
      std::vector<int64_t> in_ids;
      in_ids.reserve(ids.size());
      for (int id : ids) in_ids.push_back((int64_t)id);
      auto input = torch::tensor(in_ids, torch::TensorOptions().dtype(torch::kInt64))
                       .reshape({1, (int64_t)in_ids.size()})
                       .to(dev);
      const std::optional<int64_t> topk =
          gen_topk > 0 ? std::optional<int64_t>(gen_topk) : std::nullopt;
      auto output =
          inference_model_->generate(input, gen_max, gen_temp, topk);
      auto cpu = output[0].to(torch::kCPU).contiguous();
      const int64_t* data = cpu.data_ptr<int64_t>();
      out_ids.assign(data, data + cpu.numel());
    }
    const int out_n = (int)out_ids.size();
    QMetaObject::invokeMethod(
        this,
        [this, out_n] {
          sys_log(QStringLiteral("生成完成，输出 token 数量: %1").arg(out_n));
        },
        Qt::QueuedConnection);

    if ((int64_t)out_ids.size() <= (int64_t)ids.size()) {
      QMetaObject::invokeMethod(
          this,
          [this] {
            sys_log(QStringLiteral("模型没有生成新内容。"));
            status_label_->setText(QStringLiteral("就绪"));
            generate_btn_->setEnabled(true);
            generating_.store(false);
          },
          Qt::QueuedConnection);
      return;
    }

    std::vector<int> new_ids(out_ids.begin() + ids.size(), out_ids.end());
    std::string new_text;
    {
      const auto s2 = sp_->Decode(new_ids, &new_text);
      if (!s2.ok()) throw std::runtime_error("解码失败");
    }
    const QString new_q = QString::fromUtf8(new_text.data(), (int)new_text.size());

    QMetaObject::invokeMethod(
        this,
        [this, new_q, prompt, use_template, multi_turn] {
          sys_log(QStringLiteral("生成文本: %1").arg(new_q));
          history_text_->append(QStringLiteral("用户: %1").arg(prompt));
          history_text_->append(QStringLiteral("模型: %1").arg(new_q));
          history_text_->append(QString(40, QLatin1Char('-')));
          history_text_->verticalScrollBar()->setValue(
              history_text_->verticalScrollBar()->maximum());
          // 更新多轮上下文（保留最近约 4000 字符）
          if (multi_turn) {
            if (use_template) {
              chat_context_ += QStringLiteral("问：%1\n答：%2\n").arg(prompt, new_q);
            } else {
              chat_context_ += prompt + "\n" + new_q + "\n";
            }
            if (chat_context_.size() > 4000) chat_context_ = chat_context_.right(4000);
          }
          input_text_->clear();
          status_label_->setText(QStringLiteral("就绪"));
          generate_btn_->setEnabled(true);
          generating_.store(false);
        },
        Qt::QueuedConnection);
  } catch (const std::exception& e) {
    const QString msg = QString::fromUtf8(e.what());
    QMetaObject::invokeMethod(
        this,
        [this, msg] {
          sys_log(QStringLiteral("生成出错: %1").arg(msg));
          QMessageBox::critical(this, QStringLiteral("生成错误"), msg);
          status_label_->setText(QStringLiteral("就绪"));
          generate_btn_->setEnabled(true);
          generating_.store(false);
        },
        Qt::QueuedConnection);
  }
}

// ===========================================================================
// 清空
// ===========================================================================
void TrainingGUI::clear_chat_history() {
  history_text_->clear();
  chat_context_.clear();
}
void TrainingGUI::clear_sys_log() { sys_log_text_->clear(); }

// ===========================================================================
// 分词器页（集成自 cpp分词器）
// ===========================================================================
void TrainingGUI::setup_tokenizer_tab() {
  QWidget* tab = tabs_->widget(2);
  auto* layout = new QVBoxLayout(tab);

  // ---------- 语料选择 ----------
  auto* corpus_group = new QGroupBox(QStringLiteral("语料选择"), tab);
  auto* corpus_layout = new QVBoxLayout(corpus_group);
  auto* corpus_btn_row = new QHBoxLayout();
  auto* pick_btn = new QPushButton(QStringLiteral("📂 选择 txt 文件 (可多选)"), corpus_group);
  pick_btn->setObjectName("open_dataset_btn");
  connect(pick_btn, &QPushButton::clicked, this, &TrainingGUI::pick_tokenizer_corpus);
  corpus_btn_row->addWidget(pick_btn);
  auto* raw_btn = new QPushButton(QStringLiteral("使用 data/raw 目录"), corpus_group);
  connect(raw_btn, &QPushButton::clicked, this, &TrainingGUI::use_tokenizer_raw_dir);
  corpus_btn_row->addWidget(raw_btn);
  corpus_btn_row->addStretch();
  corpus_layout->addLayout(corpus_btn_row);
  tokenizer_corpus_label_ = new QLabel(QStringLiteral("未选择语料"), corpus_group);
  tokenizer_corpus_label_->setProperty("muted", true);
  corpus_layout->addWidget(tokenizer_corpus_label_);
  tokenizer_stats_label_ = new QLabel(QStringLiteral(""), corpus_group);
  tokenizer_stats_label_->setProperty("muted", true);
  tokenizer_stats_label_->setWordWrap(true);
  corpus_layout->addWidget(tokenizer_stats_label_);
  layout->addWidget(corpus_group);

  // ---------- 训练参数 ----------
  auto* param_group =
      new QGroupBox(QStringLiteral("训练参数 (留空 = 按语料大小自动)"), tab);
  auto* param_layout = new QVBoxLayout(param_group);
  auto* row1 = new QHBoxLayout();
  row1->addWidget(new QLabel(QStringLiteral("训练线程数:"), param_group));
  tok_threads_edit_ = new QLineEdit(param_group);
  tok_threads_edit_->setMaximumWidth(90);
  row1->addWidget(tok_threads_edit_);
  row1->addWidget(new QLabel(QStringLiteral("词表大小:"), param_group));
  tok_vocab_edit_ = new QLineEdit(param_group);
  tok_vocab_edit_->setMaximumWidth(110);
  row1->addWidget(tok_vocab_edit_);
  row1->addWidget(new QLabel(QStringLiteral("最大训练句子数 (0=全部):"), param_group));
  tok_sentences_edit_ = new QLineEdit(param_group);
  tok_sentences_edit_->setMaximumWidth(140);
  row1->addWidget(tok_sentences_edit_);
  row1->addStretch();
  param_layout->addLayout(row1);
  auto* row2 = new QHBoxLayout();
  row2->addWidget(new QLabel(QStringLiteral("输出目录:"), param_group));
  tokenizer_out_edit_ = new QLineEdit(param_group);
  row2->addWidget(tokenizer_out_edit_);
  auto* out_btn = new QPushButton(QStringLiteral("浏览"), param_group);
  connect(out_btn, &QPushButton::clicked, this, &TrainingGUI::browse_tokenizer_out);
  row2->addWidget(out_btn);
  param_layout->addLayout(row2);
  auto* tip = new QLabel(
      QStringLiteral("💡 产物：<输出目录>/models/spm.model、data/processed/token_ids.txt、data/config.json、dataset.zip"),
      param_group);
  tip->setProperty("muted", true);
  tip->setStyleSheet("font-size: 9pt;");
  param_layout->addWidget(tip);
  layout->addWidget(param_group);

  // ---------- 训练按钮 + 日志 ----------
  auto* run_row = new QHBoxLayout();
  tokenizer_start_btn_ = new QPushButton(QStringLiteral("开始训练分词器"), tab);
  tokenizer_start_btn_->setObjectName("start_btn");
  connect(tokenizer_start_btn_, &QPushButton::clicked, this,
          &TrainingGUI::start_tokenizer_training);
  run_row->addWidget(tokenizer_start_btn_);
  run_row->addStretch();
  layout->addLayout(run_row);

  auto* log_group = new QGroupBox(QStringLiteral("分词器日志"), tab);
  auto* log_layout = new QVBoxLayout(log_group);
  tokenizer_log_text_ = new QTextEdit(log_group);
  tokenizer_log_text_->setReadOnly(true);
  tokenizer_log_text_->setFont(QFont(QStringLiteral("Consolas"), 9));
  log_layout->addWidget(tokenizer_log_text_);
  layout->addWidget(log_group);

  // ---------- 测试分词 ----------
  auto* test_group = new QGroupBox(QStringLiteral("测试分词 (使用刚训练出的模型)"), tab);
  auto* test_layout = new QVBoxLayout(test_group);
  tokenizer_test_input_ = new QTextEdit(test_group);
  tokenizer_test_input_->setPlaceholderText(QStringLiteral("输入中文句子，点击分词..."));
  tokenizer_test_input_->setMaximumHeight(70);
  test_layout->addWidget(tokenizer_test_input_);
  auto* test_btn_row = new QHBoxLayout();
  tokenizer_test_btn_ = new QPushButton(QStringLiteral("分词"), test_group);
  connect(tokenizer_test_btn_, &QPushButton::clicked, this, &TrainingGUI::test_tokenize);
  test_btn_row->addWidget(tokenizer_test_btn_);
  test_btn_row->addStretch();
  test_layout->addLayout(test_btn_row);
  tokenizer_test_output_ = new QTextEdit(test_group);
  tokenizer_test_output_->setReadOnly(true);
  tokenizer_test_output_->setMaximumHeight(90);
  tokenizer_test_output_->setFont(QFont(QStringLiteral("Consolas"), 9));
  test_layout->addWidget(tokenizer_test_output_);
  layout->addWidget(test_group);
}

void TrainingGUI::refresh_tokenizer_stats() {
  QString corpus_text;
  if (tokenizer_files_.empty()) {
    const fs::path raw = tokenizer_output_dir_ / "data" / "raw";
    const fs::path corpus_txt = raw / "corpus.txt";
    if (fs::exists(corpus_txt)) {
      corpus_text = QStringLiteral("使用 corpus.txt");
    } else {
      auto files = [&]() {
        std::vector<fs::path> v;
        std::error_code ec;
        if (fs::is_directory(raw, ec)) {
          for (const auto& e : fs::directory_iterator(raw, ec)) {
            if (ec) break;
            if (e.is_regular_file(ec) && e.path().extension() == ".txt")
              v.push_back(e.path());
          }
        }
        return v;
      }();
      corpus_text = files.empty()
                        ? QStringLiteral("data/raw/ 下暂无 txt 文件，请先选择语料")
                        : QStringLiteral("使用 data/raw/ 目录 (%1 个 txt)").arg((int)files.size());
    }
  } else {
    corpus_text = QStringLiteral("已选择 %1 个 txt 文件").arg((int)tokenizer_files_.size());
  }
  tokenizer_corpus_label_->setText(corpus_text);

  // 统计（大文件为采样，可能耗时数秒）
  QApplication::setOverrideCursor(Qt::WaitCursor);
  const CorpusStats st = analyze_corpus(tokenizer_files_, tokenizer_output_dir_);
  QApplication::restoreOverrideCursor();

  if (st.total_bytes == 0) {
    tokenizer_stats_label_->setText(QStringLiteral(""));
    return;
  }
  const AutoTrainPreview p =
      preview_auto_params(st.total_bytes, st.unique_chars, st.complete);
  QString s = QStringLiteral("总大小: %1 MB | 档位: %2 | 字符集: %3 个不同字符%4\n")
                  .arg(st.total_bytes / (1024.0 * 1024.0), 0, 'f', 2)
                  .arg(QString::fromStdString(p.tier_name))
                  .arg((qulonglong)st.unique_chars)
                  .arg(st.complete ? QString() : QStringLiteral(" (采样)"));
  s += QStringLiteral("自动参数: 线程数 %1 | 词表大小 %2 | 最大句子数 %3%4")
           .arg(p.num_threads)
           .arg(p.vocab_size)
           .arg(p.input_sentence_size > 0 ? QString::number(p.input_sentence_size)
                                          : QStringLiteral("全部"))
           .arg(p.extremely_large ? QStringLiteral(" | 超大语料增量模式") : QString());
  tokenizer_stats_label_->setText(s);
}

void TrainingGUI::pick_tokenizer_corpus() {
  const QStringList files = QFileDialog::getOpenFileNames(
      this, QStringLiteral("选择语料 txt 文件（可多选）"), QString(),
      QStringLiteral("文本文件 (*.txt);;所有文件 (*)"));
  if (files.isEmpty()) return;
  tokenizer_files_.clear();
  for (const auto& f : files) tokenizer_files_.push_back(fspath(f));
  refresh_tokenizer_stats();
}

void TrainingGUI::use_tokenizer_raw_dir() {
  tokenizer_files_.clear();
  refresh_tokenizer_stats();
}

void TrainingGUI::browse_tokenizer_out() {
  const QString dir = QFileDialog::getExistingDirectory(
      this, QStringLiteral("选择输出目录"), tokenizer_out_edit_->text());
  if (!dir.isEmpty()) {
    tokenizer_out_edit_->setText(dir);
    tokenizer_output_dir_ = fspath(dir);
    refresh_tokenizer_stats();
  }
}

void TrainingGUI::start_tokenizer_training() {
  if (tokenizer_running_.load()) return;

  const QString out_text = tokenizer_out_edit_->text().trimmed();
  if (!out_text.isEmpty()) tokenizer_output_dir_ = fspath(out_text);
  if (tokenizer_output_dir_.empty()) tokenizer_output_dir_ = "data";

  // 参数（空 = 自动）
  auto parse_opt_int = [](QLineEdit* e) -> int {
    const QString t = e->text().trimmed();
    if (t.isEmpty()) return 0;
    bool ok = false;
    const int v = t.toInt(&ok);
    return ok ? v : 0;
  };
  const int threads = parse_opt_int(tok_threads_edit_);
  const int vocab = parse_opt_int(tok_vocab_edit_);
  const int sentences = parse_opt_int(tok_sentences_edit_);

  TokenizerTask task;
  task.input_files = tokenizer_files_;
  task.output_dir = tokenizer_output_dir_;
  task.num_threads = threads;
  task.vocab_size = vocab;
  task.input_sentence_size = sentences;

  tokenizer_running_.store(true);
  tokenizer_start_btn_->setEnabled(false);
  tokenizer_log_text_->clear();
  tokenizer_log_text_->append(QStringLiteral("输出目录: %1").arg(qpath(tokenizer_output_dir_)));
  tokenizer_log_text_->append(QStringLiteral("开始分词器训练流程..."));
  if (tokenizer_files_.empty()) {
    tokenizer_log_text_->append(QStringLiteral("语料来源: data/raw/ 目录"));
  }

  tokenizer_thread_ = std::thread([this, task] { run_tokenizer_training(task); });
}

void TrainingGUI::run_tokenizer_training(const TokenizerTask& task) {
  bool ok = false;
  std::string err;
  try {
    TokenizerTask t = task;
    t.log = [this](const std::string& s) {
      const QString q = QString::fromUtf8(s.data(), (int)s.size());
      QMetaObject::invokeMethod(this, [this, q] { tokenizer_log_text_->append(q); },
                                Qt::QueuedConnection);
    };
    ok = run_tokenizer_task(t, &tokenizer_result_, &err);
  } catch (const std::exception& e) {
    err = e.what();
  }
  const QString errq = QString::fromUtf8(err.data(), (int)err.size());
  QMetaObject::invokeMethod(
      this, [this, ok, errq] { on_tokenizer_done(ok, errq); },
      Qt::QueuedConnection);
}

void TrainingGUI::on_tokenizer_done(bool ok, const QString& err) {
  tokenizer_running_.store(false);
  tokenizer_start_btn_->setEnabled(true);
  if (!ok) {
    tokenizer_log_text_->append(QStringLiteral("❌ 分词器训练失败: %1").arg(err));
    return;
  }
  tokenizer_log_text_->append(QStringLiteral("✅ 分词器训练完成"));
  // 加载新模型供测试分词使用
  if (!tokenizer_result_.spm_model.empty()) {
    auto sp = std::make_shared<sentencepiece::SentencePieceProcessor>();
    const auto status = sp->Load(path_to_ansi(tokenizer_result_.spm_model));
    if (status.ok()) {
      tokenizer_sp_ = sp;
      tokenizer_log_text_->append(
          QStringLiteral("已加载: %1").arg(qpath(tokenizer_result_.spm_model)));
    } else {
      tokenizer_log_text_->append(
          QStringLiteral("加载分词模型失败: %1")
              .arg(QString::fromUtf8(status.ToString().c_str())));
    }
  }
  tokenizer_log_text_->append(
      QStringLiteral("💡 dataset.zip 已生成，可在「对话」页打开它并开始 GPT 训练。"));
}

void TrainingGUI::test_tokenize() {
  const QString text = tokenizer_test_input_->toPlainText().trimmed();
  if (text.isEmpty()) return;

  if (!tokenizer_sp_) {
    // 尝试加载上次训练产物
    const fs::path model = tokenizer_result_.spm_model;
    if (!model.empty() && fs::exists(model)) {
      auto sp = std::make_shared<sentencepiece::SentencePieceProcessor>();
      if (sp->Load(path_to_ansi(model)).ok()) tokenizer_sp_ = sp;
    }
  }
  if (!tokenizer_sp_) {
    QMessageBox::warning(this, QStringLiteral("提示"),
                         QStringLiteral("还没有可用的分词模型，请先训练或加载！"));
    return;
  }

  const std::string utf8 = text.toUtf8().toStdString();
  const std::vector<std::string> pieces = tokenizer_sp_->EncodeAsPieces(utf8);
  std::vector<int> ids;
  tokenizer_sp_->Encode(utf8, &ids);

  QStringList piece_list;
  for (const auto& p : pieces) piece_list << QString::fromUtf8(p.data(), (int)p.size());
  QStringList id_list;
  for (int id : ids) id_list << QString::number(id);

  tokenizer_test_output_->setText(
      QStringLiteral("词片段: %1\nToken ID: %2")
          .arg(piece_list.join(QStringLiteral(" | ")), id_list.join(QStringLiteral(" "))));
}

// ===========================================================================
// 关闭
// ===========================================================================
void TrainingGUI::closeEvent(QCloseEvent* event) {
  if (training_active_.load()) {
    const auto reply = QMessageBox::question(
        this, QStringLiteral("确认"), QStringLiteral("训练正在进行，确定退出吗？"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (reply != QMessageBox::Yes) {
      event->ignore();
      return;
    }
    if (auto* t = trainer_.load()) t->stop_training.store(true);
    if (training_thread_.joinable()) training_thread_.join();
  }
  if (tokenizer_running_.load()) {
    const auto reply = QMessageBox::question(
        this, QStringLiteral("确认"),
        QStringLiteral("分词器训练正在进行（无法中断），确定退出吗？"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (reply != QMessageBox::Yes) {
      event->ignore();
      return;
    }
    if (tokenizer_thread_.joinable()) tokenizer_thread_.detach();
  }
  event->accept();
}

} // namespace gpt
