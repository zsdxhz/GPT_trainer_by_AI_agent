#include "gui.h"

#include <sentencepiece_processor.h>

#include <QApplication>
#include <QCloseEvent>
#include <QComboBox>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QScreen>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QShortcut>
#include <QShowEvent>
#include <QStackedWidget>
#include <QStatusBar>
#include <QStyle>
#include <QTextEdit>
#include <QTime>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "gguf_writer.h"
#include "json_util.h"
#include "llama_prompt.h"
#include "path_util.h"
#include "train.h"
#include "ui_kit.h"
#include "ui_theme.h"

namespace fs = std::filesystem;

namespace gpt {

namespace {

// ---------------- 路径 / 文本工具 ----------------
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

// ---------------- Segoe Fluent Icons 字形 ----------------
const QString ICO_TRAIN = QStringLiteral("\uE768");   // Play
const QString ICO_CHAT = QStringLiteral("\uE8F2");    // Chat
const QString ICO_ADV = QStringLiteral("\uE713");     // Settings
const QString ICO_FOLDER = QStringLiteral("\uE8E5");  // 打开文件夹
const QString ICO_STOP = QStringLiteral("\uE71A");    // 停止
const QString ICO_LOAD = QStringLiteral("\uE896");    // 载入
const QString ICO_DELETE = QStringLiteral("\uE74D");  // 删除
const QString ICO_REFRESH = QStringLiteral("\uE72C"); // 刷新
const QString ICO_OPEN = QStringLiteral("\uE8B7");    // 文件夹
const QString ICO_EXPORT = QStringLiteral("\uE9F9");  // 导出
const QString ICO_RESTART = QStringLiteral("\uE777"); // 重试
const QString ICO_SEARCH = QStringLiteral("\uE721");  // 搜索/探测

// 参数钳制上限（防止用户输入极端值拖垮引擎）
constexpr int kMaxCtxTokens = 131072;  // 上下文长度上限
constexpr int kMaxReplyTokens = 4096;  // 单轮回复 token 上限

// ---------------- 页面外壳 ----------------
QWidget* make_page(QStackedWidget* pages, fl::NavRail* nav, const QString& glyph,
                   const QString& title, const QString& subtitle,
                   QVBoxLayout** out_body, QHBoxLayout** out_actions,
                   bool scrollable = true) {
  auto* root = new QWidget(pages);
  auto* root_layout = new QVBoxLayout(root);
  root_layout->setContentsMargins(0, 0, 0, 0);
  root_layout->setSpacing(0);

  QWidget* content = nullptr;
  QVBoxLayout* body = nullptr;
  if (scrollable) {
    auto* area = new QScrollArea(root);
    area->setWidgetResizable(true);
    area->setFrameShape(QFrame::NoFrame);
    area->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    area->viewport()->setAutoFillBackground(false);
    content = new QWidget();
    content->setAutoFillBackground(false);
    body = new QVBoxLayout(content);
    area->setWidget(content);
    root_layout->addWidget(area, 1);
  } else {
    content = new QWidget(root);
    content->setAutoFillBackground(false);
    body = new QVBoxLayout(content);
    root_layout->addWidget(content, 1);
  }

  body->setContentsMargins(30, 20, 30, 18);
  body->setSpacing(10);
  body->addWidget(fl::page_title(title));
  if (!subtitle.isEmpty()) body->addWidget(fl::page_subtitle(subtitle));

  auto* action_bar = new QWidget(root);
  action_bar->setAutoFillBackground(false);
  auto* actions = new QHBoxLayout(action_bar);
  actions->setContentsMargins(30, 4, 30, 14);
  actions->setSpacing(10);
  root_layout->addWidget(action_bar);

  root->setObjectName(title);
  pages->addWidget(root);
  nav->add_item(glyph, title);

  *out_body = body;
  *out_actions = actions;
  return root;
}

// 设置行里“标签 + 固定宽度数值框”的快捷写法
QLineEdit* add_field(fl::Card* card, const QString& label, const QString& value,
                     const QString& hint = QString(), int width = 132) {
  auto* row = card->add_row(label, hint);
  auto* edit = fl::text_field(value, QString(), width);
  row->addWidget(edit);
  return edit;
}

} // namespace

// ===========================================================================
// 构造
// ===========================================================================
TrainingGUI::TrainingGUI(QWidget* parent) : QMainWindow(parent) {
  cfg_ = Config::defaults();
  cfg_.init_dirs();
  vocab_size_ = cfg_.vocab_size;
  max_seq_len_ = cfg_.max_seq_len;

  setWindowTitle(QStringLiteral("GPT 训练与对话系统 · llama.cpp"));
  setWindowIcon(fl::app_icon());
  setMinimumSize(880, 560);
  // 默认最大化：避免高 DPI 缩放下手算尺寸超出屏幕（可手动还原窗口）
  setWindowState(windowState() | Qt::WindowMaximized);

  setup_shell(); // 左导航 + 页面容器 + 状态栏
  setup_train_tab();
  setup_chat_tab();
  setup_advanced_tab();
  nav_->set_current(0);
  if (const int page = qEnvironmentVariableIntValue("GPT_START_PAGE");
      page > 0 && page < pages_->count()) {
    nav_->set_current(page);
  }
  // 高 DPI 缩放下 frameGeometry 可能超出屏幕：显示后再钳制一次
  QTimer::singleShot(0, this, [this] {
    const QScreen* s = screen() != nullptr ? screen() : QGuiApplication::primaryScreen();
    if (s == nullptr) return;
    const QRect avail = s->availableGeometry();
    if (avail.isEmpty()) return;
    if (frameGeometry().width() > avail.width() ||
        frameGeometry().height() > avail.height()) {
      resize(std::max(minimumWidth(), std::min(width(), avail.width() - 16)),
             std::max(minimumHeight(), std::min(height(), avail.height() - 16)));
    }
    const QRect f = frameGeometry();
    if (!avail.contains(f)) {
      move(std::max(avail.left(), avail.center().x() - f.width() / 2),
           std::max(avail.top(), avail.center().y() - f.height() / 2));
    }
  });

  connect(fl::Theme::get(), &fl::Theme::changed, this,
          [this] { on_theme_changed(); }, Qt::UniqueConnection);
  on_theme_changed();

  // 默认数据路径：配置路径 → 项目内常见位置
  fs::path default_data = cfg_.data_path;
  if (!fs::exists(default_data)) {
    const fs::path root = cfg_.project_root();
    const fs::path candidates[] = {
        root / "data" / "data" / "processed" / "token_ids.txt",
        root / "data" / "processed" / "token_ids.txt",
        root / "data" / "token_ids.txt"};
    for (const auto& c : candidates) {
      if (fs::exists(c)) {
        default_data = c;
        break;
      }
    }
  }
  if (fs::exists(default_data)) {
    data_path_edit_->setText(qpath(default_data));
  }

  // 分词器默认输出目录：<项目根>/data
  tokenizer_output_dir_ = fs::path(cfg_.project_root()) / "data";
  tokenizer_out_edit_->setText(qpath(tokenizer_output_dir_));

  load_settings();
  update_stats_bar();
  update_engine_ui();
  refresh_model_list();
  refresh_tokenizer_stats();

  // 启动后自动加载列表里选中的模型（设为 GPT_NO_AUTOLOAD=1 可关闭），
  // 让用户选完模型直接发消息，不用再点「加载」
  if (!qEnvironmentVariableIsSet("GPT_NO_AUTOLOAD") && model_combo_ != nullptr &&
      model_combo_->count() > 0) {
    QTimer::singleShot(500, this, [this] {
      if (model_combo_ == nullptr || model_combo_->count() == 0) return;
      if (generating_.load()) return;
      load_selected_model();
    });
  }

  // 键盘快捷键
  auto* start_sc = new QShortcut(QKeySequence(QStringLiteral("F5")), this);
  connect(start_sc, &QShortcut::activated, this, &TrainingGUI::start_training);
  auto* stop_sc =
      new QShortcut(QKeySequence(QStringLiteral("Shift+F5")), this);
  connect(stop_sc, &QShortcut::activated, this, &TrainingGUI::stop_training);
  auto* send_sc =
      new QShortcut(QKeySequence(QStringLiteral("Ctrl+Return")), this);
  connect(send_sc, &QShortcut::activated, this, &TrainingGUI::generate_response);
}

TrainingGUI::~TrainingGUI() {
  if (auto* t = trainer_.load()) t->stop_training.store(true);
  if (training_thread_.joinable()) training_thread_.join();
  llama_.stop();
  if (engine_thread_.joinable()) engine_thread_.detach();
  if (gen_thread_.joinable()) gen_thread_.detach();
  if (tokenizer_thread_.joinable()) tokenizer_thread_.detach();
}

// ===========================================================================
// 外壳：左侧导航 + 页面容器 + 状态栏
// ===========================================================================
void TrainingGUI::setup_shell() {
  auto* central = new QWidget(this);
  setCentralWidget(central);
  auto* layout = new QHBoxLayout(central);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(0);

  nav_ = new fl::NavRail(central);
  layout->addWidget(nav_);

  pages_ = new QStackedWidget(central);
  layout->addWidget(pages_, 1);
  connect(nav_, &fl::NavRail::currentChanged, pages_,
          &QStackedWidget::setCurrentIndex);

  auto* bar = statusBar();
  bar->setSizeGripEnabled(false);
  sb_device_ = new QLabel(bar);
  sb_stats_ = new QLabel(bar);
  sb_engine_ = new QLabel(bar);
  bar->addWidget(sb_device_);
  bar->addWidget(sb_stats_);
  bar->addPermanentWidget(sb_engine_);
  bar->addPermanentWidget(new QLabel(
      QStringLiteral("F5 开始训练 · Shift+F5 停止 · Ctrl+Enter 生成"), bar));
}

// ===========================================================================
// 训练页（基础功能）
// ===========================================================================
void TrainingGUI::setup_train_tab() {
  QVBoxLayout* body = nullptr;
  QHBoxLayout* actions = nullptr;
  make_page(pages_, nav_, ICO_TRAIN, QStringLiteral("训练"),
            QStringLiteral("选好数据集，点「开始训练」即可；其余参数保持默认也能跑。"),
            &body, &actions);

  // ---------------- 数据集 ----------------
  auto* data_card = new fl::Card;
  data_card->set_title(QStringLiteral("数据集"),
                       QStringLiteral("token_ids.txt 文件，或直接用「分词器」生成的 dataset.zip"));
  {
    auto* row = data_card->add_row_wide(QStringLiteral("数据文件"));
    data_path_edit_ =
        fl::text_field(QString(), QStringLiteral("选择 token_ids.txt ..."));
    row->addWidget(data_path_edit_, 1);
    data_browse_btn_ = fl::plain_button(QStringLiteral("浏览"), ICO_FOLDER);
    connect(data_browse_btn_, &QPushButton::clicked, this,
            &TrainingGUI::browse_data_file);
    row->addWidget(data_browse_btn_);
    open_dataset_btn_ =
        fl::plain_button(QStringLiteral("打开 dataset.zip"), ICO_OPEN);
    connect(open_dataset_btn_, &QPushButton::clicked, this,
            &TrainingGUI::open_dataset);
    row->addWidget(open_dataset_btn_);
  }
  dataset_info_label_ = new QLabel(QStringLiteral("未加载数据集"));
  dataset_info_label_->setProperty("role", "hint");
  data_card->add_widget_row(dataset_info_label_, 20);
  body->addWidget(data_card);

  // ---------------- 常用参数 ----------------
  auto* param_card = new fl::Card;
  param_card->set_title(QStringLiteral("常用参数"),
                        QStringLiteral("不确定就用默认值；更多参数在「高级」页"));
  lr_edit_ = add_field(param_card, QStringLiteral("学习率"),
                       QStringLiteral("0.0003"), QStringLiteral("越小越稳"));
  batch_edit_ = add_field(param_card, QStringLiteral("批次大小"),
                          QStringLiteral("16"));
  seq_len_edit_ = add_field(param_card, QStringLiteral("序列长度"),
                            QStringLiteral("256"));
  epochs_edit_ = add_field(param_card, QStringLiteral("训练轮数 (Epoch)"),
                           QStringLiteral("10"),
                           QStringLiteral("跑完整个数据集算一轮"));
  body->addWidget(param_card);

  // ---------------- 训练状态 ----------------
  auto* status_card = new fl::Card;
  status_card->set_title(QStringLiteral("训练状态"));
  {
    auto* metrics = new QWidget;
    auto* metrics_layout = new QHBoxLayout(metrics);
    metrics_layout->setContentsMargins(0, 0, 0, 0);
    metrics_layout->setSpacing(28);
    auto add_metric = [&](const QString& caption, QLabel** out) {
      auto* col = new QVBoxLayout;
      col->setContentsMargins(0, 0, 0, 0);
      col->setSpacing(2);
      auto* value = new QLabel(QStringLiteral("--"));
      value->setMinimumWidth(64);
      value->setProperty("role", "metricValue");
      auto* label = new QLabel(caption);
      label->setProperty("role", "metricLabel");
      col->addWidget(value);
      col->addWidget(label);
      metrics_layout->addLayout(col);
      *out = value;
    };
    add_metric(QStringLiteral("Loss"), &loss_label_);
    add_metric(QStringLiteral("Epoch"), &epoch_label_);
    add_metric(QStringLiteral("步数"), &step_label_);
    metrics_layout->addStretch(1);
    status_card->add_widget_row(metrics, 44);
  }
  progress_bar_ = new QProgressBar;
  progress_bar_->setTextVisible(false);
  progress_bar_->setMinimumHeight(8);
  status_card->add_widget_row(progress_bar_, 12);
  {
    auto* row = status_card->add_row(QStringLiteral("状态"));
    status_label_ = new QLabel(QStringLiteral("就绪"));
    status_label_->setProperty("role", "chip");
    status_label_->setProperty("status", "ok");
    row->addWidget(status_label_);
  }
  body->addWidget(status_card);

  // ---------------- 训练日志 ----------------
  auto* log_card = new fl::Card;
  log_card->set_title(QStringLiteral("训练日志"));
  train_log_text_ = fl::log_view(12);
  log_card->add_widget_row(train_log_text_, 150);
  body->addWidget(log_card);
  body->addStretch(1);

  // ---------------- 底部操作 ----------------
  actions->addStretch(1);
  quit_btn_ = fl::plain_button(QStringLiteral("退出"));
  connect(quit_btn_, &QPushButton::clicked, this, &TrainingGUI::close);
  actions->addWidget(quit_btn_);
  stop_btn_ = fl::plain_button(QStringLiteral("停止训练"), ICO_STOP);
  connect(stop_btn_, &QPushButton::clicked, this, &TrainingGUI::stop_training);
  stop_btn_->setToolTip(QStringLiteral("Shift+F5"));
  stop_btn_->setEnabled(false);
  actions->addWidget(stop_btn_);
  start_btn_ = fl::accent_button(QStringLiteral("开始训练"), ICO_TRAIN);
  start_btn_->setToolTip(QStringLiteral("F5"));
  connect(start_btn_, &QPushButton::clicked, this, &TrainingGUI::start_training);
  actions->addWidget(start_btn_);
}

// ===========================================================================
// 对话页（无边框聊天页：消息区铺满 + DSH 风格输入条）
// ===========================================================================
void TrainingGUI::setup_chat_tab() {
  QVBoxLayout* body = nullptr;
  QHBoxLayout* actions = nullptr;
  make_page(pages_, nav_, ICO_CHAT, QStringLiteral("对话"),
            QStringLiteral("选一个模型点「加载」，然后在下面输入内容聊天。"),
            &body, &actions);

  // ---------------- 对话历史（无卡片边框，直接铺满页面）----------------
  chat_view_ = new fl::ChatView;
  chat_view_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  body->addWidget(chat_view_, 1);

  // ---------------- 输入（DSH 风格：大圆角输入框 + 底部工具条 + 状态行）----------------
  auto* composer_wrap = new QWidget;
  auto* wrap_layout = new QVBoxLayout(composer_wrap);
  wrap_layout->setContentsMargins(0, 0, 0, 0);
  wrap_layout->setSpacing(6);

  auto* composer = new QFrame(composer_wrap);
  composer->setObjectName(QStringLiteral("composer"));
  composer->setAttribute(Qt::WA_StyledBackground, true);
  composer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  auto* comp_layout = new QVBoxLayout(composer);
  comp_layout->setContentsMargins(16, 12, 12, 10);
  comp_layout->setSpacing(8);

  input_text_ = new QTextEdit(composer);
  input_text_->setObjectName(QStringLiteral("composerEdit"));
  input_text_->setFont(fl::ui_font(14));
  input_text_->setPlaceholderText(QStringLiteral("发消息…"));
  input_text_->setMinimumHeight(44);
  input_text_->setMaximumHeight(96);
  input_text_->setFrameShape(QFrame::NoFrame);
  input_text_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  comp_layout->addWidget(input_text_);

  auto* tools = new QHBoxLayout();
  tools->setContentsMargins(0, 0, 0, 0);
  tools->setSpacing(8);

  clear_history_btn_ = fl::plain_button(QString(), ICO_DELETE);
  clear_history_btn_->setToolTip(QStringLiteral("新建对话（清空当前对话）"));
  connect(clear_history_btn_, &QPushButton::clicked, this,
          &TrainingGUI::clear_chat_history);
  tools->addWidget(clear_history_btn_);

  auto* multi_label = new QLabel(QStringLiteral("多轮"));
  multi_label->setProperty("role", "hint");
  tools->addWidget(multi_label);
  gen_multi_turn_check_ = new fl::ToggleSwitch;
  gen_multi_turn_check_->setChecked(true);
  tools->addWidget(gen_multi_turn_check_);

  tools->addStretch(1);

  // 控制框：长度 / 温度 / 上下文（与「高级」页同步）
  auto add_tiny = [&](const QString& caption, QLineEdit* edit) {
    auto* lb = new QLabel(caption);
    lb->setProperty("role", "hint");
    tools->addWidget(lb);
    tools->addWidget(edit);
  };
  gen_max_tokens_edit_ = fl::text_field(QStringLiteral("50"), QString(), 62);
  gen_temperature_edit_ = fl::text_field(QStringLiteral("0.8"), QString(), 56);
  chat_ctx_edit_ = fl::text_field(QStringLiteral("2048"), QString(), 88);
  add_tiny(QStringLiteral("长度"), gen_max_tokens_edit_);
  add_tiny(QStringLiteral("温度"), gen_temperature_edit_);
  add_tiny(QStringLiteral("上下文"), chat_ctx_edit_);
  connect(chat_ctx_edit_, &QLineEdit::editingFinished, this, [this] {
    if (engine_ctx_edit_ != nullptr) {
      engine_ctx_edit_->setText(chat_ctx_edit_->text().trimmed());
    }
  });

  // 模型选择（选定即自动加载，无需「加载」按钮）+ 刷新
  model_combo_ = new QComboBox;
  model_combo_->setMinimumWidth(210);
  model_combo_->setMaximumWidth(320);
  tools->addWidget(model_combo_);
  // 用 activated（用户确切选定）而不是 currentIndexChanged：
  // 后者在下拉列表里用方向键浏览时也会逐项触发，导致反复切模型
  connect(model_combo_, QOverload<int>::of(&QComboBox::activated), this,
          [this](int) {
            if (model_list_filling_) return;
            if (model_combo_ == nullptr) return;
            if (model_combo_->currentData().toString().isEmpty()) return;
            load_selected_model();
          });
  refresh_btn_ = fl::plain_button(QString(), ICO_REFRESH);
  refresh_btn_->setToolTip(QStringLiteral("刷新模型列表"));
  connect(refresh_btn_, &QPushButton::clicked, this,
          &TrainingGUI::refresh_model_list);
  tools->addWidget(refresh_btn_);

  // 「多轮」开关标签：开启时用强调色，关闭时淡色，便于一眼看出状态
  auto sync_multi_label = [this, multi_label](bool on) {
    multi_label->setProperty("on", on);
    multi_label->style()->unpolish(multi_label);
    multi_label->style()->polish(multi_label);
  };
  connect(gen_multi_turn_check_, &QAbstractButton::toggled, this, sync_multi_label);
  sync_multi_label(gen_multi_turn_check_->isChecked());

  stop_gen_btn_ = fl::plain_button(QStringLiteral("停止"), ICO_STOP);
  stop_gen_btn_->setToolTip(QStringLiteral("停止生成"));
  stop_gen_btn_->setEnabled(false);
  connect(stop_gen_btn_, &QPushButton::clicked, this,
          &TrainingGUI::stop_generation);
  tools->addWidget(stop_gen_btn_);

  generate_btn_ = new QPushButton(QStringLiteral("↑"), composer);
  generate_btn_->setObjectName(QStringLiteral("sendRound"));
  generate_btn_->setFixedSize(38, 38);
  generate_btn_->setCursor(Qt::PointingHandCursor);
  generate_btn_->setToolTip(QStringLiteral("发送（Ctrl+Enter）"));
  connect(generate_btn_, &QPushButton::clicked, this,
          &TrainingGUI::generate_response);
  tools->addWidget(generate_btn_);
  comp_layout->addLayout(tools);
  wrap_layout->addWidget(composer);

  // ---------------- 状态行：tok 性能 + 模型/引擎状态 + 上下文用量环 ----------------
  auto* status_row = new QWidget(composer_wrap);
  auto* status_layout = new QHBoxLayout(status_row);
  status_layout->setContentsMargins(10, 0, 4, 2);
  status_layout->setSpacing(12);
  chat_stats_label_ = new QLabel(QStringLiteral("就绪 · 等待第一条消息"));
  chat_stats_label_->setProperty("role", "msgStats");
  status_layout->addWidget(chat_stats_label_);
  status_layout->addStretch(1);
  model_info_label_ = new QLabel(QStringLiteral("未加载模型"));
  model_info_label_->setProperty("role", "msgStats");
  status_layout->addWidget(model_info_label_);
  engine_state_label_ = new QLabel(QStringLiteral("引擎：未启动"));
  engine_state_label_->setProperty("role", "msgStats");
  status_layout->addWidget(engine_state_label_);
  ctx_ring_ = new fl::RingGauge(status_row);
  ctx_ring_->set_caption(QStringLiteral("上下文用量"));
  status_layout->addWidget(ctx_ring_);
  wrap_layout->addWidget(status_row);
  body->addWidget(composer_wrap);

  // ---------------- 底部操作（发送/停止已移入输入框工具条）----------------
  actions->addStretch(1);
}

// ===========================================================================
// 高级页：推理引擎 / 生成调参 / 训练高级参数 / 分词器 / 诊断
// ===========================================================================
void TrainingGUI::setup_advanced_tab() {
  QVBoxLayout* body = nullptr;
  QHBoxLayout* actions = nullptr;
  make_page(pages_, nav_, ICO_ADV, QStringLiteral("高级"),
            QStringLiteral("推理引擎、生成与训练的进阶参数、分词器与诊断信息都在这里。"),
            &body, &actions);

  // ---------------- 推理引擎 ----------------
  auto* engine_card = new fl::Card;
  engine_card->set_title(QStringLiteral("推理引擎"),
                         QStringLiteral("llama.cpp（推荐，速度快、支持任意 GGUF）或内置 libtorch"));
  {
    auto* row = engine_card->add_row(QStringLiteral("推理引擎"));
    engine_combo_ = new QComboBox;
    engine_combo_->addItem(QStringLiteral("自动（优先 llama.cpp）"), QStringLiteral("auto"));
    engine_combo_->addItem(QStringLiteral("llama.cpp"), QStringLiteral("llama"));
    engine_combo_->addItem(QStringLiteral("内置（libtorch）"), QStringLiteral("builtin"));
    engine_combo_->setMinimumWidth(220);
    row->addWidget(engine_combo_);
  }
  {
    auto* row = engine_card->add_row_wide(
        QStringLiteral("提示词通道"),
        QStringLiteral("自动：本项目模型走 spm token 通道，Qwen 等对话模型自动套用模型自带 chat 模板"));
    chat_mode_combo_ = new QComboBox;
    chat_mode_combo_->addItem(QStringLiteral("自动（推荐）"), QStringLiteral("auto"));
    chat_mode_combo_->addItem(QStringLiteral("对话模板（Chat 接口）"), QStringLiteral("chat"));
    chat_mode_combo_->addItem(QStringLiteral("文本补全"), QStringLiteral("text"));
    chat_mode_combo_->addItem(QStringLiteral("spm token 通道"), QStringLiteral("tokens"));
    chat_mode_combo_->setMinimumWidth(220);
    row->addWidget(chat_mode_combo_);
  }
  {
    auto* row = engine_card->add_row_wide(
        QStringLiteral("llama-server"),
        QStringLiteral("llama.cpp 的 llama-server.exe（自动探测项目内 / Ollama / PATH）"));
    server_path_edit_ = fl::text_field(QString(), QStringLiteral("自动探测"));
    row->addWidget(server_path_edit_, 1);
    server_detect_btn_ = fl::plain_button(QStringLiteral("探测"), ICO_SEARCH);
    server_detect_btn_->setToolTip(QStringLiteral("自动查找 llama-server.exe"));
    connect(server_detect_btn_, &QPushButton::clicked, this,
            &TrainingGUI::detect_server_exe);
    row->addWidget(server_detect_btn_);
    server_browse_btn_ = fl::plain_button(QStringLiteral("浏览"), ICO_FOLDER);
    connect(server_browse_btn_, &QPushButton::clicked, this,
            &TrainingGUI::browse_server_exe);
    row->addWidget(server_browse_btn_);
  }
  {
    auto* row = engine_card->add_row_wide(
        QStringLiteral("GGUF 模型"),
        QStringLiteral("直接加载任意 GGUF（不填则用对话页里 saves 模型导出的 GGUF）"));
    gguf_path_edit_ = fl::text_field(QString(), QStringLiteral("选择 .gguf 文件"));
    row->addWidget(gguf_path_edit_, 1);
    gguf_browse_btn_ = fl::plain_button(QStringLiteral("浏览"), ICO_FOLDER);
    connect(gguf_browse_btn_, &QPushButton::clicked, this,
            &TrainingGUI::browse_gguf_file);
    row->addWidget(gguf_browse_btn_);
  }
  engine_ctx_edit_ = add_field(engine_card, QStringLiteral("上下文长度"),
                              QStringLiteral("2048"),
                              QStringLiteral("越大越吃显存/内存"), 132);
  engine_threads_edit_ = add_field(engine_card, QStringLiteral("CPU 线程数"),
                                   QStringLiteral("0"),
                                   QStringLiteral("0 = 自动"), 132);
  engine_ngl_edit_ = add_field(engine_card, QStringLiteral("GPU 层数"),
                               QStringLiteral("99"),
                               QStringLiteral("99 = 尽量全部放显卡；0 = 纯 CPU"), 132);
  {
    auto* holder = new QWidget;
    auto* row = new QHBoxLayout(holder);
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(8);
    engine_detail_label_ = new QLabel(QStringLiteral("引擎未启动"));
    engine_detail_label_->setProperty("role", "hint");
    row->addWidget(engine_detail_label_, 1);
    engine_restart_btn_ =
        fl::plain_button(QStringLiteral("启动/重启"), ICO_RESTART);
    connect(engine_restart_btn_, &QPushButton::clicked, this,
            &TrainingGUI::restart_engine);
    row->addWidget(engine_restart_btn_);
    engine_stop_btn_ = fl::plain_button(QStringLiteral("停止引擎"), ICO_STOP);
    connect(engine_stop_btn_, &QPushButton::clicked, this,
            &TrainingGUI::stop_engine);
    row->addWidget(engine_stop_btn_);
    engine_card->add_widget_row(holder, 32);
  }
  body->addWidget(engine_card);

  // ---------------- 生成参数 ----------------
  auto* gen_card = new fl::Card;
  gen_card->set_title(QStringLiteral("生成参数"),
                      QStringLiteral("llama.cpp 采样参数（对话页输入条内含长度/温度/上下文）"));
  gen_topk_edit_ = add_field(gen_card, QStringLiteral("Top-K"),
                             QStringLiteral("40"), QString(), 110);
  gen_topp_edit_ = add_field(gen_card, QStringLiteral("Top-P"),
                             QStringLiteral("0.95"), QString(), 110);
  gen_repeat_edit_ = add_field(gen_card, QStringLiteral("重复惩罚"),
                               QStringLiteral("1.1"), QString(), 110);
  gen_seed_edit_ = add_field(gen_card, QStringLiteral("随机种子"),
                             QStringLiteral("-1"), QStringLiteral("-1 = 每次随机"), 110);
  {
    auto* row = gen_card->add_row_wide(
        QStringLiteral("系统提示词"),
        QStringLiteral("会拼接在对话前面（基于续写的模型留空即可）"));
    system_prompt_edit_ =
        fl::text_field(QString(), QStringLiteral("可选，例如：你是一个助手。"));
    row->addWidget(system_prompt_edit_, 1);
  }
  {
    auto* row = gen_card->add_row(QStringLiteral("问答模板"),
                                  QStringLiteral("按「问：… 答：」拼接提示词"));
    gen_template_check_ = new fl::ToggleSwitch;
    row->addWidget(gen_template_check_);
  }
  body->addWidget(gen_card);

  // ---------------- GGUF 导出 ----------------
  auto* export_card = new fl::Card;
  export_card->set_title(
      QStringLiteral("GGUF 导出"),
      QStringLiteral("把 saves 里的模型导出成 .gguf，之后就能用 llama.cpp 推理"));
  {
    auto* row = export_card->add_row(QStringLiteral("导出模型"));
    adv_model_combo_ = new QComboBox;
    adv_model_combo_->setMinimumWidth(200);
    row->addWidget(adv_model_combo_);
    delete_model_btn_ = fl::plain_button(QStringLiteral("删除"), ICO_DELETE);
    delete_model_btn_->setToolTip(QStringLiteral("删除 saves 中的该模型"));
    connect(delete_model_btn_, &QPushButton::clicked, this,
            &TrainingGUI::delete_selected_model);
    row->addWidget(delete_model_btn_);
  }
  {
    auto* row = export_card->add_row_wide(QStringLiteral("输出文件"));
    export_gguf_edit_ =
        fl::text_field(QString(), QStringLiteral("默认 models/gguf/<模型名>.gguf"));
    row->addWidget(export_gguf_edit_, 1);
    export_gguf_btn_ = fl::plain_button(QStringLiteral("导出 GGUF"), ICO_EXPORT);
    connect(export_gguf_btn_, &QPushButton::clicked, this,
            &TrainingGUI::export_current_gguf);
    row->addWidget(export_gguf_btn_);
  }
  export_result_label_ = new QLabel(QStringLiteral("尚未导出"));
  export_result_label_->setProperty("role", "hint");
  export_card->add_widget_row(export_result_label_, 20);
  body->addWidget(export_card);

  // ---------------- 训练高级参数 ----------------
  auto* train_card = new fl::Card;
  train_card->set_title(QStringLiteral("训练高级参数"),
                        QStringLiteral("改动后需要重新开始训练才生效"));
  total_steps_edit_ = add_field(train_card, QStringLiteral("训练步数"),
                                QString(),
                                QStringLiteral("留空 = 按 Epoch 训练"), 132);
  stride_edit_ = add_field(train_card, QStringLiteral("滑动步长"),
                           QStringLiteral("128"), QString(), 132);
  embed_dim_edit_ = add_field(train_card, QStringLiteral("嵌入维度"),
                              QStringLiteral("256"),
                              QStringLiteral("必须能被头数整除"), 132);
  num_heads_edit_ = add_field(train_card, QStringLiteral("注意力头数"),
                              QStringLiteral("8"), QString(), 132);
  num_layers_edit_ = add_field(train_card, QStringLiteral("层数"),
                               QStringLiteral("6"), QString(), 132);
  ckpt_interval_edit_ = add_field(train_card, QStringLiteral("断点保存间隔"),
                                  QStringLiteral("5000"),
                                  QStringLiteral("单位：步；0 = 不保存"), 132);
  {
    auto* row = train_card->add_row_wide(QStringLiteral("初始权重"));
    init_model_edit_ =
        fl::text_field(QString(), QStringLiteral("留空 = 随机初始化，可填 saves/modelN/model.pt"));
    row->addWidget(init_model_edit_, 1);
    init_model_browse_btn_ = fl::plain_button(QStringLiteral("浏览"), ICO_FOLDER);
    connect(init_model_browse_btn_, &QPushButton::clicked, this,
            &TrainingGUI::browse_init_model);
    row->addWidget(init_model_browse_btn_);
  }
  {
    auto* row = train_card->add_row(QStringLiteral("断点续训"),
                                    QStringLiteral("从 models/training_state 继续"));
    resume_check_ = new fl::ToggleSwitch;
    row->addWidget(resume_check_);
  }
  body->addWidget(train_card);

  // ---------------- 分词器 ----------------
  auto* tok_card = new fl::Card;
  tok_card->set_title(
      QStringLiteral("分词器 → 预训练（一条龙）"),
      QStringLiteral("训练 SentencePiece → 编码 token_ids.txt → 打包 dataset.zip → 自动载入训练页"));
  {
    auto* holder = new QWidget;
    auto* row = new QHBoxLayout(holder);
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(10);
    auto* pick_btn = fl::plain_button(QStringLiteral("选择 txt 语料"), ICO_FOLDER);
    connect(pick_btn, &QPushButton::clicked, this,
            &TrainingGUI::pick_tokenizer_corpus);
    row->addWidget(pick_btn);
    auto* raw_btn = fl::plain_button(QStringLiteral("使用 data/raw 目录"));
    connect(raw_btn, &QPushButton::clicked, this,
            &TrainingGUI::use_tokenizer_raw_dir);
    row->addWidget(raw_btn);
    select_tokenizer_btn_ = fl::plain_button(QStringLiteral("载入已有 spm.model"));
    connect(select_tokenizer_btn_, &QPushButton::clicked, this,
            &TrainingGUI::select_tokenizer);
    row->addWidget(select_tokenizer_btn_);
    row->addStretch(1);
    tok_card->add_widget_row(holder, 32);
  }
  {
    auto* holder = new QWidget;
    auto* col = new QVBoxLayout(holder);
    col->setContentsMargins(0, 0, 0, 0);
    col->setSpacing(3);
    tokenizer_corpus_label_ = new QLabel(QStringLiteral("未选择语料"));
    tokenizer_corpus_label_->setProperty("role", "hint");
    tokenizer_path_label_ = new QLabel(QStringLiteral("未载入分词器"));
    tokenizer_path_label_->setProperty("role", "hint");
    tokenizer_stats_label_ = new QLabel;
    tokenizer_stats_label_->setProperty("role", "hint");
    tokenizer_stats_label_->setWordWrap(true);
    col->addWidget(tokenizer_corpus_label_);
    col->addWidget(tokenizer_path_label_);
    col->addWidget(tokenizer_stats_label_);
    tok_card->add_widget_row(holder);
  }
  tok_threads_edit_ = add_field(tok_card, QStringLiteral("训练线程数"),
                                QString(), QStringLiteral("留空 = 自动"), 118);
  tok_vocab_edit_ = add_field(tok_card, QStringLiteral("词表大小"), QString(),
                              QStringLiteral("留空 = 自动"), 118);
  tok_sentences_edit_ = add_field(tok_card, QStringLiteral("最大训练句子数"),
                                  QString(), QStringLiteral("0 = 全部"), 118);
  {
    auto* row = tok_card->add_row_wide(QStringLiteral("输出目录"));
    tokenizer_out_edit_ = fl::text_field();
    row->addWidget(tokenizer_out_edit_, 1);
    auto* out_btn = fl::plain_button(QStringLiteral("浏览"), ICO_FOLDER);
    connect(out_btn, &QPushButton::clicked, this,
            &TrainingGUI::browse_tokenizer_out);
    row->addWidget(out_btn);
    tokenizer_start_btn_ =
        fl::accent_button(QStringLiteral("开始训练分词器"), ICO_TRAIN);
    connect(tokenizer_start_btn_, &QPushButton::clicked, this,
            &TrainingGUI::start_tokenizer_training);
    row->addWidget(tokenizer_start_btn_);
  }
  tok_card->add_note(QStringLiteral(
      "产物：models/spm.model、processed/token_ids.txt、config.json、dataset.zip；"
      "完成后会自动把 dataset.zip 载入「训练」页（词表/序列/数据路径全部接管），"
      "直接点「开始训练」即可预训练"));
  body->addWidget(tok_card);

  // ---------------- 分词器日志 + 测试 ----------------
  auto* tok_bottom = new QHBoxLayout;
  tok_bottom->setSpacing(10);
  auto* tok_log_card = new fl::Card;
  tok_log_card->set_title(QStringLiteral("分词器日志"));
  tokenizer_log_text_ = fl::log_view(11);
  tok_log_card->add_widget_row(tokenizer_log_text_, 150);
  tok_bottom->addWidget(tok_log_card, 3);

  auto* tok_test_card = new fl::Card;
  tok_test_card->set_title(QStringLiteral("测试分词"));
  tokenizer_test_input_ = new QTextEdit;
  tokenizer_test_input_->setProperty("role", "input");
  tokenizer_test_input_->setFont(fl::ui_font(13));
  tokenizer_test_input_->setPlaceholderText(QStringLiteral("输入中文句子测试切分"));
  tok_test_card->add_widget_row(tokenizer_test_input_, 54);
  {
    auto* holder = new QWidget;
    auto* row = new QHBoxLayout(holder);
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(10);
    tokenizer_test_btn_ = fl::plain_button(QStringLiteral("分词"), ICO_CHAT);
    connect(tokenizer_test_btn_, &QPushButton::clicked, this,
            &TrainingGUI::test_tokenize);
    row->addWidget(tokenizer_test_btn_);
    row->addStretch(1);
    tok_test_card->add_widget_row(holder, 32);
  }
  tokenizer_test_output_ = new QTextEdit;
  tokenizer_test_output_->setReadOnly(true);
  tokenizer_test_output_->setFont(fl::mono_font(11));
  tokenizer_test_output_->setProperty("role", "log");
  tokenizer_test_output_->setFrameShape(QFrame::NoFrame);
  tok_test_card->add_widget_row(tokenizer_test_output_, 78);
  tok_bottom->addWidget(tok_test_card, 2);
  body->addLayout(tok_bottom);

  // ---------------- 诊断 ----------------
  auto* diag_card = new fl::Card;
  diag_card->set_title(QStringLiteral("诊断"),
                       QStringLiteral("设备、运行环境与系统日志（排错时看这里）"));
  {
    auto* holder = new QWidget;
    auto* row = new QHBoxLayout(holder);
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(10);
    diag_label_ = new QLabel;
    diag_label_->setProperty("role", "hint");
    row->addWidget(diag_label_, 1);
    clear_syslog_btn_ = fl::plain_button(QStringLiteral("清空日志"), ICO_DELETE);
    connect(clear_syslog_btn_, &QPushButton::clicked, this,
            &TrainingGUI::clear_sys_log);
    row->addWidget(clear_syslog_btn_);
    open_log_dir_btn_ =
        fl::plain_button(QStringLiteral("打开引擎日志"), ICO_FOLDER);
    connect(open_log_dir_btn_, &QPushButton::clicked, this,
            &TrainingGUI::open_llama_log);
    row->addWidget(open_log_dir_btn_);
    diag_card->add_widget_row(holder, 32);
  }
  sys_log_text_ = fl::log_view(11);
  diag_card->add_widget_row(sys_log_text_, 90);
  body->addWidget(diag_card);
  body->addStretch(1);

  // ---------------- 底部操作 ----------------
  actions->addStretch(1);
  auto* open_root_btn = fl::plain_button(QStringLiteral("打开项目目录"), ICO_FOLDER);
  connect(open_root_btn, &QPushButton::clicked, this, [this] {
    QDesktopServices::openUrl(
        QUrl::fromLocalFile(qpath(fs::path(cfg_.project_root()))));
  });
  actions->addWidget(open_root_btn);
}

// ===========================================================================
// 主题
// ===========================================================================
void TrainingGUI::on_theme_changed() {
  // fl::Theme 负责应用级调色板与样式表；这里同步窗口装饰（深色标题栏 / 圆角）
  fl::apply_window_chrome(this);
  setWindowIcon(fl::app_icon());
  update();
}

void TrainingGUI::showEvent(QShowEvent* event) {
  QMainWindow::showEvent(event);
  fl::apply_window_chrome(this);
}

// ===========================================================================
// 状态显示
// ===========================================================================
void TrainingGUI::set_status(const QString& text, const char* tone) {
  if (status_label_ == nullptr) return;
  status_label_->setText(text);
  status_label_->setProperty("status", tone);
  status_label_->style()->unpolish(status_label_);
  status_label_->style()->polish(status_label_);
}

void TrainingGUI::update_stats_bar() {
  if (sb_device_ != nullptr) {
    sb_device_->setText(cfg_.use_cuda ? QStringLiteral("设备: CUDA")
                                      : QStringLiteral("设备: CPU"));
  }
  if (sb_stats_ != nullptr) {
    sb_stats_->setText(QStringLiteral("词表 %1 · 最大序列 %2")
                           .arg(vocab_size_)
                           .arg(max_seq_len_));
  }
  if (diag_label_ != nullptr) {
    diag_label_->setText(QStringLiteral("设备 %1 · 词表 %2 · 最大序列 %3 · 项目根 %4")
                             .arg(cfg_.use_cuda ? "CUDA" : "CPU")
                             .arg(vocab_size_)
                             .arg(max_seq_len_)
                             .arg(qpath(fs::path(cfg_.project_root()))));
  }
}

void TrainingGUI::update_engine_ui() {
  const bool running = llama_.running();
  const bool starting = engine_starting_.load();
  QString text;
  if (starting) {
    text = QStringLiteral("引擎：启动中…");
  } else if (running) {
    text = QStringLiteral("引擎：llama.cpp · %1")
               .arg(loaded_gguf_.isEmpty() ? QStringLiteral("已就绪")
                                           : QFileInfo(loaded_gguf_).fileName());
  } else if (!inference_model_.is_empty()) {
    text = QStringLiteral("引擎：内置 libtorch");
  } else {
    text = QStringLiteral("引擎：未启动");
  }
  if (sb_engine_ != nullptr) sb_engine_->setText(text);
  if (engine_state_label_ != nullptr) {
    engine_state_label_->setText(text);
  }
  if (engine_detail_label_ != nullptr) {
    engine_detail_label_->setText(running ? QString::fromStdString(llama_.model_info())
                                          : QStringLiteral("引擎未启动（点右侧「启动/重启」按当前设置启动）"));
  }
  if (engine_restart_btn_ != nullptr) engine_restart_btn_->setEnabled(!starting);
  if (engine_stop_btn_ != nullptr) engine_stop_btn_->setEnabled(running);
}

// ===========================================================================
// 设置持久化
// ===========================================================================
namespace {
QSettings& ui_settings() {
  static QSettings settings(QStringLiteral("gpt_trainer"),
                            QStringLiteral("ui-settings"));
  return settings;
}
} // namespace

void TrainingGUI::load_settings() {
  QSettings& s = ui_settings();
  const QString engine = s.value(QStringLiteral("engine"), QStringLiteral("auto")).toString();
  const int engine_index = engine_combo_->findData(engine);
  if (engine_index >= 0) engine_combo_->setCurrentIndex(engine_index);

  QString server = s.value(QStringLiteral("server_path")).toString();
  if (server.isEmpty()) {
    const fs::path detected = LlamaEngine::detect_server_exe();
    if (!detected.empty()) server = qpath(detected);
  }
  server_path_edit_->setText(server);
  gguf_path_edit_->setText(s.value(QStringLiteral("gguf_path")).toString());
  // 上下文长度 / 回复长度 / 温度：读取后钳制到合理范围（防止历史数据里存了极端值）
  {
    const long long ctx = std::clamp<long long>(
        s.value(QStringLiteral("ctx"), QStringLiteral("2048")).toLongLong(),
        512, kMaxCtxTokens);
    engine_ctx_edit_->setText(QString::number(ctx));
    if (chat_ctx_edit_ != nullptr) chat_ctx_edit_->setText(QString::number(ctx));
    const long long mt = std::clamp<long long>(
        s.value(QStringLiteral("max_tokens"), QStringLiteral("50")).toLongLong(),
        1, kMaxReplyTokens);
    gen_max_tokens_edit_->setText(QString::number(mt));
    const double temp = std::clamp(
        s.value(QStringLiteral("temperature"), QStringLiteral("0.8")).toDouble(),
        0.01, 2.0);
    gen_temperature_edit_->setText(QString::number(temp, 'g', 4));
  }
  const int chat_index =
      chat_mode_combo_->findData(s.value(QStringLiteral("chat_mode"), QStringLiteral("auto")).toString());
  if (chat_index >= 0) chat_mode_combo_->setCurrentIndex(chat_index);
  engine_threads_edit_->setText(s.value(QStringLiteral("threads"), QStringLiteral("0")).toString());
  engine_ngl_edit_->setText(s.value(QStringLiteral("ngl"), QStringLiteral("99")).toString());
  gen_topk_edit_->setText(s.value(QStringLiteral("top_k"), QStringLiteral("40")).toString());
  gen_topp_edit_->setText(s.value(QStringLiteral("top_p"), QStringLiteral("0.95")).toString());
  gen_repeat_edit_->setText(s.value(QStringLiteral("repeat"), QStringLiteral("1.1")).toString());
  gen_seed_edit_->setText(s.value(QStringLiteral("seed"), QStringLiteral("-1")).toString());
  system_prompt_edit_->setText(s.value(QStringLiteral("system_prompt")).toString());
  gen_template_check_->setChecked(s.value(QStringLiteral("template"), false).toBool());
  gen_multi_turn_check_->setChecked(s.value(QStringLiteral("multi_turn"), true).toBool());
}

void TrainingGUI::save_settings() {
  QSettings& s = ui_settings();
  s.setValue(QStringLiteral("engine"), engine_combo_->currentData().toString());
  s.setValue(QStringLiteral("server_path"), server_path_edit_->text().trimmed());
  s.setValue(QStringLiteral("gguf_path"), gguf_path_edit_->text().trimmed());
  // 保存前也钳制，避免把极端值写回设置（下次启动又被读回来）
  {
    const long long ctx = std::clamp<long long>(
        engine_ctx_edit_->text().trimmed().toLongLong(), 512, kMaxCtxTokens);
    s.setValue(QStringLiteral("ctx"), QString::number(ctx));
    if (engine_ctx_edit_->text() != QString::number(ctx)) {
      engine_ctx_edit_->setText(QString::number(ctx));
      if (chat_ctx_edit_ != nullptr) chat_ctx_edit_->setText(QString::number(ctx));
    }
    const long long mt = std::clamp<long long>(
        gen_max_tokens_edit_->text().trimmed().toLongLong(), 1, kMaxReplyTokens);
    s.setValue(QStringLiteral("max_tokens"), QString::number(mt));
    if (gen_max_tokens_edit_->text() != QString::number(mt)) {
      gen_max_tokens_edit_->setText(QString::number(mt));
    }
    const double temp = std::clamp(
        gen_temperature_edit_->text().trimmed().toDouble(), 0.01, 2.0);
    s.setValue(QStringLiteral("temperature"), QString::number(temp, 'g', 4));
    if (gen_temperature_edit_->text() != QString::number(temp, 'g', 4)) {
      gen_temperature_edit_->setText(QString::number(temp, 'g', 4));
    }
  }
  s.setValue(QStringLiteral("chat_mode"), chat_mode_combo_->currentData().toString());
  s.setValue(QStringLiteral("threads"), engine_threads_edit_->text().trimmed());
  s.setValue(QStringLiteral("ngl"), engine_ngl_edit_->text().trimmed());
  s.setValue(QStringLiteral("top_k"), gen_topk_edit_->text().trimmed());
  s.setValue(QStringLiteral("top_p"), gen_topp_edit_->text().trimmed());
  s.setValue(QStringLiteral("repeat"), gen_repeat_edit_->text().trimmed());
  s.setValue(QStringLiteral("seed"), gen_seed_edit_->text().trimmed());
  s.setValue(QStringLiteral("system_prompt"), system_prompt_edit_->text());
  s.setValue(QStringLiteral("template"), gen_template_check_->isChecked());
  s.setValue(QStringLiteral("multi_turn"), gen_multi_turn_check_->isChecked());
  s.sync();
}

// ===========================================================================
// 小工具
// ===========================================================================
int TrainingGUI::read_int(QLineEdit* edit, int fallback) const {
  if (edit == nullptr) return fallback;
  bool ok = false;
  const int v = edit->text().trimmed().toInt(&ok);
  return ok ? v : fallback;
}

double TrainingGUI::read_double(QLineEdit* edit, double fallback) const {
  if (edit == nullptr) return fallback;
  bool ok = false;
  const double v = edit->text().trimmed().toDouble(&ok);
  return ok ? v : fallback;
}

QString TrainingGUI::current_model_name() const {
  const QString data = model_combo_->currentData().toString();
  if (data.startsWith(QStringLiteral("saves|"))) return data.mid(6);
  if (data.startsWith(QStringLiteral("gguf|"))) return data.mid(5);
  return {};
}

std::filesystem::path TrainingGUI::models_gguf_dir() const {
  fs::path dir = fs::path(cfg_.project_root()) / "models" / "gguf";
  std::error_code ec;
  fs::create_directories(dir, ec);
  return dir;
}

bool TrainingGUI::engine_is_llama() const {
  const QString mode = engine_combo_->currentData().toString();
  if (mode == QLatin1String("builtin")) return false;
  if (mode == QLatin1String("llama")) return true;
  // auto：能找到 llama-server 就用 llama.cpp
  const QString server = server_path_edit_->text().trimmed();
  if (!server.isEmpty() && fs::exists(fspath(server))) return true;
  return !LlamaEngine::detect_server_exe().empty();
}

LlamaSamplingParams TrainingGUI::current_sampling() const {
  LlamaSamplingParams sp;
  sp.max_tokens = std::clamp(read_int(gen_max_tokens_edit_, 50), 1, 4096);
  sp.temperature = std::clamp(read_double(gen_temperature_edit_, 0.8), 0.01, 4.0);
  sp.top_k = std::clamp(read_int(gen_topk_edit_, 40), 0, 100000);
  sp.top_p = std::clamp(read_double(gen_topp_edit_, 0.95), 0.0, 1.0);
  sp.repeat_penalty = std::clamp(read_double(gen_repeat_edit_, 1.1), 0.5, 3.0);
  sp.seed = read_int(gen_seed_edit_, -1);
  return sp;
}

QString TrainingGUI::build_llama_prompt(const QString& user_text) const {
  QString prompt;
  const QString sys = system_prompt_edit_->text().trimmed();
  if (!sys.isEmpty()) prompt += sys + QStringLiteral("\n");
  const bool multi_turn = gen_multi_turn_check_->isChecked();
  if (multi_turn && !chat_context_.isEmpty()) prompt += chat_context_;
  if (gen_template_check_->isChecked()) {
    prompt += QStringLiteral("问：%1\n答：").arg(user_text);
  } else {
    prompt += user_text;
  }
  return prompt;
}

// ===========================================================================
// llama.cpp 引擎
// ===========================================================================
bool TrainingGUI::start_llama_for_gguf(const std::filesystem::path& gguf,
                                       bool interactive) {
  fs::path server = fspath(server_path_edit_->text().trimmed());
  if (server.empty() || !fs::exists(server)) {
    server = LlamaEngine::detect_server_exe();
  }
  if (server.empty()) {
    const QString msg =
        QStringLiteral("找不到 llama-server.exe。\n\n可以：\n"
                       "1) 在「高级」页点「探测」自动查找（支持 Ollama 自带版本）；\n"
                       "2) 下载 llama.cpp 发布包，把 llama-server.exe 放到 "
                       "third_party\\llama.cpp\\ 下；\n"
                       "3) 手动「浏览」选择 llama-server.exe。");
    if (interactive) QMessageBox::information(this, QStringLiteral("未找到推理引擎"), msg);
    sys_log(QStringLiteral("推理引擎缺失: %1").arg(msg.split('\n').first()));
    return false;
  }
  if (!server_path_edit_->text().trimmed().isEmpty() == false) {
    server_path_edit_->setText(qpath(server));
  }

  LlamaEngineOptions opts;
  opts.server_exe = server;
  opts.model_path = gguf;
  opts.n_ctx = std::clamp(read_int(engine_ctx_edit_, 2048), 128, kMaxCtxTokens);
  opts.n_threads = std::clamp(read_int(engine_threads_edit_, 0), 0, 256);
  opts.n_gpu_layers = std::clamp(read_int(engine_ngl_edit_, 99), 0, 999);
  std::string err;
  if (!llama_.start(opts, &err)) {
    sys_log(QStringLiteral("llama.cpp 启动失败: %1")
                .arg(QString::fromStdString(err)));
    return false;
  }
  loaded_gguf_ = qpath(gguf);
  llama_sp_ = load_spm_for_gguf(gguf);
  QString channel;
  if (llama_sp_ != nullptr) {
    channel = QStringLiteral("sentencepiece token 通道");
  } else if (llama_.has_chat_template()) {
    channel = QStringLiteral("对话模板通道（Chat 接口）");
  } else {
    channel = QStringLiteral("文本通道");
  }
  sys_log(QStringLiteral("llama.cpp 就绪: %1（%2）")
              .arg(QString::fromStdString(llama_.model_info()), channel));
  return true;
}

void TrainingGUI::load_gguf_and_chat(const std::filesystem::path& gguf) {
  engine_starting_.store(true);
  QMetaObject::invokeMethod(
      this,
      [this] {
        set_status(QStringLiteral("正在启动 llama.cpp…"), "busy");
        update_engine_ui();
      },
      Qt::QueuedConnection);

  const bool ok = start_llama_for_gguf(gguf, false);

  QMetaObject::invokeMethod(
      this,
      [this, ok, gguf] {
        engine_starting_.store(false);
        if (ok) {
          model_info_label_->setText(QStringLiteral("已加载: %1")
                                         .arg(QFileInfo(qpath(gguf)).fileName()));
          model_info_label_->setProperty("status", "ok");
          set_status(QStringLiteral("就绪"), "ok");
        } else {
          model_info_label_->setText(QStringLiteral("加载失败（详见高级页诊断日志）"));
          model_info_label_->setProperty("status", "warn");
          set_status(QStringLiteral("引擎启动失败"), "warn");
        }
        model_info_label_->style()->unpolish(model_info_label_);
        model_info_label_->style()->polish(model_info_label_);
        update_engine_ui();
        generate_btn_->setEnabled(ok);
      },
      Qt::QueuedConnection);
}

void TrainingGUI::run_llama_generate(const QString& prompt,
                                     const QString& full_prompt) {
  LlamaSamplingParams sp = current_sampling();

  // 提示词通道：默认自动（本项目模型走 token，Qwen 等对话模型套用 chat 模板）
  LlamaPromptMode mode = LlamaPromptMode::kAuto;
  if (chat_mode_combo_ != nullptr) {
    const QString m = chat_mode_combo_->currentData().toString();
    if (m == QStringLiteral("chat")) {
      mode = LlamaPromptMode::kChat;
    } else if (m == QStringLiteral("text")) {
      mode = LlamaPromptMode::kText;
    } else if (m == QStringLiteral("tokens")) {
      mode = LlamaPromptMode::kTokens;
    }
  }
  // 「问答模板」的停止词只对原始补全有意义，对话通道由模型模板自己收尾
  if (gen_template_check_ != nullptr && gen_template_check_->isChecked() &&
      mode != LlamaPromptMode::kChat) {
    sp.stop = {"\n问：", "\n答："};
  }
  const std::string sys =
      (system_prompt_edit_ != nullptr)
          ? system_prompt_edit_->text().trimmed().toStdString()
          : std::string();

  std::string err;
  std::string out;
  int pieces = 0;  // SSE 增量计数 ≈ 生成的 token 数（与引擎同线程，直接累加）
  size_t prompt_tokens = 0;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = llama_complete_smart(
      llama_, full_prompt.toUtf8().toStdString(), llama_sp_.get(),
      max_seq_len_, sp,
      [this, &pieces](const std::string& piece) {
        if (!piece.empty()) ++pieces;
        const QString q = QString::fromStdString(piece);
        QMetaObject::invokeMethod(
            this,
            [this, q] {
              if (chat_view_ != nullptr) chat_view_->stream_delta(q);
            },
            Qt::QueuedConnection);
      },
      &out, &err, mode, sys, &prompt_tokens);

  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  QString stats;
  if (ok && pieces > 0) {
    const double tps = pieces / std::max(0.001, secs);
    stats = QStringLiteral("⚡ %1 tok · %2 s · %3 tok/s")
                .arg(pieces)
                .arg(secs, 0, 'f', 2)
                .arg(tps, 0, 'f', 1);
  }

  QMetaObject::invokeMethod(
      this,
      [this, ok, err, prompt, out, stats, pieces, secs,
       prompt_tokens = (long long)prompt_tokens] {
        const QString text = QString::fromStdString(out);
        if (chat_view_ != nullptr) chat_view_->end_stream(stats);
        update_chat_stats(ok ? pieces : 0, secs, ok ? prompt_tokens : -1);
        if (!ok) {
          const QString msg = QString::fromStdString(err);
          sys_log(QStringLiteral("生成失败: %1").arg(msg));
          if (msg.contains(QStringLiteral("已取消"))) {
            set_status(QStringLiteral("已停止"), "ok");
          } else {
            set_status(QStringLiteral("生成失败"), "warn");
          }
        } else {
          sys_log(QStringLiteral("生成完成: %1 字符").arg(text.size()));
          set_status(QStringLiteral("就绪"), "ok");
          if (gen_multi_turn_check_->isChecked()) {
            if (gen_template_check_->isChecked()) {
              chat_context_ +=
                  QStringLiteral("问：%1\n答：%2\n").arg(prompt, text);
            } else {
              chat_context_ +=
                  prompt + QStringLiteral("\n") + text + QStringLiteral("\n");
            }
            if (chat_context_.size() > 4000) {
              chat_context_ = chat_context_.right(4000);
            }
          }
          input_text_->clear();
        }
        generating_.store(false);
        generate_btn_->setEnabled(llama_.running() || !inference_model_.is_empty());
        stop_gen_btn_->setEnabled(false);
      },
      Qt::QueuedConnection);
}

// ===========================================================================
// 高级页槽函数
// ===========================================================================
void TrainingGUI::browse_server_exe() {
  const QString file = QFileDialog::getOpenFileName(
      this, QStringLiteral("选择 llama-server.exe"), QString(),
      QStringLiteral("llama-server (llama-server.exe);;可执行文件 (*.exe)"));
  if (file.isEmpty()) return;
  server_path_edit_->setText(file);
  sys_log(QStringLiteral("llama-server: %1").arg(file));
  update_engine_ui();
}

void TrainingGUI::detect_server_exe() {
  const fs::path found = LlamaEngine::detect_server_exe(
      fspath(server_path_edit_->text().trimmed()));
  if (found.empty()) {
    QMessageBox::information(
        this, QStringLiteral("未找到 llama-server.exe"),
        QStringLiteral("没有探测到 llama.cpp 的 llama-server.exe。\n\n"
                       "可以下载 llama.cpp 发布包，把 llama-server.exe 放到 "
                       "third_party\\llama.cpp\\ 目录，或手动浏览指定。"));
    sys_log(QStringLiteral("未探测到 llama-server.exe"));
    return;
  }
  server_path_edit_->setText(qpath(found));
  sys_log(QStringLiteral("已探测到 llama-server: %1").arg(qpath(found)));
  update_engine_ui();
}

void TrainingGUI::browse_gguf_file() {
  const QString file = QFileDialog::getOpenFileName(
      this, QStringLiteral("选择 GGUF 模型"), qpath(models_gguf_dir()),
      QStringLiteral("GGUF 模型 (*.gguf);;所有文件 (*)"));
  if (file.isEmpty()) return;
  gguf_path_edit_->setText(file);
  sys_log(QStringLiteral("GGUF 模型: %1").arg(file));
}

void TrainingGUI::restart_engine() {
  QString gguf = gguf_path_edit_->text().trimmed();
  if (gguf.isEmpty()) {
    if (!last_export_gguf_.isEmpty()) {
      gguf = last_export_gguf_;
    } else {
      QMessageBox::information(
          this, QStringLiteral("还没有 GGUF"),
          QStringLiteral("请先在上方选择 GGUF 模型，或在「GGUF 导出」里把 saves 模型导出为 GGUF。"));
      return;
    }
  }
  engine_starting_.store(true);
  update_engine_ui();
  if (engine_thread_.joinable()) engine_thread_.join();
  engine_thread_ = std::thread([this, gguf] {
    const bool ok = start_llama_for_gguf(fspath(gguf), false);
    QMetaObject::invokeMethod(
        this,
        [this, ok, gguf] {
          engine_starting_.store(false);
          set_status(ok ? QStringLiteral("就绪") : QStringLiteral("引擎启动失败"),
                     ok ? "ok" : "warn");
          if (ok) {
            model_info_label_->setText(
                QStringLiteral("已加载: %1").arg(QFileInfo(gguf).fileName()));
            model_info_label_->setProperty("status", "ok");
            model_info_label_->style()->unpolish(model_info_label_);
            model_info_label_->style()->polish(model_info_label_);
          }
          update_engine_ui();
          generate_btn_->setEnabled(ok || !inference_model_.is_empty());
        },
        Qt::QueuedConnection);
  });
}

void TrainingGUI::stop_engine() {
  llama_.cancel();
  llama_.stop();
  llama_sp_.reset();
  sys_log(QStringLiteral("llama.cpp 引擎已停止"));
  update_engine_ui();
  generate_btn_->setEnabled(!inference_model_.is_empty());
}

void TrainingGUI::open_llama_log() {
  const fs::path log = llama_.log_path();
  if (log.empty() || !fs::exists(log)) {
    QDesktopServices::openUrl(QUrl::fromLocalFile(
        qpath(fs::temp_directory_path())));
    return;
  }
  QDesktopServices::openUrl(QUrl::fromLocalFile(qpath(log)));
}

void TrainingGUI::export_current_gguf() {
  const QString data = adv_model_combo_->currentData().toString();
  if (!data.startsWith(QStringLiteral("saves|"))) {
    QMessageBox::information(this, QStringLiteral("提示"),
                             QStringLiteral("请选择一个 saves 中的模型再导出。"));
    return;
  }
  const QString name = data.mid(6);
  const fs::path model_dir = fs::path(cfg_.save_root) / fspath(name);
  fs::path out = fspath(export_gguf_edit_->text().trimmed());
  if (out.empty()) out = models_gguf_dir() / fspath(name + QStringLiteral(".gguf"));

  export_gguf_btn_->setEnabled(false);
  export_result_label_->setText(QStringLiteral("正在导出…"));
  if (export_thread_.joinable()) export_thread_.join();
  export_thread_ = std::thread([this, model_dir, out, name] {
    GgufExportResult result;
    std::string err;
    bool ok = false;
    try {
      ok = export_gpt_to_gguf(model_dir, out, fs::path(), &result, &err);
    } catch (const std::exception& e) {
      err = e.what();
    }
    const QString err_q = QString::fromStdString(err);
    QMetaObject::invokeMethod(
        this,
        [this, ok, err_q, result, name, out] {
          export_gguf_btn_->setEnabled(true);
          if (ok) {
            last_export_gguf_ = qpath(result.out_path);
            export_gguf_edit_->setText(last_export_gguf_);
            export_result_label_->setText(
                QStringLiteral("已导出 %1（%2 个张量 · 词表 %3 · %4 MB）")
                    .arg(QFileInfo(last_export_gguf_).fileName())
                    .arg(result.tensor_count)
                    .arg(result.vocab_size)
                    .arg(QFileInfo(last_export_gguf_).size() / (1024.0 * 1024.0), 0,
                         'f', 1));
            sys_log(QStringLiteral("GGUF 导出成功: %1").arg(last_export_gguf_));
            if (!err_q.isEmpty()) sys_log(err_q);
            refresh_model_list();
          } else {
            export_result_label_->setText(QStringLiteral("导出失败：%1").arg(err_q));
            sys_log(QStringLiteral("GGUF 导出失败: %1").arg(err_q));
            QMessageBox::warning(this, QStringLiteral("导出失败"), err_q);
          }
          (void)name;
          (void)out;
        },
        Qt::QueuedConnection);
  });
}

// ===========================================================================
// 日志
// ===========================================================================
void TrainingGUI::train_log(const QString& text) {
  if (train_log_text_ == nullptr) return;
  train_log_text_->append(text);
  train_log_text_->verticalScrollBar()->setValue(
      train_log_text_->verticalScrollBar()->maximum());
}

void TrainingGUI::sys_log(const QString& text) {
  if (sys_log_text_ == nullptr) return;
  const QString ts = QTime::currentTime().toString("HH:mm:ss");
  sys_log_text_->append(QString("[%1] %2").arg(ts, text));
  sys_log_text_->verticalScrollBar()->setValue(
      sys_log_text_->verticalScrollBar()->maximum());
}

// ===========================================================================
// 模型列表
// ===========================================================================
void TrainingGUI::update_chat_stats(int tokens, double secs,
                                    long long prompt_tokens) {
  if (tokens > 0) {
    chat_total_tokens_ += tokens;
    ++chat_rounds_;
  }
  if (prompt_tokens >= 0) {
    ctx_used_ = prompt_tokens + std::max(0, tokens);
    // 上下文用量环：已用 / 上下文上限
    if (ctx_ring_ != nullptr) {
      const int limit = std::max(64, read_int(engine_ctx_edit_, 2048));
      const double ratio = (double)ctx_used_ / (double)limit;
      ctx_ring_->set_ratio(ratio);
      ctx_ring_->set_caption(
          QStringLiteral("上下文用量：%1 / %2 tok（%3%）")
              .arg(ctx_used_)
              .arg(limit)
              .arg((int)std::lround(ratio * 100.0)));
    }
  }
  if (chat_stats_label_ == nullptr) return;
  if (tokens <= 0 && ctx_used_ <= 0) {
    chat_stats_label_->setText(QStringLiteral("就绪 · 等待第一条消息"));
    return;
  }
  if (tokens <= 0) return;
  const double tps = secs > 0.001 ? (double)tokens / secs : 0.0;
  chat_stats_label_->setText(
      QStringLiteral("⚡ 本轮 %1 tok · %2 s · %3 tok/s    ·    累计 %4 tok · 第 %5 轮")
          .arg(tokens)
          .arg(secs, 0, 'f', 2)
          .arg(tps, 0, 'f', 1)
          .arg(chat_total_tokens_)
          .arg(chat_rounds_));
}

void TrainingGUI::refresh_model_list() {
  model_list_filling_ = true; // 填充期间不触发"选定即加载"
  const QString keep = model_combo_ ? model_combo_->currentData().toString() : QString();
  if (model_combo_ != nullptr) model_combo_->clear();
  if (adv_model_combo_ != nullptr) adv_model_combo_->clear();
  struct FillingGuard {
    bool* flag;
    ~FillingGuard() { *flag = false; }
  } guard{&model_list_filling_};

  // 1) saves 中训练出来的模型
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
  for (const auto& [num, name] : items) {
    if (model_combo_ != nullptr) {
      model_combo_->addItem(QStringLiteral("%1 · 训练模型").arg(name),
                            QStringLiteral("saves|") + name);
    }
    if (adv_model_combo_ != nullptr) {
      adv_model_combo_->addItem(QStringLiteral("%1 · 训练模型").arg(name),
                                QStringLiteral("saves|") + name);
    }
  }

  // 2) models/gguf 下已导出的 GGUF
  const fs::path gguf_dir = models_gguf_dir();
  std::vector<fs::path> ggufs;
  if (fs::exists(gguf_dir, ec)) {
    for (const auto& entry : fs::directory_iterator(gguf_dir, ec)) {
      if (entry.is_regular_file(ec) && entry.path().extension() == ".gguf") {
        ggufs.push_back(entry.path());
      }
    }
  }
  std::sort(ggufs.begin(), ggufs.end());
  for (const auto& path : ggufs) {
    if (model_combo_ != nullptr) {
      model_combo_->addItem(QStringLiteral("%1 · GGUF").arg(qpath(path.filename())),
                            QStringLiteral("gguf|") + qpath(path));
    }
    if (adv_model_combo_ != nullptr) {
      adv_model_combo_->addItem(QStringLiteral("%1 · GGUF").arg(qpath(path.filename())),
                                QStringLiteral("gguf|") + qpath(path));
    }
  }

  // 3) 高级页手动指定的 GGUF
  const QString manual = gguf_path_edit_ != nullptr ? gguf_path_edit_->text().trimmed() : QString();
  if (!manual.isEmpty() && fs::exists(fspath(manual))) {
    const QString data = QStringLiteral("gguf|") + manual;
    if (model_combo_ != nullptr && model_combo_->findData(data) < 0) {
      model_combo_->addItem(QStringLiteral("%1 · 外部 GGUF").arg(QFileInfo(manual).fileName()),
                            data);
    }
    if (adv_model_combo_ != nullptr && adv_model_combo_->findData(data) < 0) {
      adv_model_combo_->addItem(QStringLiteral("%1 · 外部 GGUF").arg(QFileInfo(manual).fileName()),
                                data);
    }
  }

  const bool empty = model_combo_ != nullptr && model_combo_->count() == 0;
  if (empty) {
    model_combo_->addItem(QStringLiteral("（暂无模型：先在「训练」页训练，或在「高级」页导出 GGUF）"),
                          QString());
    if (adv_model_combo_ != nullptr) {
      adv_model_combo_->addItem(QStringLiteral("（暂无模型）"), QString());
    }
  } else if (!keep.isEmpty() && model_combo_ != nullptr) {
    const int idx = model_combo_->findData(keep);
    if (idx >= 0) model_combo_->setCurrentIndex(idx);
  }
}

// ===========================================================================
// 模型加载
// ===========================================================================
void TrainingGUI::request_model_load(const ModelLoadJob& job) {
  {
    std::lock_guard<std::mutex> lock(load_mutex_);
    pending_job_ = job;
    has_pending_job_ = true;
  }
  // 已有加载线程在跑：它会取走最新目标（合并连续切换请求，UI 不阻塞）
  if (load_worker_running_.exchange(true)) return;
  if (engine_thread_.joinable()) engine_thread_.detach();
  engine_thread_ = std::thread([this] { model_load_worker(); });
}

// 模型加载工作线程：串行处理最新的加载请求（切换模型时自动放弃过期的中间请求）
void TrainingGUI::model_load_worker() {
  for (;;) {
    ModelLoadJob job;
    {
      std::lock_guard<std::mutex> lock(load_mutex_);
      if (!has_pending_job_) {
        load_worker_running_.store(false);
        return;
      }
      job = pending_job_;
      has_pending_job_ = false;
    }

    const QString display = QFileInfo(qpath(job.gguf)).fileName();
    QMetaObject::invokeMethod(
        this,
        [this, display] {
          engine_starting_.store(true);
          set_status(QStringLiteral("正在加载 %1 …").arg(display), "busy");
          update_engine_ui();
        },
        Qt::QueuedConnection);

    // 本项目训练出的模型：先确保 GGUF 已导出/是最新的
    if (!job.model_dir.empty()) {
      std::error_code time_ec;
      bool need_export = !fs::exists(job.gguf, time_ec);
      if (!need_export) {
        need_export = fs::last_write_time(job.gguf, time_ec) <
                      fs::last_write_time(job.model_path, time_ec);
      }
      if (need_export) {
        GgufExportResult result;
        std::string err;
        bool ok = false;
        try {
          ok = export_gpt_to_gguf(job.model_dir, job.gguf, fs::path(), &result,
                                  &err);
        } catch (const std::exception& e) {
          err = e.what();
        }
        const QString err_q = QString::fromStdString(err);
        QMetaObject::invokeMethod(
            this,
            [this, ok, err_q, gguf = job.gguf] {
              if (ok) {
                last_export_gguf_ = qpath(gguf);
                sys_log(QStringLiteral("已导出 GGUF: %1").arg(qpath(gguf)));
              } else {
                sys_log(QStringLiteral("GGUF 导出失败: %1").arg(err_q));
              }
            },
            Qt::QueuedConnection);
        if (!ok) {
          QMetaObject::invokeMethod(
              this,
              [this, err_q] {
                engine_starting_.store(false);
                set_status(QStringLiteral("GGUF 导出失败"), "warn");
                update_engine_ui();
                QMessageBox::warning(
                    this, QStringLiteral("无法用 llama.cpp 运行"),
                    QStringLiteral("导出 GGUF 失败：\n%1\n\n"
                                   "可在「高级」页把推理引擎切换为「内置（libtorch）」后重试。")
                        .arg(err_q));
              },
              Qt::QueuedConnection);
          continue;
        }
      }
    }

    const bool ok = start_llama_for_gguf(job.gguf, false);
    QMetaObject::invokeMethod(
        this,
        [this, ok, gguf = job.gguf] {
          engine_starting_.store(false);
          if (ok) {
            model_info_label_->setText(QStringLiteral("已加载: %1")
                                           .arg(QFileInfo(qpath(gguf)).fileName()));
            model_info_label_->setProperty("status", "ok");
            set_status(QStringLiteral("就绪"), "ok");
          } else {
            model_info_label_->setText(QStringLiteral("加载失败（详见高级页诊断日志）"));
            model_info_label_->setProperty("status", "warn");
            set_status(QStringLiteral("引擎启动失败"), "warn");
          }
          model_info_label_->style()->unpolish(model_info_label_);
          model_info_label_->style()->polish(model_info_label_);
          update_engine_ui();
          generate_btn_->setEnabled(ok || !inference_model_.is_empty());
        },
        Qt::QueuedConnection);
  }
}

void TrainingGUI::load_selected_model() {
  const QString data = model_combo_ != nullptr ? model_combo_->currentData().toString()
                                               : QString();
  if (data.isEmpty()) {
    QMessageBox::information(this, QStringLiteral("提示"),
                             QStringLiteral("还没有可加载的模型。先在「训练」页训练一个，"
                                            "或在「高级」页把模型导出为 GGUF。"));
    return;
  }

  // 切换模型前先停掉正在进行的生成
  if (generating_.load()) stop_generation();
  if (gen_thread_.joinable()) gen_thread_.join();

  // ---- GGUF：直接交给 llama.cpp（异步，不阻塞 UI）----
  if (data.startsWith(QStringLiteral("gguf|"))) {
    const fs::path gguf = fspath(data.mid(5));
    std::error_code ec;
    if (!fs::is_regular_file(gguf, ec)) {
      QMessageBox::warning(this, QStringLiteral("提示"),
                           QStringLiteral("GGUF 文件不存在：\n%1").arg(qpath(gguf)));
      return;
    }
    sys_log(QStringLiteral("切换模型: %1").arg(qpath(gguf)));
    ModelLoadJob job;
    job.gguf = gguf;
    model_info_label_->setText(
        QStringLiteral("正在加载: %1").arg(QFileInfo(qpath(gguf)).fileName()));
    request_model_load(job);
    return;
  }

  const QString name = data.mid(6);
  const fs::path model_dir = fs::path(cfg_.save_root) / fspath(name);
  const fs::path config_path = model_dir / "config.json";
  const fs::path model_path = model_dir / "model.pt";
  std::error_code ec;
  if (!fs::exists(config_path, ec) || !fs::exists(model_path, ec)) {
    QMessageBox::warning(this, QStringLiteral("提示"),
                         QStringLiteral("模型文件不完整，请检查 %1").arg(qpath(model_dir)));
    return;
  }

  // 读取超参（用于状态栏显示）
  try {
    Json cfg_json = Json::parse(read_text_file(config_path).toStdString());
    vocab_size_ = cfg_json.get_int("vocab_size").value_or(8000);
    max_seq_len_ = cfg_json.get_int("max_seq_len").value_or(256);
    update_stats_bar();
  } catch (...) {
  }

  // ---- llama.cpp：导出/复用 GGUF 后启动引擎（异步）----
  if (engine_is_llama()) {
    sys_log(QStringLiteral("引擎选择: llama.cpp（server=%1）")
                .arg(server_path_edit_ != nullptr
                         ? server_path_edit_->text().trimmed()
                         : QStringLiteral("<空>")));
    ModelLoadJob job;
    job.model_dir = model_dir;
    job.model_path = model_path;
    job.gguf = models_gguf_dir() / fspath(name + QStringLiteral(".gguf"));
    sys_log(QStringLiteral("切换模型: %1").arg(name));
    model_info_label_->setText(QStringLiteral("正在加载: %1").arg(name));
    request_model_load(job);
    return;
  }

  // ---- 内置 libtorch 引擎 ----
  sys_log(QStringLiteral("引擎选择: 内置 libtorch（词表 %1）").arg(vocab_size_));
  // 分词器词表必须与模型一致，否则 token id 会越过 embedding 范围
  // （CUDA 上表现为 CUBLAS_STATUS_EXECUTION_FAILED / illegal memory access）
  if (sp_ != nullptr && (int64_t)sp_->GetPieceSize() != vocab_size_) {
    sys_log(QStringLiteral("分词器词表 %1 与模型词表 %2 不一致，重新匹配")
                .arg((qlonglong)sp_->GetPieceSize())
                .arg((qlonglong)vocab_size_));
    sp_.reset();
    sp_model_path_.clear();
  }
  if (sp_ == nullptr) {
    // 自动挑选与模型词表大小匹配的 spm.model
    for (const auto& cand : find_spm_candidates(model_dir)) {
      auto sp = std::make_shared<sentencepiece::SentencePieceProcessor>();
      if (!sp->Load(path_to_ansi(cand)).ok()) continue;
      if ((int64_t)sp->GetPieceSize() != vocab_size_) continue;
      sp_ = sp;
      sp_model_path_ = cand.string();
      if (tokenizer_path_label_ != nullptr) tokenizer_path_label_->setText(qpath(cand));
      sys_log(QStringLiteral("自动载入分词器: %1").arg(qpath(cand)));
      break;
    }
  }
  if (!sp_ || (int64_t)sp_->GetPieceSize() != vocab_size_) {
    QMessageBox::warning(
        this, QStringLiteral("分词器不匹配"),
        QStringLiteral("找到的分词器词表大小与模型不一致：\n\n"
                       "模型 %1 需要 %2 词表的分词器\n"
                       "当前分词器词表：%3\n\n"
                       "请到「高级 → 分词器」选择与该模型配套的 spm.model"
                       "（训练该模型时用的那个），再重新加载模型。")
            .arg(name)
            .arg(vocab_size_)
            .arg(sp_ != nullptr ? QString::number(sp_->GetPieceSize())
                                : QStringLiteral("无")));
    set_status(QStringLiteral("分词器与模型不匹配"), "warn");
    return;
  }

  try {
    torch::Device dev = torch::cuda::is_available()
                            ? torch::Device(torch::kCUDA)
                            : torch::Device(torch::kCPU);
    inference_model_ = load_gpt_from_saves(model_dir, dev);
    loaded_model_dir_ = model_dir.string();
    chat_context_.clear();
    model_info_label_->setText(
        QStringLiteral("已加载: %1（内置引擎 · 词表 %2 · 序列 %3）")
            .arg(name)
            .arg(vocab_size_)
            .arg(max_seq_len_));
    model_info_label_->setProperty("status", "ok");
    model_info_label_->style()->unpolish(model_info_label_);
    model_info_label_->style()->polish(model_info_label_);
    generate_btn_->setEnabled(true);
    update_engine_ui();
    sys_log(QStringLiteral("内置引擎加载模型 %1 成功").arg(name));
    set_status(QStringLiteral("就绪"), "ok");
  } catch (const std::exception& e) {
    const QString msg = QString::fromUtf8(e.what());
    QMessageBox::critical(this, QStringLiteral("加载失败"),
                          QStringLiteral("加载模型时出错: %1").arg(msg));
    model_info_label_->setText(QStringLiteral("加载失败"));
    model_info_label_->setProperty("status", "warn");
    model_info_label_->style()->unpolish(model_info_label_);
    model_info_label_->style()->polish(model_info_label_);
    generate_btn_->setEnabled(false);
  }
}

void TrainingGUI::delete_selected_model() {
  const QString data = adv_model_combo_->currentData().toString();
  if (data.isEmpty()) return;
  const QString name = data.mid(6);
  if (!data.startsWith(QStringLiteral("saves|"))) {
    QMessageBox::information(
        this, QStringLiteral("提示"),
        QStringLiteral("GGUF 文件请在资源管理器里删除；这里只管理 saves 中的训练模型。"));
    return;
  }
  const fs::path model_dir = fs::path(cfg_.save_root) / fspath(name);
  if (!fs::exists(model_dir)) {
    QMessageBox::warning(this, QStringLiteral("提示"),
                         QStringLiteral("模型目录不存在: %1").arg(qpath(model_dir)));
    return;
  }
  const auto reply = QMessageBox::question(
      this, QStringLiteral("确认删除"),
      QStringLiteral("确定要删除模型 '%1' 吗？\n此操作不可恢复！").arg(name),
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
  // 顺带删掉对应的 GGUF 缓存
  std::error_code ec2;
  fs::remove(models_gguf_dir() / fspath(name + QStringLiteral(".gguf")), ec2);
  sys_log(QStringLiteral("模型 %1 已删除").arg(name));
  refresh_model_list();
  if (loaded_model_dir_ == model_dir.string()) {
    inference_model_ = nullptr;
    loaded_model_dir_.clear();
    model_info_label_->setText(QStringLiteral("未加载模型"));
    generate_btn_->setEnabled(false);
  }
  QMessageBox::information(this, QStringLiteral("成功"),
                           QStringLiteral("模型 %1 已删除。").arg(name));
}

// ===========================================================================
// 对话生成
// ===========================================================================
void TrainingGUI::generate_response() {
  if (generating_.load()) return;
  const QString prompt = input_text_->toPlainText().trimmed();
  if (prompt.isEmpty()) {
    QMessageBox::information(this, QStringLiteral("提示"),
                             QStringLiteral("请先输入内容。"));
    return;
  }
  if (gen_thread_.joinable()) gen_thread_.join();

  // ---- llama.cpp 引擎（流式输出）----
  if (llama_.running()) {
    if (chat_view_ != nullptr) {
      chat_view_->add_message(true, prompt);
      chat_view_->begin_stream(false);
    }
    if (generate_btn_->hasFocus()) input_text_->setFocus();
    generating_.store(true);
    generate_btn_->setEnabled(false);
    stop_gen_btn_->setEnabled(true);
    set_status(QStringLiteral("生成中…"), "busy");
    const QString full = build_llama_prompt(prompt);
    sys_log(QStringLiteral("llama.cpp 生成（%1 字符提示）").arg(full.size()));
    gen_thread_ = std::thread(
        [this, prompt, full] { run_llama_generate(prompt, full); });
    return;
  }

  // ---- 内置引擎 ----
  if (inference_model_.is_empty()) {
    QMessageBox::information(this, QStringLiteral("提示"),
                             QStringLiteral("请先加载模型：上面选一个模型后点「加载」。"));
    return;
  }
  if (!sp_) {
    QMessageBox::warning(this, QStringLiteral("提示"),
                         QStringLiteral("内置引擎需要分词器，请在「高级」页载入 spm.model。"));
    return;
  }
  const int gen_max = std::clamp(read_int(gen_max_tokens_edit_, 50), 1, 4096);
  const double gen_temp =
      std::clamp(read_double(gen_temperature_edit_, 0.8), 0.01, 4.0);
  const int gen_topk = std::clamp(read_int(gen_topk_edit_, 40), 0, 100000);
  const bool use_template = gen_template_check_->isChecked();
  const bool multi_turn = gen_multi_turn_check_->isChecked();

  set_status(QStringLiteral("生成中…"), "busy");
  if (generate_btn_->hasFocus()) input_text_->setFocus();
  generating_.store(true);
  generate_btn_->setEnabled(false);
  stop_gen_btn_->setEnabled(false);
  sys_log(QStringLiteral("内置引擎开始生成…"));
  if (chat_view_ != nullptr) chat_view_->add_message(true, prompt);

  QString full;
  if (use_template) {
    full = QStringLiteral("问：%1\n答：").arg(prompt);
    if (multi_turn && !chat_context_.isEmpty()) full = chat_context_ + full;
  } else {
    full = prompt;
    if (multi_turn && !chat_context_.isEmpty()) {
      full = chat_context_ + QStringLiteral("\n") + full;
    }
  }

  try {
    std::vector<int> ids;
    const auto status = sp_->Encode(full.toUtf8().toStdString(), &ids);
    if (!status.ok()) throw std::runtime_error(status.ToString());
    if ((int64_t)ids.size() > max_seq_len_) {
      const size_t drop = ids.size() - (size_t)max_seq_len_;
      ids.erase(ids.begin(), ids.begin() + drop);
      sys_log(QStringLiteral("输入较长，已截断保留最近 %1 个 token").arg(max_seq_len_));
    }
    sys_log(QStringLiteral("编码后 token 数量: %1").arg((int)ids.size()));
    // 防御：token id 必须落在模型词表范围内（否则 CUDA 上会 illegal memory access）
    int max_id = 0;
    for (int id : ids) max_id = std::max(max_id, id);
    if (max_id >= vocab_size_) {
      QMessageBox::warning(
          this, QStringLiteral("分词器与模型不匹配"),
          QStringLiteral("分词结果里有超出模型词表的 token（id=%1，词表只有 %2）。\n\n"
                         "说明当前分词器与模型不是配套的，请到「高级 → 分词器」"
                         "选择训练该模型时用的 spm.model 后重新加载模型。")
              .arg(max_id)
              .arg(vocab_size_));
      sys_log(QStringLiteral("token id %1 超出词表 %2，已阻止生成")
                  .arg(max_id)
                  .arg(vocab_size_));
      generating_.store(false);
      generate_btn_->setEnabled(true);
      set_status(QStringLiteral("分词器与模型不匹配"), "warn");
      return;
    }
    gen_thread_ = std::thread([this, ids, gen_max, gen_temp, gen_topk,
                               use_template, multi_turn, prompt] {
      run_generate(ids, gen_max, gen_temp, gen_topk, use_template, multi_turn,
                   prompt);
    });
  } catch (const std::exception& e) {
    const QString msg = QString::fromUtf8(e.what());
    sys_log(QStringLiteral("编码错误: %1").arg(msg));
    QMessageBox::critical(this, QStringLiteral("编码错误"), msg);
    set_status(QStringLiteral("就绪"), "ok");
    generating_.store(false);
    generate_btn_->setEnabled(true);
  }
}

void TrainingGUI::stop_generation() {
  llama_.cancel();
  stop_gen_btn_->setEnabled(false);
  set_status(QStringLiteral("正在停止…"), "busy");
  sys_log(QStringLiteral("用户请求停止生成"));
}

void TrainingGUI::clear_chat_history() {
  if (chat_view_ != nullptr) chat_view_->clear_messages();
  chat_context_.clear();
}

void TrainingGUI::clear_sys_log() {
  if (sys_log_text_ != nullptr) sys_log_text_->clear();
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
  set_status(QStringLiteral("训练中..."), "busy");

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

  // 上一次训练线程已结束但对象仍 joinable：必须先 join，否则对 joinable 的
  // std::thread 赋值会触发 std::terminate（表现为训练步数很小时的"闪退"）
  if (training_thread_.joinable()) training_thread_.join();
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
  set_status(QStringLiteral("正在停止..."), "busy");
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
  set_status(QStringLiteral("训练完成"), "ok");
  train_log(QStringLiteral("训练完成！"));
  train_log(QStringLiteral("=================================================="));
  refresh_model_list();
}

// 加载 dataset.zip 并接管为训练数据（分词器 / 词表 / 序列长度 / 数据路径）
bool TrainingGUI::load_dataset_zip(const QString& zip_path, bool quiet) {
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

    if (!quiet) {
      nav_->set_current(0);
      QMessageBox::information(
          this, QStringLiteral("成功"),
          QStringLiteral("数据集加载成功！\n\n词表大小: %1\n最大序列长度: %2\n数据路径: %3")
              .arg(vocab_size_)
              .arg(max_seq_len_)
              .arg(qpath(paths.token_ids)));
    }
    return true;
  } catch (const std::exception& e) {
    const QString msg = QString::fromUtf8(e.what());
    sys_log(QStringLiteral("加载失败: %1").arg(msg));
    if (!quiet) {
      QMessageBox::critical(this, QStringLiteral("加载失败"),
                            QStringLiteral("打开数据集出错:\n%1").arg(msg));
    }
    return false;
  }
}

void TrainingGUI::open_dataset() {
  const QString zip_path = QFileDialog::getOpenFileName(
      this, QStringLiteral("选择数据集压缩包"), QString(),
      QStringLiteral("Zip Files (*.zip);;All Files (*)"));
  if (zip_path.isEmpty()) return;
  load_dataset_zip(zip_path, /*quiet=*/false);
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
    const auto t0 = std::chrono::steady_clock::now();
    {
      torch::NoGradGuard no_grad;
      std::vector<int64_t> in_ids;
      in_ids.reserve(ids.size());
      // 防御：越界 token 会触发 CUDA illegal memory access，这里夹到词表范围内
      const int64_t vocab = std::max<int64_t>(1, vocab_size_);
      for (int id : ids) {
        in_ids.push_back((id < 0 || id >= vocab) ? 0 : (int64_t)id);
      }
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
    const double secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
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
            set_status(QStringLiteral("就绪"), "ok");
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

    const double tps = new_ids.empty() ? 0.0 : (double)new_ids.size() / std::max(0.001, secs);
    const QString stats = QStringLiteral("⚡ %1 tok · %2 s · %3 tok/s")
                              .arg(new_ids.size())
                              .arg(secs, 0, 'f', 2)
                              .arg(tps, 0, 'f', 1);

    QMetaObject::invokeMethod(
        this,
        [this, new_q, prompt, use_template, multi_turn, stats,
         tokens = (int)new_ids.size(), secs,
         prompt_tokens = (long long)ids.size()] {
          sys_log(QStringLiteral("生成文本: %1").arg(new_q));
          update_chat_stats(tokens, secs, prompt_tokens);
          if (chat_view_ != nullptr) {
            // 打字机动画：整体塞入，由 ChatView 逐字吐字
            chat_view_->begin_stream(false);
            chat_view_->stream_delta(new_q);
            chat_view_->end_stream(stats);
          }
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
          set_status(QStringLiteral("就绪"), "ok");
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
          set_status(QStringLiteral("就绪"), "ok");
          generate_btn_->setEnabled(true);
          generating_.store(false);
        },
        Qt::QueuedConnection);
  }
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

  if (tokenizer_thread_.joinable()) tokenizer_thread_.join();
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
  tokenizer_log_text_->append(QStringLiteral("✅ 分词训练完成"));
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

  // ---- 打包结果 + 自动接到预训练 ----
  const fs::path zip = tokenizer_result_.dataset_zip.empty()
                           ? (tokenizer_output_dir_ / "dataset.zip")
                           : tokenizer_result_.dataset_zip;
  if (fs::exists(zip)) {
    std::error_code ec;
    const auto size = fs::file_size(zip, ec);
    tokenizer_log_text_->append(
        QStringLiteral("📦 dataset.zip: %1（%2 MB，含 spm.model / token_ids.txt / config.json）")
            .arg(qpath(zip))
            .arg(ec ? 0.0 : (double)size / 1024.0 / 1024.0, 0, 'f', 1));

    // 自动载入为训练数据（词表 / 序列长度 / 数据路径全部接管）
    if (load_dataset_zip(qpath(zip), /*quiet=*/true)) {
      tokenizer_log_text_->append(
          QStringLiteral("✅ 已自动载入「训练」页：词表 %1、序列 %2 —— 直接切到「训练」页"
                         "点「开始训练」即可预训练")
              .arg(vocab_size_)
              .arg(max_seq_len_));
      nav_->set_current(0); // 跳到训练页
      set_status(QStringLiteral("分词完成，已可开始预训练"), "ok");
      sys_log(QStringLiteral("分词产物已接管为训练数据: %1").arg(qpath(zip)));
    } else {
      tokenizer_log_text_->append(
          QStringLiteral("⚠️ 自动载入训练数据失败，可在「对话」页手动打开 dataset.zip"));
    }
  } else {
    tokenizer_log_text_->append(
        QStringLiteral("⚠️ 未找到 dataset.zip（查看上方日志中的打包错误）"));
  }
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
  if (embed_dim_edit_ != nullptr) {
    // 保存高级页设置（引擎路径 / 采样参数等）
    save_settings();
  }
  llama_.cancel();
  llama_.stop();
  event->accept();
}

} // namespace gpt

