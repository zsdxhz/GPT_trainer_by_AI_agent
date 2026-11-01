#pragma once
// ui_kit.h - Windows 11「设置」风格的基础控件与便捷构造：
//   左侧导航（NavRail / NavItem）、卡片（Card）、开关（ToggleSwitch）、
//   字形图标按钮（GlyphButton）、对话气泡视图（ChatView）。
#include <QAbstractButton>
#include <QFrame>
#include <QList>
#include <QPushButton>
#include <QString>
#include <QWidget>

class QHBoxLayout;
class QLabel;
class QLineEdit;
class QPropertyAnimation;
class QScrollArea;
class QTextEdit;
class QTimer;
class QVBoxLayout;

namespace fl {

// ---------------------------------------------------------------- 左侧导航
class NavItem : public QAbstractButton {
  Q_OBJECT
 public:
  NavItem(const QString& glyph, const QString& text, QWidget* parent = nullptr);
  QSize sizeHint() const override;
  QSize minimumSizeHint() const override;

 protected:
  void paintEvent(QPaintEvent* event) override;
  void enterEvent(QEnterEvent* event) override;
  void leaveEvent(QEvent* event) override;
  void focusInEvent(QFocusEvent* event) override;
  void focusOutEvent(QFocusEvent* event) override;

 private:
  QString glyph_;
  QString text_;
  bool hover_ = false;
  bool keyboard_focus_ = false;
};

class NavRail : public QFrame {
  Q_OBJECT
 public:
  explicit NavRail(QWidget* parent = nullptr);
  int add_item(const QString& glyph, const QString& text);
  int current() const { return current_; }
  void set_current(int index);

 Q_SIGNALS:
  void currentChanged(int index);

 protected:
  void resizeEvent(QResizeEvent* event) override;
  void paintEvent(QPaintEvent* event) override;

 private:
  void update_indicator(bool animate);
  QVBoxLayout* list_ = nullptr;
  QList<NavItem*> items_;
  QWidget* indicator_ = nullptr;
  QPropertyAnimation* bar_anim_ = nullptr;
  int current_ = -1;
};

// ---------------------------------------------------------------- 开关
class ToggleSwitch : public QAbstractButton {
  Q_OBJECT
  Q_PROPERTY(qreal position READ position WRITE set_position)
 public:
  explicit ToggleSwitch(QWidget* parent = nullptr);
  QSize sizeHint() const override;
  QSize minimumSizeHint() const override;
  qreal position() const { return position_; }
  void set_position(qreal p);

 protected:
  void paintEvent(QPaintEvent* event) override;
  void checkStateSet() override;

 private:
  void animate_to(qreal target);
  qreal position_ = 0.0;
  QPropertyAnimation* anim_ = nullptr;
};

// ---------------------------------------------------------------- 卡片
class Card : public QFrame {
  Q_OBJECT
 public:
  explicit Card(QWidget* parent = nullptr);

  void set_title(const QString& title, const QString& subtitle = QString());
  QHBoxLayout* header() const { return header_; }
  QVBoxLayout* body() const { return body_; }

  // 行：左侧标签（可带说明），控件加在右侧（添加控件时自动右对齐）
  QHBoxLayout* add_row(const QString& label = QString(),
                       const QString& hint = QString(), int min_height = 42);
  // 行：标签在左，控件占满剩余宽度（用于路径 / 数值输入）
  QHBoxLayout* add_row_wide(const QString& label = QString(),
                            const QString& hint = QString(), int min_height = 42);
  // 行：整宽控件（日志 / 对话视图等）
  void add_widget_row(QWidget* widget, int height = 0, bool expand = false);
  void add_divider();
  void add_note(const QString& text);
  void add_gap(int px);

 private:
  QHBoxLayout* make_row(const QString& label, const QString& hint,
                        int min_height, bool align_right);
  QVBoxLayout* outer_ = nullptr;
  QVBoxLayout* header_text_ = nullptr;
  QHBoxLayout* header_ = nullptr;
  QVBoxLayout* body_ = nullptr;
  QLabel* title_label_ = nullptr;
  QLabel* subtitle_label_ = nullptr;
  int rows_ = 0;
};

// ---------------------------------------------------------------- 字形按钮
class GlyphButton : public QPushButton {
  Q_OBJECT
 public:
  enum class Role { Normal, Accent, Danger };
  explicit GlyphButton(const QString& text, QWidget* parent = nullptr);
  void set_role(Role role);
  void set_glyph(const QString& glyph);

 protected:
  void changeEvent(QEvent* event) override;

 private:
  void refresh_icon();
  QString glyph_;
  Role role_ = Role::Normal;
};

// ---------------------------------------------------------------- 环形用量指示
// 圆形进度环（用于对话上下文用量），中心显示百分比
class RingGauge : public QWidget {
  Q_OBJECT
 public:
  explicit RingGauge(QWidget* parent = nullptr);
  void set_ratio(double ratio);          // 0.0 ~ 1.0
  void set_caption(const QString& text); // 悬停提示
  double ratio() const { return ratio_; }
  QSize sizeHint() const override;

 protected:
  void paintEvent(QPaintEvent* event) override;
  void enterEvent(QEnterEvent* event) override;
  void leaveEvent(QEvent* event) override;

 private:
  double ratio_ = 0.0;
  QString caption_;
  bool hover_ = false;
};

// ---------------------------------------------------------------- 对话视图
// DeepSeek / ChatGPT 风格：居中内容列、头像、AI 无气泡纯文本、用户右对齐气泡、
// 逐字打字机动画、回复下方的小字性能指标（tok / 秒 / tok/s）
class ChatView : public QWidget {
  Q_OBJECT
 public:
  explicit ChatView(QWidget* parent = nullptr);
  void add_message(bool mine, const QString& text);
  void add_note(const QString& text);
  void clear_messages();
  bool is_empty() const { return count_ == 0; }

  // 流式输出：begin_stream 创建气泡，stream_delta 追加文本（打字机动画），
  // end_stream 收尾；stats 会显示在回复下方的小字里
  void begin_stream(bool mine, const QString& speaker = QString());
  void stream_delta(const QString& text);
  void end_stream(const QString& stats = QString());
  bool streaming() const { return stream_body_ != nullptr; }

 protected:
  void resizeEvent(QResizeEvent* event) override;
  void showEvent(QShowEvent* event) override;

 private:
  QWidget* add_row(bool mine, QLabel** body_out, QLabel** stats_out);
  void tick_reveal();
  void scroll_to_bottom();
  void update_bubble_widths();
  QScrollArea* scroll_ = nullptr;
  QWidget* host_ = nullptr;
  QVBoxLayout* list_ = nullptr;
  QLabel* empty_hint_ = nullptr;
  QList<QWidget*> messages_;
  QList<QFrame*> bubbles_;
  QList<QLabel*> bodies_;
  QLabel* stream_body_ = nullptr;
  QLabel* stream_stats_ = nullptr;
  QTimer* reveal_timer_ = nullptr;
  QString pending_;
  QString shown_;
  QString pending_stats_;
  bool stream_done_ = false;
  int count_ = 0;
};

// ---------------------------------------------------------------- 便捷构造
QLabel* page_title(const QString& text);
QLabel* page_subtitle(const QString& text);
QLabel* row_label(const QString& text, QWidget* parent = nullptr);
QFrame* divider_line(QWidget* parent = nullptr);
QLineEdit* text_field(const QString& value = QString(),
                      const QString& placeholder = QString(), int width = 0);
QTextEdit* log_view(int font_px = 12, QWidget* parent = nullptr);
QPushButton* plain_button(const QString& text, const QString& glyph = QString(),
                          QWidget* parent = nullptr);
QPushButton* accent_button(const QString& text, const QString& glyph = QString(),
                           QWidget* parent = nullptr);
QPushButton* danger_button(const QString& text, const QString& glyph = QString(),
                           QWidget* parent = nullptr);

} // namespace fl
