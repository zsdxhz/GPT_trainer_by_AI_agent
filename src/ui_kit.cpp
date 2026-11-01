#include "ui_kit.h"

#include <QApplication>
#include <QCursor>
#include <QEvent>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPropertyAnimation>
#include <QResizeEvent>
#include <QScrollArea>
#include <QScrollBar>
#include <QStyle>
#include <QTextEdit>
#include <QTextDocument>
#include <QTextOption>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>

#include "ui_theme.h"

namespace fl {

namespace {
constexpr int kNavItemHeight = 44;
constexpr int kSwitchWidth = 40;
constexpr int kSwitchHeight = 20;
constexpr int kNavBarWidth = 3;
constexpr int kNavBarHeight = 16;
constexpr int kNavBarLeft = 5;

QColor with_alpha(QColor c, int alpha) {
  c.setAlpha(alpha);
  return c;
}

// 导航选中指示条（独立子控件，便于在 NavRail 内做位移动画）
class NavBar : public QWidget {
 public:
  explicit NavBar(QWidget* parent = nullptr) : QWidget(parent) {
    setAttribute(Qt::WA_TransparentForMouseEvents, true);
  }

 protected:
  void paintEvent(QPaintEvent*) override {
    const Tokens& t = Theme::get()->tokens();
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setPen(Qt::NoPen);
    p.setBrush(t.accent);
    p.drawRoundedRect(QRectF(rect()), kNavBarWidth / 2.0, kNavBarWidth / 2.0);
  }
};
} // namespace

// ===========================================================================
// NavItem
// ===========================================================================
NavItem::NavItem(const QString& glyph, const QString& text, QWidget* parent)
    : QAbstractButton(parent), glyph_(glyph), text_(text) {
  setCheckable(true);
  setAutoExclusive(false);
  setCursor(Qt::PointingHandCursor);
  setFocusPolicy(Qt::StrongFocus);
  setCursor(Qt::PointingHandCursor);
  setAttribute(Qt::WA_Hover, true);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

QSize NavItem::sizeHint() const { return QSize(200, kNavItemHeight); }

QSize NavItem::minimumSizeHint() const { return QSize(140, kNavItemHeight); }

void NavItem::enterEvent(QEnterEvent* event) {
  hover_ = true;
  QAbstractButton::enterEvent(event);
  update();
}

void NavItem::leaveEvent(QEvent* event) {
  hover_ = false;
  QAbstractButton::leaveEvent(event);
  update();
}

void NavItem::focusInEvent(QFocusEvent* event) {
  keyboard_focus_ = event->reason() == Qt::TabFocusReason ||
                    event->reason() == Qt::BacktabFocusReason ||
                    event->reason() == Qt::ShortcutFocusReason;
  QAbstractButton::focusInEvent(event);
  update();
}

void NavItem::focusOutEvent(QFocusEvent* event) {
  keyboard_focus_ = false;
  QAbstractButton::focusOutEvent(event);
  update();
}

void NavItem::paintEvent(QPaintEvent*) {
  const Tokens& t = Theme::get()->tokens();
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  p.setRenderHint(QPainter::TextAntialiasing, true);

  // 背景：选中 > 按下 > 悬停
  QColor bg = Qt::transparent;
  if (isChecked()) {
    bg = t.navSelected;
  } else if (isDown()) {
    bg = t.navPressed;
  } else if (hover_ || underMouse()) {
    bg = t.navHover;
  }
  if (bg.alpha() > 0) {
    p.setPen(Qt::NoPen);
    p.setBrush(bg);
    p.drawRoundedRect(QRectF(rect()).adjusted(2, 2, -2, -2), 4, 4);
  }

  // 选中态：左侧强调色指示条由 NavRail 的浮层绘制（带切换动画），这里不再画

  const QColor fg = isEnabled() ? t.text : t.textTertiary;

  // 图标
  QFont icon_font(glyph_family());
  icon_font.setPixelSize(16);
  p.setFont(icon_font);
  p.setPen(isChecked() ? t.accent : fg);
  p.drawText(QRect(16, (height() - 16) / 2, 16, 16), Qt::AlignCenter, glyph_);

  // 文本
  QFont f = ui_font(14, isChecked() ? int(QFont::DemiBold) : int(QFont::Normal));
  p.setFont(f);
  p.setPen(fg);
  const QFontMetrics fm(f);
  const int text_left = 46;
  const int text_w = std::max(0, width() - text_left - 12);
  const QString elided = fm.elidedText(text_, Qt::ElideRight, text_w);
  p.drawText(QRect(text_left, 0, text_w, height()),
             Qt::AlignVCenter | Qt::AlignLeft, elided);

  // 键盘焦点框（鼠标点击不显示）
  if (keyboard_focus_ && hasFocus()) {
    QPen pen(with_alpha(t.text, 140), 2.0);
    p.setPen(pen);
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(QRectF(rect()).adjusted(3, 3, -3, -3), 3, 3);
  }
}

// ===========================================================================
// NavRail
// ===========================================================================
NavRail::NavRail(QWidget* parent) : QFrame(parent) {
  setObjectName(QStringLiteral("navRail"));
  setFrameShape(QFrame::NoFrame);
  setAutoFillBackground(false);
  setFixedWidth(232);

  auto* outer = new QVBoxLayout(this);
  outer->setContentsMargins(8, 12, 8, 12);
  outer->setSpacing(6);

  auto* head = new QWidget(this);
  auto* head_layout = new QVBoxLayout(head);
  head_layout->setContentsMargins(10, 4, 10, 10);
  head_layout->setSpacing(2);
  auto* title = new QLabel(QStringLiteral("GPT 训练器"), head);
  title->setProperty("role", "cardTitle");
  auto* subtitle = new QLabel(QStringLiteral("本地训练 · 对话 · 分词"), head);
  subtitle->setProperty("role", "hint");
  head_layout->addWidget(title);
  head_layout->addWidget(subtitle);
  outer->addWidget(head);

  list_ = new QVBoxLayout();
  list_->setContentsMargins(0, 0, 0, 0);
  list_->setSpacing(2);
  outer->addLayout(list_);
  outer->addStretch(1);

  connect(Theme::get(), &Theme::changed, this, [this] { update(); });
}

int NavRail::add_item(const QString& glyph, const QString& text) {
  auto* item = new NavItem(glyph, text, this);
  const int index = items_.size();
  items_.append(item);
  list_->addWidget(item);
  connect(item, &QAbstractButton::clicked, this,
          [this, index] { set_current(index); });
  connect(Theme::get(), &Theme::changed, item, [item] { item->update(); });
  if (indicator_ == nullptr) {
    indicator_ = new NavBar(this);
    indicator_->setFixedSize(kNavBarWidth, kNavBarHeight);
    bar_anim_ = new QPropertyAnimation(indicator_, "pos", this);
    bar_anim_->setDuration(180);
    bar_anim_->setEasingCurve(QEasingCurve::OutCubic);
    indicator_->raise();
    connect(Theme::get(), &Theme::changed, indicator_,
            [this] { if (indicator_ != nullptr) indicator_->update(); });
  }
  if (current_ < 0) set_current(0);
  return index;
}

void NavRail::update_indicator(bool animate) {
  if (indicator_ == nullptr || current_ < 0 || current_ >= items_.size()) return;
  NavItem* item = items_[current_];
  if (item == nullptr) return;
  const QPoint target(kNavBarLeft,
                      item->y() + (item->height() - kNavBarHeight) / 2);
  indicator_->raise();
  indicator_->show();
  if (!animate || bar_anim_ == nullptr) {
    if (bar_anim_ != nullptr) bar_anim_->stop();
    indicator_->move(target);
    indicator_->update();
    return;
  }
  bar_anim_->stop();
  bar_anim_->setStartValue(indicator_->pos());
  bar_anim_->setEndValue(target);
  bar_anim_->start();
}

void NavRail::resizeEvent(QResizeEvent* event) {
  QFrame::resizeEvent(event);
  update_indicator(false);
}

void NavRail::paintEvent(QPaintEvent* event) {
  QFrame::paintEvent(event);
}

void NavRail::set_current(int index) {
  if (index < 0 || index >= items_.size()) return;
  const bool changed = (index != current_);
  current_ = index;
  for (int i = 0; i < items_.size(); ++i) items_[i]->setChecked(i == index);
  update_indicator(changed);
  if (changed) Q_EMIT currentChanged(index);
}

// ===========================================================================
// ToggleSwitch（Win11 开关）
// ===========================================================================
ToggleSwitch::ToggleSwitch(QWidget* parent) : QAbstractButton(parent) {
  setCheckable(true);
  setCursor(Qt::PointingHandCursor);
  setFocusPolicy(Qt::StrongFocus);
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  anim_ = new QPropertyAnimation(this, "position", this);
  anim_->setDuration(130);
  anim_->setEasingCurve(QEasingCurve::OutCubic);
  connect(Theme::get(), &Theme::changed, this, [this] { update(); },
          Qt::UniqueConnection);
}

QSize ToggleSwitch::sizeHint() const { return QSize(kSwitchWidth, kSwitchHeight); }

QSize ToggleSwitch::minimumSizeHint() const {
  return QSize(kSwitchWidth, kSwitchHeight);
}

void ToggleSwitch::set_position(qreal p) {
  position_ = std::clamp<qreal>(p, 0.0, 1.0);
  update();
}

void ToggleSwitch::checkStateSet() {
  QAbstractButton::checkStateSet();
  anim_->stop();
  anim_->setStartValue(position_);
  anim_->setEndValue(isChecked() ? 1.0 : 0.0);
  anim_->start();
}

void ToggleSwitch::paintEvent(QPaintEvent*) {
  const Tokens& t = Theme::get()->tokens();
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);

  const qreal w = kSwitchWidth;
  const qreal h = kSwitchHeight;
  const qreal top = (height() - h) / 2.0;

  // 轨道
  QColor track;
  QColor knob;
  if (!isEnabled()) {
    track = with_alpha(t.text, 34);
    knob = with_alpha(t.text, 90);
  } else if (isChecked()) {
    track = t.accent;
    knob = t.onAccent;
  } else if (underMouse()) {
    track = t.scheme == Scheme::Dark ? QColor(0x7A, 0x7A, 0x7A)
                                     : QColor(0x74, 0x74, 0x74);
    knob = QColor(0xFF, 0xFF, 0xFF);
  } else {
    track = t.scheme == Scheme::Dark ? QColor(0x6A, 0x6A, 0x6A)
                                     : QColor(0x8A, 0x8A, 0x8A);
    knob = QColor(0xFF, 0xFF, 0xFF);
  }

  p.setPen(Qt::NoPen);
  p.setBrush(track);
  p.drawRoundedRect(QRectF(0, top, w, h), h / 2.0, h / 2.0);

  // 滑块
  const qreal knob_d = 14.0;
  const qreal x = 3.0 + position_ * (w - knob_d - 6.0);
  p.setBrush(knob);
  p.drawEllipse(QRectF(x, top + (h - knob_d) / 2.0, knob_d, knob_d));

  // 焦点框
  if (hasFocus()) {
    p.setPen(QPen(t.text, 2.0));
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(QRectF(rect()).adjusted(1, 1, -1, -1), h / 2.0, h / 2.0);
  }
}

// ===========================================================================
// Card
// ===========================================================================
Card::Card(QWidget* parent) : QFrame(parent) {
  setProperty("role", "card");
  setAttribute(Qt::WA_StyledBackground, true);
  setFrameShape(QFrame::NoFrame);

  outer_ = new QVBoxLayout(this);
  outer_->setContentsMargins(20, 12, 20, 12);
  outer_->setSpacing(12);

  header_ = new QHBoxLayout();
  header_->setContentsMargins(0, 0, 0, 0);
  header_->setSpacing(12);
  header_text_ = new QVBoxLayout();
  header_text_->setContentsMargins(0, 0, 0, 0);
  header_text_->setSpacing(2);

  title_label_ = new QLabel(this);
  title_label_->setProperty("role", "cardTitle");
  subtitle_label_ = new QLabel(this);
  subtitle_label_->setProperty("role", "cardSubtitle");
  subtitle_label_->setWordWrap(false);
  title_label_->hide();
  subtitle_label_->hide();
  title_label_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
  subtitle_label_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
  header_text_->addWidget(title_label_);
  header_text_->addWidget(subtitle_label_);
  header_->addLayout(header_text_, 1);
  outer_->addLayout(header_);

  body_ = new QVBoxLayout();
  body_->setContentsMargins(0, 0, 0, 0);
  body_->setSpacing(0);
  outer_->addLayout(body_, 1);
}

void Card::set_title(const QString& title, const QString& subtitle) {
  title_label_->setText(title);
  title_label_->setVisible(!title.isEmpty());
  subtitle_label_->setText(subtitle);
  subtitle_label_->setVisible(!subtitle.isEmpty());
}

QHBoxLayout* Card::make_row(const QString& label, const QString& hint,
                            int min_height, bool align_right) {
  if (rows_ > 0) add_divider();

  auto* holder = new QWidget(this);
  holder->setMinimumHeight(min_height);
  holder->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
  auto* row = new QHBoxLayout(holder);
  row->setContentsMargins(0, 4, 0, 4);
  row->setSpacing(12);

  if (!label.isEmpty()) {
    if (hint.isEmpty()) {
      auto* l = new QLabel(label, holder);
      l->setProperty("role", "rowLabel");
      row->addWidget(l);
    } else {
      auto* col = new QVBoxLayout();
      col->setContentsMargins(0, 0, 0, 0);
      col->setSpacing(2);
      auto* l = new QLabel(label, holder);
      l->setProperty("role", "rowLabel");
      auto* h = new QLabel(hint, holder);
      h->setProperty("role", "hint");
      h->setWordWrap(false);
      col->addWidget(l);
      col->addWidget(h);
      row->addLayout(col);
    }
  }
  if (align_right) row->addStretch(1);
  body_->addWidget(holder);
  ++rows_;
  return row;
}

QHBoxLayout* Card::add_row(const QString& label, const QString& hint,
                           int min_height) {
  return make_row(label, hint, min_height, true);
}

QHBoxLayout* Card::add_row_wide(const QString& label, const QString& hint,
                                int min_height) {
  return make_row(label, hint, min_height, false);
}

void Card::add_widget_row(QWidget* widget, int height, bool expand) {
  if (rows_ > 0) add_divider();
  widget->setParent(this);
  if (expand) {
    if (height > 0) widget->setMinimumHeight(height);
    widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    body_->addWidget(widget, 1);
  } else {
    // 固定行高：QTextEdit / QLabel 的 sizeHint 不可靠，直接锁高度
    if (height > 0) widget->setFixedHeight(height);
    widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    body_->addWidget(widget);
  }
  ++rows_;
}

void Card::add_divider() { body_->addWidget(divider_line(this)); }

void Card::add_note(const QString& text) {
  auto* note = new QLabel(text, this);
  note->setProperty("role", "hint");
  note->setWordWrap(true);
  note->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
  body_->addWidget(note);
}

void Card::add_gap(int px) { body_->addSpacing(px); }

// ===========================================================================
// GlyphButton
// ===========================================================================
GlyphButton::GlyphButton(const QString& text, QWidget* parent)
    : QPushButton(text, parent) {
  setProperty("role", "plain");
  setCursor(Qt::PointingHandCursor);
  setMinimumHeight(32);
  connect(Theme::get(), &Theme::changed, this, [this] { refresh_icon(); },
          Qt::UniqueConnection);
}

void GlyphButton::set_role(Role role) {
  role_ = role;
  const char* name = role == Role::Accent   ? "accent"
                     : role == Role::Danger ? "danger"
                                            : "plain";
  setProperty("role", name);
  style()->unpolish(this);
  style()->polish(this);
  update();
  refresh_icon();
}

void GlyphButton::set_glyph(const QString& glyph) {
  glyph_ = glyph;
  refresh_icon();
}

void GlyphButton::changeEvent(QEvent* event) {
  QPushButton::changeEvent(event);
  if (event->type() == QEvent::EnabledChange) refresh_icon();
}

void GlyphButton::refresh_icon() {
  if (glyph_.isEmpty()) {
    setIcon(QIcon());
    return;
  }
  const Tokens& t = Theme::get()->tokens();
  QColor color = t.text;
  if (role_ == Role::Accent) {
    color = t.onAccent;
  } else if (role_ == Role::Danger) {
    color = QColor(Qt::white);
  }
  if (!isEnabled()) color = t.textTertiary;
  setIconSize(QSize(16, 16));
  setIcon(glyph_icon(glyph_, color, 16));
}

// ===========================================================================
// RingGauge（圆形上下文用量环）
// ===========================================================================
RingGauge::RingGauge(QWidget* parent) : QWidget(parent) {
  setFixedSize(30, 30);
  setAttribute(Qt::WA_Hover, true);
  if (Theme::get() != nullptr) {
    connect(Theme::get(), &Theme::changed, this, [this] { update(); },
            Qt::UniqueConnection);
  }
}

QSize RingGauge::sizeHint() const { return QSize(30, 30); }

void RingGauge::set_ratio(double ratio) {
  ratio_ = std::clamp(ratio, 0.0, 1.0);
  update();
}

void RingGauge::set_caption(const QString& text) {
  caption_ = text;
  setToolTip(text);
}

void RingGauge::enterEvent(QEnterEvent* event) {
  hover_ = true;
  QWidget::enterEvent(event);
  update();
}

void RingGauge::leaveEvent(QEvent* event) {
  hover_ = false;
  QWidget::leaveEvent(event);
  update();
}

void RingGauge::paintEvent(QPaintEvent*) {
  const Tokens& t = Theme::get()->tokens();
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  const int size = std::min(width(), height());
  const QRectF box((width() - size) / 2.0 + 2.0, (height() - size) / 2.0 + 2.0,
                   size - 4.0, size - 4.0);
  // 底环
  QPen base(QColor(t.divider), 2.4);
  base.setCapStyle(Qt::RoundCap);
  p.setPen(base);
  p.drawArc(box, 0, 360 * 16);
  // 进度环
  if (ratio_ > 0.0) {
    QColor c = t.accent;
    if (ratio_ > 0.85) c = t.critical;
    else if (ratio_ > 0.6) c = QColor(230, 160, 40);
    QPen arc(c, 2.4);
    arc.setCapStyle(Qt::RoundCap);
    p.setPen(arc);
    p.drawArc(box, 90 * 16, -(int)(360 * 16 * ratio_));
  }
  // 百分比
  QFont f = ui_font(10);
  f.setBold(true);
  p.setFont(f);
  p.setPen(hover_ ? t.text : t.textSecondary);
  QString txt;
  const int pct = (int)std::lround(ratio_ * 100.0);
  txt = ratio_ >= 0.995 ? QStringLiteral("满") : QString::number(pct);
  p.drawText(QRectF(0, 0, width(), height()), Qt::AlignCenter, txt);
}

// ===========================================================================
// ChatView（DeepSeek / ChatGPT 风格）
// ===========================================================================
ChatView::ChatView(QWidget* parent) : QWidget(parent) {
  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(0);

  scroll_ = new QScrollArea(this);
  scroll_->setWidgetResizable(true);
  scroll_->setFrameShape(QFrame::NoFrame);
  scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  scroll_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  scroll_->viewport()->setAutoFillBackground(false);

  host_ = new QWidget();
  host_->setObjectName(QStringLiteral("chatHost"));
  host_->setAutoFillBackground(false);
  list_ = new QVBoxLayout(host_);
  list_->setContentsMargins(10, 10, 10, 10);
  list_->setSpacing(6);

  empty_hint_ = new QLabel(
      QStringLiteral("开始对话\n输入一句话，按 Ctrl+Enter 或点击「发送」"), host_);
  empty_hint_->setProperty("role", "hint");
  empty_hint_->setAlignment(Qt::AlignCenter);
  list_->addStretch(1);          // 让空状态垂直居中
  list_->addWidget(empty_hint_);
  list_->addStretch(1);

  scroll_->setWidget(host_);
  layout->addWidget(scroll_);
}

QWidget* ChatView::add_row(bool mine, QLabel** body_out, QLabel** stats_out) {
  if (count_ == 0 && empty_hint_ != nullptr) {
    empty_hint_->hide();
    list_->setStretch(0, 0); // 空状态占位收起，消息从顶部开始
  }

  auto* wrap = new QWidget(host_);
  auto* row = new QHBoxLayout(wrap);
  row->setContentsMargins(6, 3, 6, 3);
  row->setSpacing(10);

  QLabel* body = nullptr;
  QLabel* stats = nullptr;

  if (mine) {
    row->addStretch(1);
    auto* bubble = new QFrame(wrap);
    bubble->setProperty("role", "msgUserBubble");
    bubble->setAttribute(Qt::WA_StyledBackground, true);
    bubble->setFrameShape(QFrame::NoFrame);
    auto* bl = new QVBoxLayout(bubble);
    bl->setContentsMargins(14, 9, 14, 10);
    bl->setSpacing(0);
    body = new QLabel(bubble);
    body->setProperty("role", "msgText");
    body->setWordWrap(true);
    body->setTextFormat(Qt::PlainText);
    body->setTextInteractionFlags(Qt::TextSelectableByMouse);
    body->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    bl->addWidget(body);
    row->addWidget(bubble, 0, Qt::AlignTop);
    bubbles_.append(bubble);

    auto* avatar = new QLabel(QStringLiteral("你"), wrap);
    avatar->setProperty("role", "msgAvatarUser");
    avatar->setFixedSize(26, 26);
    avatar->setAlignment(Qt::AlignCenter);
    row->addWidget(avatar, 0, Qt::AlignTop);
  } else {
    auto* avatar = new QLabel(QStringLiteral("AI"), wrap);
    avatar->setProperty("role", "msgAvatarAi");
    avatar->setFixedSize(26, 26);
    avatar->setAlignment(Qt::AlignCenter);
    row->addWidget(avatar, 0, Qt::AlignTop);

    auto* col = new QWidget(wrap);
    auto* cl = new QVBoxLayout(col);
    cl->setContentsMargins(0, 1, 0, 0);
    cl->setSpacing(5);
    body = new QLabel(col);
    body->setProperty("role", "msgText");
    body->setWordWrap(true);
    body->setTextFormat(Qt::PlainText);
    body->setTextInteractionFlags(Qt::TextSelectableByMouse);
    body->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    stats = new QLabel(col);
    stats->setProperty("role", "msgStats");
    stats->setTextInteractionFlags(Qt::TextSelectableByMouse);
    stats->hide();
    cl->addWidget(body);
    cl->addWidget(stats);
    row->addWidget(col, 1);
  }

  list_->insertWidget(list_->count() - 1, wrap);
  messages_.append(wrap);
  bodies_.append(body);
  ++count_;
  update_bubble_widths();
  scroll_to_bottom();

  if (body_out != nullptr) *body_out = body;
  if (stats_out != nullptr) *stats_out = stats;
  return wrap;
}

void ChatView::add_message(bool mine, const QString& text) {
  QLabel* body = nullptr;
  add_row(mine, &body, nullptr);
  if (body != nullptr) body->setText(text);
  update_bubble_widths();
  scroll_to_bottom();
}

void ChatView::add_note(const QString& text) {
  auto* note = new QLabel(text, host_);
  note->setProperty("role", "hint");
  note->setAlignment(Qt::AlignCenter);
  note->setWordWrap(true);
  list_->insertWidget(list_->count() - 1, note);
  messages_.append(note);
  scroll_to_bottom();
}

void ChatView::clear_messages() {
  if (reveal_timer_ != nullptr) reveal_timer_->stop();
  for (QWidget* w : messages_) {
    list_->removeWidget(w);
    w->deleteLater();
  }
  messages_.clear();
  bubbles_.clear();
  bodies_.clear();
  stream_body_ = nullptr;
  stream_stats_ = nullptr;
  pending_.clear();
  shown_.clear();
  pending_stats_.clear();
  stream_done_ = false;
  count_ = 0;
  empty_hint_->show();
}

void ChatView::begin_stream(bool mine, const QString& speaker) {
  (void)speaker;
  QLabel* body = nullptr;
  QLabel* stats = nullptr;
  add_row(mine, &body, &stats);
  stream_body_ = body;
  stream_stats_ = stats;
  pending_.clear();
  shown_.clear();
  pending_stats_.clear();
  stream_done_ = false;
  if (body != nullptr) body->setText(QStringLiteral("▍"));
  if (reveal_timer_ == nullptr) {
    reveal_timer_ = new QTimer(this);
    reveal_timer_->setInterval(12);
    connect(reveal_timer_, &QTimer::timeout, this, [this] { tick_reveal(); });
  }
}

void ChatView::stream_delta(const QString& text) {
  if (stream_body_ == nullptr || text.isEmpty()) return;
  pending_ += text;
  if (reveal_timer_ != nullptr && !reveal_timer_->isActive()) {
    reveal_timer_->start();
  }
}

// 打字机：每 12ms 吐出 1~N 个字（积压越多吐得越快，避免长回复等太久）
void ChatView::tick_reveal() {
  if (stream_body_ == nullptr) {
    if (reveal_timer_ != nullptr) reveal_timer_->stop();
    return;
  }
  if (!pending_.isEmpty()) {
    int n = 1;
    const int backlog = pending_.size();
    if (backlog > 300) {
      n = std::clamp(backlog / 14, 6, 60);
    } else if (backlog > 80) {
      n = 3;
    } else if (backlog > 24) {
      n = 2;
    }
    n = std::min(n, (int)pending_.size());
    shown_ += pending_.left(n);
    pending_.remove(0, n);
    stream_body_->setText(shown_ + QStringLiteral("▍"));
    update_bubble_widths();
    scroll_to_bottom();
    return;
  }
  if (stream_done_) {
    if (reveal_timer_ != nullptr) reveal_timer_->stop();
    stream_body_->setText(shown_.isEmpty() ? QStringLiteral("（没有生成内容）")
                                           : shown_);
    if (stream_stats_ != nullptr && !pending_stats_.isEmpty()) {
      stream_stats_->setText(pending_stats_);
      stream_stats_->show();
    }
    stream_body_ = nullptr;
    stream_stats_ = nullptr;
    update_bubble_widths();
    scroll_to_bottom();
  }
}

void ChatView::end_stream(const QString& stats) {
  pending_stats_ = stats;
  if (stream_body_ == nullptr) return;
  stream_done_ = true;
  if (pending_.isEmpty()) {
    tick_reveal();
  } else if (reveal_timer_ != nullptr && !reveal_timer_->isActive()) {
    reveal_timer_->start();
  }
}

void ChatView::resizeEvent(QResizeEvent* event) {
  QWidget::resizeEvent(event);
  update_bubble_widths();
  // 布局完成后视口宽度才是最终值，再算一次
  QTimer::singleShot(0, this, [this] { update_bubble_widths(); });
}

void ChatView::showEvent(QShowEvent* event) {
  QWidget::showEvent(event);
  update_bubble_widths();
}

void ChatView::update_bubble_widths() {
  // 内容列最大宽度 860px（DeepSeek/ChatGPT 的居中长文阅读宽度）
  const int available = width() > 0 ? width() : 900;
  const int max_width = std::clamp(available - 140, 320, 860);
  const QFontMetrics fm(ui_font(14));
  for (QLabel* body : bodies_) {
    if (body == nullptr) continue;
    int natural = 0;
    const QStringList lines = body->text().split(QLatin1Char('\n'));
    for (const QString& line : lines) {
      natural = std::max(natural, fm.horizontalAdvance(line));
    }
    const int w = std::clamp(natural + 8, 140, max_width);
    body->setMaximumWidth(w);
    // wordWrap 的 QLabel 高度不会按宽度自动重算，必须显式给最小高度，
    // 否则多行回复会被压成一行（"只显示第一行"）
    const int h = body->heightForWidth(w);
    if (h > 0) body->setMinimumHeight(h);
  }
  for (QFrame* bubble : bubbles_) {
    if (bubble == nullptr) continue;
    bubble->setMaximumWidth(std::min(max_width + 30, std::max(140, available - 80)));
    bubble->adjustSize();
  }
  if (list_ != nullptr) list_->activate();
  if (host_ != nullptr) host_->adjustSize();
}

void ChatView::scroll_to_bottom() {
  QTimer::singleShot(0, this, [this] {
    QScrollBar* bar = scroll_->verticalScrollBar();
    bar->setValue(bar->maximum());
  });
}

// ===========================================================================
// 便捷构造
// ===========================================================================
QLabel* page_title(const QString& text) {
  auto* l = new QLabel(text);
  l->setProperty("role", "pageTitle");
  return l;
}

QLabel* page_subtitle(const QString& text) {
  auto* l = new QLabel(text);
  l->setProperty("role", "pageSubtitle");
  l->setWordWrap(true);
  return l;
}

QLabel* row_label(const QString& text, QWidget* parent) {
  auto* l = new QLabel(text, parent);
  l->setProperty("role", "rowLabel");
  return l;
}

QFrame* divider_line(QWidget* parent) {
  auto* f = new QFrame(parent);
  f->setProperty("role", "divider");
  f->setAttribute(Qt::WA_StyledBackground, true);
  f->setFrameShape(QFrame::NoFrame);
  f->setFixedHeight(1);
  return f;
}

QLineEdit* text_field(const QString& value, const QString& placeholder,
                      int width) {
  auto* e = new QLineEdit(value);
  e->setPlaceholderText(placeholder);
  e->setMinimumHeight(32);
  e->setFont(ui_font(14));
  if (width > 0) {
    e->setFixedWidth(width);
  } else {
    e->setMinimumWidth(120);
    e->setMaximumWidth(QWIDGETSIZE_MAX);
  }
  return e;
}

QTextEdit* log_view(int font_px, QWidget* parent) {
  auto* view = new QTextEdit(parent);
  view->setReadOnly(true);
  view->setFont(mono_font(font_px));
  view->setProperty("role", "log");
  view->setFrameShape(QFrame::NoFrame);
  view->setWordWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
  view->document()->setDocumentMargin(6);
  return view;
}

QPushButton* plain_button(const QString& text, const QString& glyph,
                          QWidget* parent) {
  auto* b = new GlyphButton(text, parent);
  b->set_role(GlyphButton::Role::Normal);
  if (!glyph.isEmpty()) b->set_glyph(glyph);
  return b;
}

QPushButton* accent_button(const QString& text, const QString& glyph,
                           QWidget* parent) {
  auto* b = new GlyphButton(text, parent);
  b->set_role(GlyphButton::Role::Accent);
  if (!glyph.isEmpty()) b->set_glyph(glyph);
  return b;
}

QPushButton* danger_button(const QString& text, const QString& glyph,
                           QWidget* parent) {
  auto* b = new GlyphButton(text, parent);
  b->set_role(GlyphButton::Role::Danger);
  if (!glyph.isEmpty()) b->set_glyph(glyph);
  return b;
}

} // namespace fl
