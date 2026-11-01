#include "ui_theme.h"

#include <QApplication>
#include <QWidget>
#include <QFontDatabase>
#include <QGuiApplication>
#include <QPainter>
#include <QPixmap>
#include <QScreen>
#include <QSettings>
#include <QStyleHints>
#include <QVariant>

#include <algorithm>

#ifdef Q_OS_WIN
#include <windows.h>
#include <dwmapi.h>
#endif

namespace fl {

namespace {

// ---------------------------------------------------------------- 颜色工具
QColor with_alpha(QColor c, int alpha) {
  c.setAlpha(alpha);
  return c;
}

double luminance(const QColor& c) {
  return (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()) / 255.0;
}

// 深色主题下 Fluent 会把强调色提亮（Win11 深色的开关 / 强调按钮是亮蓝）
QColor brighten_for_dark(const QColor& c) {
  if (c.lightnessF() >= 0.60) return c;
  const QColor hsl = c.toHsl();
  const qreal hue = hsl.hslHueF();
  const qreal sat = hsl.hslSaturationF();
  if (hue < 0.0) return c.lighter(115); // 无彩色强调色：只做提亮
  QColor out;
  out.setHslF(hue, sat, 0.62);
  return out;
}

// 系统强调色：Qt 会把它放进 QPalette::Highlight。缓存首次读到的系统值，
// 避免后续 setPalette 覆盖后反复提亮。
QColor system_accent() {
  static const QColor cached = [] {
    QColor c = QGuiApplication::palette().color(QPalette::Highlight);
    if (!c.isValid() || c == QColor(Qt::white) || c == QColor(Qt::black)) {
      c = QColor(0x00, 0x78, 0xD4);
    }
    return c;
  }();
  return cached;
}

QString css(const QColor& c) {
  return c.alpha() == 255 ? c.name(QColor::HexRgb) : c.name(QColor::HexArgb);
}

// ---------------------------------------------------------------- 主题探测
Scheme detect_scheme() {
  // 调试 / 覆盖：GPT_THEME=light|dark（默认跟随系统）
  const QString forced =
      qEnvironmentVariable("GPT_THEME").trimmed().toLower();
  if (forced == QLatin1String("dark")) return Scheme::Dark;
  if (forced == QLatin1String("light")) return Scheme::Light;

#ifdef Q_OS_WIN
  if (qApp != nullptr &&
      QGuiApplication::styleHints()->colorScheme() == Qt::ColorScheme::Dark) {
    return Scheme::Dark;
  }
  QSettings personalize(
      QStringLiteral("HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\"
                     "CurrentVersion\\Themes\\Personalize"),
      QSettings::NativeFormat);
  const QVariant light = personalize.value(QStringLiteral("AppsUseLightTheme"));
  if (light.isValid() && light.toInt() == 0) return Scheme::Dark;
  return Scheme::Light;
#else
  if (qApp != nullptr &&
      QGuiApplication::styleHints()->colorScheme() == Qt::ColorScheme::Dark) {
    return Scheme::Dark;
  }
  return Scheme::Light;
#endif
}

} // namespace

// ---------------------------------------------------------------- 设计令牌
Tokens make_tokens(Scheme scheme) {
  Tokens t;
  t.scheme = scheme;
  const QColor accent = system_accent();

  if (scheme == Scheme::Dark) {
    t.window = QColor(0x20, 0x20, 0x20);
    t.navHover = with_alpha(Qt::white, 12);
    t.navSelected = with_alpha(Qt::white, 20);
    t.navPressed = with_alpha(Qt::white, 8);

    t.card = QColor(0x2C, 0x2C, 0x2C);
    t.cardStroke = with_alpha(Qt::white, 20);
    t.divider = with_alpha(Qt::white, 18);

    t.text = QColor(0xFF, 0xFF, 0xFF);
    t.textSecondary = QColor(0xC8, 0xC8, 0xC8);
    t.textTertiary = QColor(0x92, 0x92, 0x92);

    t.accent = brighten_for_dark(accent);
    t.accentHover = t.accent.lighter(112);
    t.accentPressed = t.accent.darker(112);
    t.accentTint = with_alpha(t.accent, 38);
    t.onAccent = luminance(t.accent) >= 0.50 ? QColor(0x0B, 0x0B, 0x0B) : QColor(Qt::white);

    t.control = QColor(0x2D, 0x2D, 0x2D);
    t.controlHover = QColor(0x33, 0x33, 0x33);
    t.controlPressed = QColor(0x28, 0x28, 0x28);
    t.controlStroke = with_alpha(Qt::white, 30);

    t.viewBg = QColor(0x27, 0x27, 0x27);
    t.scrollThumb = with_alpha(Qt::white, 70);
    t.scrollThumbHover = with_alpha(Qt::white, 110);

    t.success = QColor(0x6C, 0xCB, 0x5F);
    t.successTint = with_alpha(t.success, 30);
    t.critical = QColor(0xFF, 0x99, 0xA4);
    t.criticalHover = QColor(0xFF, 0xB4, 0xBC);
    t.criticalTint = with_alpha(t.critical, 30);
  } else {
    t.window = QColor(0xF3, 0xF3, 0xF3);
    t.navHover = with_alpha(Qt::black, 8);
    t.navSelected = with_alpha(Qt::black, 14);
    t.navPressed = with_alpha(Qt::black, 5);

    t.card = QColor(0xFF, 0xFF, 0xFF);
    t.cardStroke = with_alpha(Qt::black, 18);
    t.divider = with_alpha(Qt::black, 16);

    t.text = QColor(0x1A, 0x1A, 0x1A);
    t.textSecondary = QColor(0x5D, 0x5D, 0x5D);
    t.textTertiary = QColor(0x8A, 0x8A, 0x8A);

    t.accent = accent;
    t.accentHover = accent.lighter(108);
    t.accentPressed = accent.darker(112);
    t.accentTint = with_alpha(accent, 30);
    t.onAccent = luminance(accent) >= 0.50 ? QColor(0x0B, 0x0B, 0x0B) : QColor(Qt::white);

    t.control = QColor(0xFB, 0xFB, 0xFB);
    t.controlHover = QColor(0xF3, 0xF3, 0xF3);
    t.controlPressed = QColor(0xEB, 0xEB, 0xEB);
    t.controlStroke = with_alpha(Qt::black, 30);

    t.viewBg = QColor(0xFF, 0xFF, 0xFF);
    t.scrollThumb = with_alpha(Qt::black, 60);
    t.scrollThumbHover = with_alpha(Qt::black, 100);

    t.success = QColor(0x0F, 0x7B, 0x0F);
    t.successTint = with_alpha(t.success, 26);
    t.critical = QColor(0xC4, 0x2B, 0x1C);
    t.criticalHover = QColor(0xB0, 0x23, 0x16);
    t.criticalTint = with_alpha(t.critical, 26);
  }
  return t;
}

// ---------------------------------------------------------------- 样式表
QString style_sheet(const Tokens& t) {
  QString qss = QStringLiteral(R"QSS(
/* ---------- 排版 ---------- */
QLabel { background: transparent; }
QLabel[role="pageTitle"] { font-size: 24px; font-weight: 600; color: @text; }
QLabel[role="pageSubtitle"] { font-size: 12px; color: @textSecondary; }
QLabel[role="cardTitle"] { font-size: 14px; font-weight: 600; color: @text; }
QLabel[role="cardSubtitle"] { font-size: 12px; color: @textSecondary; }
QLabel[role="rowLabel"] { font-size: 14px; color: @text; }
QLabel[role="hint"] { font-size: 12px; color: @textSecondary; }
QLabel[role="metricValue"] { font-size: 22px; font-weight: 600; color: @text; }
QLabel[role="metricLabel"] { font-size: 12px; color: @textSecondary; }
QLabel[role="bubbleSpeaker"] { font-size: 11px; color: @textSecondary; }
QLabel[role="bubbleText"] { font-size: 14px; color: @text; }
QLabel[muted="true"] { color: @textSecondary; }
QLabel[status="ok"] { color: @success; }
QLabel[status="warn"] { color: @critical; }

/* 状态徽标 */
QLabel[role="chip"] {
  background-color: @control; color: @textSecondary;
  border: 1px solid @controlStroke; border-radius: 10px;
  padding: 2px 10px; font-size: 12px;
}
QLabel[role="chip"][status="ok"] { background-color: @successTint; color: @success; border-color: transparent; }
QLabel[role="chip"][status="warn"] { background-color: @criticalTint; color: @critical; border-color: transparent; }
QLabel[role="chip"][status="busy"] { background-color: @accentTint; color: @accent; border-color: transparent; }

/* ---------- 卡片与气泡 ---------- */
QFrame[role="card"] {
  background-color: @card;
  border: 1px solid @cardStroke;
  border-radius: 8px;
}
QFrame[role="divider"] {
  background-color: @divider; border: none;
  min-height: 1px; max-height: 1px;
}
QFrame[role="bubbleMine"] {
  background-color: @accentTint; border: 1px solid transparent; border-radius: 8px;
}
QFrame[role="bubbleAi"] {
  background-color: @card; border: 1px solid @cardStroke; border-radius: 8px;
}

/* ---------- 对话（DeepSeek 风格）---------- */
QFrame[role="msgUserBubble"] {
  background-color: @accentTint;
  border: 1px solid transparent;
  border-radius: 14px;
}
QLabel[role="msgAvatarAi"] {
  background-color: @accentTint;
  color: @accent;
  border-radius: 13px;
  font-size: 12px;
  font-weight: 600;
}
QLabel[role="msgAvatarUser"] {
  background-color: @control;
  color: @textSecondary;
  border-radius: 13px;
  font-size: 12px;
  font-weight: 600;
}
QLabel[role="msgText"] { font-size: 14px; color: @text; }
QLabel[role="msgStats"] { font-size: 11px; color: @textTertiary; }

/* ---------- 输入框（DSH 风格大圆角 composer）---------- */
QFrame#composer {
  background-color: @card;
  border: 1px solid @cardStroke;
  border-radius: 22px;
}
QTextEdit#composerEdit {
  background: transparent;
  border: none;
  padding: 2px 4px;
  selection-background-color: @accent;
  selection-color: @onAccent;
}
QPushButton#sendRound {
  background-color: @accent;
  color: @onAccent;
  border: none;
  border-radius: 19px;
  font-size: 17px;
  font-weight: 600;
}
QPushButton#sendRound:hover { background-color: @accentHover; }
QPushButton#sendRound:pressed { background-color: @accentPressed; }
QPushButton#sendRound:disabled {
  background-color: @control;
  color: @textTertiary;
}

/* ---------- 文本视图 ---------- */
QTextEdit[role="log"], QPlainTextEdit[role="log"] {
  background-color: @viewBg;
  border: 1px solid @cardStroke;
  border-radius: 6px;
  padding: 6px 8px;
  selection-background-color: @accent;
  selection-color: @onAccent;
}
QTextEdit[role="input"] {
  background-color: @viewBg;
  border: 1px solid @controlStroke;
  border-bottom: 1px solid @textTertiary;
  border-radius: 6px;
  padding: 8px 10px;
  selection-background-color: @accent;
  selection-color: @onAccent;
}
QTextEdit[role="input"]:focus { border: 1px solid @accent; border-bottom: 2px solid @accent; }

/* ---------- 输入控件（Fluent 文本框） ---------- */
QLineEdit {
  background-color: @control;
  color: @text;
  border: 1px solid @controlStroke;
  border-bottom: 1px solid @textTertiary;
  border-radius: 4px;
  padding: 4px 8px;
  min-height: 22px;
  selection-background-color: @accent;
  selection-color: @onAccent;
}
QLineEdit:hover { background-color: @controlHover; }
QLineEdit:focus {
  background-color: @viewBg;
  border: 1px solid @controlStroke;
  border-bottom: 2px solid @accent;
}
QLineEdit:disabled { color: @textTertiary; border-color: @controlStroke; }
QComboBox {
  background-color: @control;
  color: @text;
  border: 1px solid @controlStroke;
  border-radius: 8px;
  padding: 4px 10px;
  min-height: 24px;
}
QComboBox:hover { background-color: @controlHover; }
QComboBox:focus { border-bottom: 2px solid @accent; }
QComboBox::drop-down {
  border: none;
  width: 22px;
  subcontrol-origin: padding;
  subcontrol-position: center right;
}
/* 下拉列表：圆角、行距、悬停高亮（DSH 同款观感） */
QComboBox QAbstractItemView {
  background-color: @card;
  color: @text;
  border: 1px solid @cardStroke;
  border-radius: 12px;
  padding: 6px;
  margin: 2px 0px;
  outline: none;
  selection-background-color: @navSelected;
  selection-color: @text;
}
QComboBox QAbstractItemView::item {
  min-height: 32px;
  padding-left: 10px;
  padding-right: 10px;
  border-radius: 8px;
}
QComboBox QAbstractItemView::item:hover {
  background-color: @navHover;
}
QLabel[on="true"] { color: @accent; }
/* ---------- 按钮（普通按钮保持原生 windows11 绘制，这里只加强调色变体） ---------- */
QPushButton[role="accent"] {
  background-color: @accent; color: @onAccent;
  border: none; border-radius: 4px;
  padding: 7px 20px; font-size: 14px; font-weight: 600;
  min-height: 22px;
}
QPushButton[role="accent"]:hover { background-color: @accentHover; }
QPushButton[role="accent"]:pressed { background-color: @accentPressed; }
QPushButton[role="accent"]:disabled {
  background-color: @control; color: @textTertiary;
  border: 1px solid @controlStroke;
}
QPushButton[role="danger"] {
  background-color: @critical; color: #FFFFFF;
  border: none; border-radius: 4px;
  padding: 7px 18px; font-size: 14px;
  min-height: 22px;
}
QPushButton[role="danger"]:hover { background-color: @criticalHover; }
QPushButton[role="danger"]:disabled {
  background-color: @control; color: @textTertiary;
  border: 1px solid @controlStroke;
}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical { background: transparent; width: 12px; margin: 4px 2px; }
QScrollBar::handle:vertical { background: @scrollThumb; border-radius: 3px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: @scrollThumbHover; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; width: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px 4px; }
QScrollBar::handle:horizontal { background: @scrollThumb; border-radius: 3px; min-width: 28px; }
QScrollBar::handle:horizontal:hover { background: @scrollThumbHover; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { height: 0px; width: 0px; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }

/* ---------- 状态栏 ---------- */
QStatusBar { background: transparent; border-top: 1px solid @divider; }
QStatusBar::item { border: none; }
QStatusBar QLabel { font-size: 12px; color: @textSecondary; }

QToolTip {
  background-color: @card; color: @text;
  border: 1px solid @cardStroke; padding: 4px 8px;
}
)QSS");

  const QList<QPair<const char*, QColor>> map = {
      {"@window", t.window},
      {"@navHover", t.navHover},
      {"@navSelected", t.navSelected},
      {"@navPressed", t.navPressed},
      {"@cardStroke", t.cardStroke},
      {"@card", t.card},
      {"@divider", t.divider},
      {"@textSecondary", t.textSecondary},
      {"@textTertiary", t.textTertiary},
      {"@text", t.text},
      {"@accentHover", t.accentHover},
      {"@accentPressed", t.accentPressed},
      {"@accentTint", t.accentTint},
      {"@accent", t.accent},
      {"@onAccent", t.onAccent},
      {"@controlHover", t.controlHover},
      {"@controlPressed", t.controlPressed},
      {"@controlStroke", t.controlStroke},
      {"@control", t.control},
      {"@viewBg", t.viewBg},
      {"@scrollThumbHover", t.scrollThumbHover},
      {"@scrollThumb", t.scrollThumb},
      {"@successTint", t.successTint},
      {"@success", t.success},
      {"@criticalHover", t.criticalHover},
      {"@criticalTint", t.criticalTint},
      {"@critical", t.critical},
  };
  // 注意：先替换长键名（@accentHover 必须先于 @accent），否则会被截断
  for (const auto& [key, color] : map) {
    qss.replace(QLatin1String(key), css(color));
  }
  return qss;
}

// ---------------------------------------------------------------- 字体与图标
QString glyph_family() {
  static const QString family = [] {
    const QStringList all = QFontDatabase::families();
    if (all.contains(QStringLiteral("Segoe Fluent Icons"))) {
      return QStringLiteral("Segoe Fluent Icons");
    }
    if (all.contains(QStringLiteral("Segoe MDL2 Assets"))) {
      return QStringLiteral("Segoe MDL2 Assets");
    }
    return QStringLiteral("Segoe UI Symbol");
  }();
  return family;
}

QIcon glyph_icon(const QString& glyph, const QColor& color, int px) {
  qreal dpr = 1.0;
  if (const QScreen* screen = QGuiApplication::primaryScreen()) {
    dpr = std::max<qreal>(1.0, screen->devicePixelRatio());
  }
  QPixmap pm(QSize(px, px) * dpr);
  pm.setDevicePixelRatio(dpr);
  pm.fill(Qt::transparent);
  QPainter p(&pm);
  p.setRenderHint(QPainter::Antialiasing, true);
  p.setRenderHint(QPainter::TextAntialiasing, true);
  QFont f(glyph_family());
  f.setPixelSize(px);
  p.setFont(f);
  p.setPen(color);
  p.drawText(QRectF(0, 0, px, px), Qt::AlignCenter, glyph);
  p.end();
  return QIcon(pm);
}

QFont ui_font(int px, int weight) {
  QFont f;
  f.setFamilies({QStringLiteral("Segoe UI Variable Text"),
                 QStringLiteral("Segoe UI"),
                 QStringLiteral("Microsoft YaHei UI"),
                 QStringLiteral("Microsoft YaHei")});
  f.setPixelSize(px);
  f.setWeight(QFont::Weight(weight));
  f.setStyleStrategy(QFont::PreferAntialias);
  return f;
}

QFont mono_font(int px) {
  QFont f;
  f.setFamilies({QStringLiteral("Cascadia Mono"), QStringLiteral("Consolas"),
                 QStringLiteral("Courier New")});
  f.setPixelSize(px);
  f.setStyleHint(QFont::Monospace);
  return f;
}

QIcon app_icon() {
  const int size = 256;
  QPixmap pm(size, size);
  pm.fill(Qt::transparent);
  QPainter p(&pm);
  p.setRenderHint(QPainter::Antialiasing, true);
  const Tokens& t = Theme::get()->tokens();
  QLinearGradient grad(0, 0, size, size);
  grad.setColorAt(0.0, t.accent.lighter(118));
  grad.setColorAt(1.0, t.accent.darker(112));
  p.setPen(Qt::NoPen);
  p.setBrush(grad);
  p.drawRoundedRect(QRectF(6, 6, size - 12, size - 12), 58, 58);
  QFont f(glyph_family());
  f.setPixelSize(150);
  p.setFont(f);
  p.setPen(t.onAccent);
  p.drawText(QRectF(0, 0, size, size), Qt::AlignCenter,
             QString(QChar(0xE9F9))); // BarChart4：训练 / 统计
  p.end();
  return QIcon(pm);
}

// ---------------------------------------------------------------- 主题单例
Theme::Theme(QObject* parent) : QObject(parent) {
  scheme_ = detect_scheme();
  tokens_ = make_tokens(scheme_);
}

Theme* Theme::get() {
  static Theme* instance = new Theme(); // 故意不析构：避免 Qt 静态析构顺序问题
  return instance;
}

Theme* Theme::install() {
  Theme* t = get();
  if (qApp == nullptr) return t;

  if (!qApp->property("fl_theme_installed").toBool()) {
    qApp->setProperty("fl_theme_installed", true);
    QApplication::setFont(ui_font(14));

    QStyleHints* hints = QGuiApplication::styleHints();
    QObject::connect(hints, &QStyleHints::colorSchemeChanged, t,
                     [t](Qt::ColorScheme) { t->refresh(); });
    QObject::connect(qApp, &QGuiApplication::paletteChanged, t,
                     [t](const QPalette&) {
                       if (t->property("fl_applying").toBool()) return;
                       if (detect_scheme() != t->scheme_) t->refresh();
                     });
  }
  t->refresh();
  return t;
}

void Theme::refresh() {
  scheme_ = detect_scheme();
  tokens_ = make_tokens(scheme_);
  apply();
}

QPalette Theme::palette_for(const Tokens& t) const {
  QPalette p;
  p.setColor(QPalette::Window, t.window);
  p.setColor(QPalette::WindowText, t.text);
  p.setColor(QPalette::Base, t.viewBg);
  p.setColor(QPalette::AlternateBase, t.control);
  p.setColor(QPalette::Text, t.text);
  p.setColor(QPalette::PlaceholderText, t.textTertiary);
  p.setColor(QPalette::Button, t.control);
  p.setColor(QPalette::ButtonText, t.text);
  p.setColor(QPalette::BrightText, t.critical);
  p.setColor(QPalette::Highlight, t.accent);
  p.setColor(QPalette::HighlightedText, t.onAccent);
  p.setColor(QPalette::Link, t.accent);
  p.setColor(QPalette::LinkVisited, t.accent);
  p.setColor(QPalette::ToolTipBase, t.card);
  p.setColor(QPalette::ToolTipText, t.text);
#if QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
  p.setColor(QPalette::Accent, t.accent);
#endif

  p.setColor(QPalette::Disabled, QPalette::Text, t.textTertiary);
  p.setColor(QPalette::Disabled, QPalette::WindowText, t.textTertiary);
  p.setColor(QPalette::Disabled, QPalette::ButtonText, t.textTertiary);
  p.setColor(QPalette::Disabled, QPalette::Base, t.control);
  p.setColor(QPalette::Disabled, QPalette::Button, t.control);
  p.setColor(QPalette::Disabled, QPalette::Highlight, t.control);
  p.setColor(QPalette::Disabled, QPalette::HighlightedText, t.textTertiary);
  return p;
}

void Theme::apply() {
  if (qApp == nullptr) return;
  setProperty("fl_applying", true);
  QApplication::setPalette(palette_for(tokens_));
  qApp->setStyleSheet(style_sheet(tokens_));
  setProperty("fl_applying", false);
  Q_EMIT changed();
}

// ---------------------------------------------------------------- DWM 装饰
void apply_window_chrome(QWidget* window) {
  if (window == nullptr) return;
#ifdef Q_OS_WIN
  const HWND hwnd = reinterpret_cast<HWND>(window->winId());
  if (hwnd == nullptr) return;

  const Tokens& t = Theme::get()->tokens();
  // 标题栏配色跟随应用主题（系统主题变化时由调用方重新应用）
  const BOOL immersive_dark = (t.scheme == Scheme::Dark) ? TRUE : FALSE;
  DwmSetWindowAttribute(hwnd, 20 /*DWMWA_USE_IMMERSIVE_DARK_MODE*/,
                        &immersive_dark, sizeof(immersive_dark));
  // Win11 圆角（标题栏与卡片一致的圆角半径）
  const int corner = 2; // DWMWCP_ROUND
  DwmSetWindowAttribute(hwnd, 33 /*DWMWA_WINDOW_CORNER_PREFERENCE*/, &corner,
                        sizeof(corner));

  // Mica 需要客户区透明绘制；Qt Widgets 下只在显式开启时尝试（GPT_MICA=1），
  // 默认使用 Fluent 回退底色，保证任何显卡 / 远程桌面下都稳定。
  if (qEnvironmentVariableIntValue("GPT_MICA") == 1) {
    const int backdrop = 2; // DWMSBT_MAINWINDOW
    DwmSetWindowAttribute(hwnd, 38 /*DWMWA_SYSTEMBACKDROP_TYPE*/, &backdrop,
                          sizeof(backdrop));
    MARGINS margins{-1, -1, -1, -1};
    DwmExtendFrameIntoClientArea(hwnd, &margins);
  }
#else
  Q_UNUSED(window);
#endif
}

} // namespace fl
