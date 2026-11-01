#pragma once
// ui_theme.h - Windows 11 (Fluent / WinUI 3) 主题：设计令牌、全局样式表、
//              系统强调色、Segoe Fluent Icons 图标与 DWM 窗口装饰。
//
// 设计目标：让 Qt Widgets 界面获得与 Windows 11「设置」一致的外观 -
//   * 原生 windows11 控件风格（由 qmodernwindowsstyle 提供，不再用 Fusion）
//   * 跟随系统浅色 / 深色主题，取系统强调色
//   * Fluent 层次配色（窗口层 / 卡片层 / 描边 / 分隔线）
//   * 圆角 + 深色标题栏（DWM），无边框拉伸、贴靠布局等系统行为保持不变
#include <QColor>
#include <QFont>
#include <QIcon>
#include <QObject>
#include <QPalette>
#include <QString>

class QWidget;

namespace fl {

enum class Scheme { Light, Dark };

// Fluent 设计令牌（浅色 / 深色两套取值）
struct Tokens {
  Scheme scheme = Scheme::Dark;

  QColor window;         // 窗口层（Mica 不可用时的回退底色）
  QColor navHover;       // 导航项悬停
  QColor navSelected;    // 导航项选中
  QColor navPressed;     // 导航项按下
  QColor card;           // 卡片层
  QColor cardStroke;     // 卡片描边
  QColor divider;        // 卡片内分隔线
  QColor text;           // 主文本
  QColor textSecondary;  // 次要文本
  QColor textTertiary;   // 占位 / 禁用
  QColor accent;         // 系统强调色
  QColor accentHover;
  QColor accentPressed;
  QColor accentTint;     // 强调色淡底色（气泡 / 徽标）
  QColor onAccent;       // 强调色之上的文字
  QColor control;        // 控件面
  QColor controlHover;
  QColor controlPressed;
  QColor controlStroke;  // 控件描边
  QColor viewBg;         // 文本 / 日志视图底色
  QColor scrollThumb;
  QColor scrollThumbHover;
  QColor success;
  QColor successTint;
  QColor critical;
  QColor criticalHover;
  QColor criticalTint;
};

Tokens make_tokens(Scheme scheme);

// 生成整窗样式表（只覆盖 Fluent 层次与排版；输入框 / 下拉框 / 按钮保持原生绘制）
QString style_sheet(const Tokens& t);

// Segoe Fluent Icons（Win11）/ Segoe MDL2 Assets（Win10 回退）
QString glyph_family();
QIcon glyph_icon(const QString& glyph, const QColor& color, int px = 16);
QFont ui_font(int px, int weight = 400);
QFont mono_font(int px);
QIcon app_icon();

class Theme : public QObject {
  Q_OBJECT
 public:
  static Theme* get();
  // 在 QApplication 构造之后调用一次：安装字体、调色板、样式表并监听系统主题切换
  static Theme* install();

  Scheme scheme() const { return scheme_; }
  const Tokens& tokens() const { return tokens_; }
  void apply();

 Q_SIGNALS:
  void changed();

 private:
  explicit Theme(QObject* parent = nullptr);
  QPalette palette_for(const Tokens& t) const;
  void refresh();

  Scheme scheme_ = Scheme::Dark;
  Tokens tokens_;
};

// 标题栏跟随主题（深色 / 浅色）、窗口圆角；GPT_MICA=1 时尝试 Mica 背景
void apply_window_chrome(QWidget* window);

} // namespace fl
