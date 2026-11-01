# third_party/zlib

本目录只随仓库提供 **zlib 头文件**（`zlib.h` / `zconf.h`），`zip_util.cpp` 通过 `#include <zlib.h>` 使用。

构建前请补齐 zlib 的**二进制库**（体积/许可原因不入库）：

| 文件 | 来源 |
|---|---|
| `z.lib` | `vcpkg install zlib:x64-windows` 后从 `installed/x64-windows/lib/zlib.lib` 复制并改名，或自行编译 zlib 静态库 |
| `z.dll` | 同上，`installed/x64-windows/bin/zlib1.dll` 复制并改名（运行时需要，`build.ps1` 会自动拷到 `build/`） |

完整步骤见仓库根目录 `README.md` 的「从源码构建」一节。

> 说明：`CMakeLists.txt` 通过 `${ZLIB_ROOT}/z.lib` 链接、`${ZLIB_ROOT}` 作为头文件目录；
> `ZLIB_ROOT` 默认指向 `build.ps1` 创建的 ASCII junction（`<盘符>:/cpptrn/zlib`），
> 也可用 `-DGPT_DEPS_ROOT=...` 覆盖。
