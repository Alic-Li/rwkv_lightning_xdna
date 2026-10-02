# RWKV Lightning XDNA

面向 Ryzen AI NPU 的 C++ 开发仓库。当前阶段已完成运行库基础、官方 C++
内核导入和实机验证；尚未实现 RWKV 模型加载或推理。

首次部署请按 [完整安装、编译与运行指南](docs/environment.md) 操作：包括系统依赖、
render 权限、memlock 锁页限制、驱动兼容设置、uv 环境、单算子和全量验证，
各步骤附官方文档或对应版本源码链接。

**本机验证：Ryzen AI 7 H 350，296 个配置通过，888 次 C++/XRT 调用。**
覆盖导入库的 61 个 C++ 源文件，包括作为初始化依赖执行的舍入设置内核。
这是源码与所选配置覆盖，不代表所有数据类型、边界输入和模板实例都已穷举。
详情见 [验证记录](docs/validation.md) 和 [机器可读报告](reports/validation-2026-10-02.json)。

## 目录

```text
include/rwkv/xdna/        C++ 公共 API
src/runtime/             XRT 会话、设备缓冲区及同步调用
apps/xdna_run.cpp        独立 C++ 测试执行器，无 Python 运行依赖
kernels/rwkv/            后续 RWKV 专用 AIE C++ 内核
third_party/mlir-aie/    复制的完整官方内核与 LUT/运行时辅助源码
third_party/iron/        固定版本 IRON 设计与编译接口
third_party/kernel_tests/ 同版本官方测试配置和级联设计
third_party/nlohmann/    固定版本 JSON 单头文件及许可证
third_party/SOURCES.json 来源版本与内核文件 SHA-256
tools/compile/          离线设计生成、编译、测试输入导出
tools/validation/       调用 C++ 执行器、离线数值比较和报告
tests/package/         已安装 CMake 包的独立消费项目
cmake/                 XRT 查找模块与安装包配置
reports/               可提交的验证摘要；runs/ 日志不进 Git
build/                 主机程序、NPU 产物、输入输出及安装测试
```

运行路径是 `C++ → XRT → amdxdna → NPU`。Python 只用于编译、生成测试数据和
离线参考数值实验；验证脚本不会调用 Python NPU 执行接口。

## 初始化环境

系统需具备已工作的 `amdxdna`、固件和 XRT 开发包；先确认 `xrt-smi examine`
能识别设备，当前用户可读写 `/dev/accel/accel0`。
Ubuntu 安装命令及固件兼容设置见 [环境说明](docs/environment.md)。

```bash
cd /home/alic-li/work_space/rwkv_lightning_xdna
bash tools/bootstrap.sh
export PATH="$PWD/.venv/bin:$PATH"
```

Python 由 uv 管理；构建依赖固定在 `requirements-build.lock`。
`requirements-build.txt` 保留直接依赖配置，升级时显式更新 lock。
项目不依赖相邻的 `../IRON` 或它的虚拟环境。

## 构建与运行

```bash
cmake --preset dev
cmake --build --preset dev

# 295 个适用于 AIE2P 的官方配置 + 1 个级联配置。
# 编译可并行，硬件提交串行。
.venv/bin/python tools/validation/sweep.py --tier all
.venv/bin/python tools/validation/report.py

# 也可以单独编译和执行双核级联矩阵乘法。
.venv/bin/python tools/compile/cascade.py
./build/host/xdna-run build/kernels/cascade-mm/manifest.json
.venv/bin/python tools/validation/check_case.py cascade-mm
```

日常开发可用 `--tier smoke`，或用 `--case <ID>` 限定配置。
`.venv/bin/python tools/compile/kernel_case.py --list all` 列出配置 ID。
`--compile-only` 只编译；`--run-only` 执行已有产物，修改内核后请重新编译。

已有产物可以在不激活 Python 环境时直接运行：

```bash
./build/host/xdna-run build/kernels/add-381f9fce11aa/manifest.json
```

程序输出 `executed` 仅表示硬件调用完成；数值正确性由离线检查器确认。
测试清单使用相对路径，可以将一个完整 case 目录连同 C++ 程序复制部署。
主机程序依赖 XRT 系统动态库，不依赖 Python、Torch 或 MLIR Python 模块。

## CTest 与 C++ 接口

编译产物存在后，重新配置可注册每个配置的提交和检查测试：

```bash
cmake --preset dev -DRWKV_XDNA_HARDWARE_TESTS=ON
ctest --preset dev
```

硬件提交测试设为串行；离线检查要求对应提交成功，避免错误地检查旧输出。

```bash
cmake --install build/host --prefix "$PWD/build/install"
cmake -S tests/package -B build/package-test -G Ninja \
  -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/package-test
```

外部 C++ 工程使用 `find_package(rwkvXdna CONFIG REQUIRED)`，链接 `rwkv::xdna`。
公共接口是 `rwkv::xdna::Session`，同一会话重复调用时复用设备缓冲区。
会话不是线程安全对象；当前测试统一串行占用 NPU。

## 开发约定

- 上游副本与许可证保持完整；自研内核放在 `kernels/rwkv/`。
- 测试采用上游数值容差，不能通过放宽容差或删掉失败配置伪造通过。
- 新的设计编译脚本和原始数据布局必须一起维护，避免主机/设备 ABI 不一致。
- 不提交 `.venv/`、编译产物、测试输入输出或模型权重。
- 下一阶段先确定 RWKV 版本、模型尺寸和精度策略，再设计权重布局与循环状态接口。

许可证：本项目基础代码为 Apache-2.0；第三方代码保留各自许可证，见
[第三方说明](third_party/README.md)。
