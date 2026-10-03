# 构建与测试

从仓库根目录执行，先完成 [环境部署](environment.md)，并将 `.venv/bin` 加入 PATH。

## 编译入口

| Preset | 内容 | 主机输出 |
|---|---|---|
| `release` | Release 运行库和 CLI，关闭测试 | `build/release/` |
| `test` | RelWithDebInfo 运行库、CLI 和测试程序；CTest 默认只跑 CPU 测试 | `build/test/` |
| `test-hardware` | 测试程序及已有设备产物的 CTest 注册 | `build/test-hardware/` |

```bash
cmake --preset release
cmake --build --preset release
cmake --build --preset release-kernels # 仅生产 kernel
# 或一次构建生产 CLI 和 kernel：
cmake --build --preset release-full

cmake --preset test
cmake --build --preset test
ctest --preset test
```

`release` 与 `release-kernels` 是独立的构建目标，共用 `release` 配置和
`build/release/` 构建目录，均须先执行 `cmake --preset release`：

- `cmake --build --preset release`：仅编译 C++ 主机运行库和 CLI。
- `cmake --build --preset release-kernels`：仅离线编译生产设备产物，不编译 CLI、不提交 NPU。
- `cmake --build --preset release-full`：构建上述两部分，满足默认 NPU 推理所需的程序和 kernel。

只完成主机构建即可生成 CLI，但不能直接开始 NPU 推理。`bootstrap.sh` 默认也只做
主机构建；首次运行还需 `release-kernels`，或使用 `release-full`。

普通主机构建不会自动编译设备 kernel。`release-full` 编译 CLI 和唯一生产 kernel
清单，不编译通用测试配置。生产 kernel 由 `rwkv7_optimized.py` 顺序编译八个阶段，
产物保留在 `build/kernels/rwkv7-bf16/`。

可选 prefill 目标（默认构建不启用这些调度）：

```bash
cmake --build --preset release --target kernels-release-prefill-batch2
cmake --build --preset release --target kernels-release-prefill-chunk4
cmake --build --preset release --target kernels-release-int8-prefill-chunk4
```

最后一个目标包含 BF16 和两种 W8A16 模式。为保留生产目录，可离线输出到独立目录：

```bash
.venv/bin/python tools/compile/rwkv7_optimized.py --output build/kernels/chunk4 \
  --prefill-chunk4 --int8-ffn-output
```

完整编译只证明产物生成；NPU 数值、状态与性能验收见[推理说明](inference.md)。

## 测试 kernel

```bash
cmake --build --preset test-kernels-smoke
# 全量：295 个 AIE2P 官方配置 + 1 个级联配置，仅离线编译。
cmake --build --preset test-kernels-all

# 对已有产物串行提交 C++ 执行器，再离线检查数值与 guard。
.venv/bin/python tools/validation/sweep.py --tier all --run-only
.venv/bin/python tools/validation/report.py
```

通用测试产物独立放在 `build/kernels/test/<case-id>/`，不会混入生产目录。
`sweep.py` 默认执行器为 `build/test/xdna-run`，可用 `--runner FILE` 指定。
也支持 `--case ID`、`--jobs N` 和 `--compile-only`；省略 `--run-only` 时先编译再验证。

单独编译和检查配置：

```bash
.venv/bin/python tools/compile/kernel_case.py --list all
.venv/bin/python tools/compile/kernel_case.py --case add-381f9fce11aa
./build/test/xdna-run build/kernels/test/add-381f9fce11aa/manifest.json
.venv/bin/python tools/validation/check_case.py add-381f9fce11aa
# 级联配置使用 tools/compile/cascade.py。
```

硬件 CTest 在配置时扫描已有测试 manifest，因此须先编译 kernel，再配置：

```bash
cmake --build --preset release-kernels # 三个 RWKV 阶段测试也需要生产产物
cmake --preset test-hardware
cmake --build --preset test-hardware
ctest --preset test-hardware
```

硬件提交串行执行，离线检查依赖对应提交成功。`executed` 仅表示调用完成，
`passed` 还要求数值检查成功。模型参考和整模型测试见 [验证说明](validation.md)。

## 配置与安装

| CMake 变量 | 默认值 / 用途 |
|---|---|
| `RWKV_XDNA_KERNEL_DIR` | `build/kernels/rwkv7-bf16`，生产编译与阶段测试目录 |
| `RWKV_XDNA_TEST_KERNEL_DIR` | `build/kernels/test`，通用测试编译与 CTest 目录 |
| `RWKV_XDNA_PYTHON` | `.venv/bin/python`，离线编译与检查 |
| `RWKV_XDNA_COMPILE_JOBS` | `4`，通用测试编译并发数 |
| `RWKV_XDNA_HARDWARE_TESTS` | 默认 OFF；要求 `BUILD_TESTING=ON` |

自定义生产目录后，CLI 需传 `--kernel-dir DIR`。直接执行通用测试脚本时，
通过环境变量 `RWKV_XDNA_TEST_KERNEL_DIR` 使用同一自定义目录。
旧 `dev` preset 已替换；旧 `build/host` 和顶层通用测试产物不会自动迁移。

```bash
cmake --install build/release --prefix "$PWD/build/install"
cmake -S tests/package -B build/package-test -G Ninja \
  -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/package-test
```

消费项目使用 `find_package(rwkvXdna CONFIG REQUIRED)`，链接 `rwkv::xdna`
或 `rwkv::inference`。部署主机程序需要 XRT 系统动态库；推理运行无需 Python。

## GitHub Actions 离线编译

`release-kernels` 适合作为 CI 中的离线设备编译步骤：编译过程无需 Ryzen AI NPU，
可在准备好依赖的 Linux x86_64 runner 上生成 `design.xclbin`、`instructions.bin`
和 `config.json`。当前仓库未验证 GitHub 托管 runner 上的完整流程。

runner 需安装锁定的 uv/Python 环境、MLIR-AIE、Peano、CMake/Ninja、Clang/LLD
和 `xclbinutil`。虽然只构建设备目标，当前顶层 CMake 配置仍查找 XRT 开发库及 uuid
头文件，因此这些依赖也必须安装；离线编译 job 无需配置设备权限、memlock 或驱动参数。
依赖安装细节见 [环境部署](environment.md)。

依赖就绪后的 job 命令：

```bash
export PATH="$PWD/.venv/bin:$PATH"
cmake --preset release
cmake --build --preset release-kernels
```

将完整的 `build/kernels/rwkv7-bf16/` 目录保存为 CI artifact，部署时保持其内部目录
结构，并让 CLI 的 `--kernel-dir` 指向该目录。`cmake --install` 目前安装主机库、CLI
和词表，不包含设备 kernel；发布包需另外携带这些产物。

离线编译通过只验证产物构建。NPU 数值、guard 和模型运行测试需在有兼容 Ryzen AI
硬件的 runner 上执行，可使用 [GitHub 自托管 runner](https://docs.github.com/en/actions/concepts/runners/self-hosted-runners)。
CI artifact 的上传或下载也不会自动完成部署，需要在发布流程中另行配置。
