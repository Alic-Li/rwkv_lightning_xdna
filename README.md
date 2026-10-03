# RWKV Lightning XDNA

面向 Ryzen AI NPU 的 C++17 RWKV-7 推理项目。生产路径使用 BF16 权重/乘法输入，
FP32 累加、归一化、残差和 WKV 状态；状态跨 token 常驻设备，默认 prefill 复用 decode。
可显式选择 `--prefill batch2` 或实验性 `--prefill chunk4`；chunk4 在 FFN 中复用四个
token 的权重流，并按层连续执行同一阶段。构建、实测范围与精度边界见[推理说明](docs/inference.md#实验性四-token-chunked-prefill)。
支持 PTH 和 safetensors，CPU FP32 后端用于显式参考。

当前优化主线是**单请求、单 token decode 内部的空间并行**：同 token 的独立 projection、
按 head/channel 切分 WKV、FFN 输出分片及阶段间流水。以重复整模型 ms/token 验收，
不开展多请求 batching 或 continuous batching；已有 prefill 功能保留。
见[空间并行实测](reports/rwkv7-spatial-w2-eight-rejected-2026-10-03.json)。

运行路径：`C++ → XRT → amdxdna → NPU`。Python 仅用于离线编译、测试数据和数值检查。

## 快速开始

先按 [环境部署](docs/environment.md) 配置 XRT、驱动、权限和 memlock。
在仓库根目录执行：

```bash
bash tools/bootstrap.sh                 # 初始化 .venv 并编译 release 主机程序
export PATH="$PWD/.venv/bin:$PATH"
cmake --build --preset release-kernels  # 只编译生产 kernel，不提交 NPU
./build/release/rwkv-cli --model /path/to/model.pth --prompt 'Hello' --max-tokens 32
```

`bootstrap.sh` 和 `cmake --build --preset release` 只编译主机程序；首次 NPU 推理前，
还必须成功执行 `release-kernels`。也可在 `cmake --preset release` 配置后用
`cmake --build --preset release-full` 一次构建主机程序与生产 kernel。
若出现 `Missing production artifact: upstream-norm`，按
[推理排障](docs/inference.md#缺少生产-kernel) 补齐设备产物。

默认产物目录为 `build/kernels/rwkv7-bf16`。当前支持的模型形状及 C++ 状态接口见
[推理说明](docs/inference.md)。

## 文档

- [构建与测试](docs/build.md)：release/test presets、kernel 编译入口、安装包。
- [环境部署](docs/environment.md)：系统依赖与设备排障。
- [推理说明](docs/inference.md)：支持范围、CLI、模块边界和状态管理。
- [验证记录](docs/validation.md)：测试流程、已有证据与数值边界。
- [执行图](docs/execution-graph.md)：设备提交机制与历史兼容性探针。

## 目录

| 目录 | 内容 |
|---|---|
| `include/`, `src/`, `apps/` | C++ 公共 API、运行库、推理与 CLI |
| `kernels/rwkv/` | 生产 BF16 / FP32 设备内核 |
| `cmake/`, `CMakePresets.json` | 主机、安装、kernel 构建配置 |
| `tests/`, `tools/validation/` | C++ 测试、离线参考和验证编排 |
| `tools/compile/` | 生产设计与通用算子测试设计的离线编译 |
| `third_party/` | 固定版本源码、许可证和来源清单 |
| `reports/` | 已提交验证摘要；详细日志在 `reports/runs/` |
| `build/` | 主机程序、设备产物、测试数据和安装结果，不提交 Git |

开发约定见 [AGENTS.md](AGENTS.md)。基础代码采用 Apache-2.0；第三方许可见
[第三方说明](third_party/README.md)。
