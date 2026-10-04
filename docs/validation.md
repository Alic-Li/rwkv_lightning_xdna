# 实机验证记录：2026-10-02

结果：**296 / 296 配置通过；888 次 C++/XRT NPU 提交；61 / 61 C++ 源文件覆盖。**

- 官方同版本测试表共 299 个配置；4 个仅支持 NPU1，在本机 AIE2P 上不适用。
- 295 个 AIE2P 配置全部编译、执行和离线校验通过。
- 单独增加官方双核 cascade_mm 设计，验证 PUT/GET 级联计算。
- set_rounding.cc 作为相关内核初始化依赖参与执行，不伪装成独立张量算子。
- 每个配置重复执行 3 次。每次输出缓冲区先填充毒化数据，检查每次输出。
- 295 个通用配置同时检查每个输出 tile 的 64 字节越界哨兵；级联测试只做数值检查。
- 数值参考和容差沿用锁定版本的官方 KernelContract，没有放宽阈值。

核对范围包括激活、逐元素运算、规约、数据搬运、归一化、矩阵乘法/向量乘法、
卷积、量化、视觉处理、prefill/attention 等测试表提供的配置。

这里的“全部”表示已复制的 61 个 C++ 源文件都有相应实机测试路径，
以及官方 295 个 AIE2P 配置全部通过。**不表示**所有模板实例、全部输入边界、
每个 IRON 高层复合调度、所有设备架构或 RWKV 整模型已经验证。
本次采用每配置固定种子的随机输入，未穷举官方 extensive 的所有 edge-data 与 seed。

## 证据

- [机器可读报告](../reports/validation-2026-10-02.json)：每个配置状态、源文件覆盖、版本与设备信息。
- `reports/runs/all/*.compile.log`：每个配置的编译日志。
- `reports/runs/all/*.run.log`：独立 C++ 程序的调用结果和时间。
- `reports/runs/all/*.check.log`：离线参考比较结果。
- `build/kernels/<case-id>/`（历史输出路径）：xclbin、指令流、manifest、输入和每次输出。
- `third_party/SOURCES.json`：复制来源、版本和源码 SHA-256。

历史硬件提交来自 `build/host/xdna-run`；Python 只进行离线设计编译、数据生成、
启动 C++ 子进程和读取输出比较。`executed` 与 `passed` 分开记录。

时间数据来自功能测试，含预热及很少的重复次数；不能直接作为稳定性能基准。
`dispatch_us` 是主机 start/wait 时间，不等同于 AIE 核心纯计算周期。

额外检查：CMake 包安装与独立 C++ 消费项目编译链接通过；592 个 CTest 项
（296 个提交 + 296 个检查）已注册，抽查 Add 和级联的 CTest 依赖路径通过。

驱动排障及已经保存的兼容设置见 [环境说明](environment.md)。

## 当前验证入口

精度编译能力可用
`.venv/bin/python tools/validation/precision_capabilities.py --output build/precision-capability`
复现。它调用当前虚拟环境的 AIE2P 编译器，先检查 BF16/INT8 正向对照，再记录
IEEE FP16 编译结果；不调度 NPU、不修改生产精度、不把编译成功当作硬件测试通过。
完整命令、诊断、目标宏和源码/object hash 保存在输出目录的 `results.json`。

主机与 kernel 的编译、通用算子 sweep 和硬件 CTest 见 [构建与测试](build.md)。
上述 2026-10-02 数据是历史实机证据，不代表每次构建都已重新验证。

```bash
cmake --preset test
cmake --build --preset test
ctest --preset test
# 生成小模型，检查 checkpoint dtype、stride、非法文件和 CPU oracle。
.venv/bin/python tools/validation/rwkv7_reference.py
./build/test/rwkv-graph-test build/tests/rwkv7/f32.pth
# 以下生产 kernel 测试串行运行。
./build/test/rwkv-channel-mix-test build/kernels/rwkv7-bf16
./build/test/rwkv-recurrence-stage-test build/kernels/rwkv7-bf16
./build/test/rwkv-projection-residual-test build/kernels/rwkv7-bf16
./build/test/rwkv-alignment-test "$MODEL" build/kernels/rwkv7-bf16
```

参考脚本可用 `--cli FILE` 指定 CLI。阶段测试使用独立 CPU FP64 点积/递推参考，
保留原数值阈值和 guard 检查。

整模型工具 `rwkv-cleanup-regression MODEL KERNELS record|verify SNAPSHOT [--int8-ffn]`
保存与比较 logits、FP32 state、reset/branch 和诊断节点。INT8 现在使用 W8A8
快照格式；旧 W8A16 快照会报 precision mismatch，不能拿新量化结果覆盖旧数值基线。
BF16 batch2/chunk4 由 `rwkv-prefill-model-test` 检查，阶段数值与 guard 由对应 C++
测试检查。当前重构结果见 [2026-10-04 清理报告](../reports/rwkv7-cleanup-2026-10-04.json)。

ReLU² 向量化使用24位 significand 的精确整数乘积及一次 nearest-even 舍入，避免
AIE 的 BF16 分解式 FP32 乘法产生1 ULP 差异。独立设备测试遍历指数为0时的全部
8,388,608个 normal significand，并覆盖随机指数、边界、subnormal、溢出、零、
无穷及 quiet NaN；同时比较原始设备 scalar 结果与独立主机 FP64 oracle。
共享 ReLU² helper 另外比较2048元素调用与 prefill 使用的32元素分块调用，所有输出
逐位一致。测试输出 ABI 扩展为6144个 FP32 元素，使用新测试程序前须重编译下述产物。

```bash
RWKV_XDNA_KERNEL_DIR="$PWD/build/kernels/activation-test" \
MLIR_AIE_KERNEL_SOURCES=third_party/mlir-aie \
  .venv/bin/python tools/compile/rwkv7_activation_test.py
cmake --build --preset test
./build/test/rwkv-activation-test build/kernels/activation-test
```

## 0.4B 动态形状与并发翻译（2026-10-04）

实测模型为 `RWKV_v7_G1d_0.4B_Translate_ctx4096_20260607.pth`：24层、C=1024、
16×64 heads、FFN=4096、vocabulary=65536，最大低秩128。
[验证摘要](../reports/rwkv7-translate-concurrency-2026-10-04.json) 包含模型元数据、
数值/guard 结果和相同 prompt 的单请求/四路并发数据。其他可编译形状尚不等于实机验证。

```bash
cmake --build --preset test
./build/test/rwkv-alignment-test "$MODEL" build/kernels/rwkv7-bf16
./build/test/rwkv-concurrent-graph-test "$MODEL" build/kernels/rwkv7-bf16
./build/test/rwkv-channel-mix-test build/kernels/rwkv7-bf16/c1024-h4096-v65536
./build/test/rwkv-recurrence-stage-test build/kernels/rwkv7-bf16/c1024-h4096-v65536
./build/test/rwkv-recurrence-stage-test build/kernels/rwkv7-bf16/c1024-h4096-v65536 1 --fused-value
./build/test/rwkv-projection-residual-test build/kernels/rwkv7-bf16/c1024-h4096-v65536
```

并发隔离测试使用两条不同 token 序列，将同步开始的并发执行与各自独立设备执行
进行逐位比较，包括每步 logits 和最终各层 attention shift、FFN shift、WKV matrix。
必须观测到至少两个同时等待完成的 NPU 提交。
CLI 实测则将同一翻译 prompt 复制四份；四路输出与单请求输出完全一致。
设备物理 tile 调度仍由 XRT 控制，提交重叠和吞吐收益不证明空间分区同时计算。
