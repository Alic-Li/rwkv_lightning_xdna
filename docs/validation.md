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

整模型工具 `rwkv-cleanup-regression MODEL KERNELS record|verify SNAPSHOT [--int8-ffn|--int8-ffn-output]`
记录首 token 全节点、128 步 logits、第 1/8/32/128 步状态，并检查分支、reset、
非法输入与主机状态接口。`verify` 必须使用改动前独立基线生成的快照。
`--int8-ffn` 用于相同 INT8 FFN 配置的逐位回归，快照带独立精度标识，拒绝跨模式使用；
INT8 与 BF16 的量化质量比较应使用 `rwkv-accuracy`。
`--int8-ffn-output` 使用第三种独立快照标识；首次 `record` 仅建立后续优化的回归基线，
不能把它称为与独立改动前实现的逐位比较。该模式的数值验证依赖独立阶段 FP64 oracle
和 BF16 teacher-forced 误差分析。
同 BF16 精度回归证据见 [清理摘要](../reports/rwkv7-cleanup-summary-2026-10-03.json)；
旧跨精度逐元素验收未通过的事实见 [历史性能报告](../reports/rwkv7-optimization-summary-2026-10-03.json)。
128 步回归不代表完整 25600 上下文已经验证。

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
