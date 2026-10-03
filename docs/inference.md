# RWKV-7 BF16 / XDNA2 推理

生产路径固定为 BF16 resident decode：矩阵权重和乘法输入使用 BF16，矩阵累加、
WKV recurrence/state、归一化、非线性输出和残差使用 FP32。保留原有统计量求值顺序，
部分归一化辅助仍用 double；本次整理没有改变算术。这里的 BF16 不是 IEEE FP16。
PTH/safetensors 支持 FP32、FP16、BF16 存储，加载为主机 FP32 后一次性排布为设备 BF16。

embedding 查表、tokenizer、sampler、权重加载和调度在 CPU；模型算术全部在 NPU。
没有 CPU 算子回退。NPU 纯 FP32、hybrid、逐节点 NPU eager 和独立 sequence prefill
已删除。CPU FP32 后端仅作为显式参考实现，继续支持 eager/graph 和分层 prefill。
NPU prefill 与 decode 复用同一个逐 token 状态转移。

当前支持 C=2048、32×64 heads、FFN=8192、词表65536、低秩维度不超过256的模型。
已验证 checkpoint 为 `rwkv7-g1k-1.5b-20260930-ctx25600.pth`，24层、146 runs/token。
其他形状会报错，不会选择旧实现。先前性能实验和跨精度漂移记录保留在 `reports/`；
其中的历史命令不再代表当前接口。

## 构建与运行

环境部署见 [environment.md](environment.md)。从仓库根目录执行：

```bash
export PATH="$PWD/.venv/bin:$PATH"
cmake --preset dev
cmake --build --preset dev
# 离线编译唯一生产配置，不调用 NPU。
.venv/bin/python tools/compile/rwkv7_optimized.py

./build/host/rwkv-cli \
  --model /home/alic-li/rwkv_weights/rwkv7-g1k-1.5b-20260930-ctx25600.pth \
  --prompt 'Question: What is the largest planet in our solar system? Answer:' \
  --top-k 1 --max-tokens 32
```

默认 `--backend npu --decode resident --prefill decode`，产物目录为
`build/kernels/rwkv7-bf16`。编译器接受 `--output DIR`，CLI 对应 `--kernel-dir DIR`。
不再需要 `RWKV_XDNA_BF16` 或一组融合开关。构图前检查全部生产 ABI、数值契约和二进制
是否存在；不再根据目录里有哪些实验文件动态选择执行计划。

`--backend cpu` 默认使用 host graph，可显式选择 `--decode eager` 或
`--prefill sequence`。NPU 上这些旧模式会被明确拒绝。
`--tokens 1,2,7,9 --dump-logits FILE` 用于导出每个输入 token 后的 FP32 logits。
CLI 单独报告建图、prefill、TTFT 和后续 decode 时间。生成32个 token 时，首 token 来自
prefill，后续只有31次 decode forward。

## 模块边界

| 文件 | 职责 |
|---|---|
| `weights.cpp` | checkpoint 解析、形状与数据检查 |
| `model.cpp`、`ops.cpp`、`backend.cpp` | CPU 参考数学流程、基础算子和 WKV |
| `graph.cpp` | 记录 RWKV 数学节点与依赖，公共图接口 |
| `graph_internal.hpp` | 图记录及持久设备资源的私有数据结构 |
| `artifacts.cpp` | 唯一生产配置的 ABI/形状检查，先检查再分配设备资源 |
| `weight_layout.cpp` | 逻辑矩阵到设备 tile 排布、BF16 nearest-even 转换，无 XRT 依赖 |
| `resident_plan.cpp` | 分配/别名绑定 arena，再将逻辑节点绑定为固定设备 run |
| `graph_execution.cpp` | 状态导入导出、replay、错误状态、诊断与计时 |
| `src/runtime/session.cpp` | XRT session、持久 BO、slice 和 prepared run |
| `kernels/rwkv/` | 当前生产阶段的 AIE kernel 及共享数学辅助 |
| `tools/compile/rwkv7_common.py` | 离线编译公共类型、设备和编译选项 |
| `tools/compile/rwkv7_optimized.py` | 按固定清单编译生产程序 |

默认设备计划为：输入 LayerNorm；每层 attention norm/mix → BF16 attention
projections → value residual（首层省略）→ FP32 recurrence stage → BF16 output
projection/residual → BF16 ChannelMix；最后 LayerNorm → BF16 vocabulary head。
每次 replay 复用全部 BO 和 run，不重新排布权重或构造 command。
960个逻辑节点仍用于依赖与逐节点诊断，实际提交146个 run；没有原生 runlist。

## C++ 接口和状态

```cpp
rwkv::inference::Weights weights(model_path);
// CPU reference is needed here only to construct an explicit zero state.
auto reference = rwkv::inference::cpu_backend();
rwkv::inference::Model model(weights, *reference);
rwkv::inference::DecodeGraph graph(weights, kernel_directory);
graph.load_state(model.initial_state());
for (int token : prompt_tokens)
  logits = graph.replay_resident(token);
logits = graph.replay_resident(next_token);
auto checkpoint = graph.export_state();
graph.load_state(checkpoint); // reset / resume / branch
```

NPU 图构造函数不接收 CPU 算术后端。`DecodeGraph(weights, backend)` 单独创建 host
参考图。Weights 必须比图活得更久；host backend 也须比 host 图活得更久。
同一图仅允许串行使用。`replay(token, State&)` 兼容显式主机状态，成功后才提交状态；
混用该接口与 resident 接口时，必须重新 `load_state`。

每 token 正常传输仅8192字节 embedding 和262144字节 logits。
FP32 state 使用 `[head,key,value]` 固定布局。`HOST_ONLY` BO 是 NPU 可访问的共享 DDR，
并非 AIE SRAM；权重仍通过 DMA 送入阵列。设备执行失败后必须重建图。

`set_trace` 读回逻辑节点和 WKV state；`set_projection_trace` 读回每个投影的实际输入，
供独立 CPU 点积 oracle 使用。性能测量必须关闭这些 hook。
`RWKV_XDNA_PROFILE=1` 输出阶段/层计时、主机 submit/wait 和传输时间。
这些时间包含调度、DMA 和 PDI 开销，不是纯 kernel cycles。

## 验证与数值边界

```bash
ctest --test-dir build/host -R '^rwkv\.' --output-on-failure
# 生成小模型，检查 FP32/F16/BF16 checkpoint、非连续 stride、非法文件和 CPU oracle。
.venv/bin/python tools/validation/rwkv7_reference.py
./build/host/rwkv-graph-test build/tests/rwkv7/f32.pth
# 以下硬件测试必须串行运行。
./build/host/rwkv-channel-mix-test build/kernels/rwkv7-bf16
./build/host/rwkv-recurrence-stage-test build/kernels/rwkv7-bf16
./build/host/rwkv-projection-residual-test build/kernels/rwkv7-bf16
./build/host/rwkv-alignment-test "$MODEL" build/kernels/rwkv7-bf16
```

`-DRWKV_XDNA_HARDWARE_TESTS=ON` 可将三个生产阶段的数值/guard测试注册到 CTest。
硬件测试用独立 CPU FP64 点积/递推参考，保留原验收阈值。
整模型回归工具 `rwkv-cleanup-regression MODEL KERNELS record|verify SNAPSHOT`
记录首 token 全节点、128步 logits、第1/8/32/128步状态，并验证状态分支、reset、
非法输入及主机状态接口。`verify` 必须使用整理前独立基线记录的快照；不能用候选版本
自身记录的快照宣称通过回归。

BF16 整模型与旧纯 FP32 轨迹不是逐值等价；旧跨精度逐元素验收未通过的事实仍保留在
[历史性能报告](../reports/rwkv7-optimization-summary-2026-10-03.json)。本次要求并检验的
是同 BF16 配置整理前后的正确性和性能，不放宽误差阈值，不以 argmax 一致代替数值检查。
128步回归也不代表已经验证完整25600上下文。

本次整理的同精度回归、阶段 oracle、性能和限制见
[清理验证摘要](../reports/rwkv7-cleanup-summary-2026-10-03.json)。
