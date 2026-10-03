# RWKV-7 BF16 / XDNA2 推理

生产路径固定为 BF16 resident decode：矩阵权重和乘法输入使用 BF16，矩阵累加、
WKV recurrence/state、归一化、非线性输出和残差使用 FP32。保留原有统计量求值顺序，
部分归一化辅助仍用 double；本次整理没有改变算术。这里的 BF16 不是 IEEE FP16。
PTH/safetensors 支持 FP32、FP16、BF16 存储，加载为主机 FP32 后一次性排布为设备 BF16。

2026-10-03 已对本机编译器做精度能力核查：`aie2p` 对应 `__AIE_ARCH__=21`，
BF16/INT8 vector multiply 正向对照均编译成功，而 `_Float16` 被目标明确拒绝。
[AMD AIE API 2026.1 矩阵模式表](https://download.amd.com/docnav/aiengine/xilinx2026_1/aiengine_api/aie_api/doc/group__group__mmul.html)
也未在 XDNA2 行列出 FP16 模式。因此 IEEE FP16 不能作为该目标的原生矩阵模式直接
替换 BF16；若要求严格 FP16 语义，需要单独实现并测量仿真路径，不能把 BF16 改名为
FP16。该能力核查不代表整个性能目标已完成，见
[精度能力证据](../reports/rwkv7-precision-capability-2026-10-03.json)。

后续 kernel 开发以 [AIE API 2026.1](https://download.amd.com/docnav/aiengine/xilinx2026_1/aiengine_api/aie_api/doc/index.html)
为文档参考，并对照本机头文件和编译器验证。必须读取 **XDNA2** 行，不能套用
AIE-MLv2 行。矩阵表中的 float 乘法通过 BF16 仿真；BF16 `8×8×8` 模式的脚注 e
要求开启 `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`，以精度换吞吐，不能直接用于
要求原有数值行为的路径。文档版本也不等于本机安装的 API 版本。

优化同时参考 `FastFlowLM/src/lib/xrt` 和 `FastFlowLM/src/lib/hrx` 的实际二进制。
已反汇编核实 HRX forward 的批量 dispatch、统一 flush/wait，以及 executable 创建路径；
这些是待实测的调度方案，不是本项目已经获得的性能收益。地址、哈希、源码交叉核对和
后续验证项见 [2026.1 / FastFlowLM 核查](../reports/amd2026-fastflow-review-2026-10-03.json)。

embedding 查表、tokenizer、sampler、权重加载和调度在 CPU；模型算术全部在 NPU。
没有 CPU 算子回退。NPU 纯 FP32、hybrid、逐节点 NPU eager 和独立 sequence prefill
已删除。CPU FP32 后端仅作为显式参考实现，继续支持 eager/graph 和分层 prefill。
NPU 默认 prefill 与 decode 复用逐 token 状态转移；显式 `--prefill batch2` 使用
分层两-token 调度，在 attention projections 和 FFN 内共享矩阵权重流；WKV 仍按 token 顺序更新。

当前支持 C=2048、32×64 heads、FFN=8192、词表65536、低秩维度不超过256的模型。
已验证 checkpoint 为 `rwkv7-g1k-1.5b-20260930-ctx25600.pth`，24层。BF16 / INT8 FFN-only
普通 decode 为100 runs/token；INT8 FFN+output 和诊断 trace 为123 runs/token。
其他形状会报错，不会选择旧实现。先前性能实验和跨精度漂移记录保留在 `reports/`；
其中的历史命令不再代表当前接口。

## 构建与运行

环境部署见 [environment.md](environment.md)。从仓库根目录执行：

```bash
export PATH="$PWD/.venv/bin:$PATH"
cmake --preset release
# 同时编译 C++ 主机程序和唯一生产 kernel 配置；编译不调用 NPU。
cmake --build --preset release-full

# BF16 推理
./build/release/rwkv-cli \
  --model "$MODEL" \
  --backend npu \
  --kernel-dir build/kernels/rwkv7-bf16 \
  --weights bf16 \
  --prompt $'English: ROCm is an open-source stack, composed primarily of open-so
urce software, designed for graphics processing unit (GPU) computation. ROCm consists of a 
collection of drivers, development tools, and APIs that enable GPU programming from low-lev
el kernel to end-user applications.\nWith ROCm, you can customize your GPU software to meet
 your specific needs.You can develop, collaborate, test, and deploy your applications in a 
free, open source, integrated, and secure software ecosystem. ROCm supports programming mod
els, such as OpenMP and OpenCL, and includes all necessary open source software compilers, 
debuggers, and libraries. ROCm is fully integrated into machine learning (ML) frameworks, s
uch as PyTorch and TensorFlow."\n\nChinese:' \
  --top-k 1 \
  --max-tokens 1024

# INT8 推理（FFN + attention output 权重量化）
./build/release/rwkv-cli \
  --model "$MODEL" \
  --backend npu \
  --kernel-dir build/kernels/rwkv7-bf16 \
  --weights int8-ffn-output \
  --prompt $'English: ROCm is an open-source stack, composed primarily of open-so
urce software, designed for graphics processing unit (GPU) computation. ROCm consists of a 
collection of drivers, development tools, and APIs that enable GPU programming from low-lev
el kernel to end-user applications.\nWith ROCm, you can customize your GPU software to meet
 your specific needs.You can develop, collaborate, test, and deploy your applications in a 
free, open source, integrated, and secure software ecosystem. ROCm supports programming mod
els, such as OpenMP and OpenCL, and includes all necessary open source software compilers, 
debuggers, and libraries. ROCm is fully integrated into machine learning (ML) frameworks, s
uch as PyTorch and TensorFlow."\n\nChinese:' \
  --top-k 1 \
  --max-tokens 1024
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

BF16 / W8A16 两-token prefill（可选，默认不变）：

```bash
cmake --build --preset release --target kernels-release-prefill-batch2
./build/release/rwkv-cli --model "$MODEL" --prompt 'Hello world' \
  --prefill batch2 --top-k 1 --max-tokens 32
./build/release/rwkv-bench "$MODEL" build/kernels/rwkv7-bf16 32 24 2 --prefill-batch2
```

`batch2` 的 attention projections、recurrence、output projection 和 FFN 均处理两行。
BF16 和 INT8 FFN-only 在非首层进一步融合 recurrence 与 output projection；INT8 FFN+output 保留独立提交。WKV 在一次提交内部仍按 token 顺序
更新 FP32 state；norm/mix 仍提交两次。共享已打包权重、session 和 FP32 recurrent state。
仅增加 activation BO，奇数尾 token
使用原 resident decode。只计算并下载最后一个 prompt token 的 logits，因此不兼容
逐 token `--dump-logits` 或 tensor trace。CPU 不支持此模式。
C++ API 使用 `DecodeGraph(weights, root, WeightMode::BFloat16, PrefillMode::Batched2)`，
先 `load_state`，再 `prefill_resident(tokens)`；空输入返回空向量且不改变状态。
返回后可直接 `replay_resident`，也可 export/reset/branch。

24层模型的 BF16 / INT8 FFN-only pair body 为123次提交，最后 logits 再加2次；
32-token prompt 共1970次（61.5625/token）。INT8 FFN+output 的 pair body 为146次，
32-token prompt 共2338次（73.0625/token）；融合前逐步路径为3936次；当前 BF16 / INT8 FFN-only 逐步路径为3200次。每 token 上传8192字节，整个 prompt 仅下载
262144字节 logits。Benchmark JSON 分开报告 prefill 提交/host BO流量和 decode 指标；
这些数值不包含设备内部 DMA，也不等于硬件带宽计数。
2026-10-03 首版仅 FFN batching 的 BF16 A/B/B/A 中，32-token prefill 平均6.532→4.009 s（1.63倍吞吐），
decode 203.599→203.689 ms/token；当时额外常驻30,193,664字节、122个 root BO（attention batching 后为49个）。
加速同时包含 FFN batching 和跳过中间 logits head，不能归因于单一 kernel。
完整 bitwise 回归、guard/oracle 和 A/B/B/A 性能见
[整模型 prefill 验证](../reports/rwkv7-model-prefill-2026-10-03.json)。

INT8 使用相同的分层 pair 调度，FFN 每个 INT8 tile 只展开一次并供两行复用。
每行输出在最后一个 K tile 后乘 FP16-derived FP32 scale；BF16 activation、FP32
累加和 recurrent state 与 resident decode 保持一致。仍是 W8A16 的 BF16 MAC，
不是原生 INT8 MAC。两种实验性 INT8 weight 模式均可使用：

```bash
cmake --build --preset release --target kernels-release-int8-prefill
./build/release/rwkv-cli --model "$MODEL" --prompt 'Hello world' \
  --weights int8-ffn-output --prefill batch2 --top-k 1 --max-tokens 32
./build/release/rwkv-bench "$MODEL" build/kernels/rwkv7-bf16 32 24 2 \
  --int8-ffn-output --prefill-batch2
# FFN-only 模式用 --weights int8-ffn / benchmark --int8-ffn。
```

离线编译器 `rwkv7_optimized.py --prefill-batch2 --int8-ffn-output` 生成 BF16 和 INT8
batch2 产物；独立 `rwkv7_prefill_channel_mix.py --int8` 仅生成 INT8 两种输入布局。
2026-10-03 的32-token A/B/B/A 中，FFN-only prefill 为6.054→3.966 s，
FFN+output 为5.941→3.885 s（降低34.6%）。三种精度各1,533个
logits/state 向量逐位回归通过；INT8 的语言质量限制仍沿用下述实验性说明。
阶段及整模型逐位回归、A/B/B/A 实测和限制见
[INT8 prefill 验证](../reports/rwkv7-int8-prefill-2026-10-03.json)。

后续 attention batching 将 R/K/V 与全部低秩分支的权重流在两个 token 间复用，
使用相邻的双份 activation arena，DMA 直接 scatter 到各 token 的 recurrence/value/rank
视图。累计常驻 BO 字节不变，root BO 从413减至340；WKV 的 FP32 state 更新次序不变。
现有 `--prefill-batch2` 编译入口会同时生成 `bf16-attention-projections-{3,4}-b2`；
旧产物目录需要重新执行对应 prefill kernel target。
该改动的32-token A/B/B/A：BF16 4.003→3.691 s，
INT8 FFN+output 3.890→3.571 s；同精度的三种模式均通过1,533个向量的
逐位 logits/state 回归。
阶段 FP64/bitwise/guard 和整模型验收见
[attention prefill 验证](../reports/rwkv7-prefill-attention-2026-10-03.json)。

BF16 output batching 通过 stride 55296/61440 的 DMA gather 直接读取两个 recurrence
视图，分别读取两个残差 BO；不增加主机拷贝或常驻 activation 内存。
重新编译 prefill target 会生成 `bf16-prefill-output-b2-s{55296,61440}`。
BF16 数值与整模型性能见
[输出投影 prefill 验证](../reports/rwkv7-prefill-output-resident-2026-10-03.json)。

INT8 output batching 复用相同的数据流，每个4160字节权重 tile 在设备上展开一次，
两个 token 共享整数到 BF16 的转换结果；行 scale 仍在八个 K tile 累加完成后应用。
这仍是 W8A16 BF16 MAC，未改变原有量化规则。INT8 prefill 编译入口会额外生成
`int8-prefill-output-b2-s{55296,61440}`；独立编译用 `rwkv7_prefill_output.py --int8`。
8-worker 版本在 MemTile 合并输入与残差流，投影 worker 内完成残差加法。
32-token A/B/B/A 为3.575→3.462 s（降低3.15%），decode 均值185.72→185.97 ms
（增加0.14%）；三种精度各1,533个向量逐位回归通过。
验证及整模型 A/B/B/A 结果见
[INT8 输出投影 prefill 验证](../reports/rwkv7-prefill-int8-output-2026-10-03.json)。

Recurrence batching 将一对 token 的状态读入/写回合并为一次，在计算核本地顺序更新。
三种精度均保持原有输出位模式和 FP32 状态；常驻 BO 数量及字节数不变。
按 DMA 描述符计算，每层每对 token 少传1,048,576字节状态，24层32-token prompt
合计减少384 MiB；这是计划传输量，不是硬件带宽计数。
重新编译 prefill target 会生成 `prefill-recurrence-b2` 与 `prefill-value-recurrence-b2`，
独立编译入口为 `rwkv7_prefill_recurrence.py`。
阶段及完整模型验证、A/B/B/A 性能和限制见
[recurrence prefill 验证](../reports/rwkv7-prefill-recurrence-2026-10-03.json)。

Recurrence/output 融合使用31个计算核，直接在原 auxiliary arena 的向量25/26写入
projection 和残差输出；FFN 以61440-float stride读取，无额外常驻 BO 或 host copy。
重新编译 prefill target 会生成 `bf16-prefill-recurrence-projection-b2` 和
`bf16/int8-prefill-ffn-b2-recurrence-input`；独立入口为
`rwkv7_prefill_recurrence_projection.py` 与 `rwkv7_prefill_channel_mix.py --recurrence-input [--int8]`。
32-token A/B/B/A：BF16 3.609→3.483 s，INT8 FFN-only 3.574→3.446 s，
decode 均值变化分别约+0.04%和−0.02%。三种精度各1,533个向量逐位回归通过。
详见[recurrence/output 融合验证](../reports/rwkv7-recurrence-projection-prefill-2026-10-03.json)。

INT8 输出权重的同类融合实验未采纳：31核版整模型仅改善0.16%，24核版筛选测试慢约3.5%，
增加至11条 recurrence lane 的32核版受到 MemTile DMA 通道与路由约束，未完成编译。
INT8 FFN+output 继续使用独立 recurrence/output 提交；详见
[INT8 融合淘汰记录](../reports/rwkv7-int8-recurrence-projection-rejected-2026-10-03.json)。

可用 `RWKV_XDNA_PROFILE=1` 诊断 batch2：

```bash
RWKV_XDNA_PROFILE=1 ./build/release/rwkv-cli --model "$MODEL" \
  --tokens 1,18,35,52,69,86,103,120 --prefill batch2
```

stderr 的 `prefill_profile` JSON 按 stage 和 `token_slot` 汇总次数、elapsed/submit/wait
微秒；slot 0/1 为成对调度的第一/第二次单-token调用，slot 2 为 fused attention/FFN pair。
另报告 pair embedding 上传和末 pair logits 下载；奇数尾的原 decode 路径单独输出
`decode_profile`。这些耗时包含调度、program switch、DMA 和算术，slot 差异不是独立
PDI 计数，不能作为硬件利用率。正常计时继续使用拒绝 profile 环境变量的 `rwkv-bench`。
2026-10-03 attention batching 之前的32-token profile 中，recurrence 约37.7 ms/token，attention projections
约33.1–33.3 ms/token，FFN 约31.7–32.2 ms/token。第一/第二 slot 的 recurrence
约2.20/0.94 ms/run，提示继续测量 program switching 与 sequence reuse；该差值
不能单独归因为 PDI。数值回归和未插桩性能检查见
[prefill profile 证据](../reports/rwkv7-prefill-profile-2026-10-03.json)。

## 实验性四-token chunked prefill

`--prefill chunk4` 支持 BF16、INT8 FFN-only 和 INT8 FFN+output，不改变默认的逐 token prefill。
每层按阶段处理四个 token：attention/WKV 使用两次原有 batch2 kernel，
WKV FP32 state 依次更新；FFN 一次权重流供四行复用。
key 核将行结果流给独立的 activation 收集核，value 核分块读取设备已写入的
activation，避免四行完整 activation 同时挤入每个 value tile。
没有新增 CPU 模型算术或 host activation 往返。尾部1–3个 token 使用原 NPU decode。

```bash
cmake --build --preset release --target kernels-release-prefill-chunk4
./build/release/rwkv-cli --model "$MODEL" --prefill chunk4 --prompt 'Hello world' --max-tokens 32
./build/release/rwkv-bench "$MODEL" build/kernels/rwkv7-bf16 32 24 2 --prefill-chunk4
```

独立编译入口是 `rwkv7_optimized.py --prefill-chunk4 --output DIR`。
它验证 output stride 变体的 PDI 逐字节相同后共享 xclbin/context，避免超过硬件
context 上限；仅保留各自 DMA 指令。原有 batch2 仍可使用同一目录。
新图额外占用60,977,152字节常驻 BO；每4个完整 token 的 body 为268次提交，
末 chunk logits 另2次。更多提交不妨碍权重复用和连续执行同一程序带来的收益。

本机首轮32-token A/B/B/A：prefill 3.437→2.590秒，9.31→12.35 tokens/s；
decode 198.73→200.81 ms/token，后者有约1%退化，尚不据此替换默认路径。
2,628个 logits/state 向量逐位验收通过，含0–64 token、各种尾部和状态分支。
完整范围与限制见[BF16 chunk4 实测](../reports/rwkv7-chunk4-model-bf16-2026-10-03.json)。
随后128-token、每侧384个 decode 样本的 B/A/A/B 复测中，prefill 为9.34→12.48 tokens/s，
decode 为198.837→198.730 ms/token。长测未复现短测约1%的 decode 退化，
也不足以确立 decode 加速；详见[长负载复测](../reports/rwkv7-chunk4-long-confirmation-2026-10-03.json)。
这不是已达硬件上限或长上下文质量验收的声明。

W8A16 chunk4 使用同样的数据流；每个 INT8 weight tile 在核内展开一次，供四个
BF16 token 复用，最后一个 K tile 后才乘 FP16-derived FP32 scale。
不量化 activation，FP32 累加与 WKV state 不变。构建和运行：

```bash
cmake --build --preset release --target kernels-release-int8-prefill-chunk4
./build/release/rwkv-cli --model "$MODEL" --weights int8-ffn-output --prefill chunk4 --prompt 'Hello' --max-tokens 32
./build/release/rwkv-bench "$MODEL" build/kernels/rwkv7-bf16 32 24 2 --int8-ffn-output --prefill-chunk4
```

32-token A/B/B/A：FFN-only prefill 9.38→12.09 tokens/s，FFN+output 9.45→12.51；
decode 分别183.82→184.11、184.20→184.31 ms/token，没有确立 decode 收益。
两种模式各2,628个 logits/state 向量逐位通过；既有 INT8 语料质量限制不变。
见[INT8 chunk4 整模型验证](../reports/rwkv7-chunk4-model-int8-2026-10-03.json)。

## 缺少生产 kernel

若 CLI 报：

```text
rwkv-cli: Missing production artifact: upstream-norm
```

表示所选 kernel 目录中缺少 `upstream-norm/config.json`，可能尚未编译生产 kernel，
或运行时指向了错误的产物目录。仅编译 `release`、执行 `bootstrap.sh`，以及编译
`test-kernels-all`，都不会生成这套生产产物。

从仓库根目录补齐生产 kernel：

```bash
export PATH="$PWD/.venv/bin:$PATH"
cmake --preset release
cmake --build --preset release-kernels
ls build/kernels/rwkv7-bf16/upstream-norm/{config.json,design.xclbin,instructions.bin}
```

确认构建成功后重新运行上面的 CLI 命令。`upstream-norm` 是生产路径的 FP32
LayerNorm，由 `rwkv7_norm.py` 生成，包含在 `release-kernels` 的编译清单中。
不要只补这个目录；推理还需要其余生产阶段的完整产物。

默认 kernel 路径相对于启动 CLI 时的工作目录。若从其他目录启动，或使用自定义
产物目录，显式传入绝对路径：

```bash
./build/release/rwkv-cli \
  --model /path/to/model.pth \
  --kernel-dir "$PWD/build/kernels/rwkv7-bf16" \
  --prompt 'Hello' --max-tokens 32
```

## 模块边界

| 文件 | 职责 |
|---|---|
| `weights.cpp` | checkpoint 解析、形状与数据检查 |
| `model.cpp`、`ops.cpp`、`backend.cpp` | CPU 参考数学流程、基础算子和 WKV |
| `graph.cpp` | 记录 RWKV 数学节点与依赖，公共图接口 |
| `graph_internal.hpp` | 图记录及持久设备资源的私有数据结构 |
| `artifacts.cpp` | 显式精度方案的 ABI/形状检查，先检查再分配设备资源 |
| `weight_layout.cpp` | 逻辑矩阵到设备 tile 排布、BF16 nearest-even 转换，无 XRT 依赖 |
| `resident_plan.cpp` | 分配/别名绑定 arena，再将逻辑节点绑定为固定设备 run |
| `graph_execution.cpp` | 状态导入导出、replay、错误状态、诊断与计时 |
| `src/runtime/session.cpp` | XRT session、持久 BO、slice 和 prepared run |
| `kernels/rwkv/` | 当前生产阶段的 AIE kernel 及共享数学辅助 |
| `tools/compile/rwkv7_common.py` | 离线编译公共类型、设备和编译选项 |
| `tools/compile/rwkv7_optimized.py` | 按固定清单编译生产程序 |

默认设备计划为：输入 LayerNorm；每层 attention norm/mix → BF16 attention
projections → FP32 recurrence 与 BF16 output projection/residual 融合 → ChannelMix；
最后 LayerNorm → BF16 vocabulary head。首层保持独立 recurrence 和 output 提交；
INT8 FFN+output 暂不使用 decode 融合。每次 replay 复用 BO 和 run，不重新排布权重。

BF16 / INT8 FFN-only 非首层使用31核 `bf16-decode-recurrence-projection`，7条 lane
处理32个 head，first-layer value 广播一次。最终 projection/residual 写入辅助 arena
的向量25/26，FFN 直接读取，无额外常驻 BO 或 host copy。24层普通 decode 从123降到
100次提交，960个逻辑节点保留用于依赖与诊断；仍没有原生 runlist。
启用 node/projection trace 时使用原逐阶段计划，恢复123次提交并输出完整中间张量；
`stats()` 反映当前计划。两种计划共享权重和请求状态，支持切换与 reset/branch。

生产 kernel preset 现包含 `bf16-decode-recurrence-projection`；独立编译入口为
`rwkv7_decode_recurrence_projection.py`。旧产物需要重新编译，包括 batch2 FFN：
两种输入 stride 的设备 PDI 经逐字节比较相同后使用同一 xclbin/UUID，各自保留 DMA
指令。C++ Session 共享同设备、UUID、kernel 的硬件 context，以避免增加 decode 程序后
batch2 初始化失败。共享缓存只持有弱引用，不延长程序生命周期。
32-token batch2 后的 decode A/B/B/A：BF16 204.667→202.220 ms/token（降低1.20%），
INT8 FFN-only 189.157→186.493 ms/token（降低1.41%）；prefill 基本持平。
两种 decode 精度各1,116个向量、三种 prefill 精度各1,533个向量精确回归通过。
阶段与整模型验证见[decode 融合记录](../reports/rwkv7-decode-recurrence-projection-2026-10-03.json)。

已测试进一步合并 attention norm/mix 与 projections 的17核图：虽然提交数从123降到99、
BO 数从291降到242，且128步逐位回归通过，整模型 A/B/B/A 仍比原图慢约0.7%。
因此未采用该 attention 融合，当时保留123次提交；孤立 warm-stage 加速不能代替整模型测量。
布局、调度失败和对比数据见 [被拒绝的 attention 融合](../reports/rwkv7-attention-fusion-rejected-2026-10-03.json)。

### 权重工作集与程序交替诊断

```bash
cmake --build --preset test --target rwkv-working-set-bench
./build/test/rwkv-working-set-bench build/kernels/rwkv7-bf16
./build/test/rwkv-working-set-bench build/kernels/rwkv7-bf16 --int8
```

此 C++ benchmark 使用128个不同权重 BO，BF16 最大工作集1 GiB。比较连续执行投影与
每次先执行另一 LayerNorm 程序的情况；投影计时不包含 LayerNorm 调用时间，也不包含
host 传输或数值检查。每种模式进行正反两轮，每组128次测量，所有 BO 做数值与保护检查。
2026-10-03 测量中，BF16 连续投影约276–282 µs，交替程序后约993–1007 µs；
W8A16 分别约173–177 µs和890–901 µs。扩大工作集的影响远小于程序交替。
缩小编译分区到投影5列、norm 2列未改善延迟，运行时仍报告共同的8列分区。
这不是实际 DDR 带宽或纯 context 切换计数。
详见[工作集与程序交替测量](../reports/rwkv7-working-set-and-program-alternation-2026-10-03.json)。

### 共用设备程序对照

将原 LayerNorm 与投影的12个 worker 放入同一个设备程序，保留各自的 DMA、BO 和独立提交。
两套指令在编译后必须具有完全相同的 PDI，才共用 xclbin/UUID 和运行时硬件上下文。
诊断编译器只输出独立测试目录：

```bash
MLIR_AIE_KERNEL_SOURCES=third_party/mlir-aie .venv/bin/python \
  tools/compile/rwkv7_shared_program_probe.py --output-root build/kernels/shared-program-bf16
./build/test/rwkv-working-set-bench build/kernels/shared-program-bf16
MLIR_AIE_KERNEL_SOURCES=third_party/mlir-aie .venv/bin/python \
  tools/compile/rwkv7_shared_program_probe.py --output-root build/kernels/shared-program-int8 --int8
./build/test/rwkv-working-set-bench build/kernels/shared-program-int8 --int8
```

2026-10-03 A/B/B/A 中，交替 norm 后的投影平均耗时：BF16 从1006.5降到273.3 µs，
W8A16 从882.0降到172.6 µs，接近连续执行投影的耗时。八个测试均通过128组权重的
FP64 oracle、重复输出与缓冲区保护检查；独立运行时查询确认只有一个硬件上下文。
这验证了共用程序在该对照中的效果，尚不能代表整模型收益或硬件极限。
模型相邻阶段的共驻布局对照见下文。
详见[共用程序对照记录](../reports/rwkv7-shared-program-control-2026-10-03.json)。

将 norm/mix 与四分支 attention 放入同一16核程序后，阶段数值/保护检查和 BF16
128步整模型1,116个向量逐位回归通过，但整模型 A/B/B/A 从200.135变为208.965 ms/token
（慢4.41%）。保留第二级低秩投影独立权重通道的版本仍为207.740 ms/token，均未采用。
独立 profile 中，attention 每 token 减少约16–18 ms，norm/mix 增加约26 ms；
微基准中程序已驻留时的收益不足以预测模型从其他程序切入时的成本。
下一步研究在不同操作间复用计算核心和 DMA 通道，并同时测量从第三个程序切入的开销。
详见[被拒绝的 attention 共驻实验](../reports/rwkv7-shared-attention-rejected-2026-10-03.json)。

### 在同一计算核心上选择操作

诊断原型把操作码放入公共输入包，让八个投影核心选择 LayerNorm 或投影分支，
复用 DMA 与输出图；共11核。独立操作码 FIFO 会超出每个核心的两个输入 DMA 通道，
因此控制字与数据共用通道。该原型的输入格式与生产模型不同，仅用于验证核心复用。

```bash
MLIR_AIE_KERNEL_SOURCES=third_party/mlir-aie .venv/bin/python \
  tools/compile/rwkv7_mode_worker_probe.py --output-root build/kernels/mode-worker
cmake --build --preset test --target rwkv-mode-worker-test
# shared-program-bf16 用上一节的共驻程序编译器生成。
./build/test/rwkv-mode-worker-test build/kernels/mode-worker \
  build/kernels/rwkv7-bf16 build/kernels/shared-program-bf16
```

测试分别覆盖连续操作和每次先执行第三个程序再切入，并计时 norm 与 projection 的总和。
2026-10-03 复用核心在后者约1.07 ms，独立核心共驻约1.16 ms，分开程序约1.19 ms。
四种输入、非均匀 affine 参数、连续不等次数的操作码切换、FP64 oracle、原程序输出比较、
不可变数据与缓冲区保护检查均通过。这里投影使用固定输入，不以 norm 输出作为输入，
因此这是程序进入与切换的对照，不代表整层或整模型加速。
详见[核心复用对照记录](../reports/rwkv7-mode-worker-control-2026-10-03.json)。

### norm/mix 与 attention 复用核心的整模型候选

`--mode-reuse` 为四分支、单 token attention 增加 norm/mix 操作模式；14个计算核心的
布局保持一致，原有权重 FIFO 传递操作码。两个指令产物共享经过字节核对的 PDI 和
xclbin，普通 decode 的第1–23层可复用同一硬件 context。trace 和 batch2 prefill
继续使用原 norm/mix 程序。FP32 算术顺序不变，runs/token 仍为100/100/123，
常驻数据增加385,024字节，不增加每 token 主机传输。

这是显式实验路径，默认编译和生产产物尚未启用。独立生成后补齐已有 kernel：

```bash
RWKV_XDNA_KERNEL_DIR="$PWD/build/kernels/norm-attention-optin" \
  .venv/bin/python tools/compile/rwkv7_attention_projections.py --mode-reuse
for artifact in "$PWD"/build/kernels/rwkv7-bf16/*; do
  target="build/kernels/norm-attention-optin/$(basename "$artifact")"
  if [ ! -e "$target" ]; then ln -s "$artifact" "$target"; fi
done
cmake --build --preset test
./build/test/rwkv-mode-attention-test build/kernels/norm-attention-optin \
  build/kernels/rwkv7-bf16
./build/release/rwkv-bench "$MODEL" build/kernels/norm-attention-optin \
  32 96 2 --prefill-batch2
```

三种权重模式的128步 decode 和各1,533个 prefill 回归向量均逐位一致。BF16 长测
B/A/A/B（每组192个 decode 样本）从201.092降至198.881 ms/token，约改善1.10%；
同批 prefill 从3.463升至3.505秒。短测收益存在明显波动，尚不足以提升为默认路径。
全部样本保留，未剔除慢轮次，见
[整模型候选验证](../reports/rwkv7-mode-attention-progress-2026-10-03.json)。

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

### 可重复性能测量

`rwkv-bench` 使用固定 token ID，先预热4步，再对每轮重新加载零状态。
计时不含权重加载、建图、reset、采样和状态导出；这些阶段单独报告或在计时区外。
默认 prefill 为逐 token resident replay；三种 NPU weight 模式均可显式开启 `batch2` FFN batching。

```bash
cmake --build --preset release
# 每轮16步 prefill、32步 decode，共3轮；输出 JSON。
./build/release/rwkv-bench "$MODEL" build/kernels/rwkv7-bf16 16 32 3
```

输出包含每轮 prefill tokens/s、decode mean/p50/p95、runs/token、常驻字节和
host BO 传输字节。`graph_nodes` / `graph_buffers` 为逻辑图节点与缓冲区数量，
`root_bos` 为实际根 BO 数，不能互相替代。每轮 `decode_samples_ms` 按执行顺序保留
原始延迟，可用于复核分位数和观察漂移；`state_reset_seconds` 单独记录状态重置，
不计入 prefill/decode。`replays` 包含预热和全部测量轮次。
禁止同时开启 `RWKV_XDNA_PROFILE`；阶段 profiling 使用 CLI
单独执行。基准期间避免并行编译或其他 NPU 任务；比较优化前后采用 A/B/B/A 顺序，
保留每轮分布。固定合成 token 是性能负载，不是语言质量验收。

`host_*_bytes_per_token` 不含共享 DDR 到 AIE 的内部 DMA。未测得的设备带宽和
全阵列计算利用率在 JSON 中为 `null`，不能用 host wait 时间冒充利用率。

### 实验性 batched projection

`rwkv7_prefill_projection.py` 为 FFN key（2048→8192）和 value（8192→2048）
编译 batch=1/2 的独立 BF16 projection。每个16×256权重 tile 在释放前用于两个
token，保持每个 token 原有 MAC reduction 和 K tile 累加顺序，输出为 token-major
FP32。batch=2 每 token 的权重 DMA payload 和提交次数均减半；这里的 payload
是静态数据量，不是设备带宽计数器。

```bash
RWKV_XDNA_KERNEL_DIR="$PWD/build/kernels/prefill-projection" \
MLIR_AIE_KERNEL_SOURCES=third_party/mlir-aie \
  .venv/bin/python tools/compile/rwkv7_prefill_projection.py
cmake --build --preset test
./build/test/rwkv-prefill-projection-test build/kernels/prefill-projection 300
```

2026-10-03 的孤立阶段 A/B/B/A 中，key 每 token 从932.75降至469.96 µs，value
从920.91降至461.81 µs。三个输入 pass（含 zero token / zero weight row）通过
FP64 oracle、batch=1/2 逐位比较、输入/权重不可变性和 BO guard 检查。
上述约2倍是孤立阶段吞吐，不能直接作为整模型加速。后续 FFN 融合和整模型接入
见下文及本文前面的 batch2 用法；attention projections 的后续 batch 接入见本文前部。详见
[batched projection 证据](../reports/rwkv7-prefill-projection-2026-10-03.json)。

在此基础上，`rwkv7_prefill_channel_mix.py` 将两个 token 的 norm/mix、key、ReLU²、
value、residual 合为一次提交。shift 在 NPU 上顺序更新；权重 tile 在两个 token 间
复用。内部 gather 使用 `[core,token,channel]` 排布，下游核直接寻址，不经 CPU
转置。输出和诊断 DMA 写回 token-major 布局；五个 BO 参数分别为输入、参数、
权重、FP32 arena、BF16 arena。arena 偏移记录在产物 `config.json`。

```bash
RWKV_XDNA_KERNEL_DIR="$PWD/build/kernels/prefill-ffn" \
MLIR_AIE_KERNEL_SOURCES=third_party/mlir-aie \
  .venv/bin/python tools/compile/rwkv7_prefill_channel_mix.py
cmake --build --preset test
./build/test/rwkv-prefill-ffn-test build/kernels/prefill-ffn build/kernels/rwkv7-bf16 300
```

阶段测试比较两次生产 ChannelMix 与一次 fused batch：mixed BF16、key/value projection、
ReLU² BF16、residual 和 final shift 均逐位检查，另做 FP64 oracle、连续 chunk、
reset/branch、不可变输入和 BO guard 检查。先前五次提交的分离方案未加速，已删除；
最终300样本/组 A/B/B/A 中，每两个 token 为3.960→2.016 ms，阶段吞吐提升1.96倍。
共享 ReLU² helper 的提取也通过 BF16 / INT8 FFN+output 的128步整模型回归。
该阶段现已通过上述 batch2 模式接入整模型，并支持 W8A16 FFN。最初 BF16 阶段结果、
驱动参数接口限制和未完成工作见 [fused FFN 证据](../reports/rwkv7-prefill-ffn-2026-10-03.json)。

2026-10-03 首轮调优将 ChannelMix value 权重 FIFO 改为双缓冲，A/B/B/A 整模型
对比为240.53→233.68 ms/token（约2.85%），128步 logits/state 逐位回归通过。
这不是整体优化任务完成或已达到硬件上限的声明；实验范围、未采用方案及待完成的
INT8/prefill 工作见 [调优进度](../reports/rwkv7-tuning-progress-2026-10-03.json)。

### ChannelMix 硬件 trace

可在独立目录编译带 trace BO 的诊断产物。生产图会拒绝此 ABI，避免将额外 trace
参数遗漏后提交。阶段测试仍执行独立数值 oracle 和 BO guard 检查。

```bash
export PATH="$PWD/.venv/bin:$PATH"
RWKV_XDNA_TRACE_BYTES=16777216 \
RWKV_XDNA_KERNEL_DIR="$PWD/build/kernels/channel-trace" \
MLIR_AIE_KERNEL_SOURCES=third_party/mlir-aie \
  .venv/bin/python tools/compile/rwkv7_channel_mix.py
cmake --build --preset test
mkdir -p reports/runs/channel-trace
./build/test/rwkv-channel-mix-test build/kernels/channel-trace 1 \
  reports/runs/channel-trace/trace.txt
.venv/bin/python -m aie.utils.trace.parse \
  --input reports/runs/channel-trace/trace.txt \
  --mlir build/kernels/channel-trace/bf16-channel-mix/design.prj/input_with_addresses.mlir \
  --output reports/runs/channel-trace/trace.json
.venv/bin/python tools/validation/trace_summary.py reports/runs/channel-trace/trace.json
```

采样一个 key core 和一个 value core，记录 vector issue、memory/stream/lock stall
及 DMA port 活动。事件时间单位为设备 cycles；统计窗口从收到初始 activation 后开始，
包含权重等待和运算。各事件比例可能重叠，vector issue 占比不是峰值 FLOPS 利用率，
也不能代表整个阵列。带 trace 的墙钟计时不用于性能对比。普通阶段微基准可用
`rwkv-channel-mix-test KERNEL_ROOT 200`，完成 oracle 后单独预热并测量200次。

INT8 trace 将编译入口换为 `tools/compile/rwkv7_channel_mix_int8.py`，并运行
`rwkv-channel-mix-int8-test TRACE_ROOT BF16_ROOT 1 TRACE_FILE`。第二个目录须为
不带 trace 的 BF16 基准产物；解析器的 MLIR 路径使用 `int8-channel-mix`。
该模式仍检查独立 FP64 oracle、输入/权重不可变性和 BO guards，但不输出性能计时。
设置 `RWKV_XDNA_TRACE_ACTIVATION=1`（须同时设置 trace bytes）可将 key core 的
计时窗口缩小到 ReLU²及 FIFO release；value core 仍记录原有矩阵窗口。
`config.json` 中的 `trace_key_region` 标明该差异，不能混用两种窗口比较周期数。

2026-10-03 另测每个 BF16 key core 缓存2048个已转换 activation，避免跨输出行重复
FP32→BF16 转换。阶段 oracle、guard 和128步整模型回归通过，但 A/B/B/A 为
203.26→203.60 ms/token，未加速，已撤回。详见
[key activation cache 实验](../reports/rwkv7-channel-key-cache-rejected-2026-10-03.json)。

2026-10-03 定向 trace 显示 scalar ReLU²占257,879 cycles；改为24位 significand
的精确整数向量平方并仅舍入一次后，降到5,780 cycles（该 key core 窗口减少97.76%）。
直接使用 AIE FP32 vector multiply 曾产生1 ULP差异，已拒绝；保留实现通过917.5万
scalar/vector/FP64逐位比较及两种精度128步整模型逐位回归。
整模型 A/B/B/A：BF16 206.953→204.137 ms/token，INT8 191.899→188.931 ms/token，
分别降低1.36%和1.55%；见 [ReLU²调优证据](../reports/rwkv7-ffn-activation-progress-2026-10-03.json)。

## 验证与数值边界

Recurrent stage 可用 `rwkv-recurrence-stage-test KERNEL_ROOT 300` 测量孤立提交延迟，
每次计时外重置 state/auxiliary；原有 FP64 oracle、连续 replay 和 BO guard 检查仍执行。
诊断编译同样支持 `RWKV_XDNA_TRACE_BYTES`：将上面 trace 命令的编译入口换为
`tools/compile/rwkv7_recurrence_stage.py`，测试换为 `rwkv-recurrence-stage-test`，
产物子目录换为 `fused-recurrence-stage`。其生产图占满16个 shim S2MM channel，
因此诊断图使用单 lane 依次处理32个 head，为 trace 留出通道；分别采样 prepare、
recurrent、finish 三个 core。marker 仅包围 kernel 调用，不含 FIFO acquire/release。
这些 cycles 用于比较核内开销，不能视为生产图的并行吞吐或完整 DMA 延迟。
早期全 `-Os` 标量核加 trace 后超出 tile program memory，历史报告中的该阶段 trace
使用 `-Oz`。当前向量化 prepare 固定用 `-Oz`，update/finish 默认 `-Os`，后两者可通过
`RWKV_XDNA_RECURRENCE_OPT` 调整；编译配置分别记录所用选项。不同版本/选项的 trace
不能互作性能证据。
2026-10-03 将 recurrent stage 从 `-Oz` 改为 `-Os` 后，孤立阶段从1.143降至0.994 ms。
整模型 A/B/B/A 中 BF16 为233.06→229.84 ms/token，INT8 FFN 为218.39→214.93 ms/token；
两种配置均保持128步逐位回归。设备 cycles、编译内存限制和验证范围见
[Recurrent stage 调优进度](../reports/rwkv7-recurrence-progress-2026-10-03.json)。

融合阶段 oracle 可运行
`rwkv-recurrence-stage-test KERNEL_ROOT 300 --fused-value`，检查连续状态更新、原始 value
输入保留和 BO guards。诊断程序切换成本时，先单独编译 `tools/compile/rwkv7_value.py`，
再用 `--alternate-value` 或 `--fused-value-alternate` 让独立 value 程序在每次计时前运行。
`mean_us` 仅含被测 recurrence 提交，`mean_pair_us` 包括用于切换的 value 提交；
`mean_wait_us` 包含调度、程序切换、DMA 和计算，不能称为纯 PDI 加载或纯计算时间。
独立 value 产物仅用于这项诊断，不再是生产图依赖。

融合后的7-lane 图有空余 trace 输出通道，可以保留生产并行布局采样：

```bash
RWKV_XDNA_TRACE_BYTES=16777216 \
RWKV_XDNA_KERNEL_DIR="$PWD/build/kernels/value-recurrence-trace" \
MLIR_AIE_KERNEL_SOURCES=third_party/mlir-aie \
  .venv/bin/python tools/compile/rwkv7_value_recurrence.py
./build/test/rwkv-recurrence-stage-test build/kernels/value-recurrence-trace \
  1 --fused-value reports/runs/value-recurrence-trace.txt
```

解析时使用 `fused-value-recurrence-stage/design.prj/input_with_addresses.mlir`。
只采样第一条 lane 的 prepare/value、state update、finish 三个 core，各有5个 head 调用。
2026-10-03 的 A/B/B/A 测量中，融合使 BF16 decode 从229.37降至216.85 ms/token，
INT8 FFN 从215.12降至202.41 ms/token。两种模式均通过128步同精度逐位回归；
数据搬运契约、trace、实验方案和限制见
[Value/recurrence 融合进度](../reports/rwkv7-value-recurrence-progress-2026-10-03.json)。

当前 prepare/value sigmoid 将相同的 FP32 range reduction 和七阶指数多项式按32 lane
执行。低于 `-69` 的指数输入保留原来的核内标量处理，维持 small-normal/subnormal 行为；
归一化统计、标量 reciprocal、FP32 recurrent state 更新及多项式系数保持不变。
prepare/update/finish 分别编译，避免将无关代码带入每个 core，并删除了无调用者的旧 DMA
入口。独立检查直接比较原标量实现与向量实现的输出 bits，另用 FP64 oracle 和 BO guards：

```bash
RWKV_XDNA_KERNEL_DIR="$PWD/build/kernels/vector-math-test" \
MLIR_AIE_KERNEL_SOURCES=third_party/mlir-aie \
  .venv/bin/python tools/compile/rwkv7_vector_exp_test.py
cmake --build --preset test
./build/test/rwkv-vector-exp-test build/kernels/vector-math-test
```

2026-10-03 的 A/B/B/A 对比中，BF16 decode 为216.91→207.68 ms/token，INT8 FFN 为
202.52→192.78 ms/token。两种模式保持128步同精度逐位回归；采样范围、代码内存取舍
和 device trace 见 [向量 prepare 进度](../reports/rwkv7-vector-prepare-progress-2026-10-03.json)。

后续分区 trace 显示 sigmoid/decay 是 prepare 中最大的已标记算术区段。
`RWKV_XDNA_PREPARE_TRACE_REGION=1|2|3|4` 分别只标记 key norm、sigmoid/decay、
prepare 输出运算、value sigmoid/blend；必须同时启用 `RWKV_XDNA_TRACE_BYTES`，
区域4仅支持 fused-value 图。默认0仍标记完整的 prepare/update/finish。
这些独立 trace build 的时间不能直接相加作为完整阶段时间，且不包含 FIFO acquire 和
未标记的 arena copy。命令和测量见 [prepare 分区 profile](../reports/rwkv7-prepare-region-profile-2026-10-03.json)。

根据该 profile，将 sigmoid 最后的单 lane vector multiply 合并成32 lane multiply，
保留每个元素的原始 native scalar reciprocal。区域2从72,349降至63,923 cycles/head，
区域4从34,979降至30,766；孤立阶段约3.6%加速，整模型 A/B/B/A 的 BF16 为
202.743→201.874 ms/token（0.43%），INT8 FFN+output 为185.481→184.825（0.35%）。
9,216个 scalar/vector 数值比较和三种精度模式的128步逐位回归均通过，BO、提交与
host 流量不变。详见 [sigmoid batching 进度](../reports/rwkv7-sigmoid-batch-progress-2026-10-03.json)。

另测 FP64 sqrt 在相邻 iterate bits 相同时提前终止：81,920个设备逐位比较和
INT8 FFN+output 的128步回归均通过，但孤立 recurrence 仅约0.44%改善，整模型
A/B/B/A 未证明稳定加速（进程间波动大于平均差异）。该方案已撤回，保留原七次迭代。
证据见 [sqrt fixed-point 实验](../reports/rwkv7-root-fixed-point-rejected-2026-10-03.json)。

测试程序、阶段 oracle 和回归命令见 [验证说明](validation.md)，编译入口见
[构建与测试](build.md)。

BF16 整模型与旧纯 FP32 轨迹不是逐值等价；旧跨精度逐元素验收未通过的事实仍保留在
[历史性能报告](../reports/rwkv7-optimization-summary-2026-10-03.json)。本次要求并检验的
是同 BF16 配置整理前后的正确性和性能，不放宽误差阈值，不以 argmax 一致代替数值检查。
128步回归也不代表已经验证完整25600上下文。

## 实验性 INT8 ChannelMix

`--weights int8-ffn` 将每层 ChannelMix 的 key/value 矩阵量化为 W8A16；attention、
低秩分支和 vocabulary head 暂时保持 BF16。这里的 A16 为 BF16，不是 IEEE FP16。
其余数值路径保持原实现：FP32 输出、矩阵累加和 WKV state，部分归一化统计使用 double。
本模式尚未完成完整语料和
长上下文质量验收，不能将阶段 oracle 通过或短序列 top-1 一致视为生产质量通过。

```bash
cmake --preset release
cmake --build --preset release
cmake --build --preset release-int8-ffn-kernels
./build/release/rwkv-cli --model "$MODEL" --weights int8-ffn \
  --prompt 'Question: What is the largest planet in the Solar System? Answer:' \
  --top-k 1 --max-tokens 32
./build/release/rwkv-bench "$MODEL" build/kernels/rwkv7-bf16 16 32 3 --int8-ffn
```

INT8 模式显式要求 `int8-channel-mix` 产物；缺失或 ABI 不一致会报错，不会回退到 BF16。
默认目录沿用历史名称 `rwkv7-bf16`，可同时保存两种 ChannelMix 产物。C++ 接口第三个参数
使用 `WeightMode::Int8FFN`；默认仍为 `WeightMode::BFloat16`。

量化遵循 CUDA RWKV 的逐输出通道对称 `[-127,127]` 算法：先将 `max_abs/127` 经原有
checkpoint I/O FP16 转换取整，最小 scale 为 `2^-24`，再 nearest-even 取整数代码并
饱和。原有 FP16 scale 转换的半值向上舍入行为也保留；它不等同于 scale 的 nearest-even。
scale 以精确展开的 FP32 存入每个 tile 尾部。权重仅在构图时量化一次，设备端只在核内
展开当前 INT8 tile，完成整行点积后融合乘 scale；不产生 DDR 中的完整反量化矩阵，
不量化 activation 或 recurrent state。每个 ChannelMix 权重 BO 为34,078,720字节，
BF16 为67,108,864字节；每 token 的 host 上传/下载字节和提交次数不变。

2026-10-03 INT8 专用图将 activation 的 FP32→BF16 nearest-even 转换移到两个
producer core，各转换一次，再经 memory tile 广播给矩阵核；保留 FP32 累加和原有
reduction tree。每阶段转换元素数从2,097,152降到10,240。孤立 INT8 阶段约
1.388→1.298 ms，整模型 A/B/B/A 为193.091→192.173 ms/token（约0.48%）；
这是小幅端到端改善，尚未达到硬件极限。同样改动使 BF16 变慢，因此 BF16 保留原图。
两种模式128步 logits/state及1,116个检查向量均逐位一致。
实验、被拒绝方案和 trace 见 [activation 转换进度](../reports/rwkv7-channel-activation-progress-2026-10-03.json)。

独立阶段测试及整模型误差工具：

```bash
cmake --build --preset test
./build/test/rwkv-channel-mix-int8-test build/kernels/rwkv7-bf16 build/kernels/rwkv7-bf16
./build/release/rwkv-accuracy "$MODEL" build/kernels/rwkv7-bf16 \
  build/kernels/rwkv7-bf16 512 tests/data/precision_smoke.txt --text > accuracy.jsonl
./build/release/rwkv-quantization-audit "$MODEL" > weight-errors.jsonl
```

`rwkv-accuracy` 先运行并释放 BF16 图，再运行 INT8 图，避免同时持有两套图触发驱动的
context 数量限制。两种模式使用同一 token 流；输出首 token 全节点误差、每步 logits
误差、KL、top-1、状态检查点和 reset/branch 检查。提供文本或 token ID 文件时还报告
next-token NLL/perplexity 差异。追加 `--checkpoint-nodes` 可在第1、8、32和最后一个
token 输出全节点误差，帮助观察量化误差随 recurrent context 的传播；重合检查点只记录一次。
状态检查点同时分别报告每层 attention shift、FFN shift 和 FP32 matrix 的误差。
诊断 trace 会增加设备读取和同步，因此这些运行不能用于性能测量。节点误差包含上游传播，
不能单独解释为该节点的局部量化误差。
512-token 实测覆盖3,840个节点比较和288个分层状态比较；已有每步输出误差指标完全一致。
末步整体 state relative L2 为1.074%，最差单层 attention shift 为2.552%，
matrix 为1.338%；整体指标不能替代分层检查。详见
[INT8 检查点误差报告](../reports/rwkv7-int8-checkpoint-accuracy-2026-10-03.json)。
`precision_smoke.txt` 是原创中英混合测试样本，仅用于
smoke 检查，不代表完整质量语料。未指定文件时使用固定控制 token 后接 BF16 贪心序列，
这种负载不能作为语言质量验收。工具使用当前目录的 `assets/rwkv_vocab_v20230424.txt`。
`rwkv-quantization-audit` 不调用 NPU，逐矩阵分离 BF16 checkpoint 存储舍入、理想
FP32 scale 下的 INT8 重建误差，以及 FP16 scale 舍入造成的权重/代码变化。

2026-10-03 的 A/B/B/A 测量中，INT8 FFN 将整模型 decode 从232.69降至218.24 ms/token
（约6.21%），常驻 BO 减少792,723,456字节。512 token smoke 文本 perplexity 从3.6261
变为3.6343（增加约0.23%）；BF16 的128步逐位回归保持通过。完整测量范围、误差归因、
产物校验值和未完成工作见 [INT8 FFN 进度](../reports/rwkv7-int8-ffn-progress-2026-10-03.json)。
当前设备核使用反量化后的 BF16 MAC，不应将其性能称为原生 INT8 MAC 吞吐。

另一个显式实验模式 `--weights int8-ffn-output` 在 FFN INT8 的基础上，将24层
attention output projection 也改为相同 per-output W8A16 量化。每个权重 tile
在核内展开，8个 K tile 完成 FP32 累加后乘行 scale，再执行 FP32 residual；不增加
提交、BO 或 activation 量化边界。编译和验证命令：

```bash
cmake --build --preset release --target kernels-release-int8-ffn-output
./build/test/rwkv-projection-residual-int8-test build/kernels/rwkv7-bf16 build/kernels/rwkv7-bf16
./build/release/rwkv-bench "$MODEL" build/kernels/rwkv7-bf16 8 24 2 --int8-ffn-output
./build/release/rwkv-accuracy "$MODEL" build/kernels/rwkv7-bf16 build/kernels/rwkv7-bf16 \
  512 tests/data/precision_smoke.txt --text --checkpoint-nodes --int8-ffn-output
./build/release/rwkv-quantization-audit "$MODEL" --attention-output
```

2026-10-03 A/B/B/A 实测相对 FFN-only INT8 从189.101降至186.747 ms/token
（1.24%，5.355 tokens/s）；resident BO 从2,124,396,544降至2,025,306,112字节。
仍为123 runs/token、291 root BO，每步 host 上传/下载8192/262144字节。
512-token smoke 的 top-1 agreement 为98.83%，mean KL 从 FFN-only 的0.001043
升至0.001201。该小语料 perplexity 略低不证明质量改善；仍未完成生产质量验收。
阶段 FP64 oracle、guard、原有两种模式的128步逐位回归均通过。详见
[INT8 output projection 进度](../reports/rwkv7-int8-output-progress-2026-10-03.json)。

本次整理的同精度回归、阶段 oracle、性能和限制见
[清理验证摘要](../reports/rwkv7-cleanup-summary-2026-10-03.json)。
