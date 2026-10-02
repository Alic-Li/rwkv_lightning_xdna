# RWKV-7 C++ / XDNA 推理

默认 `--backend npu`：**embedding 查表、tokenizer、sampler 在 CPU，模型的线性投影、
LayerNorm/GroupNorm、混合、门控、激活、残差、WKV 和输出 head 全部在 NPU 计算**。
CPU 还负责加载权重、布局整理、传输、图调度和有限值检查，没有静默 CPU 算子回退。
这是 FP32 实现；新增 `--decode resident` 将 decode 的中间激活和预排布投影权重保留在
NPU 可访问 BO 中，并跨 token 保留 FP32 recurrent state。优化路径使用阵列并行与阶段融合，
仍逐个提交预绑定 run；prefill 的 sequence 路径保持原实现。

## 构建和运行

环境部署见 [environment.md](environment.md)。本机 memlock 已按 FastFlowLM 的 Linux 指南
解决，2026-10-02 复查软/硬限制均为 `unlimited`；记录及配置见该文档第 4 节。
从仓库根目录运行：

```bash
test ! -f /opt/xilinx/xrt/setup.sh || source /opt/xilinx/xrt/setup.sh
./tools/bootstrap.sh
export PATH="$PWD/.venv/bin:$PATH"
cmake --preset dev
cmake --build --preset dev
# 离线编译基础与融合 kernel，默认保留原 FP32 算术；Python 不调用 NPU。
.venv/bin/python tools/compile/rwkv7_optimized.py
# 已有基础产物时可加 --skip-base。不要混用不同算术模式的融合产物。

./build/host/rwkv-cli \
  --model /home/alic-li/rwkv_weights/rwkv7-g1k-1.5b-20260930-ctx25600.pth \
  --vocab assets/rwkv_vocab_v20230424.txt \
  --backend npu --kernel-dir build/kernels/rwkv7-full \
  --prefill decode --decode resident \
  --prompt 'The capital of France is' \
  --max-tokens 32 --top-k 1
```

已读取该真实模型：24 层、2048 通道、32×64 head、8192 FFN、65536 词表，
1,527,668,736 个参数，原始 BF16 权重。实测能够续写 ` Paris.`。
完整验证记录见 [报告](../reports/rwkv7-validation-2026-10-02.json)。

也可用 `--prompt-file prompt.txt`。CLI 不自动添加聊天模板，需匹配具体模型训练格式。
EOS token 0 或达到 `--max-tokens` 停止；标准输出为生成文本，信息/错误在标准错误。
`--top-k 1` 为贪心，其他采样参数见 `--help`。

运行时只需要 C++ 可执行文件、XRT、词表、模型和 `--kernel-dir` 完整目录，不需要 Python、
PyTorch、CUDA 或相邻仓库。权重支持原始 state_dict 的 ZIP PTH/PT 和单文件 safetensors，
接受 BF16/F16/F32，内部转为 FP32。PTH 解析复用用户 CUDA 项目的纯 C++ loader，
本项目对连续存储做直接转换，避免生成不用的精度副本；非连续 stride 仍支持。
不支持量化 checkpoint、旧式非 ZIP PTH、任意 Python 对象或分片索引。
约 1.5B 参数的 FP32 权重需要约 6 GB 主机内存，另需加载临时空间和设备缓冲区。

## Decode：代码内图记录与重放

`rwkv::inference::DecodeGraph` 在构造时记录固定形状的算子节点、缓冲区编号、权重引用、
拓扑依赖和状态回写位置；不执行样例输入，也不保存某一次运行的张量快照。
`replay(token, state)` 更新 embedding/状态输入，重放相同计划，完整 token 成功后才提交状态。
固定图和主机缓冲区池在多个 token、不同请求以及分支状态间复用；同一图实例串行调用。

CLI 默认仍为 `--decode graph`，可选 `--decode eager` 对照，或显式开启
`--decode resident`。三种模式都复用 `Session` 的 XRT 资源；`graph` 逐节点调用后端，
中间结果通过主机传递，`resident` 使用下节的设备缓冲区计划。
**两种图模式都不是整图一次硬件提交**，各节点仍可能拆成多个 NPU run。

```cpp
rwkv::inference::Weights weights(model_path);
auto backend = rwkv::inference::full_npu_backend(kernel_directory);
rwkv::inference::Model model(weights, *backend);
rwkv::inference::DecodeGraph graph(weights, *backend); // record once

auto state = model.initial_state();
auto prompt_logits = model.prefill(prompt_token_ids, state);
auto logits = graph.replay(next_token, state); // updated token and state
```

Weights/backend 必须比 Model/DecodeGraph 活得更久。State 是显式对象，可复制用于分支，
通过 `initial_state()` 清零。Model eager/prefill 执行异常后应丢弃该请求状态。

## Resident decode：固定 BO、融合与设备状态

```cpp
backend->release_device_cache(); // prefill 后释放空闲后端 context
rwkv::inference::DecodeGraph graph(weights, *backend, kernel_directory);
graph.load_state(state);                      // reset 或接续 prefill，一次上传
auto logits = graph.replay_resident(token);   // 只传 embedding / logits
state = graph.export_state();                 // 仅在 checkpoint / 分支时读回
```

构造时分配固定 arena、预排布并上传权重、创建并绑定全部 run。token 之间复用这些资源，
state 的 `[head,key,value]` FP32 布局和地址保持固定；没有逐 token 的 BO/command 构建。
`replay(token, State&)` 兼容接口仍每次同步完整状态，在成功后更新调用者 State。
混用接口时，重新 `load_state` 再调用 `replay_resident`。图实例仅允许串行使用；
设备执行失败后须重建，不能继续使用可能部分更新的设备状态。

节点之间没有应用层中间张量读回。`HOST_ONLY` BO 是 NPU 可访问的共享系统 DDR，
不是 AIE SRAM；权重仍需通过 DMA 送入阵列。arena 合并了分配，尚未实现完整的
临时张量生命周期复用。960 个逻辑节点、1492 个逻辑 buffer 仍保留作依赖和诊断记录，
不是实际 run 数和根 BO 数。`set_trace` 可读回每个逻辑节点和 WKV state；诊断传输不计入性能模式。

当前优化布局针对本模型 C=2048、32×64 heads、FFN=8192；其他受支持的小形状使用
基础 kernel。缺少可用 GEMV 形状时报错，没有静默 CPU 算子回退。

### AIE 阵列与融合

- 普通投影：8 workers 并行输出行，K 维累加留在核内；65536 词表 head 使用
  32 workers、每次8192输出行。32 workers 在小投影/FFN 上实测更慢，因此不全局启用。
- 六路 attention mix 合并一次，并更新 shift；FFN mix 同时更新 shift。
- 低秩投影 → activation → 投影在一次 dispatch 的多 worker 数据流中完成。
- key normalize、alpha、decay、key scaling 合并准备阶段；32 个 WKV heads
  用8个 workers、每个连续4 heads；state 原位写回。
- GroupNorm、RKV residual、gate 合并结束阶段；value residual 分8 workers。
- FFN key projection 与 ReLU² 合并，后接 value projection。RWKV-7 此模型的 FFN
  不额外引入其他 RWKV 版本的 receptance 公式。

这些是阶段级融合，**尚未达到一层一个 NPU program，也未达到整 token 一次提交**。
当前固定图复用 prepared `xrt::run`，没有启用原生 runlist。

### 严格默认与官方 kernel 复用

默认离线编译设置 `RWKV_XDNA_EXACT=1`，保持原参考 FP32 表达式、WKV 累加顺序，
以及 exp/tanh/norm 的原求值方法。`exact` 表示保留参考算术，不能推导所有输入或
CPU/NPU 都逐 bit 一致。原有 `2e-6 + 2e-5*abs(reference)` 逐元素验收阈值没有放宽。

`third_party/mlir-aie/aie_kernels/transformer/layer_norm_f32.cc` 已通过包装实例化
FP32 affine 输出（上游公开 affine 包装输出 BF16），单核数值/guard 测试通过；
`common/scalar_f32.h` 的原生标量辅助也已复用。上游文件保持不变，版本和校验值见
[third_party/SOURCES.json](../third_party/SOURCES.json)。

这些原生算法与更快的向量/FP32非线性组合在真实模型上有微小累计舍入差异，
没有通过旧严格逐元素验收，因此仅在显式 `rwkv7_optimized.py --native-fp32` 实验模式启用。
**不要用该模式的更快计时作为严格版本的结果。** 恢复默认须重新运行编译脚本（不加该选项）。
运行时检查融合产物的算术标记，拒绝部分重编译造成的混用。官方 BF16/INT8/量化 kernel
不能直接当成同精度 FP32 替换；W8A16/INT8 尚未接入。

### 本机严格版本结果（2026-10-03）

真实模型相同，串行先跑原逐节点 NPU 图4个 token `1,2,7,9`，释放空闲 context，
再跑 resident 图。第一个 token 预热，后三个计时；状态校验读回排除在 decode 计时外。

| 项目 | 本轮结果 |
|---|---:|
| 原逐节点图 / 新 resident 平均 decode | 7.5926 / 1.0150 秒/token |
| 相对原逐节点图加速 | 7.48× |
| 相对旧 resident 5.908 秒/token | 约5.82×（跨轮比较） |
| 每 token runs | 4280 → 488 |
| 数据根 BO 数 / 总字节 | 584 / 5,796,215,936 |
| 每 token 应用层上传 / 下载 | 8,192 / 262,144 字节 |
| 四步最终 logits / 全部 state 最大误差，相对原 NPU | 0 |
| 图准备时间，不含模型加载 | 3.339 秒 |

旧 resident 总数据 BO 为5,793,261,632字节；本轮主要减少分配碎片、提交和传输，
没有减少 FP32 权重体积。传输统计为应用有效字节，不含驱动缓存粒度或片上 DMA；
BO 不含 XRT 内部资源，也不含主机 FP32 权重。CLI 单独报告 sequence prefill 后的
图准备耗时，避免把一次性建图算进稳态生成。短提示可用上面的 `--prefill decode`；
sequence prefill 尚未优化。详细验证见 [本轮报告](../reports/rwkv7-decode-2026-10-03.json)。

原命令 `--prefill sequence` 实测：prefill39.382秒、随后建图3.304秒、32输出token生成31.454秒
（31次decode推进；加上建图共34.758秒，对照用户原183.526秒）。四组知识题CPU/NPU
文本逐字一致，decode-prefill路径的32输出token约31.4秒；France问答在15token遇到EOS。

同一 checkpoint、相同 prompt 和 greedy 设置下，CPU 也会复读原 France 句子。
CPU/NPU 短生成一致不等于已经改善模型质量；CLI 不会用 repetition penalty 掩盖数值错误。
聊天格式需要匹配 checkpoint，本轮未验证长上下文或聊天质量。

### 性能目标与尚未完成的工作

50 INT8 TOPS 是 INT8 峰值，不是此 FP32 decode 的可用算力。扣除 embedding 查表后，
约1.39B活跃线性权重对应每 token 约2.8G乘加计数操作，理想 INT8 纯计算上界约18k token/s；
这忽略带宽、WKV、非线性、调度等，不能作为实际 decode 目标。

batch=1 每步大致读取5.6 GB FP32权重（W8约1.4 GB）。若有效权重带宽分别为
20/40/80 GB/s，则仅权重读取给出 FP32约3.6/7.1/14.3 token/s、W8约14/29/57 token/s。
这是带宽假设下的估算，并非本机带宽测量。当前约0.985 token/s **尚未接近已证明的硬件极限**。
后续仍需整层调度/共用 context、<100 runs、临时内存复用，以及保留 FP32 state 验证的
权重量化。16-worker WKV 拆分尝试超过 MemTile DMA 通道预算，未作为可用实现保留。

参考本地 FastFlowLM `b0d41a03411470373f849b25e3ce9356d716a483` 中的持久 BO、
固定 run 参数与 DMA 地址绑定接口；其本 checkout 的 Qwen3 forward 在预编译库里，
不能声称已读到完整实现。历史 runlist 探针限制见 [execution-graph.md](execution-graph.md)。

## BF16 主投影实验（2026-10-03）

488-run 历史基线实际是 **FP32**，不能标成 FP16。新实验使用 BF16 权重和
BF16 乘法、FP32 累加；当前低秩投影、激活和 recurrent state 仍为 FP32。
它也不是 IEEE FP16 或 BFP16，不使用 INT8/INT4、CPU/GPU arithmetic fallback。
上游 `linalg/mv_bf16.cc` 保持原样，RWKV 包装复用其四行 MAC/归约；权重加载时
一次性按现有 tile 顺序打包为 BF16。输入在 AIE 上以 nearest-even 转为 BF16。

```bash
export MLIR_AIE_KERNEL_SOURCES="$PWD/third_party/mlir-aie"
.venv/bin/python tools/compile/rwkv7_array.py --bf16
.venv/bin/python tools/compile/rwkv7_array.py --bf16 --k 2048 --rows 8192
.venv/bin/python tools/compile/rwkv7_ffn.py --bf16
cmake --build --preset dev
./build/host/rwkv-array-test build/kernels/rwkv7-full --bf16
./build/host/bench_xdna_fp16 build/kernels/rwkv7-full bf16 2048 2048
# 显式打开；仅 resident、C=2048、32 heads。
RWKV_XDNA_BF16=1 ./build/host/rwkv-cli --model "$MODEL" \
  --decode resident --prefill decode --prompt 'The capital of France is' \
  --max-tokens 32 --top-k 1
```

`bench_xdna_fp16` 当前覆盖主投影形状，实际精度由 `fp32|bf16` 参数及输出字段说明。
预热20次、计时30次，报告 min/median/max、主机 submit/wait、有效带宽、数值误差和
guard 检查。其时间包含 DMA 和调度，不能视为纯设备 kernel cycles。
`RWKV_XDNA_PROFILE=1` 另报告每类算子、每层、提交/等待、embedding/logits 传输时间。
wait 包含设备计算、DMA 和调度/PDI 开销，尚未被设备 trace 分拆。

首轮 2048×2048 GEMV 最小耗时 699→290 μs；两种 FFN 大形状约2586→952/962 μs。
128步固定输入/greedy对照（首步不计时）为1.0175→0.8908秒/token，仍488 runs/token。
这是精度切换实验，不能当成同精度FP32优化。详细条件见
[实验记录](../reports/rwkv7-bf16-projections-2026-10-03.json)。

数值验收区分三个问题：随机/边界单算子；实际设备输入上的独立CPU FP64点积；
以及不同精度/独立轨迹的模型漂移。前两者已通过，580次真实输入主投影的最大误差
为1.53e-5，沿用原单算子阈值。独立CPU BF16整图在第二层未通过旧FP32阈值；
未放宽阈值，不能声称这条整图严格检查通过。128步FP32/BF16对照的末步state相对L2
约0.0023。更多提示与数值契约仍需验证，因此该模式不作为默认路径。
自然语言 planet 提示的128步 teacher-forced 对照为127/128次 argmax一致，末步
state相对L2约0.00145；不能把它写成自由生成文本完全相同。

```bash
./build/host/rwkv-alignment-test "$MODEL" build/kernels/rwkv7-full --bf16-projections
./build/host/rwkv-precision-benchmark "$MODEL" build/kernels/rwkv7-full 128 \
  'Question: What is the largest planet in our solar system? Answer:'
```

`set_projection_trace` 会读回实际投影输入/输出，仅用于 oracle；其诊断传输和CPU
点积不属于生产推理路径。性能测试必须关闭此 hook。

后续实验可在独立目录编译，避免覆盖严格基线：

```bash
.venv/bin/python tools/compile/rwkv7_optimized.py --native-fp32 --bf16 \
  --bf16-rank --rkv --output build/kernels/rwkv7-bf16
RWKV_XDNA_BF16=1 ./build/host/rwkv-cli --model "$MODEL" \
  --kernel-dir build/kernels/rwkv7-bf16 --decode resident --prefill decode \
  --prompt 'The capital of France is' --max-tokens 32 --top-k 1
```

`--native-fp32` 仍为显式数值契约切换。该路径复用向量WKV、官方FP32 LayerNorm和
原生FP32辅助，BF16主投影下自然语言128步平均约0.419秒/token，末步state相对L2约
0.00151；与严格FP32参考的126/128次argmax相同。没有改变旧FP32验收阈值。
`--bf16-rank` 将低秩矩阵也改为BF16，并扩展实际输入oracle到1340次投影；
`--rkv` 合并三路投影，runs从488到440。32步同精度对照logits/state最大误差0，
平均约0.396→0.392秒/token（约0.9%，收益小，不能用run降幅代替延迟收益）。
测量与未完成的长序列验收见[融合记录](../reports/rwkv7-bf16-fusion-2026-10-03.json)。

同精度比较时可给 `rwkv-precision-benchmark` 设置
`RWKV_XDNA_REFERENCE_BF16=1` 和 `RWKV_XDNA_REFERENCE_KERNEL_DIR=基线目录`。
oracle的 `--bf16-all-projections` 使用BF16低秩参考；`--bf16-projections` 保留FP32低秩
参考。这些均为CPU诊断计算，不参与生产推理。

## Prefill：独立序列路径

`Model::prefill` 按最多 16 token 的 chunk 分层执行：先准备一层的投影和门控序列，
再调用独立的 `wkv7_prefill_fp32.cc`，最后完成该层的归一化、残差与 FFN。
每个 head 的 16 步 WKV 状态在一次 NPU dispatch 内连续更新；只在 chunk 边界传出状态。
尾部 padding 显式标记无效，不更新状态，接续 decode 可使用相同 `[head,key,value]` 布局。

这是稳定的顺序状态递推 prefill kernel，不是 CUDA 仓库中所有 chunk/GEMM/scan 优化的移植。
非 WKV 投影目前仍逐 token 使用 GEMV，尚未实现多 token GEMM 的权重复用。
`--prefill decode` 提供逐 token 参考路径，可结合 `--decode eager|graph|resident` 对照。
CLI 分块处理提示词，避免保存整个长提示词的所有 logits。

## 计算实现与约束

- 基础 GEMV：每次输出256行；优化阵列路径一次2048/8192行。内部16×256权重tile，K维累加在NPU内完成。
  默认离线编译 K=256/2048/8192，较短低秩维度以零 padding 补到 256。
- 通用算子：2048 元素一包，支持完整 2048 通道 LayerNorm 和 64 通道 GroupNorm。
- 基础 Decode WKV 一次一个 head；优化路径一次32 heads，状态始终为64×64 FP32，无隐含设备全局状态。
- Prefill WKV：每 head、每 chunk 一次提交，返回最终状态和每步输出。
- 线性投影使用 AIE FP32 向量计算，不将激活或权重再量化为 BF16。
- 门控 exp/tanh 在 AIE 上用 double 范围缩减及多项式求值；sqrt 使用 double Newton 迭代。
  它们与主机 libm 按数值容差对照，不宣称所有输入逐 bit 一致。

模型数学流程参考用户 `rwkv_lightning_cuda` 中的
`src/backend/rwkv7_fast_v4.cu`、`cuda/rwkv7_fast_ops_fp16.cu`、
`cuda/rwkv7_wkv_fp32_v2.cu` 和序列/state-passing 内核，未复制 CUDA 专用调度到 NPU。
每个 head 的主要递推（`a=-kk`, `b=kk*alpha`）为：

```text
decay = exp(-exp(-0.5) * sigmoid(w0 + tanh(xw @ w1) @ w2))
S_new = diag(decay) @ S_old + b[:,None] @ (a[None,:] @ S_old) + k[:,None] @ v[None,:]
y     = r[None,:] @ S_new
```

## 工程结构

| 位置 | 职责 |
|---|---|
| `include/rwkv/inference/model.hpp` | 权重、状态、模型及计算后端公共接口 |
| `include/rwkv/inference/graph.hpp` | 固定 decode 图记录/replay API |
| `src/inference/weights.cpp` | PTH/safetensors 适配与检查 |
| `src/inference/model.cpp` | eager decode / 分层 prefill 流程 |
| `src/inference/graph.cpp` | 固定图、依赖、状态绑定和 resident BO/run 计划 |
| `src/inference/ops.cpp` | CPU 对照基础算子 |
| `src/inference/backend.cpp` | CPU / NPU decode WKV |
| `src/inference/npu_ops.cpp` | 全 NPU 算子和 prefill 适配 |
| `kernels/rwkv/` | 自研 AIE C++ kernel |
| `src/runtime/session.cpp` | XRT 资源、持久 BO、无复制切片及 run 复用 |
| `tools/compile/rwkv7_resident.py` | resident 算子的 DMA join/strided gather/split 接口 |
| `src/utils/`, `apps/rwkv_cli.cpp` | 用户 tokenizer/sampler 适配及 CLI |

`--backend cpu` 为全部 CPU 对照；`--backend hybrid` 保留先前仅 WKV 上 NPU 的实现。
默认 `npu` 不会自动退回这些后端。模型缺少所需形状的编译产物时会报错。

## 验证

```bash
cmake --preset dev -DRWKV_XDNA_HARDWARE_TESTS=ON
cmake --build --preset dev
# 单独 recurrent 测试仍使用旧的独立 decode 产物：
.venv/bin/python tools/compile/rwkv7.py
ctest --test-dir build/host -R '^rwkv\.' --output-on-failure
.venv/bin/python tools/validation/rwkv7_reference.py --npu
./build/host/rwkv-prefill-test build/tests/rwkv7/f32.pth build/kernels/rwkv7-full
./build/host/rwkv-graph-test build/tests/rwkv7/f32.pth build/kernels/rwkv7-full
./build/host/rwkv-resident-test build/kernels/rwkv7-full
./build/host/rwkv-resident-benchmark \
  /home/alic-li/rwkv_weights/rwkv7-g1k-1.5b-20260930-ctx25600.pth \
  build/kernels/rwkv7-full
```

包括独立 FP64 参考、FP32/F16/BF16 PTH 与 safetensors、非连续 stride、非法文件、
非线性饱和/常量归一化、GEMV、状态 reset，以及 1/15/16/17/33 token prefill 与 decode
的全部 logits、最终状态和后续 decode 对照。图测试覆盖动态 token、清零、分支状态、
prefill 接续、图结构不变和非法输入不修改状态；同时覆盖 resident 模式。
设备缓冲区测试覆盖无主机读回的链式运算、嵌套切片、首/末 head、原位状态重放及边界哨兵。Python 仅做离线参考与编译，NPU 由 C++ 调用。

本轮额外验证：

```bash
./build/host/rwkv-drift-test "$MODEL" build/kernels/rwkv7-full
./build/host/rwkv-alignment-test "$MODEL" build/kernels/rwkv7-full
.venv/bin/python tools/validation/real_alignment.py --model "$MODEL"
.venv/bin/python tools/validation/knowledge_smoke.py --model "$MODEL"
```

`MODEL` 设置为上面的 checkpoint 路径。drift 比较原 NPU 与融合图全部960节点和WKV state；
alignment 比较CPU与NPU的8步逐层输出/state/greedy；独立FP64 oracle比较5个提示token的logits。
四组知识题每组最多32输出token，CPU/NPU串行生成。硬件测试请串行运行。
真实模型测试是短序列正确性和生成检查，不代表已验证完整25600上下文或长文本质量。
