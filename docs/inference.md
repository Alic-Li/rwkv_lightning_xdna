# RWKV-7 XDNA2 推理

运行路径为 `C++17 → XRT → amdxdna → NPU`。CPU 负责权重加载、embedding 查表、
tokenizer、sampler 和调度；NPU 执行模型算术，无 CPU 算子回退。Python 只用于离线
编译和验证。显式 `--backend cpu` 提供 FP32 参考。

模型加载器从权重读取层数、通道数、头数、FFN 和低秩矩阵形状。
BF16 tiled decode 接受 512≤C≤2048（256 的倍数）、head_size=64、
统一 FFN hidden≤8192（256 的倍数）、词表为128的倍数、低秩维度≤256。
低秩矩阵补零到256；其逻辑维度仍取自权重。已验证形状为 C=2048 / FFN=8192，
以及翻译专用 0.4B 的 C=1024 / FFN=4096 / 16 heads / 24层 / 词表65536。
INT8 和 batch2/chunk4 保持原来的 C=2048 / FFN=8192 支持范围。
PTH/safetensors 的 FP32、FP16、BF16 存储统一加载为主机 FP32，再一次性打包设备权重。

## 权重与数值

只有两种权重模式，默认 BF16：

| CLI / C++ | FFN | 其他矩阵 | 状态与残差 |
|---|---|---|---|
| `--weights bf16` / `WeightMode::BFloat16` | BF16 权重和乘法输入，FP32 累加 | BF16 | FP32 |
| `--weights int8-ffn` / `WeightMode::Int8FFN` | W8A8 streaming | BF16 | FP32 |

INT8 只量化 FFN key/value：每输出行对称权重码 `[-127,127]`，权重 scale 先舍入为
FP16 再展开为 FP32；activation 按256元素动态量化，nearest-even。
INT32 block dot 转成 FP32 后按 activation/weight scale 缩放，再按原 K 顺序累加。
W1 的256输出块直接产生 activation packet，W2 随后消费；W2 权重固定采用四 worker
K-major 排布，加载时打包一次，replay 不在 CPU 量化或搬运 activation。
归一化、ReLU²、WKV state/update 和残差保留原来的 FP32 算术与统计量求值顺序。

旧 W8A16、INT8 attention output、grouped/pair/local 等变体已删除。
`--weights int8-ffn` 和 `WeightMode::Int8FFN` **现在指 W8A8**；旧 W8A16 产物会被拒绝。
BF16 保持默认。W8A8 已通过阶段 oracle、128-step 状态/输出回归和短文本筛选，
尚未完成独立语料及长上下文质量评估。历史性能约178 ms/token，W8A16 约184 ms/token；
这是同机对照结果，不是所有输入的速度保证。证据见
[W8A8 streaming 报告](../reports/rwkv7-w8a8-stream-2026-10-04.json)。

## 构建与运行

先完成 [环境部署](environment.md)，在仓库根目录执行：

```bash
export PATH="$PWD/.venv/bin:$PATH"
cmake --preset release
cmake --build --preset release-full
./build/release/rwkv-cli --model "$MODEL" --prompt 'Hello' --max-tokens 32

# 同一产物目录中补齐唯一 INT8 方案。
cmake --build --preset release-int8-ffn-kernels
./build/release/rwkv-cli --model "$MODEL" --weights int8-ffn \
  --prompt 'Hello' --top-k 1 --max-tokens 32
```

默认 `--backend npu --decode resident --prefill decode`，产物目录
`build/kernels/rwkv7-bf16`。自定义产物目录用离线编译器的 `--output DIR`，
运行时用 `--kernel-dir DIR`。主机构建不会隐式编译 kernel；首次运行需完成对应 kernel target。

CLI 用 `--prompt-file FILE` 读取较长输入；`--tokens 1,2,7,9 --dump-logits FILE`
可导出逐 token FP32 logits。CPU 参考支持 `--decode graph|eager` 和
`--prefill decode|sequence`。NPU 只支持 resident decode。
CLI 分开报告权重加载、建图、prefill、TTFT 和后续 decode；首个生成 token 来自 prefill。

## 按模型形状编译与并发翻译

首次使用不同模型形状，在仓库根目录执行：

```bash
export PATH="$PWD/.venv/bin:$PATH"
cmake --build --preset release
export MODEL=/home/alic-li/rwkv_weights/RWKV_v7_G1d_0.4B_Translate_ctx4096_20260607.pth
./build/release/rwkv-cli --model "$MODEL" --inspect-model
.venv/bin/python tools/compile/rwkv7_optimized.py --model "$MODEL" \
  --output build/kernels/rwkv7-bf16

./build/release/rwkv-cli --model "$MODEL" --backend npu \
  --kernel-dir build/kernels/rwkv7-bf16 --weights bf16 \
  --prompt-file /home/alic-li/work_space/translate.txt --unescape-prompt \
  --concurrency 4 --top-k 1 --max-tokens 128
```

离线编译器调用 C++ 权重加载器读取实际形状，写入 `c1024-h4096-v65536/`
子目录。运行时自动按形状选择此目录，也可直接指定子目录。旧默认形状的根目录
产物继续可用。不同层数使用同一组内核；图按实际层数展开。

`--concurrency N` 把同一条 `--prompt` 或 `--prompt-file` 输入复制给 N 个请求。
每个请求拥有独立 DecodeGraph、设备 BO、递归状态、sampler 和 penalties。
各请求线程同步启动，独立完成 prefill，再同步开始生成；建图和权重上传不计入吞吐。
没有跨请求的推理互斥锁，不会循环完成一个请求后再启动下一个。
输出按 `[request 0]` 等分组，避免字符交错。诊断模式的 logits 分别写入
`FILE.request-0` 等文件。CPU 并发参考只支持 graph/decode。

统计包含各请求 prefill/TTFT、总 prompt token/s、总生成 token/s、decode forward/s
和端到端生成 token/s。**总吞吐用所有请求的 token 数除以同一段墙钟时间**，
不会相加各请求的 token/s。首个生成 token 使用 prefill logits，故生成数量和
后续 forward 次数分别报告。EOS 使请求提前结束，`--max-tokens` 是上限。

运行时报告 `peak` 个已提交但尚未完成的 NPU 命令，验证提交发生重叠。
这表示并发请求/异步设备队列；同一 xclbin 的 Session 共享硬件 context，
实际计算 tile 的执行顺序由 XRT/驱动调度。此实现不声称多个请求在不同 tile
上同时计算，也不保证吞吐随并发数增长。

`--unescape-prompt` 是可选的输入转换：把字面量 `\n`、`\r`、`\t`、`\\`
转为对应字符。本次 `translate.txt` 末尾含字面量 `\n\nChinese:`，
需要此选项才形成模型预期的实际换行；文件本身不被修改。

## Prefill

默认逐 token prefill 与 decode 使用相同状态转移。BF16 另有两个已验证调度：

| 模式 | 工作方式 | 编译 target |
|---|---|---|
| `decode` | 每 token resident replay | `kernels-release` |
| `batch2` | 按层处理两 token，共享矩阵权重流 | `kernels-release-prefill-batch2` |
| `chunk4` | 四 activation slots，成对 attention/WKV，四 token FFN | `kernels-release-prefill-chunk4` |

```bash
cmake --build --preset release --target kernels-release-prefill-chunk4
./build/release/rwkv-cli --model "$MODEL" --prefill chunk4 \
  --prompt-file prompt.txt --top-k 1 --max-tokens 32
```

batch2/chunk4 只返回最后一个 prompt token 的 logits，不兼容逐 token dump 或 trace。
WKV 按 token 顺序更新 FP32 state；不足一组的尾 token 使用 resident decode。
chunk4 包含 batch2 产物，并检查 stride variants 的 PDI 相同后共享 xclbin。
BF16 chunk4 已有约33%的 prefill 吞吐收益及逐位回归证据，见
[验证记录](../reports/rwkv7-throughput-handoff-2026-10-03.json)。
INT8 仅支持 `--prefill decode`，不隐式切换量化方案。

## C++ 状态接口

```cpp
using namespace rwkv::inference;
auto reference = cpu_backend();
Weights weights(model_path);
Model model(weights, *reference);
DecodeGraph graph(weights, kernel_dir, WeightMode::Int8FFN);
graph.load_state(model.initial_state());
auto logits = graph.replay_resident(token);
auto checkpoint = graph.export_state();
graph.load_state(checkpoint);  // reset / branch
```

Weights 必须比 graph 活得更久；一个 graph 实例只允许串行调用。
BF16 batch2/chunk4 构造时传 `PrefillMode::Batched2|Chunked4`，再调用
`prefill_resident(tokens)`；空输入不修改状态。之后可继续 `replay_resident`。

普通 replay 只上传 embedding（8192 bytes）并下载 logits（262144 bytes），
FP32 recurrent state 留在设备。`replay(token, State&)` 则显式同步完整状态。
24层正常 decode 为100 runs/token，诊断路径为123；这是可复用 XRT run 的逐阶段提交，
不是一次硬件图提交。`set_trace` / `set_projection_trace` 与
`RWKV_XDNA_PROFILE=1` 用于诊断，不用于性能测量。

## 模块边界

| 模块 | 职责 |
|---|---|
| `model`, `weights`, `backend`, `ops` | checkpoint、模型数学与 CPU 参考 |
| `graph` | 记录逻辑节点、公共 API 和状态绑定 |
| `artifacts` | 固定 decode/prefill ABI 与数值契约检查 |
| `weight_layout`, `quantization` | BF16 排布、唯一 W8A8 权重打包 |
| `resident_buffers` | session、BO 分配、activation arena 视图 |
| `resident_plan` | 逻辑节点映射至固定设备阶段、decode fusion |
| `graph_execution` | replay、状态上传/导出、trace |
| `prefill_plan`, `chunked_plan` | BF16 batch2/chunk4 调度 |
| `src/runtime/session.cpp` | XRT session、buffer、run 生命周期 |
| `tools/compile/rwkv7_optimized.py` | 唯一离线生产编译清单 |

## 产物排障与迁移

`Missing production artifact: upstream-norm` 表示只编译了主机或用了错误目录，
执行 `release-kernels`；INT8 还需 `release-int8-ffn-kernels`。
ABI 不匹配应重新编译，不要手工修改 config 绕过检查。
实验标记、mode-worker、trace ABI、旧量化布局都会被拒绝。

本轮 chunk4 产物改名为 `bf16-prefill-ffn-b4-projection-input`，旧名字不再加载。
升级后重新编译相应 target，建议先输出至新目录并验证。
历史报告保留用于追溯，报告中的旧 CLI、编译器文件名与实验命令不再是当前接口。
IEEE FP16 不是 BF16；该目标精度核查见
[精度能力报告](../reports/rwkv7-precision-capability-2026-10-03.json)。
