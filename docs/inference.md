# RWKV-7 C++ / XDNA 推理

默认 `--backend npu`：**embedding 查表、tokenizer、sampler 在 CPU，模型的线性投影、
LayerNorm/GroupNorm、混合、门控、激活、残差、WKV 和输出 head 全部在 NPU 计算**。
CPU 还负责加载权重、布局整理、传输、图调度和有限值检查，没有静默 CPU 算子回退。
这是正确性优先的 FP32 实现；仍有大量主机传输和逐算子提交，尚不是性能优化版本。

## 构建和运行

环境部署见 [environment.md](environment.md)。从仓库根目录运行：

```bash
source /opt/xilinx/xrt/setup.sh
./tools/bootstrap.sh
export PATH="$PWD/.venv/bin:$PATH"
cmake --preset dev
cmake --build --preset dev
# 离线编译 decode、prefill、通用算子和 GEMV；不从 Python 调用 NPU。
.venv/bin/python tools/compile/rwkv7_full.py

./build/host/rwkv-cli \
  --model /home/alic-li/rwkv_weights/rwkv7-g1k-1.5b-20260930-ctx25600.pth \
  --vocab assets/rwkv_vocab_v20230424.txt \
  --backend npu --kernel-dir build/kernels/rwkv7-full \
  --prefill sequence --decode graph \
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

CLI 默认 `--decode graph`，可选 `--decode eager` 对照。底层 `Session` 复用 XRT BO、
`xrt::run` 和参数绑定。当前图执行器逐节点调用后端，各节点可能拆成多个 NPU run。
**它不是一次硬件提交的 CUDA Graph 等价实现**：中间结果目前仍通过主机缓冲区传递，
整个模型的权重、激活和状态尚未常驻设备；不能据此宣称已经消除了提交或传输开销。

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

## Prefill：独立序列路径

`Model::prefill` 按最多 16 token 的 chunk 分层执行：先准备一层的投影和门控序列，
再调用独立的 `wkv7_prefill_fp32.cc`，最后完成该层的归一化、残差与 FFN。
每个 head 的 16 步 WKV 状态在一次 NPU dispatch 内连续更新；只在 chunk 边界传出状态。
尾部 padding 显式标记无效，不更新状态，接续 decode 可使用相同 `[head,key,value]` 布局。

这是稳定的顺序状态递推 prefill kernel，不是 CUDA 仓库中所有 chunk/GEMM/scan 优化的移植。
非 WKV 投影目前仍逐 token 使用 GEMV，尚未实现多 token GEMM 的权重复用。
`--prefill decode` 提供逐 token 参考路径，可结合 `--decode eager|graph` 对照。
CLI 分块处理提示词，避免保存整个长提示词的所有 logits。

## 计算实现与约束

- GEMV：每次输出 256 行，内部 16×256 权重 tile，K 维累加在 NPU 内完成。
  默认离线编译 K=256/2048/8192，较短低秩维度以零 padding 补到 256。
- 通用算子：2048 元素一包，支持完整 2048 通道 LayerNorm 和 64 通道 GroupNorm。
- Decode WKV：64×64 FP32 状态，一次处理一个 head，无隐含设备全局状态。
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
| `src/inference/graph.cpp` | 固定图、依赖和状态绑定，不包含设备实现 |
| `src/inference/ops.cpp` | CPU 对照基础算子 |
| `src/inference/backend.cpp` | CPU / NPU decode WKV |
| `src/inference/npu_ops.cpp` | 全 NPU 算子和 prefill 适配 |
| `kernels/rwkv/` | 自研 AIE C++ kernel |
| `src/runtime/session.cpp` | XRT 资源和 run 复用 |
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
```

包括独立 FP64 参考、FP32/F16/BF16 PTH 与 safetensors、非连续 stride、非法文件、
非线性饱和/常量归一化、GEMV、状态 reset，以及 1/15/16/17/33 token prefill 与 decode
的全部 logits、最终状态和后续 decode 对照。图测试覆盖动态 token、清零、分支状态、
prefill 接续、图结构不变和非法输入不修改状态。Python 仅做离线参考与编译，NPU 由 C++ 调用。

真实模型测试是短序列正确性和生成检查，不代表已验证完整 25600 上下文或长文本质量。
