# 代码内 decode graph 与 XRT 执行

按用户要求，推理使用进程内 `DecodeGraph`，不依赖 `xrt-capture` / `xrt-replay` 命令。
图在构造时记录算子、buffer 依赖和状态回写；replay 绑定新 token 和 State。
细节及使用方法见 [inference.md](inference.md)。

当前使用已验证的 `xrt::run` 复用。它复用对象和参数绑定，但仍逐个提交 run，
不是整个 decode 的单次硬件图提交。默认 NPU resident 路径使用 BF16 权重、固定 arena、
阶段融合和阵列并行。通过 `load_state / replay_resident / export_state`，FP32 状态跨
 token 留在设备 BO，稳态只传 embedding/logits；兼容 `replay(token, State&)` 同步完整状态。
当前24层模型普通 BF16 / INT8 FFN-only decode 为100 runs/token，INT8 FFN+output
和诊断 trace 为123 runs/token。延迟取决于权重模式和测试条件，生产实现及各次测量见
[inference.md](inference.md)。

2026-10-03 补查 FastFlowLM 的 `src/lib/hrx/libllama_npu.so`：其 forward 中存在
循环调用 `hrx_stream_dispatch` 后统一 `hrx_stream_flush` / `hrx_stream_wait` 的路径，
随后单独运行输出 head。本地 HRX 包装头文件也提供 executable 缓存和批量 record 接口。
FastFlowLM 的 `FLM_USE_HRX` 默认关闭，不能把 HRX 与 XRT 库的行为混为一谈。
这说明应独立验证 HRX 的兼容性及收益；下面的旧 XRT runlist 失败不能排除 HRX 路径。
统一 flush 也不证明只有一条硬件命令。可复查的二进制地址、哈希及证据边界见
[FastFlowLM HRX 核查](../reports/amd2026-fastflow-review-2026-10-03.json)。

以下是历史 runlist/capture 探针记录，不代表当前接口或本轮重新测试。

## 本机实验记录（2026-10-02，XRT 2.25）

- 普通 run 重复执行并改变输入：通过数值检查。
- 两条 run 组成的原生 runlist，期望复用两次并改变输入：第一次执行返回
  `ERT_CMD_STATE_ABORT`。历史探针源码已随实验清理删除；推理没有启用该路径。
- `xrt-capture --frames 2` 包装已验证的 C++ WKV 测试：应用数值检查通过，
  产生 `capture_*.bin`，但工具报告没有 `replay.json`。
- 使用已知 WKV ABI 手动构建 replay 描述后，工具报告指令参数 bank 连接不匹配，
  随后以 `No host side buffer in destination buffer: Invalid argument` 失败。
  这是当前安装版本/旧式 xclbin 路径的实测兼容性限制，尚未定位完整根因，
  不能归纳为所有 XRT/NPU 都不支持图或 runlist。

这些探针之后，普通 NPU 算子、模型和状态测试仍正常通过。没有为此修改驱动配置。

## 参考资料

[XRT 官方 capture/replay 说明](https://github.com/Xilinx/XRT/blob/master/src/runtime_src/core/common/runner/replay.md)
将 frame 定义为一次 run.start 或 runlist.execute，并记录资源、缓冲区快照和同步点。
它适合复现和分析，录制内容并不会自动包含采样器和新 token 的状态推进逻辑。
官方文档的命令名称与本机版本可能不同，实际 CLI 以本机 `--help` 为准。

[XRT Native APIs](https://xilinx.github.io/XRT/master/html/xrt_native_apis.html)
及 [runlist API 头文件](https://github.com/Xilinx/XRT/blob/master/src/runtime_src/core/include/xrt/experimental/xrt_kernel.h)
描述原生执行接口。runlist 中的 run 必须关联同一个硬件 context；它与代码内的模型图、
DMA 数据流和设备缓存布局是不同层次，不能仅换一个命令就把当前模型变成单次硬件提交。
