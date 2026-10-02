# 代码内 decode graph 与 XRT 执行

按用户要求，推理使用进程内 `DecodeGraph`，不依赖 `xrt-capture` / `xrt-replay` 命令。
图在构造时记录算子、buffer 依赖和状态回写；replay 绑定新 token 和 State。
细节及使用方法见 [inference.md](inference.md)。

当前使用已验证的 `xrt::run` 复用。它复用对象和参数绑定，但仍逐个提交 run，
不是整个 decode 的单次硬件图提交。默认 `--decode graph` 的中间结果仍有主机传输；
新增 `--decode resident` 使用持久 BO、预排布权重和 DMA 拼接/拆分，中间节点之间
不再进行应用层主机传输，状态和 logits 在 token 边界同步。
真实 1.5B 模型短序列实测从 7.691 降到 5.908 秒/token，全部 logits/状态最大误差为 0。
编译、运行、测量条件及限制见 [resident decode](inference.md#resident-decode中间数据保留在设备缓冲区)。

此实现参考 FastFlowLM 的 BO、run 和 DMA 工具接口，仍使用本项目 RWKV FP32 kernel。
没有启用原生 runlist，也没有修改驱动参数；以下 runlist/capture 结果为此前的历史探针记录，
不应理解为本轮重新测试或对所有硬件版本的判断。

## 本机实验记录（2026-10-02，XRT 2.25）

- 普通 run 重复执行并改变输入：通过数值检查。
- 两条 run 组成的原生 runlist，期望复用两次并改变输入：第一次执行返回
  `ERT_CMD_STATE_ABORT`。探针源码在 `tools/probes/xrt_runlist.cpp`，不纳入默认自动测试，
  也未在推理中启用失败路径。
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
