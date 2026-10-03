# Decode graph 与 XRT 执行

`DecodeGraph` 在进程内记录逻辑节点、buffer 依赖及状态绑定。
CPU graph 通过参考 backend 执行；NPU graph 将节点映射为固定 resident 阶段。
运行时不根据目录中的实验产物选择其他计划。

构造时一次完成 ABI 检查、权重打包、session/BO 分配和 XRT run 参数绑定。
非首层的 recurrence 与 BF16 output projection 融合，FFN 直接读取融合输出。
24层正常 decode 为100次提交；诊断 trace 使用保留中间结果的123次提交计划。
每次仍逐个执行可复用 `xrt::run`，没有启用 runlist 或 HRX。

`load_state` 建立设备请求状态，`replay_resident` 只同步 embedding/logits，
`export_state` 显式下载 checkpoint。reset/branch 通过再次 `load_state` 实现。
执行失败后 graph 不再接受 replay，须重新加载状态。

BF16 prefill 使用同一组权重、session 和 recurrent state：batch2 使用两份 activation
视图，chunk4 使用四份。WKV 保持 token 顺序；组内只运行最后一次 logits head，
尾 token 走普通 resident decode。INT8 只使用逐 token 状态转移。

模块结构与 API 见 [推理说明](inference.md)。历史 runlist、HRX、mode-worker 和共享程序
探针的结论保留在 `reports/`，探针代码及生产选择分支已移除。
