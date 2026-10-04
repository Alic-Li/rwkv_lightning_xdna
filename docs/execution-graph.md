# Decode graph 与 XRT 执行

`DecodeGraph` 在进程内记录逻辑节点、buffer 依赖及状态绑定。
CPU graph 通过参考 backend 执行；NPU graph 将节点映射为固定 resident 阶段。
运行时不根据目录中的实验产物选择其他计划。

构造时一次完成 ABI 检查、权重打包、session/BO 分配和 XRT run 参数绑定。
非首层的 recurrence 与 BF16 output projection 融合，FFN 直接读取融合输出。
C=2048 / FFN=8192 的24层正常 decode 为100次提交；诊断 trace 使用保留
中间结果的123次提交计划。C=1024 / FFN=4096 的0.4B目前使用123次提交计划，
不要求原来只为2048通道编译的 recurrence/output 融合内核。
编译器和运行时根据模型形状选择同一套 tile ABI；图按实际层数展开。
`--concurrency N` 用 N 个独立图及状态并发提交，图实例本身仍只能由单个线程操作。
请求共享只读主机权重和 XRT program context；所有可写 BO 和 prepared run 独立。
每次仍逐个执行可复用 `xrt::run`，没有启用 runlist 或 HRX。

`load_state` 建立设备请求状态，`replay_resident` 只同步 embedding/logits，
`export_state` 显式下载 checkpoint。reset/branch 通过再次 `load_state` 实现。
执行失败后 graph 不再接受 replay，须重新加载状态。

BF16 prefill 使用同一组权重、session 和 recurrent state：batch2 使用两份 activation
视图，chunk4 使用四份。WKV 保持 token 顺序；组内只运行最后一次 logits head，
尾 token 走普通 resident decode。INT8 只使用逐 token 状态转移。

模块结构与 API 见 [推理说明](inference.md)。历史 runlist、HRX、mode-worker 和共享程序
探针的结论保留在 `reports/`，探针代码及生产选择分支已移除。
