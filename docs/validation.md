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
- `build/kernels/<case-id>/`：xclbin、指令流、manifest、输入和每次输出。
- `third_party/SOURCES.json`：复制来源、版本和源码 SHA-256。

所有硬件提交都来自 `build/host/xdna-run`；Python 只进行离线设计编译、数据生成、
启动 C++ 子进程和读取输出比较。`executed` 与 `passed` 分开记录。

时间数据来自功能测试，含预热及很少的重复次数；不能直接作为稳定性能基准。
`dispatch_us` 是主机 start/wait 时间，不等同于 AIE 核心纯计算周期。

额外检查：CMake 包安装与独立 C++ 消费项目编译链接通过；592 个 CTest 项
（296 个提交 + 296 个检查）已注册，抽查 Add 和级联的 CTest 依赖路径通过。

驱动排障及已经保存的兼容设置见 [环境说明](environment.md)。
