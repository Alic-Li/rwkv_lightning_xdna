# 从环境部署到 C++ NPU 算子运行

适用目标：Ubuntu 24.04、Ryzen AI 350 / AIE2P。本指南包含系统依赖、设备权限、
内存锁定限制、uv 构建环境、编译、C++ 执行及离线验证。
从项目根目录执行命令；系统安装步骤需要 sudo，普通编译和运行使用当前用户。

## 1. 本机已验证的版本

2026-10-02 的实际验证环境：

| 组件 | 版本 |
|---|---|
| 系统 | Ubuntu 24.04.5 LTS |
| 内核 | 7.0.0-38-generic，内置 amdxdna 0.7.0 |
| XRT | 2.25.00 |
| NPU | RyzenAI-npu6，AIE2P，XRT 拓扑 6×8 |
| 固件 | 1.0.0.63 |
| MLIR-AIE | 1.4.4.dev53+gd53582d |
| Peano | 22.0.0.2026091701+773413fb |
| 编译用 Python | 3.12.3，uv 管理 |

XRT 拓扑的 6 行包含 shim 和 memory tile；计算核为 4 行。
上述版本是本机测试记录，上游安装页面会更新，复现时使用仓库的 lock 和源码快照。
本机已完成 296 个配置、888 次 C++/XRT 调用，见 [验证记录](validation.md)。

## 2. 检查设备、内核和驱动

```bash
uname -r
lspci -nn | grep -i 'signal processing'
lsmod | grep amdxdna
modinfo amdxdna
ls -l /dev/accel/
journalctl -k -b --no-pager | grep -Ei 'amdxdna|amdnpu'
```

预期看到 NPU PCI 设备、已加载的 amdxdna、成功加载固件的日志及
`/dev/accel/accel0`。本机 PCI ID 是 `1022:17f0`。
若 PCI 层没有设备，先检查 BIOS 的 NPU/IPU 开关。
本机内核驱动已经工作；保留它并安装 XRT 用户态组件即可。
其它内核/系统的驱动构建和依赖按 [AMD XDNA 驱动说明](https://github.com/amd/xdna-driver#system-requirements)
处理，不能仅凭没有安装 DKMS 就判定没有驱动。

## 3. 安装系统依赖与 XRT

本机使用 [MLIR-AIE 官方安装入口](https://github.com/Xilinx/mlir-aie#install-the-xdna-driver-and-xrt)
列出的 AMD XRT PPA：

```bash
sudo apt update
sudo apt install -y software-properties-common ca-certificates curl git \
  python3 python3-venv build-essential clang clang-14 lld lld-14 uuid-dev
sudo add-apt-repository -y ppa:amd-team/xrt
sudo apt update
sudo apt install -y libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu python3-xrt
sudo usermod -aG render "$USER"
```

修改组后完整退出登录再登录，重新打开 IDE/终端，然后检查：

```bash
id
ls -l /dev/accel/accel0
test -r /dev/accel/accel0 && test -w /dev/accel/accel0 && echo 'NPU access OK'
xrt-smi examine
command -v xclbinutil
```

本机 XRT 输出设备名 `RyzenAI-npu6`，架构 `aie2p`。
PPA 包将工具安装到系统路径。若使用上游 `/opt/xilinx/xrt` 布局，先在当前 shell
执行 `source /opt/xilinx/xrt/setup.sh`，再执行后面的编译/运行命令。
C++ 主机程序需要 libxrt-dev，`.xclbin` 打包需要 libxrt-utils 提供的 xclbinutil。

`uuid-dev` 提供 XRT 头文件间接需要的 `uuid/uuid.h`。
本机初次缺少此包时，bootstrap 将 Ubuntu 包解压到 `.cache/sdk/` 供 CMake 使用；
新环境直接安装系统包更清晰。CMake、Ninja 和格式化工具由后面的 uv 环境提供。

## 4. 内存页锁定上限：memlock

你提到的限制对应 `RLIMIT_MEMLOCK`：控制一个进程可锁定的内存字节数，
而具体内核路径按页计数。Linux 7.0 的 amdxdna 用户缓冲区代码读取该限制，
在无 `CAP_IPC_LOCK` 且累计固定页数超过上限时返回 `-ENOMEM`。
依据：[Linux getrlimit 文档](https://man7.org/linux/man-pages/man2/getrlimit.2.html)、
[amdxdna 固定用户页代码](https://github.com/torvalds/linux/blob/v7.0/drivers/accel/amdxdna/amdxdna_ubuf.c#L186-L194)。
[FastFlowLM 官方 Linux 指南](https://github.com/ROCm/FastFlowLM/blob/main/docs/linux-getting-started.md)
也将检查和提高 memlock 列为 NPU 部署步骤；该项目自己的固件要求不等同于本仓库的要求。

### 查看实际限制

```bash
ulimit -S -l                  # Bash 输出软限制，单位 KiB
ulimit -H -l                  # 硬限制，单位 KiB
getconf PAGESIZE              # 单页大小，单位字节
grep 'Max locked memory' /proc/$$/limits  # 当前 shell 的限制，单位字节
```

本轮助手 shell 实测：软/硬限制均为 `3993344 KiB`，即 `4089184256` 字节、约
`3.81 GiB`；页大小为 `4096` 字节。不同 IDE、终端和服务可能继承不同限制，
应在真正启动程序的环境检查，不能假定所有 Linux 都是某个固定默认值。

当前算子测试已通过；本项目此前未通过修改 memlock 解决命令中止。
后续 RWKV 大模型长期固定较多权重/状态缓冲区时，应重新检查这个上限。
设置 unlimited 只解除资源上限，不会预分配物理内存。

### 临时提高当前 shell 的限制

硬限制已经是 unlimited 时，可直接执行 `ulimit -S -l unlimited`。
若硬限制也有限，需要管理员提升；下面命令修改当前 shell，随后启动的 C++
程序会继承它：

```bash
sudo prlimit --pid "$$" --memlock=unlimited:unlimited
ulimit -S -l
ulimit -H -l
./build/host/xdna-run build/kernels/add-381f9fce11aa/manifest.json
```

这里最后一行要求先完成第 7 节编译。该方法不修改已运行的其它终端/IDE，
也不会在重新登录后永久保留。soft/hard limit 的权限规则见
[getrlimit / prlimit](https://man7.org/linux/man-pages/man2/getrlimit.2.html)。

### 持久设置：普通登录会话

为当前用户创建专用 PAM limits 文件，避免扩大到所有用户：

```bash
printf '%s soft memlock unlimited\n%s hard memlock unlimited\n' "$USER" "$USER" \
  | sudo tee /etc/security/limits.d/90-rwkv-xdna-memlock.conf
```

完整退出登录再登录，重新启动 IDE/终端，复查 `ulimit -S -l` 和 `ulimit -H -l`。
limits 文件通过 `pam_limits` 在新登录会话中应用，不会追溯修改现有进程。
文件格式和生效机制见 [Linux-PAM limits.conf](https://man7.org/linux/man-pages/man5/limits.conf.5.html)。

### systemd 服务或桌面/IDE 仍继承旧限制时

若未来将推理程序作为 systemd 系统服务运行，在该服务的 drop-in 中设置：

```ini
[Service]
LimitMEMLOCK=infinity
```

执行 `sudo systemctl daemon-reload` 并重启相应服务后，查看其
`LimitMEMLOCK` / `LimitMEMLOCKSoft`，以及实际进程的 `/proc/<PID>/limits`。
这是服务部署示例；本仓库当前没有创建名为 RWKV 的推理服务。

用户级 systemd 服务不能把限制提高到用户管理器继承的硬限制以上。
如果重新登录后 IDE/用户服务仍有限制，可为当前用户的系统级管理器配置：

```bash
xdna_uid=$(id -u)
sudo mkdir -p "/etc/systemd/system/user@${xdna_uid}.service.d"
printf '[Service]\nLimitMEMLOCK=infinity\n' \
  | sudo tee "/etc/systemd/system/user@${xdna_uid}.service.d/90-rwkv-xdna-memlock.conf"
sudo systemctl daemon-reload
```

然后在保存工作后重新启动用户管理器；重启系统是使整个桌面进程树重新继承设置的
简单方式。最后在 IDE 新终端里再次检查限制，而不是只看配置文件。
依据：[systemd.exec 官方资源限制说明](https://github.com/systemd/systemd/blob/main/man/systemd.exec.xml#L1034-L1105)。
通常先采用用户级 PAM 配置，仅在启动链需要时再补 systemd 设置。

## 5. 本机另一个独立问题：force_cmdlist

本机初次 C++ 提交返回 `ERT_CMD_STATE_ABORT (6)`，日志含
`Message callback ret -22`。在保持 xclbin、输入及主机程序不变的情况下，
只将 `amdxdna.force_cmdlist` 从 `Y` 改为 `N`，Add/Zero 随即通过，后续完整测试通过。
因此本次已验证的解决措施针对命令提交路径，不能把该现象直接归因为 memlock。

Linux 7.0 源码将此参数默认设为 true，并据此选择强制命令列表或直接提交路径，见
[参数定义](https://github.com/torvalds/linux/blob/v7.0/drivers/accel/amdxdna/aie2_ctx.c#L26-L28) 和
[提交分支](https://github.com/torvalds/linux/blob/v7.0/drivers/accel/amdxdna/aie2_ctx.c#L318-L323)。
本机的对照结果记录于本仓库；不将其泛化为上游要求所有系统关闭命令列表。

相同版本组合遇到相同问题时：

```bash
sudo bash tools/configure_driver.sh
sudo cat /sys/module/amdxdna/parameters/force_cmdlist
cat /etc/modprobe.d/rwkv-lightning-xdna.conf
```

预期为 `N` 和 `options amdxdna force_cmdlist=N`。脚本立即修改当前参数并保存
后续模块加载配置。本机 amdxdna 不在现有 initramfs 内；若其它安装把驱动放进了
initramfs，需在更新配置后运行 `sudo update-initramfs -u` 才能让早期加载也读取新配置。

恢复默认时，删除本项目专用的 `/etc/modprobe.d/rwkv-lightning-xdna.conf`，
并以管理员权限将当前模块参数写回 `Y`；若曾更新 initramfs，也需同步重建。
升级内核或固件后重新验证此兼容措施。

## 6. 安装 uv 并创建构建环境

尚未安装 uv 时，使用 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
sh /tmp/uv-install.sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

切换到项目目录，执行构建环境初始化：

```bash
cd /home/alic-li/work_space/rwkv_lightning_xdna
bash tools/bootstrap.sh
export PATH="$PWD/.venv/bin:$PATH"
```

`bootstrap.sh` 执行以下步骤：创建/复用 `.venv`、安装锁定依赖、检查 uuid 头文件，
然后运行 CMake 配置与主机 C++ 编译。它不安装驱动、不修改 memlock、也不运行 NPU。
手动展开主要步骤为：

```bash
uv venv --allow-existing .venv --python /usr/bin/python3
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-build.lock
export PATH="$PWD/.venv/bin:$PATH"
cmake --preset dev
cmake --build --preset dev
```

依赖管理依据：[uv 环境选择](https://docs.astral.sh/uv/pip/environments/)、
[uv requirements 锁定流程](https://docs.astral.sh/uv/pip/compile/)。
本项目 `.lock` 是固定版本的 requirements 文本，不是 `uv.lock` 项目模式文件。
原始依赖范围位于 `requirements-build.txt`，实际复现使用 `requirements-build.lock`。

使用多索引解析是为了匹配 IRON 的依赖配置，安装 PyTorch CPU 索引上的固定版本。
锁定的 llvm-aie wheel 下载文件名为 py3-none，但内部元数据标为 cp310，故
Python 3.12 下 `uv pip check` 会报告平台标签问题；本机原生 Peano 编译器已经
完成全部实测编译。没有修改上游 wheel 元数据来隐藏这个问题。

## 7. 先跑一个算子：编译、C++ 提交、数值校验

以 BF16 Add 为例，先完成离线编译和输入生成：

```bash
.venv/bin/python tools/compile/kernel_case.py --case add-381f9fce11aa
```

输出位于 `build/kernels/add-381f9fce11aa/`，包括 `design.xclbin`、
`instructions.bin`、`manifest.json` 和输入二进制。实际计算使用已复制的
`third_party/mlir-aie/aie_kernels/` C++ 源码，编译器由固定版本 wheel 提供。

随后直接启动 C++ 程序，进行 3 次硬件提交：

```bash
./build/host/xdna-run build/kernels/add-381f9fce11aa/manifest.json
```

成功返回 JSON `"status":"executed"` 和计时，输出保存在同目录的
`output-*.bin.0`、`.1`、`.2`。此阶段没有 Python 运行时参与 NPU 提交。
最后离线比较每次输出与官方参考值，并检查越界哨兵：

```bash
.venv/bin/python tools/validation/check_case.py add-381f9fce11aa
```

预期得到 `"status":"passed"`。`executed` 表示硬件调用完成，`passed`
还要求数值和哨兵检查成功。编译与参考实验使用 Python，运行算子使用 C++。
源码/测试定义的来源见 [第三方清单](../third_party/README.md)，对应
[官方 kernel_cases](https://github.com/Xilinx/mlir-aie/blob/d53582d/test/python/npu/kernel_cases.py)
和 [官方 kernel_design](https://github.com/Xilinx/mlir-aie/blob/d53582d/python/iron/algorithms/kernel_design.py)。

## 8. 全量验证与重新运行

```bash
# 首次：编译 + C++ 串行执行 + 离线检查；默认 4 个编译任务并行。
.venv/bin/python tools/validation/sweep.py --tier all --jobs 4
.venv/bin/python tools/validation/report.py

# 内核/编译参数未变化，已有产物只需重新运行时：
.venv/bin/python tools/validation/sweep.py --tier all --run-only
```

全量包含 295 个适用于 AIE2P 的官方配置和 1 个双核级联配置。
典型进度行为 `[296/296] passed: ...`，报告生成器汇总为
`passed: 296`、`source_count: 61`、`covered_source_count: 61`、`dispatch_count: 888`。
每配置 3 次调用；未穷举所有数据分布与模板实例，具体范围见 [验证记录](validation.md)。

常用选项：`--tier smoke` 用于日常基础覆盖，`--case <ID>` 选择单个配置，
`--compile-only` 只生成设备产物。修改 C++ 内核后应先重新编译，不能只用 run-only。

还可以将现有产物注册为 CTest：

```bash
cmake --preset dev -DRWKV_XDNA_HARDWARE_TESTS=ON
cmake --build --preset dev
ctest --preset dev
```

测试把 C++ 提交和离线检查设为有依赖的两个步骤，296 个配置共 592 项。
详细日志位于 `reports/runs/all/`，按日期汇总的报告位于 `reports/validation-YYYY-MM-DD.json`。

## 9. 常见故障定位

| 现象 | 优先检查 |
|---|---|
| 没有 `/dev/accel/accel0` | PCI/BIOS 开关、内核驱动和固件加载日志 |
| 打开设备 Permission denied | render 组和实际进程继承的组权限 |
| `uuid/uuid.h` 缺失 | 安装 uuid-dev，或使用 bootstrap 的本地 SDK 回退 |
| `xclbinutil` 找不到 | 安装 libxrt-utils；/opt 布局先 source setup.sh |
| 固定大缓冲区失败、ENOMEM | 实际可用内存、进程 memlock、内核固定页日志；单凭 ENOMEM 不能确定原因 |
| `ulimit -l unlimited` 被拒绝 | 硬限制有限，需管理员 prlimit 或重新登录加载配置 |
| limits 文件已改，IDE 中仍有限制 | 旧进程继承链或 systemd 用户管理器的硬限制 |
| 本机 ERT state=6、消息回调 -22 | 对照本机 force_cmdlist 排障记录；该错误状态本身不是唯一根因证明 |
| 返回 executed 但校验失败 | 读取对应 check.log，核对输入、产物、布局及源码版本 |

外部文档链接按本次查阅补充；涉及本机特定行为的结论以实际日志和验证报告为依据。
