# 环境部署

目标为 Ubuntu 24.04、Ryzen AI 350 / AIE2P。从仓库根目录执行；系统安装使用 sudo，
普通编译和运行使用当前用户。

## 已验证环境

以下为 2026-10-02 实机记录，复现使用仓库 lock 和源码快照：

| 组件 | 版本 |
|---|---|
| 系统 / 内核 | Ubuntu 24.04.5 LTS / 7.0.0-38-generic |
| 驱动 / XRT | amdxdna 0.7.0 / XRT 2.25.00 |
| NPU / 固件 | RyzenAI-npu6，AIE2P / 1.0.0.63 |
| MLIR-AIE | 1.4.4.dev53+gd53582d |
| Peano | 22.0.0.2026091701+773413fb |
| 构建 Python | 3.12.3，由 uv 管理 |

XRT 拓扑 6×8 中包含 shim 和 memory tile，计算核为 4 行。
验证范围和报告见 [验证记录](validation.md)。

## 系统依赖与设备权限

先确认 BIOS 已启用 NPU、内核已加载 amdxdna，且固件正常：

```bash
lspci -nn | grep -i 'signal processing'
lsmod | grep amdxdna
ls -l /dev/accel/
journalctl -k -b --no-pager | grep -Ei 'amdxdna|amdnpu'
```

本机使用现有内核驱动，XRT 来自 AMD PPA：

```bash
sudo apt update
sudo apt install -y software-properties-common ca-certificates curl git \
  python3 python3-venv build-essential clang clang-14 lld lld-14 uuid-dev
sudo add-apt-repository -y ppa:amd-team/xrt
sudo apt update
sudo apt install -y libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu python3-xrt
sudo usermod -aG render "$USER"
```

退出登录再登录后，在实际运行推理的终端检查：

```bash
id
ls -l /dev/accel/accel0
xrt-smi examine
command -v xclbinutil
```

当前用户需可读写设备。使用 `/opt/xilinx/xrt` 安装布局时，先执行
`source /opt/xilinx/xrt/setup.sh`。`libxrt-dev` 用于主机编译，`xclbinutil` 用于
设备产物打包。其他系统/内核按 [AMD 驱动说明](https://github.com/amd/xdna-driver#system-requirements)
部署；安装入口参考 [MLIR-AIE](https://github.com/Xilinx/mlir-aie#install-the-xdna-driver-and-xrt)。

## memlock

NPU 固定主机缓冲区受到进程 `RLIMIT_MEMLOCK` 限制。检查实际启动终端的软/硬限制：

```bash
ulimit -S -l
ulimit -H -l
grep 'Max locked memory' /proc/$$/limits
```

硬限制已是 unlimited 时，可执行 `ulimit -S -l unlimited`。持久设置当前用户：

```bash
printf '%s soft memlock unlimited\n%s hard memlock unlimited\n' "$USER" "$USER" \
  | sudo tee /etc/security/limits.d/90-rwkv-xdna-memlock.conf
```

完整退出登录再登录并重新启动 IDE。systemd 系统服务在 drop-in 中使用
`LimitMEMLOCK=infinity`；用户服务还受用户管理器继承的硬限制约束。
配置不会追溯修改现有进程，须再次检查实际终端/服务的 `/proc/<PID>/limits`。
本机 2026-10-02 已确认软/硬限制均为 unlimited。

资源限制说明见 [getrlimit](https://man7.org/linux/man-pages/man2/getrlimit.2.html)、
[PAM limits](https://man7.org/linux/man-pages/man5/limits.conf.5.html) 和
[systemd 资源限制](https://github.com/systemd/systemd/blob/main/man/systemd.exec.xml)。

## 本机驱动兼容设置

本机曾出现 `ERT_CMD_STATE_ABORT (6)`、`Message callback ret -22`。
保持程序、产物和输入不变，将 `amdxdna.force_cmdlist` 从 Y 改为 N 后测试通过。
这是本机版本组合的对照结果，与 memlock 分别排查。
遇到相同组合和症状时：

```bash
sudo bash tools/configure_driver.sh
cat /sys/module/amdxdna/parameters/force_cmdlist
cat /etc/modprobe.d/rwkv-lightning-xdna.conf
```

预期为 N 和 `options amdxdna force_cmdlist=N`。脚本修改当前参数并保存模块加载配置。
若驱动在 initramfs 中，更新配置后还需 `sudo update-initramfs -u`。
恢复时删除上述项目专用配置、将当前参数写回 Y，并同步更新曾修改的 initramfs。
升级内核或固件后重新验证。参数定义见 [Linux 7.0 源码](https://github.com/torvalds/linux/blob/v7.0/drivers/accel/amdxdna/aie2_ctx.c#L26-L28)。

## 构建工具链

按 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv，然后：

```bash
bash tools/bootstrap.sh         # 默认 release 主机构建
# 或 bash tools/bootstrap.sh test
export PATH="$PWD/.venv/bin:$PATH"
```

bootstrap 创建/复用项目 `.venv`，同步 `requirements-build.lock`，检查 uuid 头文件，
并配置、编译所选主机 preset。缺少 uuid-dev 时解压本地 SDK 到 `.cache/sdk/`。
驱动、memlock 和 NPU 提交由上述步骤及后续测试处理。

直接依赖在 `requirements-build.txt`，精确版本在 requirements 格式的 `.lock`。
项目编译 `third_party/mlir-aie` 快照，无需相邻 IRON 仓库或 amd/IRON 包。
CMake、Ninja、MLIR-AIE 和 Peano 由锁定环境提供。
已知锁定 llvm-aie wheel 的内部 cp310 元数据导致 Python 3.12 下 `uv pip check`
报告标签问题；已验证的原生编译器可完成编译，未修改上游元数据。

接下来按 [构建与测试](build.md) 编译生产 kernel 或通用测试配置，再按
[推理说明](inference.md) 运行模型。

## 排障

| 现象 | 检查 |
|---|---|
| 无设备 | BIOS、PCI、驱动及固件日志 |
| Permission denied | render 组及实际进程继承的权限 |
| uuid 头文件缺失 | uuid-dev 或 bootstrap 本地 SDK |
| xclbinutil 缺失 | libxrt-utils 或 XRT setup.sh |
| 缓冲区分配 ENOMEM | 实际内存、memlock、内核固定页日志 |
| IDE 仍有 memlock 限制 | 旧进程继承链、systemd 用户管理器 |
| ERT state=6 / callback -22 | 本机 force_cmdlist 对照记录，错误状态本身不能确定根因 |
| executed 但数值失败 | check.log、输入、布局、产物及源码版本 |
