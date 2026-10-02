#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compatibility setting verified with kernel 7.0.0-38 and NPU firmware 1.0.0.63.
set -euo pipefail
if [[ ${EUID} -ne 0 ]]; then
    echo "Run with sudo or pkexec to configure the amdxdna module." >&2
    exit 1
fi
config=/etc/modprobe.d/rwkv-lightning-xdna.conf
if [[ -e "$config" ]] && ! grep -qx 'options amdxdna force_cmdlist=N' "$config"; then
    echo "Refusing to replace an existing, different configuration: $config" >&2
    exit 1
fi
echo N > /sys/module/amdxdna/parameters/force_cmdlist
printf '%s\n' '# RWKV Lightning XDNA: legacy firmware command submission compatibility.' \
    'options amdxdna force_cmdlist=N' > "$config"
echo "Disabled forced command lists now and on future module loads: $config"
