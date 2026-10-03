#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
uv venv --allow-existing .venv --python /usr/bin/python3
uv pip sync --python .venv/bin/python --index-strategy unsafe-best-match requirements-build.lock
if [[ ! -f /usr/include/uuid/uuid.h ]]; then
    # XRT's Ubuntu development package does not pull in this header dependency.
    # Extract an Ubuntu package locally; no administrator access is needed.
    mkdir -p .cache/sdk
    (
        cd .cache/sdk
        apt-get download uuid-dev
        for package in uuid-dev_*.deb; do dpkg-deb -x "$package" .; done
    )
fi
export PATH="$project_root/.venv/bin:$PATH"
build_preset=${1:-release}
cmake --preset "$build_preset"
cmake --build --preset "$build_preset"
