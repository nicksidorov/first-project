#!/usr/bin/env bash
set -euo pipefail
cd /workspace/first-project
# Изолированное окружение задачи уже существует; используем текущую копию Git.
# Системные NumPy/SciPy разрешено переиспользовать, если они удовлетворяют pyproject.
if [ ! -x .venv/bin/python ]; then
    python3 -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir -e '.[test]'
