#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK_TEMPLATE="${REPO_ROOT}/.githooks/pre-commit"
HOOK_TEMPLATE_PS1="${REPO_ROOT}/.githooks/pre-commit.ps1"
TARGET_HOOK="${REPO_ROOT}/.git/hooks/pre-commit"
TARGET_HOOK_PS1="${REPO_ROOT}/.git/hooks/pre-commit.ps1"

if [[ ! -f "${HOOK_TEMPLATE}" ]]; then
  echo "[install-hook] FEHLER: Hook-Template nicht gefunden: ${HOOK_TEMPLATE}" >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET_HOOK}")"
cp "${HOOK_TEMPLATE}" "${TARGET_HOOK}"
chmod +x "${TARGET_HOOK}"

if [[ -f "${HOOK_TEMPLATE_PS1}" ]]; then
  cp "${HOOK_TEMPLATE_PS1}" "${TARGET_HOOK_PS1}"
fi

echo "[install-hook] Pre-Commit-Hook installiert: ${TARGET_HOOK}"
if [[ -f "${TARGET_HOOK_PS1}" ]]; then
  echo "[install-hook] PowerShell-Hook installiert: ${TARGET_HOOK_PS1}"
fi
echo "[install-hook] Teste den Hook jetzt mit: .git/hooks/pre-commit"
