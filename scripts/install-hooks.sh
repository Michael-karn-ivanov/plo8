#!/usr/bin/env bash
# Point git at the versioned hooks in .githooks/ for this clone.
set -euo pipefail
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
git -C "$repo_root" config core.hooksPath .githooks
chmod +x "$repo_root"/.githooks/*
echo "Git hooks enabled: core.hooksPath=.githooks"
