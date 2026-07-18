#!/bin/sh
# Install the repo's git hooks into .git/hooks.
# Run once after cloning:  bash scripts/install-hooks.sh
set -e
ROOT="$(git rev-parse --show-toplevel)"
cp "$ROOT/scripts/git-hooks/pre-commit" "$ROOT/.git/hooks/pre-commit"
chmod +x "$ROOT/.git/hooks/pre-commit"
echo "Installed pre-commit hook -> .git/hooks/pre-commit"
