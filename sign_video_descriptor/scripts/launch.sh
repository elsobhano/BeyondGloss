#!/bin/bash
# Minimal launcher: set PYTHONPATH to the repo root, then exec the given command.
#   ./launch.sh python describe_segments.py --json_path ...
export PYTHONPATH=.
exec "$1" "${@:2}"
