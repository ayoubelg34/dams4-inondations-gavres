#!/usr/bin/env bash
set -e
exec gunicorn dash_app:server --bind 0.0.0.0:${PORT:-8050} --workers 1 --threads 4 --timeout 120
