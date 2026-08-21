#!/usr/bin/env bash

python -m venv .venv
source .venv/bin/activate

pip install uv
uv pip install -r requirements.txt
