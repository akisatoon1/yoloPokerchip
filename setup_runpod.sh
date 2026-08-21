#!/usr/bin/env bash

git clone https://github.com/akisatoon1/yoloPokerchip.git
cd yoloPokerchip
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
