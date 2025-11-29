#!/bin/sh
sudo apt install python3-pip  -y &&
sudo apt install python3.12-venv  -y &&
sudo apt-get install poppler-utils -y &&
python3 -m venv pdf &&
source pdf/bin/activate &&
pip install marker-pdf[all] &&
pip install pytesseract &&
pip install psutil &&
pip install poppler-utils &&
pip install ollama &&
pip install pillow imagehash
