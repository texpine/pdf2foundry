#!/bin/sh
sudo apt update -y &&
sudo apt upgrade -y &&
# required nvidia drivers
sudo apt install linux-headers-$(uname -r) -y &&
sudo add-apt-repository ppa:graphics-drivers/ppa -y &&
sudo apt install nvidia-driver-570 -y &&
sudo apt install nvidia-cuda-toolkit -y &&
#installs ollama
curl https://ollama.com/install.sh | sh &&
sudo systemctl start ollama &&
# main llm for chat and reasoning
ollama pull qwen3-coder:30b &&
ollama pull gpt-oss:20b &&
ollama pull gpt-oss:120b &&
# LLM to auto-complete code
ollama pull vanta-research/wraith-coder-7b:latest &&
# LLM for embedding (text to vector)
ollama pull nomic-embed-text &&
# LLM for re ranking
ollama pull dengcao/Qwen3-Reranker-4B:Q4_K_M &&
# vision LLM
ollama pull llava:34b
