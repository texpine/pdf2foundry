#!/bin/sh
sudo systemctl restart ollama &&
ollama serve &&
export CUDA_VISIBLE_DEVICES=0,1,2,3 &&
export OLLAMA_GPU_MEMORY_FRACTION=1 &&
export OLLAMA_NO_GPU=0 &&
export PATH=/usr/lib/cuda/bin:$PATH &&
export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH &&
sudo apt-get install poppler-utils -y &&
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&
bash Miniconda3-latest-Linux-x86_64.sh -b &&
export PATH="$HOME/miniconda3/bin:$PATH" &&
conda config --set plugins.auto_accept_tos yes &&
conda create --name pdf python=3.12 --yes &&
conda init &&
sudo miniconda3/envs/pdf/bin/pip install marker-pdf[all] &&
conda install -y -n pdf pytesseract &&
conda install -y -n pdf psutil &&
sudo miniconda3/envs/pdf/bin/pip install poppler-utils &&
sudo miniconda3/envs/pdf/bin/pip install ollama &&
conda install -y -n pdf pillow &&
conda install -y -n pdf imagehash &&
conda install -c nvidia -y -n pdf cudatoolkit &&
# main llm for chat and reasoning
echo 'Ollama: downloading qwen3-coder:30b' &&
ollama pull qwen3-coder:30b && # needs 4 A4000
echo 'Ollama: downloading gpt-oss:20b' &&
ollama pull gpt-oss:20b && # needs 2 A4000
echo 'Ollama: downloading gpt-oss:120b' &&
ollama pull gpt-oss:120b && # needs 4 A4000
echo 'Ollama: downloading deepseek-r1:70b' &&
ollama pull deepseek-r1:70b && # needs 4 A4000
echo 'Ollama: downloading deepseek-r1:14b' &&
ollama pull deepseek-r1:14b  && # needs 2 A4000
# LLM to auto-complete code
echo 'Ollama: downloading wraith-coder-7b' &&
ollama pull vanta-research/wraith-coder-7b:latest &&
# LLM for embedding (text to vector)
echo 'Ollama: downloading nomic-embed-text' &&
ollama pull nomic-embed-text &&
# LLM for re ranking
echo 'Ollama: downloading Qwen3-Reranker-4B' &&
ollama pull dengcao/Qwen3-Reranker-4B:Q4_K_M &&
# vision LLMs
echo 'Ollama: downloading llava:34b' &&
ollama pull llava:34b && # needs 2 A4000
echo 'Ollama: downloading qwen3-vl:32b' &&
ollama pull qwen3-vl:32b && # needs 4 A4000






# #!/bin/sh
# sudo apt upgrade -y &&
# # required nvidia drivers
# sudo apt install software-properties-common &&
# sudo apt install linux-headers-$(uname -r) -y &&
# sudo add-apt-repository ppa:graphics-drivers/ppa -y &&
# wget https://developer.download.nvidia.com/compute/nvidia-driver/570.124.06/local_installers/nvidia-driver-local-repo-ubuntu2404-570.124.06_1.0-1_amd64.deb &&
# sudo dpkg -i nvidia-driver-local-repo-ubuntu2404-570.124.06_1.0-1_amd64.deb &&
# sudo apt update -y &&
# cp /var/nvidia-driver-local-repo-ubuntu2404-570.124.06/nvidia-driver-local-*-keyring.gpg /usr/share/keyrings/


# sudo apt install nvidia-driver-570 -y &&
# sudo apt install nvidia-cuda-toolkit -y &&
# sudo ubuntu-drivers autoinstall &&
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
# sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
# sudo dpkg -i cuda-keyring_1.1-1_all.deb &&

# sudo apt install nvidia-driver-pinning-570
# sudo apt-get -y install cuda-toolkit-12-3 &&
# echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc &&
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc &&
# source ~/.bashrc
# #installs ollama
# curl https://ollama.com/install.sh | sh &&
# sudo systemctl start ollama &&
# # main llm for chat and reasoning
# ollama pull qwen3-coder:30b &&
# ollama pull gpt-oss:20b &&
# ollama pull gpt-oss:120b &&
# # LLM to auto-complete code
# ollama pull vanta-research/wraith-coder-7b:latest &&
# # LLM for embedding (text to vector)
# ollama pull nomic-embed-text &&
# # LLM for re ranking
# ollama pull dengcao/Qwen3-Reranker-4B:Q4_K_M &&
# # vision LLM
# ollama pull llava:34b
# # Python part!
# sudo apt-get install poppler-utils -y &&
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&
# bash Miniconda3-latest-Linux-x86_64.sh -b &&
# export PATH="$HOME/miniconda3/bin:$PATH" &&
# conda config --set plugins.auto_accept_tos yes &&
# conda create --name pdf python=3.12 --yes &&
# conda init &&
# $ conda install cuda -c nvidia
# sudo miniconda3/envs/pdf/bin/pip install marker-pdf[all] &&
# sudo miniconda3/envs/pdf/bin/pip install pytesseract &&
# sudo miniconda3/envs/pdf/bin/pip install psutil &&
# sudo miniconda3/envs/pdf/bin/pip install poppler-utils &&
# sudo miniconda3/envs/pdf/bin/pip install ollama &&
# sudo miniconda3/envs/pdf/bin/pip install pillow &&
# sudo miniconda3/envs/pdf/bin/pip install imagehash
# # we need to reboot for the drivers to work, and the GPU and VRAM
# # to process the models, otherwise they go to the CPU and RAM
# sudo reboot
