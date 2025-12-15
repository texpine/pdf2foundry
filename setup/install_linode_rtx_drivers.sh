#!/bin/sh
sudo apt install linux-headers-$(uname -r) -y &&
sudo add-apt-repository ppa:graphics-drivers/ppa -y &&
sudo apt update -y &&
sudo apt upgrade -y &&
sudo apt-get -y install ubuntu-drivers-common &&
sudo ubuntu-drivers autoinstall &&
sudo apt install nvidia-cuda-toolkit -y &&
sudo apt install nvtop &&
cd ~ &&
# #installs ollama
curl -fsSL https://ollama.com/install.sh | sh &&
# installs miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&
bash Miniconda3-latest-Linux-x86_64.sh -b &&
export PATH="$HOME/miniconda3/bin:$PATH" &&
conda config --set plugins.auto_accept_tos yes &&
sudo reboot