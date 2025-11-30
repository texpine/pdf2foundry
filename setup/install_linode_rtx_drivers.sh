#!/bin/sh
sudo apt install linux-headers-$(uname -r) -y &&
sudo add-apt-repository ppa:graphics-drivers/ppa -y &&
sudo apt update -y &&
sudo apt upgrade -y &&
sudo apt-get -y install ubuntu-drivers-common &&
sudo ubuntu-drivers autoinstall &&
sudo apt install nvidia-cuda-toolkit -y &&
sudo apt install nvtop &&
# #installs ollama
curl -fsSL https://ollama.com/install.sh | sh &&
sudo reboot
