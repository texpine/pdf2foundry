#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3 &&
export OLLAMA_GPU_MEMORY_FRACTION=1 &&
export OLLAMA_SCHED_SPREAD=1 &&
export OLLAMA_NO_GPU=0 &&
export PATH=/usr/lib/cuda/bin:$PATH &&
export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH &&
sudo apt-get install poppler-utils -y &&
export PATH="$HOME/miniconda3/bin:$PATH" &&
conda create --name pdf python=3.12 --yes &&
conda init &&
conda install -c nvidia -y -n pdf cudatoolkit &&
conda install -y -n pdf pytesseract &&
conda install -y -n pdf psutil &&
sudo miniconda3/envs/pdf/bin/pip install poppler-utils &&
sudo miniconda3/envs/pdf/bin/pip install PyMuPDF &&
sudo miniconda3/envs/pdf/bin/conda install scikit-image -y &&
sudo miniconda3/envs/pdf/bin/pip install opencv-python &&
sudo miniconda3/envs/pdf/bin/pip install ollama &&
conda install -y -n pdf pillow &&
conda install -y -n pdf imagehash &&
sudo /root/miniconda3/envs/pdf/bin/pip install poppler-utils &&
sudo /root/miniconda3/envs/pdf/bin/pip install ollama &&
sudo /root/miniconda3/envs/pdf/bin/pip install marker-pdf[all] &&
# we create manipulated Modelfiles to force the LLMs to use all GPUs
mkdir ~/modelfile/ &&
# main llm for chat and reasoning
# qwen3-coder:30b
echo 'Ollama: downloading qwen3-coder:30b' &&
echo 'FROM qwen3-coder:30b' >> ~/modelfile/Modelfile_qwen3-coder &&
echo 'PARAMETER num_gpu 999' >> ~/modelfile/Modelfile_qwen3-coder &&
ollama pull qwen3-coder:30b && # needs 4 A4000
ollama create qwen3-coder:30b -f ~/modelfile/Modelfile_qwen3-coder &&
# gpt-oss:20b
echo 'Ollama: downloading gpt-oss:20b' &&
echo 'FROM gpt-oss:20b' >> ~/modelfile/Modelfile_gpt-oss_20b &&
echo 'PARAMETER num_gpu 999' >> ~/modelfile/Modelfile_gpt-oss_20b &&
ollama pull gpt-oss:20b && # needs 2 A4000
ollama create gpt-oss:20b -f ~/modelfile/Modelfile_gpt-oss_20b &&
# gpt-oss:120b
echo 'Ollama: downloading gpt-oss:120b' &&
echo 'FROM gpt-oss:120b' >> ~/modelfile/Modelfile_gpt-oss_120b &&
echo 'PARAMETER num_gpu 999' >> ~/modelfile/Modelfile_gpt-oss_120b &&
ollama pull gpt-oss:120b && # needs 4 A4000
ollama create gpt-oss:120b -f ~/modelfile/Modelfile_gpt-oss_120b &&
# deepseek models
echo 'Ollama: downloading deepseek-r1:70b' &&
echo 'FROM deepseek-r1:70b' >> ~/modelfile/Modelfile_deepseek_70b &&
echo 'PARAMETER num_gpu 999' >> ~/modelfile/Modelfile_deepseek_70b &&
ollama pull deepseek-r1:70b && # needs 4 A4000
ollama create deepseek-r1:70b -f ~/modelfile/Modelfile_deepseek_70b &&
# deepseek-r1:14b
echo 'Ollama: downloading deepseek-r1:14b' &&
echo 'FROM deepseek-r1:14b' >> ~/modelfile/Modelfile_deepseek_14b &&
echo 'PARAMETER num_gpu 999' >> ~/modelfile/Modelfile_deepseek_14b &&
ollama pull deepseek-r1:14b  && # needs 2 A4000
ollama create deepseek-r1:14b -f ~/modelfile/Modelfile_deepseek_14b &&
# wraith-coder-7b
# LLM to auto-complete code
echo 'Ollama: downloading wraith-coder-7b' &&
echo 'FROM vanta-research/wraith-coder-7b:latest' >> ~/modelfile/Modelfile_wraith_coder_7b &&
echo 'PARAMETER num_gpu 999' >> ~/modelfile/Modelfile_wraith_coder_7b &&
ollama pull vanta-research/wraith-coder-7b:latest &&
ollama create wraith-coder-7b -f ~/modelfile/Modelfile_wraith_coder_7b &&
# nomic-embed-text
# LLM for embedding (text to vector)
echo 'Ollama: downloading nomic-embed-text' &&
echo 'FROM nomic-embed-text' >> ~/modelfile/Modelfile_nomic_embed_text &&
echo 'PARAMETER num_gpu 999' >> ~/modelfile/Modelfile_nomic_embed_text &&
ollama pull nomic-embed-text &&
ollama create nomic-embed-text -f ~/modelfile/Modelfile_nomic_embed_text &&
# Qwen3-Reranker-4B
# LLM for re ranking
echo 'Ollama: downloading Qwen3-Reranker-4B' &&
echo 'FROM dengcao/Qwen3-Reranker-4B:Q4_K_M' >> ~/modelfile/Modelfile_Qwen3_Reranker_4B &&
echo 'PARAMETER num_gpu 999' >> ~/modelfile/Modelfile_Qwen3_Reranker_4B &&
ollama pull dengcao/Qwen3-Reranker-4B:Q4_K_M &&
ollama create Qwen3-Reranker-4B -f ~/modelfile/Modelfile_Qwen3_Reranker_4B &&
# llava:34b
# vision LLMs
echo 'Ollama: downloading llava:34b' &&
echo 'FROM llava:34b' >> ~/modelfile/Modelfile_llava_34b &&
echo 'PARAMETER num_gpu 999' >> ~/modelfile/Modelfile_llava_34b &&
ollama pull llava:34b && # needs 2 A4000
ollama create llava:34b -f ~/modelfile/Modelfile_llava_34b &&
# qwen3-vl:32b
echo 'Ollama: downloading qwen3-vl:32b' &&
echo 'FROM qwen3-vl:32b' >> ~/modelfile/Modelfile_qwen3_vl_32b &&
echo 'PARAMETER num_gpu 999' >> ~/modelfile/Modelfile_qwen3_vl_32b &&
ollama pull qwen3-vl:32b && # needs 4 A4000
ollama create qwen3-vl:32b -f ~/modelfile/Modelfile_qwen3_vl_32b