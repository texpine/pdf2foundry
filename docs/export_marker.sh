#!/bin/sh
marker_single  ../pdf2foundry_input/Skull Wizards of the Chaos Caverns.pdf --output_format markdown --output_dir ../pdf2foundry_output/ --use_llm --llm_service marker.services.ollama.OllamaService --ollama_model gpt-oss:120b --ollama_base_url http://localhost:11434 --paginate_output --workers 4
# multi-file export prompt
# marker /root/input/numenera --output_format markdown --output_dir /root/output/numenera --use_llm --llm_service marker.services.claude.ClaudeService --claude_model_name claude-haiku-4-5-20251001 --claude_api_key <insert> --paginate_output --workers 2
