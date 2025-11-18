# Cost-optimized model configuration
MODEL_CONFIG = {
    "demo_tier": {
        # Fast, cheap models for 50k CCU
        "content_analysis": "ollama:mistral:7b",
        "entity_extraction": "gpt-3.5-turbo",  # Only if Ollama fails
        "vision": "moondream2",  # Lightweight local
        "image_generation": "sdxl-turbo",  # Fast local generation
    },
    "full_tier": {
        # Better quality for paid users (100 CCU)
        "content_analysis": "ollama:llama3.3:70b",
        "entity_extraction": "ollama:mixtral:8x7b",
        "complex_parsing": "gpt-4o-mini",  # Only for difficult content
        "vision": "llava:34b",
        "image_generation": "stable-diffusion-3",
    },
}

# Ollama models configuration
OLLAMA_MODELS = {
    "content_analysis": "llama3.3:70b",  # Best open-source for complex reasoning
    "entity_extraction": "mixtral:8x7b",  # Good balance of speed/quality
    "simple_tasks": "mistral:7b",  # Fast, efficient for basic extractions
    "code_generation": "qwen2.5-coder:32b",  # For future JS/TS generation
}
