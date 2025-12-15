# # Infrastructure for your CCU requirements
# class ScalingConfig:
#     DEMO_TIER = {
#         "redis_connections": 100,
#         "celery_workers": 50,  # For 50k CCU demo
#         "llm_connections": {
#             "ollama": 20,  # Local instances
#             "openai": 5,  # Fallback only
#         },
#         "timeout": 120,  # 2 minutes max
#     }

#     FULL_TIER = {
#         "celery_workers": 10,  # For 100 CCU full processing
#         "llm_connections": {"ollama": 10, "openai": 2},
#         "timeout": 7200,  # 2 hours max
#     }
