import os


ROOT = os.path.dirname(__file__)

DATA_DIR = os.path.join(ROOT, "data")
INDEX_CACHE_DIR = os.path.join(ROOT, ".index_cache")

GROUND_TRUTH_ACTUAL_JSON = os.path.join(
    DATA_DIR, "ground_truth_20240313_111832_actual_prompt_1_4.json"
)
GROUND_TRUTH_USER_JSON = os.path.join(
    DATA_DIR, "ground_truth_20240313_111832_user_prompt_1_4.json"
)


EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
TOGETHER_META_LLAMA_70B_FREE = "together:meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
TOGETHER_META_LLAMA_VISION_FREE = "together:meta-llama/Llama-Vision-Free"
TOGETHER_DEEPSEEK_DISTILL_LLAMA_70B_FREE = (
    "together:deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
)


dummy_full_list_of_devices = [
    "LIFEPAK 15",
    "LIFEPAK 20",
    "Mizuho 6800",
    "Philips M3002",
    "Defib Misc",
    "Philips M3015",
    "Philips M4841",
    "Philips V60 Vent",
]
dummy_full_list_of_devices_lower = tuple(
    map(lambda x: x.lower(), dummy_full_list_of_devices)
)
