import os


ROOT = os.path.dirname(__file__)

DATA_DIR = os.path.join(ROOT, "data")
INDEX_CACHE_DIR = os.path.join(ROOT, ".index_cache")
LP_DOCLING_CACHE_DIR = os.path.join(ROOT, ".lp_doc_cache")
LP_PLUMBER_CACHE_DIR = os.path.join(ROOT, ".lp_plum_cache")

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
GROQ_GEMMA_9B = "groq:gemma2-9b-it"
GROQ_LLAMA_90B = "groq:llama-3.2-90b-vision-preview"

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = "11434"

LP_DOCLING_COLLECTION_NAME = "lp_docling_collection"
LP_PLUMBER_COLLECTION_NAME = "lp_plumber_collection"
