import os
from dotenv import load_dotenv

load_dotenv()

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Agent settings
MAX_TOKENS = 4096
TEMPERATURE = 0.5

# Search settings
MAX_ARXIV_RESULTS = 3
MAX_SEMANTIC_RESULTS = 3
MAX_WIKI_SENTENCES = 10