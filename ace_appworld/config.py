"""
======================================================================
==               MASTER CONFIGURATION FILE (ACE)                    ==
======================================================================
All parameters referenced in the paper are controlled from here.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# --- Base Setup ---
# Load .env file (OPENROUTER_API_KEY, GEMINI_API_KEY, etc.)
load_dotenv()

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# --- 1. Dataset & Paths (D_train, P) ---
# This is D_train (AppWorld training set)
# Set to "train_small" for a quick test run (10 tasks)
# Set to "train" for the full run (13,101 tasks)
APPWORLD_DATA_SPLIT = "train"

# Path to the AppWorld data directory
APPWORLD_DATA_DIR = BASE_DIR / "data"

# Path to the Playbook (P)
PLAYBOOK_PATH = BASE_DIR / "playbook.json"

# Paths for offline adaptation outputs
TRACE_OUTPUT_DIR = OUTPUT_DIR / "traces"
REFLECTION_OUTPUT_DIR = OUTPUT_DIR / "reflections"

TRACE_OUTPUT_DIR.mkdir(exist_ok=True)
REFLECTION_OUTPUT_DIR.mkdir(exist_ok=True)


# --- 2. Online Adaptation Parameters (Algorithm 2) ---

# (K) Max retries for a single task in online mode
MAX_ONLINE_RETRIES = 3


# --- 3. Agent & LLM Parameters (M_gen, M_ref, M_cur, T) ---

# (T) Max steps per episode
MAX_EPISODE_STEPS = 40

# LLM sampling temperature
TEMPERATURE = 0.7
MAX_TOKENS = 4000

# --- Generator (M_gen) ---
GENERATOR_PROVIDER = "openrouter" # "openrouter", "gemini", "ollama"
GENERATOR_MODEL = "anthropic/claude-haiku-4.5"

# --- Reflector (M_ref) ---
REFLECTOR_PROVIDER = "openrouter"
REFLECTOR_MODEL = "anthropic/claude-haiku-4.5"
# Max refinement rounds for reflection (if enabled in reflector)
REFLECTOR_MAX_REFINEMENT = 1 # Set to 1 for no iterative refinement

# --- Curator (M_cur) ---
CURATOR_PROVIDER = "openrouter"
CURATOR_MODEL = "anthropic/claude-haiku-4.5" # Can be a faster model


# --- 4. Curation & Pruning Parameters ---

# Semantic similarity threshold for deduplicating new insights
CURATOR_DEDUP_THRESHOLD = 0.9

# Pruning: Remove bullets if `harmful` > this value
CURATOR_PRUNE_HARMFUL_THRESHOLD = 3

# Pruning: Remove bullets if `helpful / (helpful + harmful)` < this ratio
# (Only triggers if HARMFUL_THRESHOLD is met)
CURATOR_PRUNE_HELPFUL_RATIO = 0.33


# --- 5. Logging Configuration ---
LOG_FILE = LOG_DIR / "ace_appworld.log"
LOG_LEVEL = logging.INFO # DEBUG, INFO, WARNING, ERROR

# --- 6. API Keys (Loaded from .env) ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

