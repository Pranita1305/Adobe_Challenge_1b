"""
utils.py

This module serves as a repository for utility functions and global constants
that are used across different parts of the application.

Centralizing constants here makes the system easier to configure and tune without
needing to modify the core logic of the classes.
"""

# --- Ranking Engine Constants ---

# The number of top-scoring chunks within a section to average for the section's
# final aggregated score. This value balances relevance density with section length.
# A value of 3-5 is generally a good starting point.
TOP_K_CHUNKS_FOR_AGGREGATION = 3

# The number of top-ranked sections to select for the detailed
# "sub_section_analysis" in the final JSON output.
TOP_N_SECTIONS_FOR_ANALYSIS = 5


# --- Model Constants ---

# The identifier for the sentence-transformer model. This is used by both the
# download script and the embedding engine to ensure consistency.
MODEL_NAME = "thenlper/gte-small"

# The local directory where the model will be saved and loaded from for
# offline execution.
MODEL_SAVE_PATH = "models/thenlper/gte-small"

