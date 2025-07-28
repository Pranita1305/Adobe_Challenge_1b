import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel #

# Define the models to download
MODELS_TO_DOWNLOAD = [
    "sentence-transformers/all-MiniLM-L6-v2",
    # "distilbert-base-uncased" # Can add DistilBERT if specifically needed later
]

# Define the directory to save the models
SAVE_DIR = Path("transformers_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def download_model(model_name: str, save_path: Path):
    """Downloads a model and its tokenizer to a specified local path."""
    print(f"Downloading {model_name} to {save_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name) #
        model = AutoModel.from_pretrained(model_name) #
        
        tokenizer.save_pretrained(save_path) #
        model.save_pretrained(save_path) #
        print(f"✅ Successfully downloaded and saved {model_name}.")
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")

if __name__ == "__main__":
    print("--- Starting Model Download ---")
    for model_name in MODELS_TO_DOWNLOAD:
        model_save_path = SAVE_DIR / model_name.replace("/", "_")
        if model_save_path.exists() and any(model_save_path.iterdir()):
            print(f"Skipping {model_name}: already exists at {model_save_path}")
        else:
            download_model(model_name, model_save_path)
    print("--- Model Download Complete ---")
