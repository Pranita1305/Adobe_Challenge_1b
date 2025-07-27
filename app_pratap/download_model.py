"""
download_model.py

This script is a one-time utility to download the required sentence-transformer
model from the Hugging Face Hub and save it to a local directory.

Running this script ensures that the main application can run in a fully
offline environment by loading the model from the local file system instead of
the internet.

This script needs to be run once before executing main.py.
"""
import os
from sentence_transformers import SentenceTransformer
from utils import MODEL_NAME, MODEL_SAVE_PATH

def download_model():
    """
    Downloads and saves the specified sentence-transformer model.
    """
    print(f"Attempting to download model: '{MODEL_NAME}'")
    
    # Check if the model directory already exists
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Model directory already exists at '{MODEL_SAVE_PATH}'.")
        # You can perform a more thorough check here if needed, e.g., verify contents
        print("Skipping download. If you need to re-download, please delete the directory first.")
        return

    try:
        # Instantiate the model. This will download it from the Hub to a cache.
        print("Downloading model from Hugging Face Hub... (this may take a moment)")
        model = SentenceTransformer(MODEL_NAME)
        
        # Create the target directory if it doesn't exist
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        # Save the model from the cache to our specified local directory
        print(f"Saving model to local directory: '{MODEL_SAVE_PATH}'")
        model.save(MODEL_SAVE_PATH)
        
        print("\n" + "="*50)
        print("Model downloaded and saved successfully!")
        print(f"The application can now run offline using the model at '{MODEL_SAVE_PATH}'")
        print("="*50)

    except Exception as e:
        print(f"\nAn error occurred during the download process: {e}")
        print("Please check your internet connection and ensure you have the 'sentence-transformers' library installed.")
        print("You can install it with: pip install -U sentence-transformers")

if __name__ == "__main__":
    download_model()

