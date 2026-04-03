import os

class Config:
    # Default dataset path (can be overridden)
    # Assuming user might put data in a 'data' folder in the project root
    # Default dataset path (Always use local data folder)
    DATASET_PATH = os.path.join(os.getcwd(), "dataset")
    
    # Image preprocessing settings
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CHANNELS = 3
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    
    # Feature Extraction
    MODEL_NAME = "ResNet50"
    WEIGHTS = "imagenet"
    INCLUDE_TOP = False
    POOLING = "avg"
    FINE_TUNE = True # Enable minimal fine-tuning logic if we were training, but for inference we rely on better pre-processing
    
    # Dataset specific
    # We will improve data loading to ensure we don't mix categories randomly
    
    # Storage paths
    ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")
    EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
    FILENAMES_PATH = os.path.join(ARTIFACTS_DIR, "filenames.npy")
    FEATURE_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "feature_index.faiss")
    
    # Create artifacts dir if it doesn't exist
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

config = Config()
