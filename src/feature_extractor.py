import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import os
from tqdm import tqdm
from src.config import config
import pickle

class FeatureExtractor:
    def __init__(self):
        # Initialize ResNet50 model
        base_model = ResNet50(weights=config.WEIGHTS, 
                              include_top=config.INCLUDE_TOP, 
                              pooling=config.POOLING,
                              input_shape=config.INPUT_SHAPE)
        self.model = base_model
        print("Feature Extractor Model (ResNet50) loaded successfully.")

    def preprocess_image(self, img_path):
        """
        Load and preprocess image for ResNet50.
        Includes Center Crop to focus on object and avoid background noise.
        """
        try:
            # 1. Load at slightly larger scale
            img = image.load_img(img_path, target_size=(256, 256))
            img_array = image.img_to_array(img)
            
            # 2. Center Crop to 224x224
            h, w, _ = img_array.shape
            start_x = w//2 - (config.IMG_WIDTH//2)
            start_y = h//2 - (config.IMG_HEIGHT//2)
            img_array = img_array[start_y:start_y+config.IMG_HEIGHT, start_x:start_x+config.IMG_WIDTH, :]
            
            # 3. Preprocess
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None

    def extract_single_embedding(self, img_path):
        """
        Extract embedding for a single image (used for query).
        """
        preprocessed_img = self.preprocess_image(img_path)
        if preprocessed_img is None:
            return None
        
        # Predict returns (1, 2048) vector for ResNet50 with avg pooling
        features = self.model.predict(preprocessed_img)
        
        # Flatten and normalize
        features = features.flatten()
        features = features / np.linalg.norm(features) # L2 Normalization
        return features

    def extract_dataset_embeddings(self, image_paths):
        """
        Extract embeddings for a list of images.
        Returns: numpy array of embeddings.
        """
        embeddings = []
        valid_paths = []
        
        print("Extracting features from dataset...")
        for img_path in tqdm(image_paths):
            emb = self.extract_single_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)
                valid_paths.append(img_path)
        
        return np.array(embeddings), valid_paths

    def save_embeddings(self, embeddings, filenames, categories, genders=None):
        """
        Save embeddings, filenames, categories, and genders to disk.
        """
        np.save(config.EMBEDDINGS_PATH, embeddings)
        
        if genders is None:
            genders = ['Unisex'] * len(filenames)

        # Save metadata
        metadata = {
            'filenames': filenames,
            'categories': categories,
            'genders': genders
        }
        
        metadata_path = os.path.join(config.ARTIFACTS_DIR, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"Saved {len(embeddings)} embeddings to {config.EMBEDDINGS_PATH}")
        print(f"Saved metadata to {metadata_path}")

if __name__ == "__main__":
    # Test block
    if os.path.exists(config.DATASET_PATH):
        from src.data_loader import DataLoader
        loader = DataLoader()
        paths, cats = loader.validate_dataset()
        
        # We need to map paths to categories if we want to save them
        # Re-deriving categories for the valid paths list
        valid_cats = []
        # This is a bit inefficient, better to let DataLoader return tuples or dicts
        # But for now, we'll infer again or just run the extractor
        
        extractor = FeatureExtractor()
        # For demonstration, only processing first 5 if many
        embeddings, valid_files = extractor.extract_dataset_embeddings(paths[:10])
        print(f"Extracted shape: {embeddings.shape}")
