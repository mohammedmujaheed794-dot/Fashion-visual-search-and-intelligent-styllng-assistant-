import os
import cv2
from glob import glob
from tqdm import tqdm
import collections
from src.config import config

class DataLoader:
    def __init__(self, dataset_path=None):
        # Force use of config.DATASET_PATH, ignoring argument for safety
        self.dataset_path = config.DATASET_PATH
        self.categories = []
        self.image_paths = []
        
    def validate_dataset(self):
        """
        Traverses the dataset directory, infers categories, and checks for valid images.
        Also infers GENDER from the folder structure (e.g. 'men', 'women').
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        print(f"Scanning dataset at: {self.dataset_path}")
        
        stats = collections.defaultdict(int)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        data_items = []

        # Assume structure: dataset_path/gender/category/image.jpg
        # Or dataset_path/category/image.jpg (uncategorized/unisex)
        
        for root, dirs, files in os.walk(self.dataset_path):
            # Infer category from the immediate parent folder name
            category = os.path.basename(root)
            
            # Skip root dataset folder
            if os.path.abspath(root) == os.path.abspath(self.dataset_path):
                category = "uncategorized"

            # Infer Gender from path
            # Split path into parts to avoid partial matches (e.g. 'women' contains 'men')
            path_parts = os.path.normpath(root).lower().split(os.sep)
            
            gender = 'Unisex' # Default
            if 'men' in path_parts or 'male' in path_parts:
                gender = 'Male'
            # Check female AFTER male to handle cases, but since we use processed parts, 
            # we must be careful. 'women' part implies female.
            # If both exist in path, usually the top level one wins or it's ambiguous.
            if 'women' in path_parts or 'female' in path_parts:
                gender = 'Female'
            
            # Simple override: if 'women' was detected, it overwrites 'men' possibility 
            # (unlikely to have both unless nested weirdly)

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(root, file)
                    
                    if self.is_valid_image(full_path):
                        # Return tuple with gender
                        data_items.append((full_path, category, gender))
                        stats[f"{gender} - {category}"] += 1
                        
                        if category not in self.categories and category != "uncategorized":
                            self.categories.append(category)
                    else:
                        try:
                            print(f"Skipping corrupted file: {full_path}")
                        except UnicodeEncodeError:
                             pass
        
        self.print_stats(stats)
        return data_items

    def is_valid_image(self, path):
        """
        Quick check if image can be opened.
        """
        try:
            img = cv2.imread(path)
            if img is not None:
                return True
            return False
        except Exception:
            return False

    def print_stats(self, stats):
        print("\n=== Dataset Statistics ===")
        print(f"Total Categories: {len(stats)}")
        total_images = sum(stats.values())
        print(f"Total Images: {total_images}")
        print("-" * 30)
        for category, count in stats.items():
            print(f"{category}: {count} images")
        print("==========================\n")

if __name__ == "__main__":
    loader = DataLoader()
    try:
        loader.validate_dataset()
    except Exception as e:
        print(f"Error: {e}")
