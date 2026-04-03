import os
import shutil
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Mapping ImageNet classes to our Fashion Categories
# This is a heuristic mapping based on common ImageNet fashion labels
FASHION_MAPPINGS = {
    'tops': [
        'jersey', 'sweatshirt', 'cardigan', 'trench_coat', 'suit', 'velvet', 'wool', 
        'cloak', 'poncho', 'kimono', 'shirt', 'blouse', 'features', 'bulletproof_vest',
        'lab_coat', 'military_uniform', 'pajama', 't-shirt'
    ],
    'bottoms': [
        'jean', 'miniskirt', 'swimming_trunks', 'maillot', 'bikini', 'skirt'
    ],
    'shoes': [
        'running_shoe', 'sneaker', 'loafer', 'cowboy_boot', 'sandal', 'clog', 'shoe_shop',
        'boot', 'high_heel', 'moccasin', 'soccer_ball' # soccer ball often misclassified for sporty shoes
    ],
    'accessories': [
        'purse', 'backpack', 'sunglasses', 'tie', 'umbrella', 'wallet', 'bag',
        'lipstick', 'perfume', 'necklace', 'hair_slide'
    ],
    'watches': [
        'digital_watch', 'analog_clock', 'stopwatch'
    ]
}

def organize_images(source_dir, dest_root):
    # Load Model with Top Layer for Classification
    print("Loading Classifier (ResNet50)...")
    model = ResNet50(weights='imagenet')
    
    # Ensure source exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} not found.")
        return

    # Get valid images
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in os.listdir(source_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    print(f"Analyzing {len(files)} images for auto-categorization...")
    
    count_moved = 0
    
    for filename in tqdm(files):
        img_path = os.path.join(source_dir, filename)
        
        try:
            # Preprocess
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Predict
            preds = model.predict(x, verbose=0)
            decoded = decode_predictions(preds, top=3)[0]
            
            # Check for matches
            # decoded is list of (id, label, prob)
            # We check the top predictions
            
            best_label = decoded[0][1].lower()
            target_category = None
            
            # Check mappings
            for cat, keywords in FASHION_MAPPINGS.items():
                # Check if closest label contains any keyword or is in list
                if any(k in best_label for k in keywords):
                    target_category = cat
                    break
                
                # Double check top 3 for specific strong signals
                if target_category is None:
                    for _, label, prob in decoded:
                        if prob > 0.1: # Threshold
                            if any(k in label.lower() for k in keywords):
                                target_category = cat
                                break
                    if target_category: break

            if target_category:
                # Create dest folder
                cat_dir = os.path.join(dest_root, target_category)
                os.makedirs(cat_dir, exist_ok=True)
                
                # Move
                final_path = os.path.join(cat_dir, filename)
                shutil.move(img_path, final_path)
                count_moved += 1
                
        except Exception as e:
            print(f"Error organizing {filename}: {e}")

    print(f"Organization Complete. Moved {count_moved} / {len(files)} items.")
    print("Remaining items in imported_gallery are likely non-fashion or ambiguous.")

if __name__ == "__main__":
    # Source: The flat folder where we dumped everything
    src = os.path.join(os.getcwd(), 'data', 'imported_gallery')
    # Dest: The root data folder where we want 'tops', 'shoes', etc.
    dst = os.path.join(os.getcwd(), 'data')
    
    organize_images(src, dst)
