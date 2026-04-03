import os
import streamlit as st
from src.search_engine import SearchEngine
from src.config import config
import cv2

def debug_search():
    print("Initializing Search Engine...")
    engine = SearchEngine()
    
    query_image = "temp_query.jpg"
    if not os.path.exists(query_image):
        print(f"{query_image} not found. Picking a random image from dataset...")
        # Pick one from dataset
        import random
        from glob import glob
        all_images = glob(os.path.join(config.DATASET_PATH, "*", "*.jpg"))
        if not all_images:
            print("No images found in dataset.")
            return
        query_image = random.choice(all_images)
        print(f"Selected query image: {query_image}")

    print(f"Testing search with: {query_image}")
    
    # 1. Prediction
    feature = engine.feature_extractor.extract_single_embedding(query_image)
    if feature is None:
        print("Failed to extract features.")
        return

    feature = feature.reshape(1, -1)
    
    print("\n--- Category Prediction ---")
    if engine.classifier:
        try:
            pred = engine.classifier.predict(feature)[0]
            probs = engine.classifier.predict_proba(feature)
            print(f"Predicted Category: {pred}")
            print(f" probabilities: {probs}")
        except Exception as e:
            print(f"Prediction error: {e}")
    else:
        print("No classifier available.")

    # 2. Search
    print("\n--- Search Results ---")
    results = engine.search(query_image, top_k=5)
    for i, res in enumerate(results):
        print(f"Rank {i+1}:")
        print(f"  Path: {res['image_path']}")
        print(f"  Category: {res['category']}")
        print(f"  Score: {res['score']:.4f}")
        print(f"  Predicted Semantic: {res.get('predicted_semantic')}")

if __name__ == "__main__":
    debug_search()
