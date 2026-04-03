import argparse
import os
import sys
from src.config import config
from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.search_engine import SearchEngine
from src.stylist import Stylist
from src.evaluation import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Fashion Visual Search & Intelligent Styling Assistant")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init/Load Data Command
    init_parser = subparsers.add_parser("init", help="Scan dataset and extract features")
    init_parser.add_argument("--dataset", type=str, help="Path to dataset directory", default=config.DATASET_PATH)

    # Search Command
    search_parser = subparsers.add_parser("search", help="Search for similar items")
    search_parser.add_argument("image_path", type=str, help="Path to query image")
    search_parser.add_argument("--top_k", type=int, default=10, help="Number of results")

    # Recommend/Style Command
    style_parser = subparsers.add_parser("style", help="Get outfit recommendations")
    style_parser.add_argument("image_path", type=str, help="Path to query image")

    # Evaluate Command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model accuracy")
    eval_parser.add_argument("--top_k", type=int, default=5, help="Top K results to check")
    eval_parser.add_argument("--samples", type=int, default=100, help="Number of samples to test")

    args = parser.parse_args()

    if args.command == "init":
        # Update config path if provided
        dataset_path = args.dataset
        # Ideally, we update config or pass it down
        
        print(f"Initializing ML Pipeline with dataset: {dataset_path}")
        init_index(dataset_path)

    elif args.command == "search":
        if not os.path.exists(config.EMBEDDINGS_PATH):
            print("Error: Model not initialized. Run 'init' first.")
            return
            
        engine = SearchEngine()
        results = engine.search(args.image_path, top_k=args.top_k)
        
        print(f"\nSearch Results for: {args.image_path}")
        if not results:
            print("No matching items found.")
        else:
            for idx, res in enumerate(results):
                print(f"{idx+1}. {res['category']} - Similarity: {res['score']:.4f}")
                print(f"   Path: {res['image_path']}")

    elif args.command == "style":
        if not os.path.exists(config.EMBEDDINGS_PATH):
            print("Error: Model not initialized. Run 'init' first.")
            return

        stylist = Stylist()
        recommendation = stylist.get_outfit_recommendations(args.image_path)
        
        if "error" in recommendation:
            print(f"Error: {recommendation['error']}")
            return
            
        print("\n=== Intelligent Styling Assistant ===")
        print(f"Input Detected Category: {recommendation['input_category']}")
        print(f"Explanation: {recommendation['explanation']}")
        print("\n--- Visual Matches ---")
        for m in recommendation['visual_matches']:
            print(f"- {m['category']} ({m['score']:.2f})")
            
        print("\n--- Outfit Suggestions ---")
        for sugg in recommendation['styling_suggestions']:
            print(f"Category: {sugg['category']}")
            for item in sugg['items']:
                print(f"  * {item['image_path']}")

    elif args.command == "evaluate":
        if not os.path.exists(config.EMBEDDINGS_PATH):
            print("Error: Model not initialized. Run 'init' first.")
            return

        evaluator = Evaluator()
        evaluator.evaluate_top_k_accuracy(k=args.top_k, sample_size=args.samples)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
