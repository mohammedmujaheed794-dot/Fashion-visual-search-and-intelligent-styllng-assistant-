import numpy as np
from src.search_engine import SearchEngine
from tqdm import tqdm
import random

class Evaluator:
    def __init__(self):
        self.engine = SearchEngine()
        
    def evaluate_top_k_accuracy(self, k=5, sample_size=100):
        """
        Evaluate the retrieval accuracy based on category matching.
        Accuracy = (Relevant items in Top-K) / K, averaged over queries.
        Precision@K.
        """
        if self.engine.embeddings is None:
            print("Model not ready.")
            return

        total_precision = 0
        
        # Select random samples to test
        num_samples = min(sample_size, len(self.engine.filenames))
        indices = np.random.choice(len(self.engine.filenames), num_samples, replace=False)
        
        print(f"Evaluating Top-{k} Accuracy on {num_samples} samples...")
        
        for idx in tqdm(indices):
            query_path = self.engine.filenames[idx]
            true_category = self.engine.categories[idx]
            
            # Search (excluding the query image itself ideally, but NearestNeighbors might return it at dist=0)
            # We ask for k+1 and ignore the first if it's the same image
            results = self.engine.search(query_path, top_k=k+1)
            
            relevant_count = 0
            retrieved_count = 0
            
            for res in results:
                # Skip self-match if it occurs (distance ~ 0)
                if res['image_path'] == query_path:
                    continue
                
                if retrieved_count >= k:
                    break
                    
                if res['category'] == true_category:
                    relevant_count += 1
                retrieved_count += 1
                
            precision = relevant_count / k if k > 0 else 0
            total_precision += precision
            
        avg_precision = total_precision / num_samples
        print(f"Average Precision@{k} (Category Match): {avg_precision:.4f}")
        return avg_precision

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate_top_k_accuracy()
