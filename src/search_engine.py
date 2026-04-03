import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from src.config import config
from src.feature_extractor import FeatureExtractor

class SearchEngine:
    def __init__(self):
        self.embeddings = None
        self.filenames = None
        self.categories = None
        self.genders = None # NEW
        self.neighbors_model = None
        self.classifier = None # NEW: To predict category
        self.feature_extractor = FeatureExtractor()
        
        self.load_artifacts()

    def load_artifacts(self):
        """
        Load pre-computed embeddings and metadata.
        """
        if not os.path.exists(config.EMBEDDINGS_PATH):
            print("Embeddings not found. Please run feature extraction first.")
            return

        self.embeddings = np.load(config.EMBEDDINGS_PATH)
        
        metadata_path = os.path.join(config.ARTIFACTS_DIR, "metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.filenames = metadata['filenames']
                self.categories = metadata['categories']
                self.genders = metadata.get('genders') # Load genders
        else:
            print("Metadata not found.")
            
        if self.embeddings is not None:
            self._build_index()

    def _build_index(self):
        """
        Build NearestNeighbors index and a Classifier for robust category prediction.
        """
        num_items = len(self.embeddings)
        print(f"Building index for {num_items} items...")
        
        # 1. Similarity Index (Global)
        # Dynamic neighbors count to prevent crash on small datasets
        nn_k = min(num_items, 20)
        self.neighbors_model = NearestNeighbors(n_neighbors=nn_k, metric='cosine', algorithm='brute')
        self.neighbors_model.fit(self.embeddings)
        
        # 2. Classifier (Robust Category Prediction)
        # We use kNN classifier to vote based on neighbors. 
        if self.categories and len(set(self.categories)) > 1:
            # Dynamic neighbors for classifier too
            clf_k = min(num_items, 7)
            self.classifier = KNeighborsClassifier(n_neighbors=clf_k, metric='cosine')
            self.classifier.fit(self.embeddings, self.categories)
            print("Classifier trained for category prediction.")

    def search(self, query_img_path, top_k=5, gender_filter=None):
        """
        Search for visually similar items.
        Args:
            gender_filter: 'Male', 'Female', or None.
        """
        # 1. Extract feature
        query_emb = self.feature_extractor.extract_single_embedding(query_img_path)
        if query_emb is None:
            return []
        
        query_emb = query_emb.reshape(1, -1)
        
        # 2. Predict Category (Semantic Step)
        predicted_category = None
        if self.classifier:
            try:
                predicted_category = self.classifier.predict(query_emb)[0]
            except Exception as e:
                print(f"Prediction error: {e}")

        # 3. Constrained Search
        # 3. Hybrid Search Strategy
        # Approach: 
        # A. Semantic Search: Get candidates from predicted category
        # B. Global Search: Get candidates purely by visual similarity
        # C. Merge & Rerank
        
        candidates_indices = set()
        
        # A. Semantic/Constrained Candidates
        if predicted_category:
            filtered_indices = [i for i, cat in enumerate(self.categories) if cat == predicted_category]
            # Take all or a subset
            candidates_indices.update(filtered_indices)
            
        # B. Global Candidates (Visual Fallback)
        # Increased to 50*top_k to capture more cross-category visual matches
        # This fixes the issue where relevant items were missed because they weren't in the predicted category
        global_k = min(len(self.embeddings), max(100, top_k * 20))
        distances, global_indices = self.neighbors_model.kneighbors(query_emb, n_neighbors=global_k)
        
        # Add global indices
        for idx in global_indices[0]:
            candidates_indices.add(idx)
            
        # C. Rerank
        final_candidates_idx = list(candidates_indices)
        if not final_candidates_idx:
            return []
            
        final_emb = self.embeddings[final_candidates_idx]
        
        # Calculate Cosine Similarity Manually for the merged set
        # query_emb is (1, 2048), final_emb is (N, 2048)
        # Cosine Sim = dot(A, B) / (normA * normB) -> already normalized
        scores = np.dot(final_emb, query_emb.T).flatten()
        
        # Sort descending
        
        sorted_local_args = np.argsort(scores)[::-1]
        
        results = []
        count = 0
        for local_idx in sorted_local_args:
            if count >= top_k:
                break
                
            original_idx = final_candidates_idx[local_idx]
            path = self.filenames[original_idx]
            cat = self.categories[original_idx]
            gen = self.genders[original_idx] if self.genders else "Unisex"
            score = scores[local_idx]
            
            # --- GENDER FILTERING ---
            # If user selected Male, show Male + Unisex. Hide Female.
            # If user selected Female, show Female + Unisex. Hide Male.
            if gender_filter:
                # If item is specifically the OTHER gender, skip it
                if gender_filter.lower() == 'male' and gen.lower() == 'female':
                    continue
                if gender_filter.lower() == 'female' and gen.lower() == 'male':
                    continue
            
            # --- END FILTERING ---
            
            results.append({
                'image_path': path,
                'category': cat,
                'gender': gen,
                'score': float(score),
                'predicted_semantic': predicted_category
            })
            count += 1
            
        return results

    def get_random_by_category(self, category, count=1, gender_filter=None):
        """
        Helper to get random items from a specific category.
        """
        if self.categories is None:
            return []
            
        candidates = []
        for i, cat in enumerate(self.categories):
            if cat.lower() == category.lower():
                # Apply Gender Filter
                gen = self.genders[i] if self.genders else "Unisex"
                if gender_filter:
                     if gender_filter.lower() == 'male' and gen.lower() == 'female':
                        continue
                     if gender_filter.lower() == 'female' and gen.lower() == 'male':
                        continue
                candidates.append(i)
        
        if not candidates:
            return []
            
        selected_indices = np.random.choice(candidates, min(len(candidates), count), replace=False)
        results = []
        for idx in selected_indices:
             results.append({
                'image_path': self.filenames[idx],
                'category': self.categories[idx],
                'gender': self.genders[idx] if self.genders else "Unisex",
                'score': 1.0 
            })
        return results
