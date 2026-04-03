import random
from src.search_engine import SearchEngine

class Stylist:
    def __init__(self):
        self.search_engine = SearchEngine()
        
        # Simple Compatibility Rules
        # Keys should match the folder names found in dataset (normalized to lowercase)
        # This is flexible; we should update these based on actual dataset categories
        # Gender-Specific Compatibility Rules
        # Keys should match the folder names found in dataset (normalized to lowercase)
        self.male_compatibility_rules = {
            'jacket': ['pants', 'shirt', 'tshirt'],
            'shirt': ['pants', 'jacket'],
            'tshirt': ['pants', 'jacket'],
            'pants': ['shirt', 'tshirt', 'jacket'],
            # If male uploads skirt/dress by accident, recommend nothing or generic
            'skirt': [], 
            'dress': []
        }

        self.female_compatibility_rules = {
            'jacket': ['pants', 'skirt', 'dress', 'shirt', 'tshirt'],
            'shirt': ['pants', 'skirt', 'jacket'],
            'tshirt': ['pants', 'skirt', 'jacket'],
            'pants': ['shirt', 'tshirt', 'jacket'],
            'skirt': ['shirt', 'tshirt', 'jacket'],
            'dress': ['jacket']
        }
        
        # Fallback default
        self.default_rules = self.female_compatibility_rules

    def get_outfit_recommendations(self, query_img_path, gender='Female'):
        """
        Generate a complete outfit recommendation based on a query image.
        Uses Robust Semantic Classification and Gender Filtering.
        """
        # 1. Similarity Search (Now includes Semantic Prediction)
        # Pass gender filter to ensure we get relevant visual matches
        similar_items = self.search_engine.search(query_img_path, top_k=5, gender_filter=gender)
        
        if not similar_items:
            return {
                "error": "No similar items found in the dataset. Please ensure the dataset is populated."
            }
            
        # Use simple voting or the explicit prediction from the engine
        ref_item = similar_items[0]
        # Robustly use the predicted category if available, else fall back to the neighbor's tag
        ref_category = ref_item.get('predicted_semantic') or ref_item.get('category', 'uncategorized')
        ref_category = str(ref_category).lower()
        
        print(f"Detected Category: {ref_category}, Context: {gender}")
        
        # 2. Determine complementary categories based on Gender
        rules = self.female_compatibility_rules # default
        if gender and gender.lower() == 'male':
            rules = self.male_compatibility_rules
            
        complementary_cats = rules.get(ref_category, [])
        if not complementary_cats:
            # Fallback for partial matches
            for key in rules:
                if key in ref_category:
                    complementary_cats = rules[key]
                    break
        
        if not complementary_cats:
            # If no rules found (e.g. Male uploaded Skirt), maybe don't return anything or just shoes
            # For now, let's just return nothing distinctive to avoid bad recs, or 'shoes' if valid
            if gender.lower() == 'male' and ref_category in ['skirt', 'dress']:
                complementary_cats = [] 
            else:
                 complementary_cats = ['shoes'] # Default fallback (if shoes existed)

        # 3. Select items from complementary categories
        recommendations = {}
        
        recommendations['input_category'] = ref_category
        recommendations['visual_matches'] = similar_items
        recommendations['styling_suggestions'] = []
        
        explanation = [f"Since you selected a {ref_category} ({gender}), we suggest:"]
        
        if not complementary_cats:
             explanation = [f"We don't have standard recommendations for {ref_category} for {gender}."]

        for comp_cat in complementary_cats:
            # For now, get random items from that category
            items = self.search_engine.get_random_by_category(comp_cat, count=3)
            if items:
                recommendations['styling_suggestions'].append({
                    'category': comp_cat,
                    'items': items
                })
                explanation.append(f"- {comp_cat} to complete the look.")
                
        recommendations['explanation'] = " ".join(explanation)
        
        return recommendations

if __name__ == "__main__":
    # Test
    pass
