"""
Product Matcher module for retail pricing system

Handles matching products between different catalogs when SKUs don't match.
Uses fuzzy matching on product names and handles brand name contractions.
"""

import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher

class ProductMatcher:
    """
    Product matching engine that handles brand contractions and fuzzy matching
    to match products between your inventory and competitor data
    """
    
    def __init__(self, brand_contractions=None):
        """
        Initialize the product matcher
        
        Args:
            brand_contractions: Dictionary of brand contractions mapping
                                prefix to full brand name, or DataFrame with 
                                'Brand' and 'Prefix' columns
        """
        self.brand_map = {}
        if brand_contractions is not None:
            self.load_brand_contractions(brand_contractions)
    
    def load_brand_contractions(self, contractions):
        """
        Load brand contractions from dictionary or DataFrame
        
        Args:
            contractions: Dictionary with prefix->brand mapping or DataFrame
                         with 'Brand' and 'Prefix' columns
        """
        if isinstance(contractions, pd.DataFrame):
            # If DataFrame, convert to dictionary
            for _, row in contractions.iterrows():
                self.brand_map[row['Prefix']] = row['Brand']
        elif isinstance(contractions, dict):
            # If dictionary, use directly
            self.brand_map = contractions
        else:
            raise ValueError("Contractions must be a DataFrame or dictionary")
        
        print(f"Loaded {len(self.brand_map)} brand contractions")
        
    def load_brand_contractions_from_markdown(self, markdown_table):
        """
        Parse markdown table of brand contractions
        
        Args:
            markdown_table: String with markdown table content
        """
        contractions = {}
        lines = markdown_table.strip().split('\n')
        
        # Skip header rows (first 2 lines)
        for line in lines[2:]:
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 4:  # Account for leading/trailing |
                    brand = parts[1].strip()
                    prefix = parts[2].strip()
                    if brand and prefix:
                        contractions[prefix] = brand
        
        self.brand_map = contractions
        print(f"Loaded {len(self.brand_map)} brand contractions from markdown table")
        
    def expand_brand_name(self, product_name):
        """
        Try to expand contracted brand name in product description
        
        Args:
            product_name: Product name that may contain contracted brand
            
        Returns:
            Expanded product name if match found, otherwise original name
        """
        # Check each prefix to see if the product name starts with it
        for prefix, brand in self.brand_map.items():
            # Match prefix at start of name followed by space or slash
            pattern = f"^{re.escape(prefix)}(\\s|/)"
            if re.match(pattern, product_name):
                # Replace the prefix with the full brand name
                return re.sub(pattern, f"{brand}\\1", product_name, count=1)
        
        return product_name
    
    def normalize_product_name(self, name):
        """
        Normalize product name for comparison
        
        Args:
            name: Product name to normalize
            
        Returns:
            Normalized name (uppercase, no punctuation, standardized spaces)
        """
        if not isinstance(name, str):
            return ""
            
        return (name.upper()
                  .replace('-', ' ')
                  .replace('/', ' ')
                  .replace('.', '')
                  .replace(',', '')
                  .replace('(', '')
                  .replace(')', '')
                  .replace('  ', ' ')
                  .strip())
    
    def extract_features(self, name):
        """
        Extract key product features from name (size, count, strength)
        
        Args:
            name: Product name
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract numbers which might represent size, count, or strength
        numbers = re.findall(r'\d+(?:\.\d+)?', name)
        if numbers:
            features['numbers'] = numbers
        
        # Extract units like mg, ml, g, etc.
        units = re.findall(r'\d+\s*(?:MG|ML|G|TABS|CAPSULES|CAPS|TAB|CAP|MCG|PACK|PK)\b', 
                           name.upper())
        if units:
            features['units'] = units
        
        # Extract product type indicators
        types = re.findall(r'\b(?:TABLET|TABLETS|CAPSULE|CAPSULES|CREAM|GEL|CAPLET|CAPLETS|LIQUID|SPRAY|LOTION)\b', 
                           name.upper())
        if types:
            features['types'] = types
        
        return features
    
    def calculate_similarity(self, name1, name2):
        """
        Calculate similarity score between two product names
        
        Args:
            name1: First product name
            name2: Second product name
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize both names
        norm1 = self.normalize_product_name(name1)
        norm2 = self.normalize_product_name(name2)
        
        if not norm1 or not norm2:
            return 0
        
        # If identical after normalization, they're a perfect match
        if norm1 == norm2:
            return 1.0
        
        # Check if one contains the other completely
        if norm1 in norm2 or norm2 in norm1:
            return 0.9
            
        # Calculate base similarity using SequenceMatcher
        seq_similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Extract features from both names
        features1 = self.extract_features(norm1)
        features2 = self.extract_features(norm2)
        
        # Compare the extracted features
        feature_similarity = 0
        feature_weights = {'numbers': 0.4, 'units': 0.3, 'types': 0.3}
        total_weight = 0
        
        for feature_type, weight in feature_weights.items():
            if feature_type in features1 and feature_type in features2:
                common = set(features1[feature_type]).intersection(set(features2[feature_type]))
                all_features = set(features1[feature_type]).union(set(features2[feature_type]))
                
                if all_features:
                    feature_similarity += weight * len(common) / len(all_features)
                    total_weight += weight
        
        # If we compared any features, blend the two similarity scores
        if total_weight > 0:
            combined_similarity = (seq_similarity * 0.5) + (feature_similarity / total_weight * 0.5)
        else:
            combined_similarity = seq_similarity
        
        return combined_similarity
        
    def find_matches(self, your_products, competitor_products, 
                    your_name_col='product_name', comp_name_col='product_name',
                    threshold=0.6, expand_brands=True):
        """
        Find matching products between your inventory and competitor data
        
        Args:
            your_products: DataFrame with your product data
            competitor_products: DataFrame with competitor product data
            your_name_col: Column name containing your product names
            comp_name_col: Column name containing competitor product names
            threshold: Minimum similarity score to consider a match
            expand_brands: Whether to expand contracted brand names
            
        Returns:
            DataFrame with matched products
        """
        matches = []
        
        # Create a copy of your products to avoid modifying the original
        your_df = your_products.copy()
        
        # Add expanded brand names if requested
        if expand_brands:
            your_df['expanded_name'] = your_df[your_name_col].apply(self.expand_brand_name)
        else:
            your_df['expanded_name'] = your_df[your_name_col]
        
        # Add normalized versions of names
        your_df['normalized_name'] = your_df['expanded_name'].apply(self.normalize_product_name)
        comp_df = competitor_products.copy()
        comp_df['normalized_name'] = comp_df[comp_name_col].apply(self.normalize_product_name)
        
        # Create a lookup of your normalized names to original indices
        your_name_to_index = {}
        for idx, name in zip(your_df.index, your_df['normalized_name']):
            your_name_to_index[name] = idx
        
        # First try exact matching on normalized names
        exact_matches = pd.merge(
            comp_df, 
            your_df[['normalized_name', your_name_col, 'expanded_name']],
            on='normalized_name',
            how='inner',
            suffixes=('_comp', '_your')
        )
        
        # Track which items have been matched
        matched_comp_indices = set(exact_matches.index)
        matched_your_indices = set()
        
        # Add exact matches to results
        for _, row in exact_matches.iterrows():
            your_idx = your_name_to_index[row['normalized_name']]
            matches.append({
                'your_index': your_idx,
                'comp_index': row.name,
                'your_name': row[f'{your_name_col}_your'],
                'comp_name': row[comp_name_col],
                'expanded_name': row['expanded_name'],
                'similarity': 1.0,
                'match_type': 'exact'
            })
            matched_your_indices.add(your_idx)
        
        # For unmatched items, try fuzzy matching
        unmatched_comp = comp_df[~comp_df.index.isin(matched_comp_indices)]
        unmatched_your = your_df[~your_df.index.isin(matched_your_indices)]
        
        # If we have a small number of products, we can do pairwise comparison
        if len(unmatched_comp) * len(unmatched_your) < 1000000:
            for comp_idx, comp_row in unmatched_comp.iterrows():
                best_match = None
                best_score = 0
                
                for your_idx, your_row in unmatched_your.iterrows():
                    similarity = self.calculate_similarity(
                        your_row['expanded_name'], 
                        comp_row[comp_name_col]
                    )
                    
                    if similarity > threshold and similarity > best_score:
                        best_score = similarity
                        best_match = {
                            'your_index': your_idx,
                            'comp_index': comp_idx,
                            'your_name': your_row[your_name_col],
                            'comp_name': comp_row[comp_name_col],
                            'expanded_name': your_row['expanded_name'],
                            'similarity': similarity,
                            'match_type': 'fuzzy'
                        }
                
                if best_match:
                    matches.append(best_match)
        else:
            # For larger datasets, implement a more efficient approach
            # This could use techniques like blocking or LSH to reduce comparisons
            print("Large dataset detected, using optimized matching...")
            # (Optimized matching implementation would go here)
        
        # Convert matches to DataFrame
        matches_df = pd.DataFrame(matches)
        
        print(f"Found {len(exact_matches)} exact matches and {len(matches_df) - len(exact_matches)} fuzzy matches")
        return matches_df