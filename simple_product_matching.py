"""
Simplified product matching script
"""

import pandas as pd
import os
import re
from difflib import SequenceMatcher

def normalize_product_name(name):
    """Normalize product name for comparison"""
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

def calculate_similarity(name1, name2):
    """Calculate similarity between two product names"""
    # Normalize names
    norm1 = normalize_product_name(name1)
    norm2 = normalize_product_name(name2)
    
    if not norm1 or not norm2:
        return 0
    
    # Use SequenceMatcher for fuzzy matching
    return SequenceMatcher(None, norm1, norm2).ratio()

def expand_brand_name(name, brand_map):
    """Expand contracted brand names based on mapping"""
    if not isinstance(name, str):
        return ""
        
    for prefix, brand in brand_map.items():
        if name.startswith(prefix + " ") or name.startswith(prefix + "/"):
            return name.replace(prefix, brand, 1)
    
    return name

def load_brand_contractions_from_markdown(markdown_table):
    """Parse markdown table with brand contractions"""
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
    
    return contractions

def main():
    # Load files
    print("Loading data files...")
    try:
        # Load product data
        product_file = "pricing_test.xlsx"
        if not os.path.exists(product_file):
            print(f"ERROR: Product file not found: {product_file}")
            return
            
        products_df = pd.read_excel(product_file)
        print(f"Loaded {len(products_df)} products")
        print(f"Product columns: {list(products_df.columns)}")
        
        # Load competitor data
        competitor_file = "competitor_prices.csv"
        if not os.path.exists(competitor_file):
            print(f"ERROR: Competitor file not found: {competitor_file}")
            return
            
        competitor_df = pd.read_csv(competitor_file)
        print(f"Loaded {len(competitor_df)} competitor products")
        print(f"Competitor columns: {list(competitor_df.columns)}")
        
        # Load brand contractions
        brand_contractions = """
        |          Brand          |  Prefix  |
        |:-----------------------:|:--------:|
        | Blackmores              | BM       |
        | L'Oreal Paris           | LOR      |
        | Apotex                  | APO      |
        """
        
        brand_map = load_brand_contractions_from_markdown(brand_contractions)
        print(f"Loaded {len(brand_map)} brand contractions")
        
        # Determine column names
        product_name_col = None
        for col in ['product_name', 'Item Description', 'description', 'name']:
            if col in products_df.columns:
                product_name_col = col
                break
                
        if not product_name_col:
            print("ERROR: Could not find product name column")
            return
            
        competitor_name_col = None
        for col in ['product_name', 'Item Description', 'description', 'name']:
            if col in competitor_df.columns:
                competitor_name_col = col
                break
                
        if not competitor_name_col:
            print("ERROR: Could not find competitor name column")
            return
            
        print(f"Using product name columns: '{product_name_col}' and '{competitor_name_col}'")
        
        # Add expanded brand names
        products_df['expanded_name'] = products_df[product_name_col].apply(
            lambda x: expand_brand_name(x, brand_map)
        )
        
        # Create normalized names
        products_df['normalized_name'] = products_df['expanded_name'].apply(normalize_product_name)
        competitor_df['normalized_name'] = competitor_df[competitor_name_col].apply(normalize_product_name)
        
        # Initialize results
        matches = []
        
        # Try exact matching
        print("\nLooking for exact matches...")
        
        # First create a lookup for quick matching
        product_lookup = {}
        for idx, name in zip(products_df.index, products_df['normalized_name']):
            product_lookup[name] = idx
            
        # Find exact matches
        exact_match_count = 0
        for comp_idx, comp_row in competitor_df.iterrows():
            comp_norm_name = comp_row['normalized_name']
            if comp_norm_name in product_lookup:
                exact_match_count += 1
                
        print(f"Found {exact_match_count} exact matches")
        
        # Try fuzzy matching
        print("\nTrying fuzzy matching for sample products...")
        
        # Get 5 random products for demonstration
        sample_products = products_df.sample(min(5, len(products_df)))
        sample_competitors = competitor_df.sample(min(5, len(competitor_df)))
        
        print("\nSample similarity scores:")
        for _, prod_row in sample_products.iterrows():
            for _, comp_row in sample_competitors.iterrows():
                prod_name = prod_row[product_name_col]
                comp_name = comp_row[competitor_name_col]
                expanded_name = prod_row['expanded_name']
                
                similarity = calculate_similarity(expanded_name, comp_name)
                
                print(f"Similarity: {similarity:.2f}")
                print(f"  Your product: {prod_name}")
                print(f"  Expanded name: {expanded_name}")
                print(f"  Competitor product: {comp_name}")
                print()
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()