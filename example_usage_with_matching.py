"""
Fixed example script for pricing system with product matching

Uses the actual column names from your data files.
"""

import os
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re

class ProductMatcher:
    """
    Simplified product matching class
    """
    
    def __init__(self):
        self.brand_map = {}
        
    def load_brand_contractions_from_markdown(self, markdown_table):
        """Parse markdown table of brand contractions"""
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
        print(f"Loaded {len(self.brand_map)} brand contractions")
        
    def expand_brand_name(self, product_name):
        """Expand contracted brand name"""
        if not isinstance(product_name, str):
            return ""
            
        # Check each prefix
        for prefix, brand in self.brand_map.items():
            # Match prefix at start of name followed by space or slash
            pattern = f"^{re.escape(prefix)}(\\s|/)"
            if re.match(pattern, product_name):
                # Replace the prefix with the full brand name
                return re.sub(pattern, f"{brand}\\1", product_name, count=1)
        
        return product_name
    
    def normalize_product_name(self, name):
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
    
    def calculate_similarity(self, name1, name2):
        """Calculate similarity between two product names"""
        # Normalize names
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
            
        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, norm1, norm2).ratio()
        
    def find_matches(self, your_products, competitor_products, 
                  your_name_col, comp_name_col, threshold=0.6):
        """Find matching products between your data and competitor data"""
        matches = []
        
        # Add expanded names
        your_products['expanded_name'] = your_products[your_name_col].apply(self.expand_brand_name)
        
        # Add normalized names
        your_products['normalized_name'] = your_products['expanded_name'].apply(self.normalize_product_name)
        competitor_products['normalized_name'] = competitor_products[comp_name_col].apply(self.normalize_product_name)
        
        # Create a lookup of your normalized names to original indices
        your_name_to_index = {}
        for idx, name in zip(your_products.index, your_products['normalized_name']):
            your_name_to_index[name] = idx
        
        # First find exact matches
        print("Finding exact matches...")
        
        # Create a lookup of competitor names
        comp_norm_names = set(competitor_products['normalized_name'])
        
        # Find exact matches
        exact_matches = 0
        matched_indices = set()  # Track matched products
        
        for your_idx, your_row in your_products.iterrows():
            norm_name = your_row['normalized_name']
            
            if norm_name in comp_norm_names:
                # Find all matching competitor products
                for comp_idx, comp_row in competitor_products[
                    competitor_products['normalized_name'] == norm_name].iterrows():
                    
                    matches.append({
                        'your_index': your_idx,
                        'comp_index': comp_idx,
                        'your_name': your_row[your_name_col],
                        'comp_name': comp_row[comp_name_col],
                        'expanded_name': your_row['expanded_name'],
                        'competitor_price': comp_row['competitor_price'],
                        'similarity': 1.0,
                        'match_type': 'exact'
                    })
                    
                    matched_indices.add(your_idx)
                    exact_matches += 1
                    break  # Just use the first match for each product
        
        print(f"Found {exact_matches} exact matches")
        
        # Find fuzzy matches for remaining products
        print("Finding fuzzy matches...")
        fuzzy_matches = 0
        
        # Only process unmatched products
        unmatched_products = your_products[~your_products.index.isin(matched_indices)]
        
        # For each of your unmatched products
        for your_idx, your_row in unmatched_products.iterrows():
            best_match = None
            best_score = threshold  # Only consider matches above threshold
            
            # Compare with each competitor product
            for comp_idx, comp_row in competitor_products.iterrows():
                similarity = self.calculate_similarity(
                    your_row['expanded_name'], 
                    comp_row[comp_name_col]
                )
                
                # Keep the best match above threshold
                if similarity > best_score:
                    best_score = similarity
                    best_match = {
                        'your_index': your_idx,
                        'comp_index': comp_idx,
                        'your_name': your_row[your_name_col],
                        'comp_name': comp_row[comp_name_col],
                        'expanded_name': your_row['expanded_name'],
                        'competitor_price': comp_row['competitor_price'],
                        'similarity': similarity,
                        'match_type': 'fuzzy'
                    }
            
            # Add the best match if found
            if best_match:
                matches.append(best_match)
                fuzzy_matches += 1
        
        print(f"Found {fuzzy_matches} fuzzy matches")
        print(f"Total matches: {len(matches)}")
        
        return matches

def main():
    print("Starting simplified pricing system with product matching...")
    
    # File paths
    product_file = "pricing_test.xlsx"
    competitor_file = "competitor_prices.csv"
    output_file = "recommended_prices.xlsx"
    match_report_file = "product_matches.xlsx"
    
    # Load product data
    print("\nLoading product data...")
    products_df = pd.read_excel(product_file)
    print(f"Loaded {len(products_df)} products")
    
    # Rename columns to standardized names (preserving originals)
    column_mapping = {
        'Item Code': 'sku',
        'Item Description': 'product_name',
        'Item Price': 'current_price',
        'Item Cost': 'cost',
        'Department Description': 'department',
        'Category Description': 'category',
        'Generic': 'is_generic'
    }
    
    # Create a copy with standardized names
    products_std = products_df.copy()
    products_std = products_std.rename(columns=column_mapping)
    
    # Load competitor data
    print("\nLoading competitor data...")
    competitor_df = pd.read_csv(competitor_file)
    print(f"Loaded {len(competitor_df)} competitor products")
    
    # Brand contractions table
    brand_contractions = """
    |          Brand          |   Prefix  |
    |:-----------------------:|:---------:|
    | Actavis                 | ACT       |
    | Adnohr Marketing        | AND       |
    | Allersearch             | A/SEARCH  |
    | Allersearch             | ALLSCH    |
    | Almay Australia         | ALM       |
    | Alphapharm              | APF       |
    | Alphapharm              | AP        |
    | Amneal Pharmaceuticals  | AN        |
    | Apo Health              | APH       |
    | Apotex                  | APO       |
    | Aurobindo Pharma        | AURO      |
    | Aussie Bodies           | AB        |
    | Banana Boat             | B/BOAT    |
    | Billie Goat             | BG        |
    | Bio Organics            | BIO       |
    | Bio Organics            | BIO/ORG   |
    | Blackmores              | BM        |
    | Body Plus               | BP        |
    | Bonne Bell              | BB        |
    | Bourjois                | BJ        |
    | Cancer Council          | CC        |
    | Cancer Council          | CAN/C     |
    | Care Plus               | C/PLUS    |
    | Celebrity Slim          | C/SLIM    |
    | Cenovis                 | CEN       |
    | Chemist Own             | CO        |
    | Chemist Own             | CHEM OWN  |
    | Chemmart                | CM        |
    | Closer to Nature        | CTN       |
    | Closer to Nature        | C2N       |
    | Colgate                 | C/GATE    |
    | Cover Girl              | CG        |
    | Coverplast              | C/PLAST   |
    | David Bull Laboratories | DBL       |
    | David Craig             | D/CRAIG   |
    | Dermal Therapy          | D/THERAPY |
    | Dermaveen               | DV        |
    | Dr Reddy's Laboratories | DRLA      |
    | Dr Reddy's Laboratories | DR        |
    | Elastoplast             | E/PLAST   |
    | Essensce                | ES        |
    | Fat Blaster             | FBX       |
    | Garnier                 | GAR       |
    | Generic Health          | GH        |
    | GenRx                   | GRX       |
    | Gold Cross              | G/CROSS   |
    | Gold Cross              | GOLDX     |
    | Herbal Essence          | H/ESS     |
    | Innoxa                  | INN       |
    | Invisible Zinc          | INV/ZINC  |
    | Invisible Zinc          | I/ZINC    |
    | Jack Black              | JB        |
    | Jean Arthes             | JA        |
    | John Freida             | JF        |
    | Johnson and Johnson     | J&J       |
    | Johnson and Johnson     | JJ        |
    | Leukoplast              | L/PLAST   |
    | Leukoplast              | LP        |
    | L'Oreal Paris           | LOR       |
    | L'Oreal Paris           | LV        |
    | Manicare                | MCARE     |
    | Mason Pearson           | M/PERSON  |
    | Max Factor              | MF        |
    | Maybelline              | MB        |
    | McGloins                | MG        |
    | Napoleon Perdis         | NP        |
    | Nature's Own            | NO        |
    | Nature's Own            | N/OWN     |
    | Nexcare                 | NEX/C     |
    | Original Source         | O/S       |
    | PAIN AWAY               | ATH       |
    | Pfizer                  | PF        |
    | Pharmacor               | PC        |
    | Pharmacy Action         | PA        |
    | Pharmacy Action         | P/ACTON   |
    | Pharmacy Choice         | PC        |
    | Pharmacy Health         | PH        |
    | Priceline               | PL        |
    | Ranbaxy                 | RBX       |
    | Revlon                  | REV       |
    | Sandoz                  | SZ        |
    | Scholl                  | SC        |
    | Skin Basics             | SK/BAS    |
    | Terry White             | TW        |
    | Thatcher Eyewear        | TE        |
    | Thermoskin              | T/SKIN    |
    | Thursday Plantation     | TP        |
    | Thursday Plantation     | T/PL      |
    | Tommee Tippee           | TT        |
    | Trilogy                 | TRI       |
    | Ulta3                   | UL        |
    | Warner Brothers         | WB        |
        """
    
    # Create product matcher
    print("\nInitializing product matcher...")
    matcher = ProductMatcher()
    matcher.load_brand_contractions_from_markdown(brand_contractions)
    
    # Find matches
    print("\nFinding product matches...")
    matches = matcher.find_matches(
        products_df,
        competitor_df,
        your_name_col='Item Description',
        comp_name_col='product_name',
        threshold=0.75  # Higher threshold for better quality matches
    )
    
    # Convert matches to DataFrame
    matches_df = pd.DataFrame(matches)
    
    # Export match report
    print("\nExporting match report...")
    if len(matches_df) > 0:
        # Add columns for match validation
        matches_df['match_confirmed'] = False
        matches_df['review_notes'] = ''
        
        # Export to Excel
        matches_df.to_excel(match_report_file, index=False)
        print(f"Exported {len(matches_df)} matches to {match_report_file}")
    else:
        print("No matches found to export")
    
    # Add competitor prices to product data
    print("\nAdding competitor prices to product data...")
    products_df['competitor_price'] = np.nan
    products_df['competitor_name'] = ''
    products_df['match_quality'] = np.nan
    
    for _, match in matches_df.iterrows():
        your_idx = match['your_index']
        products_df.loc[your_idx, 'competitor_price'] = match['competitor_price']
        products_df.loc[your_idx, 'competitor_name'] = match['comp_name']
        products_df.loc[your_idx, 'match_quality'] = match['similarity']
    
    # Calculate price recommendations
    print("\nCalculating price recommendations...")
    # Use default target margin
    target_margin = 0.38
    higher_margin = 0.45
    
    # Add a base price column using target margin
    products_df['base_price'] = products_df['Item Cost'] / (1 - target_margin)
    
    # Use higher margin for products without competitor data
    no_comp_mask = products_df['competitor_price'].isna()
    products_df.loc[no_comp_mask, 'base_price'] = (
        products_df.loc[no_comp_mask, 'Item Cost'] / (1 - higher_margin)
    )
    
    # Apply competitive adjustment (max 5% above competitor)
    products_df['comp_adjusted_price'] = products_df['base_price']
    comp_mask = products_df['competitor_price'].notna()
    
    if comp_mask.sum() > 0:
        # Maximum price (competitor price + 5% premium)
        max_price = products_df.loc[comp_mask, 'competitor_price'] * 1.05
        
        # Take minimum of our calculated price and competitive ceiling
        products_df.loc[comp_mask, 'comp_adjusted_price'] = np.minimum(
            products_df.loc[comp_mask, 'base_price'],
            max_price
        )
    
    # Apply price ending rules
    products_df['final_price'] = products_df['comp_adjusted_price']
    
    # Apply price ending rules
    for idx, row in products_df.iterrows():
        price = row['comp_adjusted_price']
        
        if price < 20:
            # Items under $20: round to nearest .49 or .99
            cents = (price * 100) % 100
            
            # Determine which ending (.49 or .99) is closer
            if abs(cents - 49) <= abs(cents - 99):
                # .49 is closer
                new_price = np.floor(price) + 0.49
            else:
                # .99 is closer
                new_price = np.floor(price) + 0.99
        else:
            # Items $20+: round to nearest .99
            new_price = np.floor(price) + 0.99
            
        # Store the final price
        products_df.loc[idx, 'final_price'] = new_price
    
    # Calculate margins
    products_df['current_margin'] = (
        (products_df['Item Price'] - products_df['Item Cost']) / 
        products_df['Item Price'] * 100
    )
    
    products_df['recommended_margin'] = (
        (products_df['final_price'] - products_df['Item Cost']) / 
        products_df['final_price'] * 100
    )
    
    products_df['margin_change'] = (
        products_df['recommended_margin'] - products_df['current_margin']
    )
    
    # Flag items for review
    products_df['needs_review'] = False
    
    # Flag items with price > 20% above competitor
    if comp_mask.sum() > 0:
        products_df['price_difference_pct'] = (
            (products_df.loc[comp_mask, 'final_price'] / 
             products_df.loc[comp_mask, 'competitor_price'] - 1) * 100
        )
        
        high_price_mask = products_df['price_difference_pct'] > 20
        products_df.loc[high_price_mask, 'needs_review'] = True
    
    # Flag items with significant margin drop
    margin_drop_mask = (products_df['current_margin'] - products_df['recommended_margin']) > 20
    products_df.loc[margin_drop_mask, 'needs_review'] = True
    
    # Flag uncertain matches
    uncertain_matches = (products_df['match_quality'] < 0.9) & (products_df['match_quality'] > 0)
    products_df.loc[uncertain_matches, 'needs_review'] = True
    
    # Export results
    print("\nExporting price recommendations...")
    
    # Select columns for output
    output_cols = [
        'Item Code', 'Item Description', 'Item Price', 'final_price',
        'Item Cost', 'current_margin', 'recommended_margin', 'margin_change',
        'competitor_price', 'competitor_name', 'match_quality', 'needs_review'
    ]
    
    # Make sure all requested columns exist
    final_cols = [col for col in output_cols if col in products_df.columns]
    
    # Export to Excel
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    products_df[final_cols].to_excel(writer, sheet_name='Recommended Prices', index=False)
    
    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Recommended Prices']
    
    # Add formats
    money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
    pct_fmt = workbook.add_format({'num_format': '0.00'})
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9D9D9'})
    
    # Apply formats to specific columns
    for col_idx, col_name in enumerate(final_cols):
        # Apply header format to all columns
        worksheet.write(0, col_idx, col_name, header_fmt)
        
        # Apply number formats
        if col_name in ['Item Price', 'final_price', 'Item Cost', 'competitor_price']:
            worksheet.set_column(col_idx, col_idx, 12, money_fmt)
        elif col_name in ['current_margin', 'recommended_margin', 'margin_change']:
            worksheet.set_column(col_idx, col_idx, 12, pct_fmt)
        elif col_name == 'match_quality':
            # Format match quality as percentage
            match_fmt = workbook.add_format({'num_format': '0.00%'})
            worksheet.set_column(col_idx, col_idx, 12, match_fmt)
        else:
            worksheet.set_column(col_idx, col_idx, 15)
    
    # Add conditional formatting for review items
    if 'needs_review' in final_cols:
        review_col = final_cols.index('needs_review')
        worksheet.conditional_format(1, review_col, len(products_df), review_col, {
            'type': 'cell',
            'criteria': 'equal to',
            'value': True,
            'format': workbook.add_format({'bg_color': '#FFC7CE'})
        })
    
    # Add conditional formatting for match quality
    if 'match_quality' in final_cols:
        match_col = final_cols.index('match_quality')
        worksheet.conditional_format(1, match_col, len(products_df), match_col, {
            'type': 'cell',
            'criteria': 'less than',
            'value': 0.9,
            'format': workbook.add_format({'bg_color': '#FFEB9C'})
        })
    
    # Save the file
    writer.close()
    
    # Display summary
    print(f"\nExported price recommendations to {output_file}")
    print(f"Exported {len(matches_df)} product matches to {match_report_file}")
    
    matched_count = (products_df['competitor_price'].notna()).sum()
    print(f"\nSummary:")
    print(f"  Total products: {len(products_df)}")
    print(f"  Products with competitor prices: {matched_count} ({matched_count/len(products_df)*100:.1f}%)")
    
    if comp_mask.sum() > 0:
        avg_price_diff = products_df.loc[comp_mask, 'price_difference_pct'].mean()
        print(f"  Average price difference: {avg_price_diff:.2f}%")
    
    review_count = products_df['needs_review'].sum()
    print(f"  Items flagged for review: {review_count}")
    
    # Show a sample of recommendations
    print("\nSample price recommendations:")
    sample_products = products_df[products_df['competitor_price'].notna()].head(3)
    
    for _, row in sample_products.iterrows():
        print(f"\n  Product: {row['Item Description']}")
        print(f"    Current Price: ${row['Item Price']:.2f}")
        print(f"    Recommended Price: ${row['final_price']:.2f}")
        print(f"    Competitor Price: ${row['competitor_price']:.2f}")
        print(f"    Competitor Product: {row['competitor_name']}")
        print(f"    Match Quality: {row['match_quality']*100:.1f}%")
        print(f"    Current Margin: {row['current_margin']:.2f}%")
        print(f"    Recommended Margin: {row['recommended_margin']:.2f}%")
    
    print("\nNext steps:")
    print("1. Review the product matches in the match report")
    print("2. Check items flagged for review")
    print("3. Implement approved price changes in your system")

if __name__ == "__main__":
    main()