"""
Enhanced example script for pricing system with product matching and pack size detection

Adds specific handling for package size indicators in product names.
"""

import os
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re

class ProductMatcher:
    """
    Enhanced product matching class with pack size detection
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
    
    def extract_pack_size(self, name):
        """
        Extract package size information from product name
        
        Looks for patterns like:
        - "30 tablets"
        - "60s"
        - "100 caps"
        - "20 caplets"
        - "500ml"
        - "60pk"
        
        Returns:
            Dictionary with unit type and count, or None if not found
        """
        if not isinstance(name, str):
            return None
            
        # Normalize name to uppercase for consistent matching
        name_upper = name.upper()
        
        # Try to extract various packaging patterns
        
        # Pattern: number followed by unit type (e.g., "30 TABLETS", "100 CAPSULES", "60 CAPS")
        tablet_pattern = r'(\d+)\s*(?:TABLET|TABLETS|TAB|TABS)\b'
        tablet_match = re.search(tablet_pattern, name_upper)
        if tablet_match:
            return {
                'type': 'tablets',
                'count': int(tablet_match.group(1))
            }
            
        capsule_pattern = r'(\d+)\s*(?:CAPSULE|CAPSULES|CAP|CAPS)\b'
        capsule_match = re.search(capsule_pattern, name_upper)
        if capsule_match:
            return {
                'type': 'capsules',
                'count': int(capsule_match.group(1))
            }
            
        caplet_pattern = r'(\d+)\s*(?:CAPLET|CAPLETS)\b'
        caplet_match = re.search(caplet_pattern, name_upper)
        if caplet_match:
            return {
                'type': 'caplets',
                'count': int(caplet_match.group(1))
            }
            
        # Pattern: number followed by "s" at word boundary (e.g., "60s", "100s")
        s_pattern = r'(\d+)S\b'
        s_match = re.search(s_pattern, name_upper)
        if s_match:
            return {
                'type': 'count',
                'count': int(s_match.group(1))
            }
            
        # Pattern: number followed by pack/pk (e.g., "60 PACK", "30PK")
        pack_pattern = r'(\d+)\s*(?:PACK|PK)\b'
        pack_match = re.search(pack_pattern, name_upper)
        if pack_match:
            return {
                'type': 'pack',
                'count': int(pack_match.group(1))
            }
            
        # Pattern: number followed by ml/g (e.g., "500ML", "100G")
        volume_pattern = r'(\d+)\s*(?:ML|G)\b'
        volume_match = re.search(volume_pattern, name_upper)
        if volume_match:
            return {
                'type': 'volume',
                'count': int(volume_match.group(1)),
                'unit': volume_match.group(0)[-2:].upper()  # ML or G
            }
            
        # If no patterns match, return None
        return None
    
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
            
        # Extract pack sizes
        pack1 = self.extract_pack_size(name1)
        pack2 = self.extract_pack_size(name2)
        
        # Use sequence matcher for base similarity
        base_similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # If both have pack sizes, compare them
        if pack1 and pack2:
            # Same type and count is a very strong indicator
            if pack1['type'] == pack2['type'] and pack1['count'] == pack2['count']:
                # Boost similarity score significantly if pack sizes match
                return min(1.0, base_similarity + 0.3)
            
            # Different count for same type should reduce similarity
            elif pack1['type'] == pack2['type'] and pack1['count'] != pack2['count']:
                # Penalize for different pack sizes of same type
                return max(0.0, base_similarity - 0.2)
        
        return base_similarity
        
    def find_matches(self, your_products, competitor_products, 
                  your_name_col, comp_name_col, threshold=0.6):
        """Find matching products between your data and competitor data"""
        matches = []
        
        # Add expanded names
        your_products['expanded_name'] = your_products[your_name_col].apply(self.expand_brand_name)
        
        # Add normalized names
        your_products['normalized_name'] = your_products['expanded_name'].apply(self.normalize_product_name)
        competitor_products['normalized_name'] = competitor_products[comp_name_col].apply(self.normalize_product_name)
        
        # Extract pack sizes (for later comparison)
        your_products['pack_size'] = your_products[your_name_col].apply(self.extract_pack_size)
        competitor_products['pack_size'] = competitor_products[comp_name_col].apply(self.extract_pack_size)
        
        # Print some examples of pack size extraction
        print("\nPack size extraction examples:")
        sample_your = your_products.sample(min(3, len(your_products)))
        for _, row in sample_your.iterrows():
            print(f"Product: {row[your_name_col]}")
            print(f"Detected pack: {row['pack_size']}")
            
        sample_comp = competitor_products.sample(min(3, len(competitor_products)))
        for _, row in sample_comp.iterrows():
            print(f"Competitor: {row[comp_name_col]}")
            print(f"Detected pack: {row['pack_size']}")
        
        # Create a lookup of your normalized names to original indices
        your_name_to_index = {}
        for idx, name in zip(your_products.index, your_products['normalized_name']):
            your_name_to_index[name] = idx
        
        # First find exact matches
        print("\nFinding exact matches...")
        
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
                    
                    # Compare pack sizes if available
                    pack_match_quality = "Unknown"
                    if your_row['pack_size'] is not None and comp_row['pack_size'] is not None:
                        if (your_row['pack_size']['type'] == comp_row['pack_size']['type'] and 
                            your_row['pack_size']['count'] == comp_row['pack_size']['count']):
                            pack_match_quality = "Perfect"
                        elif your_row['pack_size']['type'] == comp_row['pack_size']['type']:
                            pack_match_quality = "Type only"
                        else:
                            pack_match_quality = "Mismatch"
                    
                    matches.append({
                        'your_index': your_idx,
                        'comp_index': comp_idx,
                        'your_name': your_row[your_name_col],
                        'comp_name': comp_row[comp_name_col],
                        'expanded_name': your_row['expanded_name'],
                        'competitor_price': comp_row['competitor_price'],
                        'similarity': 1.0,
                        'match_type': 'exact',
                        'your_pack': str(your_row['pack_size']),
                        'comp_pack': str(comp_row['pack_size']),
                        'pack_match': pack_match_quality
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
                    # Compare pack sizes if available
                    pack_match_quality = "Unknown"
                    if your_row['pack_size'] is not None and comp_row['pack_size'] is not None:
                        if (your_row['pack_size']['type'] == comp_row['pack_size']['type'] and 
                            your_row['pack_size']['count'] == comp_row['pack_size']['count']):
                            pack_match_quality = "Perfect"
                            # Boost similarity for perfect pack match
                            similarity = min(1.0, similarity + 0.1)
                        elif your_row['pack_size']['type'] == comp_row['pack_size']['type']:
                            pack_match_quality = "Type only"
                        else:
                            pack_match_quality = "Mismatch"
                            # Reduce similarity for pack type mismatch
                            similarity = max(0.0, similarity - 0.1)
                    
                    best_score = similarity
                    best_match = {
                        'your_index': your_idx,
                        'comp_index': comp_idx,
                        'your_name': your_row[your_name_col],
                        'comp_name': comp_row[comp_name_col],
                        'expanded_name': your_row['expanded_name'],
                        'competitor_price': comp_row['competitor_price'],
                        'similarity': similarity,
                        'match_type': 'fuzzy',
                        'your_pack': str(your_row['pack_size']),
                        'comp_pack': str(comp_row['pack_size']),
                        'pack_match': pack_match_quality
                    }
            
            # Add the best match if found
            if best_match:
                matches.append(best_match)
                fuzzy_matches += 1
        
        print(f"Found {fuzzy_matches} fuzzy matches")
        print(f"Total matches: {len(matches)}")
        
        return matches

def main():
    print("Starting enhanced pricing system with product matching and pack size detection...")
    
    # File paths
    product_file = "pricing_test.xlsx"
    competitor_file = "competitor_prices.csv"
    output_file = "recommended_prices_with_pack.xlsx"
    match_report_file = "product_matches_with_pack.xlsx"
    
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
    |          Brand          |  Prefix  |
    |:-----------------------:|:--------:|
    | Actavis                 | ACT      |
    | Adnohr Marketing        | AND      |
    | Allersearch             | A/SEARCH |
    | Allersearch             | ALLSCH   |
    | Almay Australia         | ALM      |
    | Alphapharm              | APF      |
    | Alphapharm              | AP       |
    | Amneal Pharmaceuticals  | AN       |
    | Apo Health              | APH      |
    | Apotex                  | APO      |
    | Aurobindo Pharma        | AURO     |
    | Aussie Bodies           | AB       |
    | Banana Boat             | B/BOAT   |
    | Billie Goat             | BG       |
    | Bio Organics            | BIO      |
    | Bio Organics            | BIO/ORG  |
    | Blackmores              | BM       |
    | Body Plus               | BP       |
    | Bonne Bell              | BB       |
    | Bourjois                | BJ       |
    | Cancer Council          | CC       |
    | Cancer Council          | CAN/C    |
    | Care Plus               | C/PLUS   |
    | Celebrity Slim          | C/SLIM   |
    | Cenovis                 | CEN      |
    | Chemist Own             | CO       |
    | Chemist Own             | CHEM OWN |
    | Chemmart                | CM       |
    | Closer to Nature        | CTN      |
    | Closer to Nature        | C2N      |
    | Colgate                 | C/GATE   |
    | Cover Girl              | CG       |
    | Coverplast              | C/PLAST  |
    | David Bull Laboratories | DBL      |
    | David Craig             | D/CRAIG  |
    | Dermaveen               | DV       |
    | Dr Reddy's Laboratories | DRLA     |
    | Dr Reddy's Laboratories | DR       |
    | Elastoplast             | E/PLAST  |
    | Essensce                | ES       |
    | Garnier                 | GAR      |
    | Generic Health          | GH       |
    | GenRx                   | GRX      |
    | Gold Cross              | G/CROSS  |
    | Gold Cross              | GOLDX    |
    | Herbal Essence          | H/ESS    |
    | Innoxa                  | INN      |
    | Invisible Zinc          | INV/ZINC |
    | Invisible Zinc          | I/ZINC   |
    | Jack Black              | JB       |
    | Jean Arthes             | JA       |
    | John Freida             | JF       |
    | Johnson and Johnson     | J&J      |
    | Johnson and Johnson     | JJ       |
    | Leukoplast              | L/PLAST  |
    | L'Oreal Paris           | LOR      |
    | L'Oreal Paris           | LV       |
    | Manicare                | MCARE    |
    | Mason Pearson           | M/PERSON |
    | Max Factor              | MF       |
    | Maybelline              | MB       |
    | McGloins                | MG       |
    | Napoleon Perdis         | NP       |
    | Nature's Own            | NO       |
    | Nature's Own            | N/OWN    |
    | Original Source         | O/S      |
    | Pfizer                  | PF       |
    | Pharmacor               | PC       |
    | Pharmacy Action         | PA       |
    | Pharmacy Action         | P/ACTON  |
    | Pharmacy Choice         | PC       |
    | Pharmacy Health         | PH       |
    | Ranbaxy                 | RBX      |
    | Revlon                  | REV      |
    | Sandoz                  | SZ       |
    | Scholl                  | SC       |
    | Skin Basics             | SK/BAS   |
    | Terry White             | TW       |
    | Thatcher Eyewear        | TE       |
    | Thermoskin              | T/SKIN   |
    | Thursday Plantation     | TP       |
    | Thursday Plantation     | T/PL     |
    | Tommee Tippee           | TT       |
    | Trilogy                 | TRI      |
    | Ulta3                   | UL       |
    | Warner Brothers         | WB       |
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
    
    # Count pack match types
    if 'pack_match' in matches_df.columns:
        pack_match_counts = matches_df['pack_match'].value_counts()
        print("\nPack size match statistics:")
        for match_type, count in pack_match_counts.items():
            print(f"  {match_type}: {count}")
    
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
    products_df['pack_match'] = ''
    
    for _, match in matches_df.iterrows():
        your_idx = match['your_index']
        products_df.loc[your_idx, 'competitor_price'] = match['competitor_price']
        products_df.loc[your_idx, 'competitor_name'] = match['comp_name']
        products_df.loc[your_idx, 'match_quality'] = match['similarity']
        products_df.loc[your_idx, 'pack_match'] = match['pack_match']
    
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
    
    # Special handling for pack size mismatches
    pack_mismatch_mask = (products_df['pack_match'] == "Type only") & comp_mask
    if pack_mismatch_mask.sum() > 0:
        print(f"Adjusting prices for {pack_mismatch_mask.sum()} products with mismatched pack sizes...")
        
        # For these products, we'll try to normalize by pack size
        for idx, row in products_df[pack_mismatch_mask].iterrows():
            try:
                # Extract pack size info from match dataframe
                match_info = matches_df[matches_df['your_index'] == idx].iloc[0]
                your_pack_str = match_info['your_pack']
                comp_pack_str = match_info['comp_pack']
                
                # Parse back from string representation
                import ast
                your_pack = ast.literal_eval(your_pack_str)
                comp_pack = ast.literal_eval(comp_pack_str)
                
                if your_pack and comp_pack and your_pack['count'] > 0 and comp_pack['count'] > 0:
                    # Calculate ratio for normalization
                    ratio = your_pack['count'] / comp_pack['count']
                    
                    # Adjust competitor price by ratio
                    adjusted_comp_price = row['competitor_price'] * ratio
                    
                    # Apply competitive adjustment with normalized price
                    max_adjusted_price = adjusted_comp_price * 1.05
                    products_df.loc[idx, 'comp_adjusted_price'] = min(
                        row['base_price'],
                        max_adjusted_price
                    )
            except (IndexError, ValueError, KeyError, SyntaxError):
                # Skip if we can't parse or process
                continue
    
    # Apply price ending rules
    products_df['final_price'] = products_df['comp_adjusted_price']
    
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
    
    # Flag pack size mismatches
    pack_mismatch = products_df['pack_match'] == "Type only"
    products_df.loc[pack_mismatch, 'needs_review'] = True
    
    # Export results
    print("\nExporting price recommendations...")
    
    # Select columns for output
    output_cols = [
        'Item Code', 'Item Description', 'Item Price', 'final_price',
        'Item Cost', 'current_margin', 'recommended_margin', 'margin_change',
        'competitor_price', 'competitor_name', 'match_quality', 'pack_match', 'needs_review'
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
    
    # Add conditional formatting for pack match
    if 'pack_match' in final_cols:
        pack_col = final_cols.index('pack_match')
        worksheet.conditional_format(1, pack_col, len(products_df), pack_col, {
            'type': 'cell',
            'criteria': 'equal to',
            'value': '"Type only"',
            'format': workbook.add_format({'bg_color': '#FFEB9C'})
        })
        worksheet.conditional_format(1, pack_col, len(products_df), pack_col, {
            'type': 'cell',
            'criteria': 'equal to',
            'value': '"Mismatch"',
            'format': workbook.add_format({'bg_color': '#FFC7CE'})
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
    
    if 'pack_match' in products_df.columns:
        pack_stats = products_df['pack_match'].value_counts()
        print("\nPack match statistics:")
        for match_type, count in pack_stats.items():
            if match_type:
                print(f"  {match_type}: {count}")
    
    # Show a sample of recommendations
    print("\nSample price recommendations with pack size matching:")
    sample_products = products_df[products_df['competitor_price'].notna()].head(3)
    
    for _, row in sample_products.iterrows():
        print(f"\n  Product: {row['Item Description']}")
        print(f"    Current Price: ${row['Item Price']:.2f}")
        print(f"    Recommended Price: ${row['final_price']:.2f}")
        print(f"    Competitor Price: ${row['competitor_price']:.2f}")
        print(f"    Competitor Product: {row['competitor_name']}")
        print(f"    Match Quality: {row['match_quality']*100:.1f}%")
        if 'pack_match' in row and row['pack_match']:
            print(f"    Pack Match: {row['pack_match']}")
        print(f"    Current Margin: {row['current_margin']:.2f}%")
        print(f"    Recommended Margin: {row['recommended_margin']:.2f}%")
    
    print("\nNext steps:")
    print("1. Review the product matches in the match report")
    print("2. Check items flagged for review, especially those with pack size mismatches")
    print("3. Implement approved price changes in your system")

if __name__ == "__main__":
    main()