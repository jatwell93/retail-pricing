"""
Improved Product Matcher Class implementation

This is the standalone class file to be imported by the main script.
"""

import re
from difflib import SequenceMatcher


class HybridProductMatcher:
    """
    Hybrid product matcher class that provides more balanced handling of pack size information
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
        Extract package size information from product name with improved pattern matching
        
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
        # Be more careful with this pattern as it can cause false matches
        s_pattern = r'(\d+)S\b'
        s_match = re.search(s_pattern, name_upper)
        if s_match:
            # Only match if the number is reasonably large
            # (to avoid matching things like "1s" or "2s" which might not be pack sizes)
            count = int(s_match.group(1))
            if count >= 10:  # Threshold to avoid false matches
                return {
                    'type': 'count',
                    'count': count
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
    
    def calculate_similarity(self, name1, name2, use_pack_info=True):
        """
        Calculate similarity between two product names
        
        Args:
            name1: First product name
            name2: Second product name
            use_pack_info: Whether to use pack size information
        
        Returns:
            Similarity score between 0 and 1
        """
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
            # One name is a subset of the other (e.g. "Panadol" vs "Panadol Extra")
            return 0.9
            
        # Use sequence matcher for base similarity
        base_similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # If pack info should not be used, return base similarity
        if not use_pack_info:
            return base_similarity
            
        # Extract pack sizes
        pack1 = self.extract_pack_size(name1)
        pack2 = self.extract_pack_size(name2)
        
        # If both have pack sizes and they're of the same type, compare them
        if pack1 and pack2 and pack1['type'] == pack2['type']:
            # Same type and count is a good indicator
            if pack1['count'] == pack2['count']:
                # Boost similarity score if pack sizes match, but more conservatively
                return min(1.0, base_similarity + 0.15)  # Reduced from 0.3
            
            # Different count for same type should slightly reduce similarity
            else:
                # Calculate what percentage different the counts are
                count_ratio = min(pack1['count'], pack2['count']) / max(pack1['count'], pack2['count'])
                
                # If counts are close (within 20%), apply a smaller penalty
                if count_ratio > 0.8:
                    return max(0.0, base_similarity - 0.05)  # Smaller penalty for small differences
                # If counts are very different, apply a larger penalty
                else:
                    return max(0.0, base_similarity - 0.1)  # Reduced from 0.2
        
        return base_similarity
        
    def find_matches(self, your_products, competitor_products, 
                  your_name_col, comp_name_col, threshold=0.6,
                  use_pack_info=True, confidence_levels=True):
        """
        Find matching products between your data and competitor data
        
        Args:
            your_products: DataFrame with your products
            competitor_products: DataFrame with competitor products
            your_name_col: Column name for your product names
            comp_name_col: Column name for competitor product names
            threshold: Minimum similarity score to consider a match
            use_pack_info: Whether to use pack size information in matching
            confidence_levels: Add confidence level classifications to matches
            
        Returns:
            List of product matches
        """
        matches = []
        
        # Add expanded names
        your_products['expanded_name'] = your_products[your_name_col].apply(self.expand_brand_name)
        
        # Add normalized names
        your_products['normalized_name'] = your_products['expanded_name'].apply(self.normalize_product_name)
        competitor_products['normalized_name'] = competitor_products[comp_name_col].apply(self.normalize_product_name)
        
        # Extract pack sizes (for later comparison)
        if use_pack_info:
            your_products['pack_size'] = your_products[your_name_col].apply(self.extract_pack_size)
            competitor_products['pack_size'] = competitor_products[comp_name_col].apply(self.extract_pack_size)
            
            # Print some examples of pack size extraction
            print("\nPack size extraction examples:")
            sample_your = your_products.head(3)
            for _, row in sample_your.iterrows():
                print(f"Product: {row[your_name_col]}")
                print(f"Detected pack: {row['pack_size']}")
                
            sample_comp = competitor_products.head(3)
            for _, row in sample_comp.iterrows():
                print(f"Competitor: {row[comp_name_col]}")
                print(f"Detected pack: {row['pack_size']}")
        
        # Create a lookup of competitor names
        comp_norm_names = set(competitor_products['normalized_name'])
        
        # First find exact matches
        print("\nFinding exact matches...")
        exact_matches = 0
        matched_indices = set()  # Track matched products
        
        for your_idx, your_row in your_products.iterrows():
            norm_name = your_row['normalized_name']
            
            if norm_name in comp_norm_names:
                # Find all matching competitor products
                comp_matches = competitor_products[competitor_products['normalized_name'] == norm_name]
                
                # If pack info is available and should be used, try to find the best pack match
                best_comp_row = None
                pack_match_quality = "Unknown"
                
                if use_pack_info and 'pack_size' in your_products.columns and your_row['pack_size'] is not None:
                    best_pack_diff = float('inf')
                    
                    for comp_idx, comp_row in comp_matches.iterrows():
                        if comp_row['pack_size'] is not None:
                            # Check if pack types match
                            if your_row['pack_size']['type'] == comp_row['pack_size']['type']:
                                # Calculate difference in counts
                                count_diff = abs(your_row['pack_size']['count'] - comp_row['pack_size']['count'])
                                
                                # Track the competitor product with the closest pack size
                                if count_diff < best_pack_diff:
                                    best_pack_diff = count_diff
                                    best_comp_row = comp_row
                                    
                                    # Determine pack match quality
                                    if count_diff == 0:
                                        pack_match_quality = "Perfect"
                                    else:
                                        pack_match_quality = "Type only"
                
                # If no pack match was found, use the first match
                if best_comp_row is None and len(comp_matches) > 0:
                    best_comp_row = comp_matches.iloc[0]
                
                # Add the match if found
                if best_comp_row is not None:
                    match_data = {
                        'your_index': your_idx,
                        'comp_index': best_comp_row.name,  # Get the index from the row
                        'your_name': your_row[your_name_col],
                        'comp_name': best_comp_row[comp_name_col],
                        'expanded_name': your_row['expanded_name'],
                        'competitor_price': best_comp_row['competitor_price'],
                        'similarity': 1.0,
                        'match_type': 'exact'
                    }
                    
                    # Add pack info if available
                    if use_pack_info:
                        match_data['your_pack'] = str(your_row['pack_size'])
                        match_data['comp_pack'] = str(best_comp_row['pack_size'])
                        match_data['pack_match'] = pack_match_quality
                    
                    # Add the match
                    matches.append(match_data)
                    matched_indices.add(your_idx)
                    exact_matches += 1
        
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
            best_pack_match = "Unknown"
            
            # Compare with each competitor product
            for comp_idx, comp_row in competitor_products.iterrows():
                # Calculate similarity with or without pack info
                similarity = self.calculate_similarity(
                    your_row['expanded_name'], 
                    comp_row[comp_name_col],
                    use_pack_info=use_pack_info
                )
                
                # Manual check of pack sizes if needed (for transparency)
                pack_match_quality = "Unknown"
                if use_pack_info and similarity > best_score:
                    if your_row['pack_size'] is not None and comp_row['pack_size'] is not None:
                        if (your_row['pack_size']['type'] == comp_row['pack_size']['type'] and 
                            your_row['pack_size']['count'] == comp_row['pack_size']['count']):
                            pack_match_quality = "Perfect"
                        elif your_row['pack_size']['type'] == comp_row['pack_size']['type']:
                            pack_match_quality = "Type only"
                        else:
                            pack_match_quality = "Mismatch"
                
                # Keep the best match above threshold
                if similarity > best_score:
                    best_score = similarity
                    best_pack_match = pack_match_quality
                    
                    # Create match data
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
                    
                    # Add pack info if available
                    if use_pack_info:
                        best_match['your_pack'] = str(your_row['pack_size'])
                        best_match['comp_pack'] = str(comp_row['pack_size'])
                        best_match['pack_match'] = pack_match_quality
            
            # Add the best match if found
            if best_match:
                matches.append(best_match)
                fuzzy_matches += 1
        
        print(f"Found {fuzzy_matches} fuzzy matches")
        print(f"Total matches: {len(matches)}")
        
        # Add confidence levels based on similarity scores and pack matching
        if confidence_levels and len(matches) > 0:
            for match in matches:
                # Start with similarity-based confidence
                if match['similarity'] >= 0.95:
                    confidence = 'High'
                elif match['similarity'] >= 0.85:
                    confidence = 'Medium'
                else:
                    confidence = 'Low'
                    
                # Adjust based on pack match quality (if available)
                if use_pack_info and 'pack_match' in match:
                    if match['pack_match'] == 'Perfect' and confidence != 'High':
                        # Upgrade confidence by one level if we have a perfect pack match
                        if confidence == 'Low':
                            confidence = 'Medium'
                        elif confidence == 'Medium':
                            confidence = 'High'
                    elif match['pack_match'] == 'Mismatch' and confidence != 'Low':
                        # Downgrade confidence by one level if we have a pack mismatch
                        if confidence == 'High':
                            confidence = 'Medium'
                        elif confidence == 'Medium':
                            confidence = 'Low'
                
                match['confidence'] = confidence
        
        return matches