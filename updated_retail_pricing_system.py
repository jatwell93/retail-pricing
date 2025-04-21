"""
Enhanced Retail Pricing System with product matching capabilities

Extends the original RetailPricingSystem with functions to match products
between your inventory and competitor data when SKUs don't match.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from product_matcher import ProductMatcher

class EnhancedRetailPricingSystem:
    """
    Enhanced pricing management system with product matching capabilities
    
    Extends the original RetailPricingSystem to handle product matching
    when SKUs don't align between your system and competitor data.
    """
    
    def __init__(self, original_system=None, target_margin=0.38, no_competitor_margin=0.45):
        """
        Initialize the enhanced pricing system
        
        Args:
            original_system: Instance of RetailPricingSystem to wrap (optional)
            target_margin: Default target margin (38%)
            no_competitor_margin: Higher margin for products without competitor data (45%)
        """
        if original_system:
            # If we're wrapping an existing system, copy its attributes
            self.__dict__ = original_system.__dict__.copy()
        else:
            # Otherwise initialize as a new system
            self.target_margin = target_margin
            self.no_competitor_margin = no_competitor_margin
            self.products_df = None
            self.competitor_data = None
            self.last_update = None
        
        # Initialize the product matcher
        self.product_matcher = ProductMatcher()
        
    def load_brand_contractions(self, contractions):
        """
        Load brand contractions into the product matcher
        
        Args:
            contractions: Dictionary, DataFrame, or markdown table with contractions
        """
        if isinstance(contractions, str) and '|' in contractions:
            # If it looks like a markdown table, parse it
            self.product_matcher.load_brand_contractions_from_markdown(contractions)
        else:
            # Otherwise pass it to the regular loader
            self.product_matcher.load_brand_contractions(contractions)
            
    def load_competitor_data_with_matching(self, file_path, match_threshold=0.6):
        """
        Load competitor pricing data with product matching
        
        Matches competitor products to your products based on name similarity
        when SKUs don't match directly.
        
        Args:
            file_path: Path to competitor data file
            match_threshold: Minimum similarity score for fuzzy matching (0.0-1.0)
            
        Returns:
            DataFrame with matched competitor data merged with your products
        """
        # Ensure product data is loaded first
        if self.products_df is None:
            raise ValueError("Product data must be loaded before competitor data")
            
        # Load competitor data
        if file_path.endswith('.csv'):
            competitor_df = pd.read_csv(file_path)
        else:
            competitor_df = pd.read_excel(file_path)
            
        print(f"Loaded competitor data with columns: {list(competitor_df.columns)}")
        
        # Store original competitor data
        self.original_competitor_data = competitor_df.copy()
        
        # Try direct SKU matching first if both dataframes have a 'sku' column
        if 'sku' in competitor_df.columns and 'sku' in self.products_df.columns:
            print("Attempting direct SKU matching...")
            
            # Count matches by SKU
            direct_matches = competitor_df[competitor_df['sku'].isin(self.products_df['sku'])]
            if len(direct_matches) > 0:
                print(f"Found {len(direct_matches)} direct SKU matches")
                
                # If we have a good number of direct matches, use them
                if len(direct_matches) >= 0.25 * len(competitor_df):
                    print(f"Using direct SKU matching ({len(direct_matches)} matches)")
                    
                    # Ensure competitor_price column exists
                    price_col = None
                    for col in ['competitor_price', 'Price', 'price']:
                        if col in competitor_df.columns:
                            price_col = col
                            break
                    
                    if price_col is None:
                        print(f"Warning: No price column found in competitor data. Available columns: {list(competitor_df.columns)}")
                        return self.products_df
                    
                    # Rename the price column if needed
                    if price_col != 'competitor_price':
                        competitor_df = competitor_df.rename(columns={price_col: 'competitor_price'})
                    
                    # Merge with product data
                    self.products_df = pd.merge(
                        self.products_df,
                        competitor_df[['sku', 'competitor_price']],
                        on='sku',
                        how='left'
                    )
                    
                    print(f"Added competitor data for {self.products_df['competitor_price'].notna().sum()} products")
                    return self.products_df
        
        # If direct matching didn't work or found too few matches, use name matching
        print("Using product name matching...")
        
        # Determine the product name column in your products dataframe
        your_name_col = None
        for col in ['product_name', 'Item Description', 'description', 'name']:
            if col in self.products_df.columns:
                your_name_col = col
                break
                
        if your_name_col is None:
            print(f"Error: Could not find product name column in your data. Available columns: {list(self.products_df.columns)}")
            return self.products_df
        
        # Determine the product name column in competitor dataframe
        comp_name_col = None
        for col in ['product_name', 'Item Description', 'description', 'name']:
            if col in competitor_df.columns:
                comp_name_col = col
                break
                
        if comp_name_col is None:
            print(f"Error: Could not find product name column in competitor data. Available columns: {list(competitor_df.columns)}")
            return self.products_df
            
        # Determine the price column in competitor dataframe
        comp_price_col = None
        for col in ['competitor_price', 'Price', 'price']:
            if col in competitor_df.columns:
                comp_price_col = col
                break
                
        if comp_price_col is None:
            print(f"Error: Could not find price column in competitor data. Available columns: {list(competitor_df.columns)}")
            return self.products_df
            
        print(f"Using columns: Your products: '{your_name_col}', Competitor: '{comp_name_col}', Price: '{comp_price_col}'")
        
        # Debug column presence
        print(f"Checking columns in competitor data: {list(competitor_df.columns)}")
        sample_row = competitor_df.iloc[0] if len(competitor_df) > 0 else None
        if sample_row is not None:
            print(f"Sample competitor data row: {dict(sample_row)}")
            
        print(f"Checking columns in product data: {list(self.products_df.columns)}")
        sample_row = self.products_df.iloc[0] if len(self.products_df) > 0 else None
        if sample_row is not None:
            print(f"Sample product data row: {dict(sample_row)}")
            
        try:
            # Find matches between your products and competitor products
            matches = self.product_matcher.find_matches(
                self.products_df, 
                competitor_df,
                your_name_col=your_name_col,
                comp_name_col=comp_name_col,
                threshold=match_threshold,
                expand_brands=True
            )
        except Exception as e:
            print(f"Error during product matching: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Return empty matches to continue without matching
            matches = pd.DataFrame(columns=['your_index', 'comp_index', 'your_name', 'comp_name', 
                                         'expanded_name', 'similarity', 'match_type'])
        
        if len(matches) == 0:
            print("No matches found between your products and competitor data")
            return self.products_df
        
        # Create a mapping from your product index to competitor price
        price_map = {}
        for _, match in matches.iterrows():
            your_idx = match['your_index']
            comp_idx = match['comp_index']
            
            # Get the competitor price
            comp_price = competitor_df.loc[comp_idx, comp_price_col]
            
            # Only add if we have a valid price
            if pd.notna(comp_price):
                price_map[your_idx] = {
                    'competitor_price': comp_price,
                    'competitor_name': match['comp_name'],
                    'match_quality': match['similarity']
                }
        
        # Add competitor data to your products
        if 'competitor_price' not in self.products_df.columns:
            self.products_df['competitor_price'] = np.nan
            
        if 'competitor_name' not in self.products_df.columns:
            self.products_df['competitor_name'] = ''
            
        if 'match_quality' not in self.products_df.columns:
            self.products_df['match_quality'] = np.nan
        
        for idx, data in price_map.items():
            self.products_df.loc[idx, 'competitor_price'] = data['competitor_price']
            self.products_df.loc[idx, 'competitor_name'] = data['competitor_name']
            self.products_df.loc[idx, 'match_quality'] = data['match_quality']
        
        # Store match results for reference
        self.match_results = matches
        
        print(f"Added competitor prices for {len(price_map)} products via name matching")
        
        # Flag uncertain matches for review
        if 'match_quality' in self.products_df.columns:
            uncertain_matches = (
                (self.products_df['match_quality'] < 0.8) & 
                (self.products_df['match_quality'] >= match_threshold)
            )
            uncertain_count = uncertain_matches.sum()
            
            if 'needs_review' not in self.products_df.columns:
                self.products_df['needs_review'] = False
                
            self.products_df.loc[uncertain_matches, 'needs_review'] = True
            
            print(f"Flagged {uncertain_count} uncertain matches (similarity score < 0.8) for review")
        
        return self.products_df
    
    def export_match_report(self, output_path="product_matches.xlsx"):
        """
        Export a report of product matches for review
        
        Args:
            output_path: Path to save the match report
            
        Returns:
            Path to saved file
        """
        if not hasattr(self, 'match_results') or self.match_results is None:
            print("No match results available - run load_competitor_data_with_matching() first")
            return None
            
        # Prepare data for export
        match_df = self.match_results.copy()
        
        # Add columns for match validation
        match_df['match_confirmed'] = False
        match_df['review_notes'] = ''
        
        # Export to Excel with formatting
        if output_path.endswith('.csv'):
            match_df.to_csv(output_path, index=False)
        else:
            writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
            match_df.to_excel(writer, sheet_name='Product Matches', index=False)
            
            # Apply formatting
            workbook = writer.book
            worksheet = writer.sheets['Product Matches']
            
            # Define formats
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9D9D9'})
            highlight_fmt = workbook.add_format({'bg_color': '#FFEB9C'})
            
            # Format headers
            for col_idx, col_name in enumerate(match_df.columns):
                worksheet.write(0, col_idx, col_name, header_fmt)
                worksheet.set_column(col_idx, col_idx, 20)
            
            # Highlight low confidence matches
            similarity_col = match_df.columns.get_loc('similarity')
            worksheet.conditional_format(1, similarity_col, len(match_df), similarity_col, {
                'type': 'cell',
                'criteria': 'less than',
                'value': 0.8,
                'format': highlight_fmt
            })
            
            writer.close()
            
        print(f"Exported match report to {output_path}")
        return output_path
        
    def get_unmatched_products(self):
        """
        Get a list of products without competitor matches
        
        Returns:
            DataFrame of products without competitor data
        """
        if self.products_df is None:
            print("No product data loaded")
            return None
            
        if 'competitor_price' not in self.products_df.columns:
            print("No competitor data loaded")
            return self.products_df
            
        return self.products_df[self.products_df['competitor_price'].isna()]
        
    def get_match_quality_summary(self):
        """
        Get a summary of match quality statistics
        
        Returns:
            Dictionary with match quality statistics
        """
        if not 'match_quality' in self.products_df.columns:
            print("No match quality data available")
            return None
            
        # Filter to only products with a match
        matched = self.products_df[self.products_df['match_quality'].notna()]
        
        # Categorize matches
        high_quality = (matched['match_quality'] >= 0.9).sum()
        medium_quality = ((matched['match_quality'] >= 0.8) & (matched['match_quality'] < 0.9)).sum()
        low_quality = ((matched['match_quality'] >= 0.6) & (matched['match_quality'] < 0.8)).sum()
        
        # Create summary statistics
        summary = {
            'total_products': len(self.products_df),
            'matched_products': len(matched),
            'match_coverage': len(matched) / len(self.products_df) * 100,
            'high_quality_matches': high_quality,
            'medium_quality_matches': medium_quality,
            'low_quality_matches': low_quality,
            'avg_match_quality': matched['match_quality'].mean()
        }
        
        return summary
        
    def generate_pricing(self):
        """
        Run the complete pricing pipeline from the original RetailPricingSystem
        with additional match quality columns
        """
        # First use the original generate_pricing method
        from retail_pricing_system import RetailPricingSystem
        
        # Check if we're already running in an instance of RetailPricingSystem
        if isinstance(self, RetailPricingSystem):
            # Call the original method using super()
            final_df = RetailPricingSystem.generate_pricing(self)
        else:
            # Create a temporary instance to call the method
            temp_system = RetailPricingSystem(
                target_margin=self.target_margin,
                no_competitor_margin=self.no_competitor_margin
            )
            
            # Copy our data to the temporary system
            temp_system.products_df = self.products_df.copy()
            
            # Call generate_pricing on the temporary system
            final_df = temp_system.generate_pricing()
            
            # Update our instance with the processed data
            self.products_df = temp_system.products_df.copy()
        
        # Add match quality columns if available
        match_columns = ['match_quality', 'competitor_name']
        for col in match_columns:
            if col in self.products_df.columns and col not in final_df.columns:
                final_df[col] = self.products_df[col]
        
        return final_df
    
    def export_pricing(self, output_path="recommended_prices.xlsx"):
        """
        Export pricing recommendations with match quality information
        """
        # Create a wrapper around the original method
        from retail_pricing_system import RetailPricingSystem
        
        # Get the final dataframe
        if 'final_price' not in self.products_df.columns:
            print("No recommended prices generated - run generate_pricing() first")
            return None
            
        # Select columns for output
        output_cols = [
            'sku', 'product_name', 'current_price', 'final_price', 
            'cost', 'current_margin', 'recommended_margin', 'margin_change',
            'needs_review'
        ]
        
        # Add competitor and match quality columns if available
        extra_cols = ['competitor_price', 'price_difference_pct', 
                     'competitor_name', 'match_quality']
        
        for col in extra_cols:
            if col in self.products_df.columns:
                output_cols.append(col)
                
        # Make sure all requested columns exist
        final_cols = [col for col in output_cols if col in self.products_df.columns]
        
        # Export based on file extension
        if output_path.endswith('.csv'):
            self.products_df[final_cols].to_csv(output_path, index=False)
        else:
            # Create a formatted Excel export
            writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
            self.products_df[final_cols].to_excel(writer, sheet_name='Recommended Prices', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Recommended Prices']
            
            # Add formats
            money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
            pct_fmt = workbook.add_format({'num_format': '0.00'})
            match_fmt = workbook.add_format({'num_format': '0.00%'})
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9D9D9'})
            
            # Apply formats to specific columns
            for col_idx, col_name in enumerate(final_cols):
                # Apply header format to all columns
                worksheet.write(0, col_idx, col_name, header_fmt)
                
                # Apply number formats
                if col_name in ['current_price', 'final_price', 'cost', 'competitor_price']:
                    worksheet.set_column(col_idx, col_idx, 12, money_fmt)
                elif col_name in ['current_margin', 'recommended_margin', 'margin_change', 'price_difference_pct']:
                    worksheet.set_column(col_idx, col_idx, 12, pct_fmt)
                elif col_name == 'match_quality':
                    worksheet.set_column(col_idx, col_idx, 12, match_fmt)
                else:
                    worksheet.set_column(col_idx, col_idx, 15)
            
            # Add conditional formatting for flagged items
            if 'needs_review' in final_cols:
                review_col = final_cols.index('needs_review')
                worksheet.conditional_format(1, review_col, len(self.products_df), review_col, {
                    'type': 'cell',
                    'criteria': 'equal to',
                    'value': True,
                    'format': workbook.add_format({'bg_color': '#FFC7CE'})
                })
                
            # Add conditional formatting for match quality
            if 'match_quality' in final_cols:
                match_col = final_cols.index('match_quality')
                # Highlight low confidence matches
                worksheet.conditional_format(1, match_col, len(self.products_df), match_col, {
                    'type': 'cell',
                    'criteria': 'less than',
                    'value': 0.8,
                    'format': workbook.add_format({'bg_color': '#FFEB9C'})
                })
            
            # Save the file
            writer.close()
        
        print(f"Exported pricing recommendations to {output_path}")
        return output_path
        
    def export_review_items(self, output_path="items_for_review.xlsx"):
        """
        Export items flagged for manual review with match quality information
        """
        if 'needs_review' not in self.products_df.columns:
            print("No review flags found - run generate_pricing() first")
            return None
            
        # Filter items needing review
        review_items = self.products_df[self.products_df['needs_review'] == True].copy()
        
        if len(review_items) == 0:
            print("No items flagged for review")
            return None
            
        # Add review reason column
        review_items['review_reason'] = ''
        
        # Flag competitive price issues
        if 'price_difference_pct' in review_items.columns:
            high_price_mask = review_items['price_difference_pct'] >= 20
            review_items.loc[high_price_mask, 'review_reason'] = 'Price >20% above competitor'
        
        # Flag margin drop issues
        margin_drop_mask = (
            review_items['current_margin'].notna() & 
            review_items['recommended_margin'].notna() &
            (review_items['current_margin'] - review_items['recommended_margin'] > 20)
        )
        
        # Update review reasons
        review_items.loc[margin_drop_mask & (review_items['review_reason'] == ''), 'review_reason'] = 'Margin drop >20%'
        
        # Flag negative margins
        negative_margin_mask = review_items['recommended_margin'] < 0
        review_items.loc[negative_margin_mask & (review_items['review_reason'] == ''), 'review_reason'] = 'Negative margin'
        
        # Flag low match quality
        if 'match_quality' in review_items.columns:
            low_match_mask = (review_items['match_quality'] < 0.8) & (review_items['match_quality'].notna())
            review_items.loc[low_match_mask & (review_items['review_reason'] == ''), 'review_reason'] = 'Low match confidence'
            
            # Update multiple reasons
            both1_mask = high_price_mask & low_match_mask
            if both1_mask.any():
                review_items.loc[both1_mask, 'review_reason'] = 'Price >20% above competitor AND Low match confidence'
                
            both2_mask = margin_drop_mask & low_match_mask
            if both2_mask.any():
                review_items.loc[both2_mask, 'review_reason'] = 'Margin drop >20% AND Low match confidence'
        
        # Handle multiple issues between price and margin
        if 'price_difference_pct' in review_items.columns:
            both_mask = high_price_mask & margin_drop_mask
            if both_mask.any():
                review_items.loc[both_mask, 'review_reason'] = 'Price >20% above competitor AND Margin drop >20%'
        
        # Select columns for output
        output_cols = [
            'sku', 'product_name', 'current_price', 'final_price', 
            'cost', 'current_margin', 'recommended_margin', 'margin_change',
            'review_reason'
        ]
        
        # Add competitor and match quality columns if available
        extra_cols = ['competitor_price', 'price_difference_pct', 
                     'competitor_name', 'match_quality']
        
        for col in extra_cols:
            if col in review_items.columns:
                # Insert at appropriate position
                if col in ['competitor_price', 'price_difference_pct']:
                    idx = output_cols.index('final_price') + 1
                    output_cols.insert(idx, col)
                else:
                    # Add at the end before review_reason
                    idx = output_cols.index('review_reason')
                    output_cols.insert(idx, col)
                
        # Make sure all requested columns exist
        final_cols = [col for col in output_cols if col in review_items.columns]
        
        # Export to file
        if output_path.endswith('.csv'):
            review_items[final_cols].to_csv(output_path, index=False)
        else:
            # Create Excel file with proper formatting
            writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
            review_items[final_cols].to_excel(writer, sheet_name='Items for Review', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Items for Review']
            
            # Add formats
            money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
            pct_fmt = workbook.add_format({'num_format': '0.00'})
            match_fmt = workbook.add_format({'num_format': '0.00%'})
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9D9D9'})
            
            # Apply formats to specific columns
            for col_idx, col_name in enumerate(final_cols):
                # Apply header format to all columns
                worksheet.write(0, col_idx, col_name, header_fmt)
                
                # Apply number formats
                if col_name in ['current_price', 'final_price', 'cost', 'competitor_price']:
                    worksheet.set_column(col_idx, col_idx, 12, money_fmt)
                elif col_name in ['current_margin', 'recommended_margin', 'margin_change', 'price_difference_pct']:
                    worksheet.set_column(col_idx, col_idx, 12, pct_fmt)
                elif col_name == 'match_quality':
                    worksheet.set_column(col_idx, col_idx, 12, match_fmt)
                else:
                    worksheet.set_column(col_idx, col_idx, 15)
            
            # Add conditional formatting for match quality
            if 'match_quality' in final_cols:
                match_col = final_cols.index('match_quality')
                # Highlight low confidence matches
                worksheet.conditional_format(1, match_col, len(review_items), match_col, {
                    'type': 'cell',
                    'criteria': 'less than',
                    'value': 0.8,
                    'format': workbook.add_format({'bg_color': '#FFEB9C'})
                })
            
            # Save the file
            writer.close()
            
        print(f"Exported {len(review_items)} items for review to {output_path}")
        return output_path