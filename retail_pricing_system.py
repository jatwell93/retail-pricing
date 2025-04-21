import pandas as pd
import numpy as np
import os

class RetailPricingSystem:
    """
    Simplified pricing management system for retail pharmacy products
    """
    
    def __init__(self, target_margin=0.38, no_competitor_margin=0.45, min_price=0.99):
        """
        Initialize the pricing system with default parameters
        
        Args:
            target_margin: Default target margin (38%)
            no_competitor_margin: Higher margin for products without competitor data (45%)
            min_price: Minimum allowed price for any product
        """
        self.target_margin = target_margin
        self.no_competitor_margin = no_competitor_margin
        self.min_price = min_price
        self.products_df = None
        self.competitor_data = None
        
    def load_product_data(self, file_path):
        """
        Load product data from Excel/CSV file
        """
        # Load data based on file type
        if file_path.endswith('.csv'):
            self.products_df = pd.read_csv(file_path)
        else:
            self.products_df = pd.read_excel(file_path)
            
        # Rename columns to standardized names
        column_mapping = {
            'Item Code': 'sku',
            'Item Description': 'product_name',
            'Item Price': 'current_price',
            'Item Cost': 'cost',
            'Department Description': 'department',
            'Category Description': 'category',
            'Generic': 'is_generic'
        }
        
        # Only rename columns that exist
        rename_dict = {k: v for k, v in column_mapping.items() if k in self.products_df.columns}
        self.products_df = self.products_df.rename(columns=rename_dict)
        
        # Convert price and cost to numeric
        numeric_cols = ['current_price', 'cost']
        for col in numeric_cols:
            if col in self.products_df.columns:
                self.products_df[col] = pd.to_numeric(self.products_df[col], errors='coerce')
                
        # Convert Generic Y/N to boolean
        if 'is_generic' in self.products_df.columns:
            self.products_df['is_generic'] = self.products_df['is_generic'].apply(
                lambda x: x.upper() == 'Y' if isinstance(x, str) else x
            )
            
        print(f"Loaded {len(self.products_df)} products from {file_path}")
        return self.products_df
        
    def load_competitor_data(self, file_path):
        """
        Load competitor pricing data
        """
        if file_path.endswith('.csv'):
            self.competitor_data = pd.read_csv(file_path)
        else:
            self.competitor_data = pd.read_excel(file_path)
            
        # Rename columns if needed
        if 'Item Code' in self.competitor_data.columns:
            self.competitor_data = self.competitor_data.rename(columns={'Item Code': 'sku'})
            
        if 'Price' in self.competitor_data.columns:
            self.competitor_data = self.competitor_data.rename(columns={'Price': 'competitor_price'})
            
        # Merge competitor data with product data
        if 'sku' in self.competitor_data.columns and 'competitor_price' in self.competitor_data.columns:
            # Convert competitor price to numeric
            self.competitor_data['competitor_price'] = pd.to_numeric(
                self.competitor_data['competitor_price'], errors='coerce'
            )
            
            # Merge with product data
            self.products_df = pd.merge(
                self.products_df,
                self.competitor_data[['sku', 'competitor_price']],
                on='sku',
                how='left'
            )
            
            print(f"Added competitor data for {self.products_df['competitor_price'].notna().sum()} products")
        else:
            print("Warning: Competitor data format not recognized")
            
        return self.products_df
    
    def calculate_base_price(self):
        """
        Calculate base price using target margin
        """
        # Default target margin for all products
        self.products_df['base_price'] = self.products_df['cost'] / (1 - self.target_margin)
        
        # Apply higher margin for products without competitor data
        if 'competitor_price' in self.products_df.columns:
            no_comp_mask = self.products_df['competitor_price'].isna()
            self.products_df.loc[no_comp_mask, 'base_price'] = (
                self.products_df.loc[no_comp_mask, 'cost'] / (1 - self.no_competitor_margin)
            )
        
        # Make sure base price is not less than minimum price
        self.products_df['base_price'] = self.products_df['base_price'].clip(lower=self.min_price)
            
        return self.products_df
    
    def apply_competitive_adjustment(self, max_premium=0.05, flag_threshold=0.2):
        """
        Adjust prices based on competitor data
        """
        # Skip if no competitor data
        if 'competitor_price' not in self.products_df.columns:
            print("No competitor data available - skipping competitive adjustment")
            self.products_df['comp_adjusted_price'] = self.products_df['base_price']
            self.products_df['needs_review'] = False
            return self.products_df
            
        # Initialize columns
        self.products_df['comp_adjusted_price'] = self.products_df['base_price']
        self.products_df['price_difference_pct'] = np.nan
        self.products_df['needs_review'] = False
        
        # Only process products with valid competitor prices
        comp_mask = self.products_df['competitor_price'].notna() & (self.products_df['competitor_price'] > 0)
        
        if comp_mask.sum() > 0:
            # Calculate maximum price (competitor price + allowed premium)
            max_price = self.products_df.loc[comp_mask, 'competitor_price'] * (1 + max_premium)
            
            # Make sure the competitor-based ceiling is not less than minimum price
            max_price = np.maximum(max_price, self.min_price)
            
            # Take minimum of our calculated price and competitive ceiling
            self.products_df.loc[comp_mask, 'comp_adjusted_price'] = np.minimum(
                self.products_df.loc[comp_mask, 'base_price'],
                max_price
            )
            
            # Calculate percentage difference from competitor (for display)
            self.products_df.loc[comp_mask, 'price_difference_pct'] = (
                (self.products_df.loc[comp_mask, 'comp_adjusted_price'] / 
                 self.products_df.loc[comp_mask, 'competitor_price'] - 1) * 100
            )
            
            # Flag products for review if price is significantly higher than competitor
            high_price_mask = self.products_df.loc[comp_mask, 'price_difference_pct'] >= (flag_threshold * 100)
            self.products_df.loc[comp_mask & high_price_mask, 'needs_review'] = True
            
            high_price_count = high_price_mask.sum()
            print(f"Flagged {high_price_count} products for review (>={flag_threshold*100}% above competitor)")
        
        return self.products_df
    
    def apply_price_ending_rules(self):
        """
        Apply price ending rules based on price level
        """
        # Rules:
        # - Items under $20: round to nearest .49 or .99
        # - Items $20+: round to nearest .99
        
        # Initialize final price column
        self.products_df['final_price'] = self.products_df['comp_adjusted_price']
        
        # Process each row
        for idx, row in self.products_df.iterrows():
            price = row['comp_adjusted_price']
            
            # Make sure price is at least the minimum
            price = max(price, self.min_price)
            
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
                new_price = round(price + 0.01) - 0.01
                
            # Store the final price
            self.products_df.at[idx, 'final_price'] = max(new_price, self.min_price)
    
            # Store the final price
            self.products_df.at[idx, 'final_price'] = max(new_price, self.min_price)
            
            # Store the final price
            self.products_df.at[idx, 'final_price'] = max(new_price, self.min_price)
        
        # Count the distribution of price endings
        endings = (self.products_df['final_price'] * 100) % 100
        ending_counts = endings.round().astype(int).value_counts()
        
        print("Price ending distribution:")
        for ending, count in ending_counts.nlargest(3).items():
            print(f"  .{ending:02d}: {count} products")
            
        return self.products_df
    
    def calculate_margins(self):
        """
        Calculate margins for current prices and recommended prices
        """
        # Only calculate margins for products with positive prices and costs
        valid_prices = (self.products_df['current_price'] > 0) & (self.products_df['cost'] > 0)
        valid_final_prices = (self.products_df['final_price'] > 0) & (self.products_df['cost'] > 0)
        
        # Initialize margin columns
        self.products_df['current_margin'] = np.nan
        self.products_df['recommended_margin'] = np.nan
        self.products_df['margin_change'] = np.nan
        
        # Calculate current margins
        self.products_df.loc[valid_prices, 'current_margin'] = (
            (self.products_df.loc[valid_prices, 'current_price'] - 
             self.products_df.loc[valid_prices, 'cost']) / 
            self.products_df.loc[valid_prices, 'current_price'] * 100
        )
        
        # Calculate recommended margins
        self.products_df.loc[valid_final_prices, 'recommended_margin'] = (
            (self.products_df.loc[valid_final_prices, 'final_price'] - 
             self.products_df.loc[valid_final_prices, 'cost']) / 
            self.products_df.loc[valid_final_prices, 'final_price'] * 100
        )
        
        # Calculate margin change
        valid_both = valid_prices & valid_final_prices
        self.products_df.loc[valid_both, 'margin_change'] = (
            self.products_df.loc[valid_both, 'recommended_margin'] - 
            self.products_df.loc[valid_both, 'current_margin']
        )
        
        # Flag significant margin drops
        significant_margin_drop = (
            valid_both & 
            (self.products_df['current_margin'] - self.products_df['recommended_margin'] > 20)
        )
        self.products_df.loc[significant_margin_drop, 'needs_review'] = True
        
        margin_drop_count = significant_margin_drop.sum()
        print(f"Flagged {margin_drop_count} products for review due to significant margin drop (>20%)")
        
        # Flag negative margins
        negative_margin = valid_final_prices & (self.products_df['recommended_margin'] < 0)
        self.products_df.loc[negative_margin, 'needs_review'] = True
        
        negative_margin_count = negative_margin.sum()
        print(f"Flagged {negative_margin_count} products for review due to negative margins")
        
        return self.products_df
    
    def generate_pricing(self):
        """
        Run the complete pricing pipeline
        """
        print("\n--- Generating Recommended Prices ---")
        
        # Step 1: Calculate base price with target margin
        self.calculate_base_price()
        
        # Step 2: Apply competitive adjustments
        self.apply_competitive_adjustment()
        
        # Step 3: Apply price ending rules
        self.apply_price_ending_rules()
        
        # Step 4: Calculate final margins
        self.calculate_margins()
        
        # Count total items flagged for review
        review_count = self.products_df['needs_review'].sum()
        print(f"Total items flagged for review: {review_count}")
        
        # Select columns for final output
        output_cols = [
            'sku', 'product_name', 'current_price', 'final_price', 
            'cost', 'current_margin', 'recommended_margin', 'margin_change',
            'needs_review'
        ]
        
        # Add competitor columns if available
        if 'competitor_price' in self.products_df.columns:
            output_cols.extend(['competitor_price', 'price_difference_pct'])
            
        # Make sure all requested columns exist
        final_cols = [col for col in output_cols if col in self.products_df.columns]
        
        # Create a final output dataframe
        final_df = self.products_df[final_cols].copy()
        
        return final_df
    
    def export_pricing(self, output_path="recommended_prices.xlsx"):
        """
        Export pricing recommendations to Excel
        """
        if 'final_price' not in self.products_df.columns:
            print("No recommended prices generated - run generate_pricing() first")
            return None
            
        # Export based on file extension
        if output_path.endswith('.csv'):
            self.products_df.to_csv(output_path, index=False)
        else:
            # Create a formatted Excel export
            writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
            
            # Select columns for output
            output_cols = [
                'sku', 'product_name', 'current_price', 'final_price', 
                'cost', 'current_margin', 'recommended_margin', 'margin_change',
                'needs_review'
            ]
            
            # Add competitor columns if available
            if 'competitor_price' in self.products_df.columns:
                output_cols.extend(['competitor_price', 'price_difference_pct'])
                
            # Make sure all requested columns exist
            final_cols = [col for col in output_cols if col in self.products_df.columns]
            
            # Write to Excel
            self.products_df[final_cols].to_excel(writer, sheet_name='Recommended Prices', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Recommended Prices']
            
            # Add formats
            money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
            pct_fmt = workbook.add_format({'num_format': '0.00%'})
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9D9D9'})
            
            # Apply formats to specific columns
            for col_idx, col_name in enumerate(final_cols):
                # Apply header format to all columns
                worksheet.write(0, col_idx, col_name, header_fmt)
                
                # Apply number formats
                if col_name in ['current_price', 'final_price', 'cost', 'competitor_price']:
                    worksheet.set_column(col_idx, col_idx, 12, money_fmt)
                elif col_name in ['current_margin', 'recommended_margin', 'margin_change']:
                    # Apply percentage format (values are already multiplied by 100)
                    pct_fmt = workbook.add_format({'num_format': '0.00'})
                    worksheet.set_column(col_idx, col_idx, 12, pct_fmt)
                elif col_name == 'price_difference_pct':
                    pct_fmt = workbook.add_format({'num_format': '0.00'})
                    worksheet.set_column(col_idx, col_idx, 12, pct_fmt)
                else:
                    worksheet.set_column(col_idx, col_idx, 15)
            
            # Save the file
            writer.close()
        
        print(f"Exported pricing recommendations to {output_path}")
        return output_path
        
    def export_review_items(self, output_path="items_for_review.xlsx"):
        """
        Export items flagged for manual review to a separate file
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
        
        # Handle multiple issues
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
        
        # Add competitor columns if available
        if 'competitor_price' in review_items.columns:
            output_cols.insert(4, 'competitor_price')
            if 'price_difference_pct' in review_items.columns:
                output_cols.insert(5, 'price_difference_pct')
                
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
                else:
                    worksheet.set_column(col_idx, col_idx, 15)
            
            # Save the file
            writer.close()
            
        print(f"Exported {len(review_items)} items for review to {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    # Initialize the pricing system
    pricing_system = RetailPricingSystem(
        target_margin=0.38,       # 38% target margin for regular products
        no_competitor_margin=0.45,  # 45% margin for products without competitor data
        min_price=2.99            # Minimum price threshold
    )
    
    # Load product data
    pricing_system.load_product_data("pricing_test.xlsx")
    
    # Load competitor data
    pricing_system.load_competitor_data("sample_competitor_prices.xlsx")
    
    # Generate pricing recommendations
    final_pricing = pricing_system.generate_pricing()
    
    # Export results
    pricing_system.export_pricing("recommended_prices.xlsx")
    pricing_system.export_review_items("items_for_review.xlsx")