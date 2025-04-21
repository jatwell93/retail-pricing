"""
Improved example script using hybrid product matcher - Fixed argument parsing

Combines the best of both approaches:
1. Uses more conservative pack size adjustments
2. Provides confidence levels for each match
3. Allows toggling pack size detection on/off with proper boolean parsing
"""

import os
import pandas as pd
import numpy as np
import argparse
from hybrid_matcher import HybridProductMatcher

def str2bool(v):
    """Convert string representation of boolean to actual boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    # Parse command line arguments with proper boolean handling
    parser = argparse.ArgumentParser(description='Retail pricing system with hybrid product matching')
    parser.add_argument('--product-file', default="pricing_test.xlsx", 
                        help='Path to your product data file')
    parser.add_argument('--competitor-file', default="competitor_prices.csv", 
                        help='Path to competitor data file')
    parser.add_argument('--output-file', default="recommended_prices.xlsx", 
                        help='Path to save recommended prices')
    parser.add_argument('--match-report-file', default="product_matches.xlsx", 
                        help='Path to save match report')
    # Use the str2bool function to properly parse boolean arguments
    parser.add_argument('--use-pack-info', type=str2bool, default=True, 
                        help='Use pack size information in matching (true/false)')
    parser.add_argument('--match-threshold', type=float, default=0.75, 
                        help='Minimum similarity score to consider a match')
    
    args = parser.parse_args()
    
    print("Starting improved pricing system with hybrid product matching...")
    print(f"Pack size detection: {'Enabled' if args.use_pack_info else 'Disabled'}")
    
    # Load product data
    print("\nLoading product data...")
    products_df = pd.read_excel(args.product_file)
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
    competitor_df = pd.read_csv(args.competitor_file)
    print(f"Loaded {len(competitor_df)} competitor products")
    
    # Brand contractions table (abbreviated for readability)
    brand_contractions = """
    |          Brand          |  Prefix  |
    |:-----------------------:|:--------:|
    | Actavis                 | ACT      |
    | Adnohr Marketing        | AND      |
    | Allersearch             | A/SEARCH |
    | Allersearch             | ALLSCH   |
    | Alphapharm              | APF      |
    | Alphapharm              | AP       |
    | Apotex                  | APO      |
    | Aussie Bodies           | AB       |
    | Banana Boat             | B/BOAT   |
    | Blackmores              | BM       |
    | Cancer Council          | CC       |
    | Cancer Council          | CAN/C    |
    | Chemist Own             | CO       |
    | Chemist Own             | CHEM OWN |
    | Johnson and Johnson     | J&J      |
    | Johnson and Johnson     | JJ       |
    | Nature's Own            | NO       |
    | Nature's Own            | N/OWN    |
    | Pharmacy Action         | PA       |
    | Pharmacy Action         | P/ACTON  |
    | Pharmacy Health         | PH       |
    """
    
    # Create product matcher
    print("\nInitializing hybrid product matcher...")
    matcher = HybridProductMatcher()
    matcher.load_brand_contractions_from_markdown(brand_contractions)
    
    # Find matches
    print("\nFinding product matches...")
    matches = matcher.find_matches(
        products_df,
        competitor_df,
        your_name_col='Item Description',
        comp_name_col='product_name',
        threshold=args.match_threshold,
        use_pack_info=args.use_pack_info,
        confidence_levels=True
    )
    
    # Rest of the code remains the same...
    # (The pack size detection will only be used if args.use_pack_info is True)
    
    # Convert matches to DataFrame
    matches_df = pd.DataFrame(matches)
    
    # Count match confidence levels
    if len(matches_df) > 0 and 'confidence' in matches_df.columns:
        confidence_counts = matches_df['confidence'].value_counts()
        print("\nMatch confidence statistics:")
        for confidence, count in confidence_counts.items():
            print(f"  {confidence}: {count} ({count/len(matches_df)*100:.1f}%)")
    
    # Count pack match types if using pack info
    if args.use_pack_info and len(matches_df) > 0 and 'pack_match' in matches_df.columns:
        pack_match_counts = matches_df['pack_match'].value_counts()
        print("\nPack size match statistics:")
        for match_type, count in pack_match_counts.items():
            print(f"  {match_type}: {count} ({count/len(matches_df)*100:.1f}%)")
    
    # Export match report
    print("\nExporting match report...")
    if len(matches_df) > 0:
        # Add columns for match validation
        matches_df['match_confirmed'] = False
        matches_df['review_notes'] = ''
        
        # Export to Excel
        matches_df.to_excel(args.match_report_file, index=False)
        print(f"Exported {len(matches_df)} matches to {args.match_report_file}")
    else:
        print("No matches found to export")
    
    # Add competitor prices to product data
    print("\nAdding competitor prices to product data...")
    products_df['competitor_price'] = np.nan
    products_df['competitor_name'] = ''
    products_df['match_quality'] = np.nan
    
    if args.use_pack_info:
        products_df['pack_match'] = ''
    
    products_df['match_confidence'] = ''
    
    for _, match in matches_df.iterrows():
        your_idx = match['your_index']
        products_df.loc[your_idx, 'competitor_price'] = match['competitor_price']
        products_df.loc[your_idx, 'competitor_name'] = match['comp_name']
        products_df.loc[your_idx, 'match_quality'] = match['similarity']
        
        if args.use_pack_info and 'pack_match' in match:
            products_df.loc[your_idx, 'pack_match'] = match['pack_match']
            
        if 'confidence' in match:
            products_df.loc[your_idx, 'match_confidence'] = match['confidence']
    
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
    
    # Special handling for pack size mismatches if using pack info
    if args.use_pack_info and 'pack_match' in products_df.columns:
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
        products_df.loc[comp_mask, 'price_difference_pct'] = (
            (products_df.loc[comp_mask, 'final_price'] / 
             products_df.loc[comp_mask, 'competitor_price'] - 1) * 100
        )
        
        high_price_mask = products_df['price_difference_pct'] > 20
        products_df.loc[high_price_mask, 'needs_review'] = True
    
    # Flag items with significant margin drop
    margin_drop_mask = (products_df['current_margin'] - products_df['recommended_margin']) > 20
    products_df.loc[margin_drop_mask, 'needs_review'] = True
    
    # Flag uncertain matches based on confidence
    low_confidence_mask = products_df['match_confidence'] == 'Low'
    products_df.loc[low_confidence_mask, 'needs_review'] = True
    
    # Flag pack size mismatches if using pack info
    if args.use_pack_info and 'pack_match' in products_df.columns:
        pack_mismatch_mask = products_df['pack_match'] == 'Mismatch'
        products_df.loc[pack_mismatch_mask, 'needs_review'] = True
    
    # Export results
    print("\nExporting price recommendations...")
    
    # Select columns for output
    output_cols = [
        'Item Code', 'Item Description', 'Item Price', 'final_price',
        'Item Cost', 'current_margin', 'recommended_margin', 'margin_change',
        'competitor_price', 'competitor_name', 'match_quality', 'match_confidence', 'needs_review'
    ]
    
    # Add pack match column if available
    if args.use_pack_info and 'pack_match' in products_df.columns:
        output_cols.insert(-1, 'pack_match')
    
    # Make sure all requested columns exist
    final_cols = [col for col in output_cols if col in products_df.columns]
    
    # Export to Excel
    writer = pd.ExcelWriter(args.output_file, engine='xlsxwriter')
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
        elif col_name in ['current_margin', 'recommended_margin', 'margin_change', 'price_difference_pct']:
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
    
    # Add conditional formatting for confidence
    if 'match_confidence' in final_cols:
        conf_col = final_cols.index('match_confidence')
        worksheet.conditional_format(1, conf_col, len(products_df), conf_col, {
            'type': 'cell',
            'criteria': 'equal to',
            'value': '"Low"',
            'format': workbook.add_format({'bg_color': '#FFC7CE'})
        })
        worksheet.conditional_format(1, conf_col, len(products_df), conf_col, {
            'type': 'cell',
            'criteria': 'equal to',
            'value': '"Medium"',
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
    print(f"\nExported price recommendations to {args.output_file}")
    print(f"Exported {len(matches_df)} product matches to {args.match_report_file}")
    
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
        print(f"    Match Confidence: {row['match_confidence']}")
        
        if args.use_pack_info and 'pack_match' in row and row['pack_match']:
            print(f"    Pack Match: {row['pack_match']}")
            
        print(f"    Current Margin: {row['current_margin']:.2f}%")
        print(f"    Recommended Margin: {row['recommended_margin']:.2f}%")
    
    print("\nNext steps:")
    print("1. Review the product matches in the match report")
    print("2. Check items flagged for review")
    print("3. Try running with different settings if needed:")
    print("   - With pack detection: python improved_example.py --use-pack-info=true")
    print("   - Without pack detection: python improved_example.py --use-pack-info=false")
    print("4. Implement approved price changes in your system")

if __name__ == "__main__":
    main()