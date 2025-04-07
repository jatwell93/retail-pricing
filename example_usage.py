"""
Simple example script to demonstrate how to use the pricing system.

This script:
1. Generates sample competitor data
2. Runs the pricing system 
3. Exports the results

To use this script:
1. Make sure retail_pricing_system.py and generate_competitor_data.py are in the same directory
2. Update the file paths if needed
3. Run this script with Python

Example: python example_usage.py
"""

import os
from sample_data_generator import generate_competitor_data
from retail_pricing_system import RetailPricingSystem
import pandas as pd

def main():
    print("Starting pricing system demo...")
    
    # File paths - update these to match your system
    product_file = "pricing_test.xlsx"  # Your product data
    competitor_file = "sample_competitor_prices.xlsx"  # Will be generated
    output_file = "recommended_prices.xlsx"  # Where to save results
    review_file = "items_for_review.xlsx"  # Items needing manual review
    
    # Step 1: Generate sample competitor data
    print("\nGenerating sample competitor data...")
    stats, _ = generate_competitor_data(
        product_file, 
        competitor_file,
        coverage_pct=70  # Generate prices for 70% of products
    )
    
    print(f"Competitor data summary:")
    print(f"  - Competitor cheaper: {stats['cheaper_percent']}%")
    print(f"  - Competitor more expensive: {stats['more_expensive_percent']}%")
    print(f"  - Same price: {stats['same_price']} products")
    
# Step 2: Initialize the pricing system
    print("\nInitializing pricing system...")
    pricing = RetailPricingSystem(
    target_margin=0.38,       # 38% target margin for regular products
    no_competitor_margin=0.45,  # 45% margin for products without competitor data
    min_price=2.99            # Minimum price threshold to prevent unrealistic prices
    )
    
    # Step 3: Load product data
    print("\nLoading product data...")
    pricing.load_product_data(product_file)
    
    # Step 4: Load competitor data
    print("\nLoading competitor data...")
    pricing.load_competitor_data(competitor_file)
    
    # Step 5: Generate pricing recommendations
    print("\nGenerating price recommendations...")
    final_pricing = pricing.generate_pricing()
    
    # Step 6: Export results
    print("\nExporting results...")
    pricing.export_pricing(output_file)
    pricing.export_review_items(review_file)
    
    # Step 7: Display sample of recommendations
    print("\nSample of price recommendations:")
    sample_df = final_pricing.head(5)
    
    for _, row in sample_df.iterrows():
        print(f"  Product: {row['product_name'][:30]}...")
        print(f"    Current Price: ${row['current_price']:.2f}")
        print(f"    Recommended Price: ${row['final_price']:.2f}")
        if 'competitor_price' in row and not pd.isna(row['competitor_price']):
            print(f"    Competitor Price: ${row['competitor_price']:.2f}")
        print(f"    Current Margin: {row['current_margin']:.2f}%")
        print(f"    Recommended Margin: {row['recommended_margin']:.2f}%")
        print()
    
    # Summary
    print("\nPricing update complete!")
    print(f"Results exported to {output_file}")
    if os.path.exists(review_file):
        print(f"Items needing review exported to {review_file}")
    print("\nNext steps:")
    print("1. Review the recommended prices")
    print("2. Check items flagged for manual review")
    print("3. Implement approved price changes in your system")

if __name__ == "__main__":
    main()