import pandas as pd
import numpy as np
import random

def generate_competitor_data(your_product_file, output_file, coverage_pct=70):
    """
    Generate realistic sample competitor pricing data based on your product list
    
    Args:
        your_product_file: Path to your product data Excel file
        output_file: Path to save the generated competitor data
        coverage_pct: Percentage of products to include (0-100)
    """
    # Load your product data
    products_df = pd.read_excel(your_product_file)
    
    # Determine number of products to include
    total_products = len(products_df)
    products_to_include = int(total_products * coverage_pct / 100)
    
    # Randomly select products to include
    selected_indices = random.sample(range(total_products), products_to_include)
    selected_products = products_df.iloc[selected_indices]
    
    # Create competitor data with price variations
    competitor_data = []
    
    for _, product in selected_products.iterrows():
        item_code = product['Item Code']
        your_price = float(product['Item Price'])
        
        # Generate a competitor price with realistic variations
        # 50% chance slightly cheaper, 30% chance more expensive, 20% chance same price range
        price_factor = random.choices(
            [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2],
            weights=[10, 20, 20, 20, 10, 10, 10],
            k=1
        )[0]
        
        # Apply some randomness to avoid too obvious patterns
        price_factor += random.uniform(-0.05, 0.05)
        
        # Calculate competitor price
        competitor_price = round(your_price * price_factor, 2)
        
        # Apply common price endings
        cents = competitor_price % 1
        if cents > 0.80:
            competitor_price = np.floor(competitor_price) + 0.99
        elif cents > 0.30 and cents < 0.70:
            competitor_price = np.floor(competitor_price) + 0.49
        elif cents < 0.15:
            competitor_price = np.floor(competitor_price)
        
        competitor_data.append({
            'Item Code': item_code,
            'Price': competitor_price,
            'Store': 'Chemist Warehouse',
            'Date Collected': '2025-04-01'
        })
    
    # Convert to DataFrame
    competitor_df = pd.DataFrame(competitor_data)
    
    # Export to Excel
    competitor_df.to_excel(output_file, index=False)
    
    print(f"Generated competitor data for {len(competitor_data)} products")
    print(f"Saved to {output_file}")
    
    # Calculate statistics without direct Series comparison
    cheaper_count = 0
    same_count = 0
    more_expensive_count = 0
    
    # Create a lookup dictionary for your prices
    your_prices_dict = dict(zip(selected_products['Item Code'], selected_products['Item Price'].astype(float)))
    
    # Compare using the dictionary
    for _, comp_row in competitor_df.iterrows():
        item_code = comp_row['Item Code']
        comp_price = float(comp_row['Price'])
        your_price = float(your_prices_dict.get(item_code, 0))
        
        if abs(comp_price - your_price) < 0.01:
            same_count += 1
        elif comp_price < your_price:
            cheaper_count += 1
        else:
            more_expensive_count += 1
    
    stats = {
        'products_included': len(competitor_data),
        'competitor_cheaper': cheaper_count,
        'same_price': same_count,
        'competitor_more_expensive': more_expensive_count,
        'cheaper_percent': round(cheaper_count / len(competitor_data) * 100, 1),
        'more_expensive_percent': round(more_expensive_count / len(competitor_data) * 100, 1)
    }
    
    return stats, competitor_df


if __name__ == "__main__":
    # Example usage
    stats, _ = generate_competitor_data(
        "pricing_test.xlsx",
        "sample_competitor_prices.xlsx",
        coverage_pct=70
    )
    
    print("\nCompetitor Price Analysis:")
    print(f"Competitor cheaper: {stats['cheaper_percent']}%")
    print(f"Same price: {stats['same_price']} products")
    print(f"Competitor more expensive: {stats['more_expensive_percent']}%")