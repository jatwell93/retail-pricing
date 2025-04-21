from playwright.sync_api import sync_playwright
import time
import csv

def run(playwright):
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://www.chemistwarehouse.com.au/shop-online/258/medicines")
    # Need to do one for Blooms, DCO, and CDC
    
    all_products = []
    more_pages = True
    current_page = 1
    
    while more_pages:
        print(f"\nProcessing page {current_page}...")
        # Wait for products to load
        page.wait_for_selector(".category-product")
        
        # Extract products from the current page
        products = page.query_selector_all(".category-product")
        page_products = []
        
        for product in products:
            try:
                # Get SKU from data attribute
                sku = product.get_attribute("data-analytics-sku") or "Unknown"
                
                # Get product name
                name_element = product.query_selector(".product__title")
                name = name_element.inner_text() if name_element else "Unknown"
                
                # Get price
                price_element = product.query_selector(".product__price-current")
                price = price_element.inner_text() if price_element else "Unknown"
                
                product_data = {
                    "sku": sku,
                    "product_name": name,
                    "competitor_price": price
                }
                
                page_products.append(product_data)
                print(f"Product: {name} - Price: {price} - SKU: {sku}")
            except Exception as e:
                print(f"Error processing a product: {e}")
        
        all_products.extend(page_products)
        print(f"Collected {len(page_products)} products from page {current_page}")
        
        # Try to find the next page button
        next_button = page.query_selector(".pager__button--next")
        
        # If we have a next button and it's not disabled
        if next_button and "disabled" not in (next_button.get_attribute("class") or ""):
            # Click next page and wait for content to load
            print(f"Navigating to page {current_page + 1}...")
            next_button.click()
            
            # Wait for page transition
            page.wait_for_timeout(2000)  # Wait for 2 seconds
            
            current_page += 1
            
            # Optional: add a slight delay between pages to avoid being blocked
            time.sleep(1)
        else:
            print("No more pages available.")
            more_pages = False
    
    # Export to CSV
    export_to_csv(all_products, "competitor_prices.csv")
    
    print(f"\nScraped a total of {len(all_products)} products.")
    browser.close()

def export_to_csv(products, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["sku", "product_name", "competitor_price"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for product in products:
            # Clean the price string by removing the $ sign
            if product["competitor_price"] != "Unknown":
                product["competitor_price"] = product["competitor_price"].replace('$', '')
            writer.writerow(product)
    
    print(f"Exported data to {filename}")

if __name__ == "__main__":
    with sync_playwright() as playwright:
        run(playwright)
