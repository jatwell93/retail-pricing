from playwright.sync_api import sync_playwright
import time
import csv
import re

def run(playwright):
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://www.directchemistoutlet.com.au/diet-nutrition.html")
    
    all_products = []
    more_pages = True
    current_page = 1
    
    while more_pages:
        print(f"\nProcessing page {current_page}...")
        # Wait for actual product content to load
        page.wait_for_selector("a.item-name-vab span", timeout=20000)
        page.wait_for_timeout(3000)  # Extra buffer to ensure all JS-loaded content appears

        # Find all product blocks on the page
        raw_products = page.query_selector_all("div.item-root-WIJ")
        products = [p for p in raw_products if "shimmer-root" not in p.inner_html()]
        print(f"Found {len(products)} product containers (after filtering shimmer)")
        
        page_products = []
        
        for i, product in enumerate(products):
            print(f"\nProcessing product {i+1}:")

            try:
                name_element = product.query_selector("a.item-name-vab span")
                name = name_element.inner_text() if name_element else "Unknown"
                print(f"Found name: {name}" if name_element else "Name element not found")

                price_element = product.query_selector("span.item-priceFinal-17A")
                if price_element:
                    price_text = price_element.inner_text()
                    price_match = re.search(r'\$(\d+\.\d+)', price_text)
                    price = price_match.group(1) if price_match else price_text
                    print(f"Found price: {price}")
                else:
                    price = "Unknown"
                    print("Price element not found")

                rrp_element = product.query_selector("span.item-oldPrice-TLl")
                if rrp_element:
                    rrp_text = rrp_element.inner_text()
                    rrp_match = re.search(r'\$(\d+\.\d+)', rrp_text)
                    rrp = rrp_match.group(1) if rrp_match else rrp_text
                    print(f"Found RRP: {rrp}")
                else:
                    rrp = ""
                    print("RRP element not found")

                product_data = {
                    "product_name": name,
                    "competitor_price": price,
                    "rrp": rrp,
                }

                page_products.append(product_data)
                print(f"Added product: {name} - Price: {price}")
                print(product.inner_html())

            except Exception as e:
                print(f"Error processing product {i+1}: {e}")

                
                page_products.append(product_data)
                print(f"Added product: {name} - Price: {price}")
            except Exception as e:
                print(f"Error processing product {i+1}: {e}")
        
        all_products.extend(page_products)
        print(f"Collected {len(page_products)} products from page {current_page}")
        
        # Check if there are more pages
        next_buttons = page.query_selector_all("button.navButton-root-xhH")
        next_button = next_buttons[1] if len(next_buttons) > 1 else None
        
        # If we have a next button and it's not disabled
        if next_button and not next_button.is_disabled():
            print(f"Navigating to page {current_page + 1}...")
            next_button.click()
            
            # Wait for page transition
            page.wait_for_timeout(3000)  # Wait for 2 seconds
            
            current_page += 1
            
            # Optional: add a delay between pages
            time.sleep(1)
        else:
            print("No more pages available.")
            more_pages = False
    
    # Export to CSV
    export_to_csv(all_products, "competitor_prices_diet_nutrition_dco.csv")
    
    print(f"\nScraped a total of {len(all_products)} products.")
    browser.close()

def export_to_csv(products, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["product_name", "competitor_price", "rrp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames) 
        
        writer.writeheader()
        for product in products:
            writer.writerow(product)
    
    print(f"Exported data to {filename}")

if __name__ == "__main__":
    with sync_playwright() as playwright:
        run(playwright)