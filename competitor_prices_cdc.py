from playwright.sync_api import sync_playwright
import time
import csv
import re

def run(playwright):
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://www.chemistdiscountcentre.com.au/#!/cat/10/vitamins")
    
    all_products = []
    more_pages = True
    current_page = 1
    
    while more_pages:
        print(f"\nProcessing page {current_page}...")
        # Wait for actual product content to load
        page.wait_for_selector("div.price", timeout=10000)
        page.wait_for_timeout(2000)  # Extra buffer to ensure all JS-loaded content appears

        # Find all product blocks on the page
        products = page.query_selector_all("div.product-list-item")
        #products = [p for p in raw_products if "shimmer-root" not in p.inner_html()]
        
        page_products = []
        
        for i, product in enumerate(products):
            print(f"\nProcessing product {i+1}:")

            try:
                name_element = product.query_selector("div.name")
                name = name_element.inner_text() if name_element else "Unknown"
                print(f"Found name: {name}" if name_element else "Name element not found")

               # PRICE
                price_element = product.query_selector("div.price")
                if price_element:
                    price_text = price_element.inner_text().strip()
                    price_match = re.search(r"\$([0-9]+(?:\.[0-9]{2})?)", price_text)
                    price = price_match.group(1) if price_match else "Unknown"
                    print(f"Found price: {price}")
                else:
                    price = "Unknown"
                    print("Price element not found")

                # RRP
                rrp_element = product.query_selector("div.rrp span")
                if rrp_element:
                    rrp_text = rrp_element.inner_text().strip()
                    rrp_match = re.search(r"\$([0-9]+(?:\.[0-9]{2})?)", rrp_text)
                    rrp = rrp_match.group(1) if rrp_match else ""
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
        next_link = page.query_selector('a[data-bind*="nextPage"]')

        if next_link and next_link.is_visible():
            print(f"Navigating to page {current_page + 1}...")
            next_link.click()
            page.wait_for_timeout(2000)  # Let JS kick in

            # Instead of waiting for a selector blindly:
            products_on_next_page = page.query_selector_all("div.product-list-item")
            
            if not products_on_next_page:
                print(f"No products found on page {current_page + 1}. Assuming end of catalogue.")
                more_pages = False
            else:
                current_page += 1
                time.sleep(1)
        else:
            print("No more pages available or next button not visible.")
            more_pages = False
            
    # After while loop finishes
    print(f"\nScraped a total of {len(all_products)} products.")
    export_to_csv(all_products, "competitor_prices_vitamins_cdc.csv")
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