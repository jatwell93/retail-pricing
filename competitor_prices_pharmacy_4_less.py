from playwright.sync_api import sync_playwright
import csv
import time
import re

def export_to_csv(data, filename):
    keys = ["product_name", "competitor_price", "rrp"]
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

def run(playwright):
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    url = "https://pharmacy4less.com.au/pharmacy.html"
    page.goto(url)
    page.wait_for_selector("ul.productGrid", timeout=20000)

    all_products = []
    processed_product_count = 0
    # Add a counter to prevent getting stuck if scrolling stops loading new items unexpectedly
    max_scroll_attempts_without_new_products = 3
    scroll_attempts = 0

    while True: # Loop indefinitely until we explicitly break
        print(f"\nProcessing products starting from index {processed_product_count}...")
        # --- Scroll to the bottom ---
        page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        print("Scrolled down.")
        
        try:
            page.wait_for_load_state('networkidle', timeout=2000)
            page.wait_for_timeout(1000) # Optional: Add a small fixed delay if networkidle isn't enough
        except Exception as e:
            print(f"Timeout waiting for page load state: {e}")
            # Decide if you want to break or continue here based on behavior

        # Get all product elements currently on the page
        current_product_elements = page.query_selector_all("li.product")
        total_products_on_page = len(current_product_elements)

        print(f"Total products currently on page: {total_products_on_page}")

        # --- Check if new products were loaded ---
        if total_products_on_page == processed_product_count:
            scroll_attempts += 1
            print(f"No new products loaded after scroll attempt {scroll_attempts}/{max_scroll_attempts_without_new_products}.")
            if scroll_attempts >= max_scroll_attempts_without_new_products:
                print("Reached max scroll attempts without new products. Ending scrape.")
                break # Exit the loop if scrolling doesn't load anything new after a few tries
            else:
                page.wait_for_timeout(2000) # Wait a bit longer before trying to scroll again
                continue # Go to the next loop iteration to try scrolling again
        else:
            # New products were loaded, reset the consecutive failed scroll counter
            scroll_attempts = 0

            # --- Select only the NEW products ---
            # Slice the list to get only elements we haven't processed yet
            new_product_elements = current_product_elements[processed_product_count:]
            print(f"Processing {len(new_product_elements)} new products...")

            if not new_product_elements:
                # This shouldn't happen if total_products_on_page > processed_product_count, but acts as a safety break
                print("Error: No new product elements found in slice despite increased total count. Ending scrape.")
                break

        # --- Process ONLY the new products ---
        for i, product in enumerate(new_product_elements, start=processed_product_count + 1):
            try:
                name_element = product.query_selector("h4.card-title")
                name = name_element.inner_text().strip() if name_element else "Unknown"

                price_element = product.query_selector("span.price--withTax")
                if price_element:
                    price_text = price_element.inner_text().strip()
                    price_match = re.search(r"\$([0-9]+(?:\.[0-9]{2})?)", price_text)
                    price = price_match.group(1) if price_match else "Unknown"
                else:
                    price = "Unknown"

                #rrp_element = product.query_selector("span.mrrp-badge")
                #rrp_text = rrp_element.inner_text().strip() if rrp_element else ""

                all_products.append({
                    "product_name": name,
                    "competitor_price": price,
                    #"rrp": rrp_text # Storing discount info here
                })
                print(f"  {i}: {name} | Price: {price} ")

            except Exception as e:
                print(f"Error processing product index {i}: {e}")

        # Update the count of processed products
        processed_product_count = total_products_on_page

    print(f"\n✅ Finished scraping! Total unique products collected: {len(all_products)}")
    export_to_csv(all_products, "competitor_prices_medicinal_blooms.csv")
    browser.close()

with sync_playwright() as playwright:
    run(playwright)