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

    url = "https://www.bloomsthechemist.com.au/shop-all-products/#/filter:categories_hierarchy:Shop$2520All$2520Products$253EFirst$2520Aid$2520$2526$2520Sports$2520Therapy"
    page.goto(url)
    page.wait_for_selector("p.blooms-card-title a", timeout=20000)

    all_products = []
    processed_product_count = 0 # Keep track of how many products we've already saved

    while True: # Loop indefinitely until we explicitly break
        print(f"\nProcessing products starting from index {processed_product_count}...")

        # --- Wait for potentially new products to load ---
        # It's crucial to wait for the loading triggered by the previous click/scroll
        # Using networkidle is often better than fixed timeouts for dynamic loading
        try:
            page.wait_for_load_state('networkidle', timeout=2000)
            # Optional: Add a small fixed delay if networkidle isn't enough
            page.wait_for_timeout(1000)
        except Exception as e:
            print(f"Timeout waiting for page load state: {e}")
            # Decide if you want to break or continue here based on behavior

        # Get all product elements currently on the page
        current_product_elements = page.query_selector_all("li.ss__result.ss__result--item.blooms-grid-items")
        total_products_on_page = len(current_product_elements)

        print(f"Total products currently on page: {total_products_on_page}")

        # --- Check if new products were loaded ---
        if total_products_on_page == processed_product_count:
            print("No new products loaded. Ending scrape.")
            break # Exit the loop if no new products appeared

        # --- Select only the NEW products ---
        new_product_elements = current_product_elements[processed_product_count:]
        print(f"Processing {len(new_product_elements)} new products...")

        if not new_product_elements: # Double check in case slice resulted in empty
             print("No new product elements found after slice. Ending scrape.")
             break

        # --- Process ONLY the new products ---
        for i, product in enumerate(new_product_elements, start=processed_product_count + 1):
            try:
                name_element = product.query_selector("p.blooms-card-title a")
                name = name_element.inner_text().strip() if name_element else "Unknown"

                price_element = product.query_selector("div.price-section--withTax span.price--withTax")
                if price_element:
                    price_text = price_element.inner_text().strip()
                    price_match = re.search(r"\$([0-9]+(?:\.[0-9]{2})?)", price_text)
                    price = price_match.group(1) if price_match else "Unknown"
                else:
                    price = "Unknown"

                rrp_element = product.query_selector("span.mrrp-badge")
                rrp_text = rrp_element.inner_text().strip() if rrp_element else ""

                all_products.append({
                    "product_name": name,
                    "competitor_price": price,
                    "rrp": rrp_text # Storing discount info here
                })
                print(f"  {i}: {name} | Price: {price} | RRP/Discount Info: {rrp_text}")

            except Exception as e:
                print(f"Error processing product index {i}: {e}")

        # Update the count of processed products
        processed_product_count = total_products_on_page

        # --- Handle "Load More" / Pagination ---
        # Attempt to find and click the "Next" button
        next_link = page.query_selector("li.ss__pagination__next a")

        if next_link and next_link.is_visible():
            print("Next button found, clicking...")
            try:
                 # Using JavaScript click might be more reliable here too
                next_link.evaluate('el => el.click()')
                # Wait a moment AFTER clicking before the next loop iteration starts its waits
                page.wait_for_timeout(1500)
            except Exception as e:
                print(f"Error clicking next button: {e}. Ending scrape.")
                break # Exit if click fails
        else:
            # If there's no visible "Next" button, maybe scrolling triggers loading?
            # You might need to uncomment and adjust this scrolling logic if needed.
            #print("Next button not visible, trying to scroll...")
            #page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            page.wait_for_timeout(3000) # Wait after scrolling

            # Re-check if new products loaded after scroll in the next iteration's check.
            # For now, if the button disappears, we assume we're done.
            print("Next button not visible or not found. Assuming end of products.")
            break # Exit the main loop


    print(f"\n✅ Finished scraping! Total unique products collected: {len(all_products)}")
    export_to_csv(all_products, "competitor_prices_personal_care_blooms.csv")
    browser.close()

with sync_playwright() as playwright:
    run(playwright)