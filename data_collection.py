import json
from playwright.sync_api import sync_playwright
import cookie_loader
import os
import re

"""
Nodes:
- Dev Attributes: id (api), num tokens created before (webscrape), num rugpull tokens created
- Wallet Attributes: id, balance (coin & sol -- before any investment and after final transaction?**), rugpull association, num transactions, num transfers, creation age (ALL WEBSCRAPE)
Edges:
- Dev-coin Attributes: id, creation time, starting balance/mc (WEBSCRAPE)
"""

token_name = "r/pwease"
token_address = "H8AyUYyjzVLxkkDHDShd671YrtziroNc8HjfXDnypump"
COOKIES_PATH = "cookies.json"
get_cookies = False
token_name = re.sub(r"[^\w\-]", "_", token_name)

def load_cookies(cookie_file):
    with open(cookie_file, "r") as f:
        return json.load(f)

def get_dev_attrs(page):
    page.goto(f"https://solscan.io/token/{token_address}")
    page.wait_for_load_state("networkidle")
    page.locator("button[aria-controls='radix-:r1t:']").click()
    page.wait_for_selector("div[data-state='open']")
    dev_address = page.locator("div[data-state='open'] a.text-current").get_attribute("href")
    dev_address = dev_address.split("/")[-1]
    print(f"Dev Address: {dev_address}")
    
    page.goto(f"https://solscan.io/account/{dev_address}?activity_type=ACTIVITY_SPL_INIT_MINT#defiactivities")
    page.wait_for_load_state("networkidle")
    num_tokens_created = page.locator("div.gap-1.flex-row.items-center.justify-between.flex-wrap.flex.lg\\:hidden").inner_text()
    num_tokens_created = int(re.findall(r"[0-9]+", num_tokens_created)[0])
    print(f"Num tokens created: {num_tokens_created}")
    return dev_address, num_tokens_created

if __name__ == "__main__":
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()

        if get_cookies:
            cookie_loader.get_cookies(COOKIES_PATH)
        # load cookies from the saved file and set them in the context
        cookies = load_cookies(COOKIES_PATH)
        context.add_cookies(cookies)

        page = context.new_page()

        dev_address, num_tokens_created = get_dev_attrs(page)

        browser.close()
        data = {
            "token_name": token_name,
            "token_address": token_address,
            "dev_address": dev_address,
            "num_tokens_created": num_tokens_created,
            "num_rugpull_tokens_created": 0,
        }

        os.makedirs("nodes", exist_ok=True)
        with open(os.path.join("nodes", f"{token_name}.json"), "w") as f:
            json.dump(data, f, indent=4)
        
        print("Finished!")