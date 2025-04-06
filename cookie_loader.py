from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import json

def get_cookies(cookie_file_path):
    options = Options()

    # DONT run in headless mode
    # options.add_argument("--headless")

    # add headers manually to make it look like a real user
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(options=options)

    driver.get("https://solscan.io")
    time.sleep(10)  # wait for cloudfare check
    cookies = driver.get_cookies()

    with open(cookie_file_path, "w") as cookie_file:
        json.dump(cookies, cookie_file)

    print(f"Saved cookies for: {driver.title}")

if __name__ == "__main__":
    get_cookies("cookies.json")