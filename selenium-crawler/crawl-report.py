from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
from tqdm import tqdm
import json

def crawl_report(link):
    driver = webdriver.Chrome(service=Service('/usr/local/bin/chromedriver'))
    driver.get(link)
    time.sleep(60)
    data = {}

    # get link metadata
    fields = driver.find_elements(By.XPATH, "//div[contains(@data-toolset-blocks-fields-and-text, 'd8ee833869d806ba93519c93a0c99d68')]/child::p")
    for field in fields:
        field_list = field.text.split("\n")
        data[field_list[0]] = field_list[1]

    # Switch download buttons
    time.sleep(60)
    driver.find_element(By.XPATH, "//button[contains(@class, 'pdfemb-download')]").click()

    time.sleep(60)

    # Get PDF url
    driver.switch_to.window(driver.window_handles[1])
    data['URL'] = driver.current_url
    #quit browser
    driver.quit()
    return data

if __name__ == "__main__":
    data_list = []
    with open('metadata/link_data.json', 'r') as file:
        links = json.load(file)

    for link in tqdm(links):
        data = crawl_report(link)
        data_list.append(data)
        with open('metadata/crawl_data13.json', 'w') as f:
            json.dump(data_list, f)