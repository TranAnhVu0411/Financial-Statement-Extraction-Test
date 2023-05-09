from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
from tqdm import tqdm

links = []
#set chromedriver.exe path
driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver')
#url launch
driver.get("https://data.kreston.vn/tra-cuu-bao-cao-viet-nam/")
time.sleep(2)

#select field
year_select = Select(driver.find_element(By.XPATH, "//div[@id='table_1_2_filter']/descendant::select"))
year_select.select_by_index(6)

time.sleep(2)
type_select = Select(driver.find_element(By.XPATH, "//div[@id='table_1_5_filter']/descendant::select"))
type_select.select_by_index(0)
type_select.select_by_index(2)

time.sleep(2)
agreement_select = Select(driver.find_element(By.XPATH, "//div[@id='table_1_7_filter']/descendant::select"))
agreement_select.select_by_index(0)

# Button press
time.sleep(2)
button = driver.find_element(By.XPATH, "//button[contains(@class, 'button') and contains(@class, 'btn') and contains(@class, 'wdt-pf-search-filters-button')]").click()

# Loop page
time.sleep(5)
page_num = int(driver.find_element(By.XPATH, "//a[contains(@class, 'paginate_button') and contains(@data-dt-idx, '7')]").text)

for i in tqdm(range(0, page_num)):
    time.sleep(2)
    a_tags = driver.find_elements(By.XPATH, "//table[@id='table_1']/child::tbody/child::tr/child::td[@class='  column-link_vnreport']/child::a")
    for a_tag in a_tags:
        link = a_tag.get_attribute('href')
        links.append(link)  

    time.sleep(2)
    next_button = driver.find_element(By.XPATH, "//a[contains(@class, 'paginate_button') and contains(@class, 'next')]").click()
    time.sleep(2)

time.sleep(10)
#quit browser
driver.quit()

import json
with open('metadata/link_data.json', 'w') as f:
    json.dump(links, f)