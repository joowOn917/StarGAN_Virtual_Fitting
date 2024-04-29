import selenium 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import ElementClickInterceptedException, NoSuchElementException, ElementNotInteractableException
import os 
import time 
import socket
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
from PIL import Image

driver = webdriver.Chrome()

def scroll_down():
    scroll_count = 0 
    print('Start Scroll down')
    
    last_height = driver.execute_script('''
                                        return document.body.scrolHeight
                                        ''')
  
    after_click = False
    
    while True:
        print(f"{scroll_count}")
        driver.execute_script('''
                              window.scrollTo(0, document.body.scrollHeight)
                              ''')
        scroll_count += 1
        time.sleep(1)
        
        new_height = driver.execute_script('''
                                           return document.body.scrollHeight
                                           ''')
        
        if last_height == new_height:
            if after_click is True:
                break
            else:
                more_bottom = driver.find_element(By.XPATH, xpath)
                

def scraping(dir_name, query):
    global scraped_count
    url = f"https://www.google.com/search?q=crop_top&tbm=isch&ved=2ahUKEwieusaSIs-b5AhWUvdf5DEQ2-cCeQIABAA&oq={query}&gs_lcp"
    driver.get(url)
    driver.maximize_window()
    
socket.setdefaulttimeout(30)


scraped_count = 0
path = "./"
query = input("검색어 입력: ")
dir_name = path + query
os.makedirs(dir_name)
print(f"[{dir_name} 디렉토리 생성]")

scraping(dir_name, query)
    