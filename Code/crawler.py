import requests        #导入requests包
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys

# get Html_text from url
def driver_getHtml(url):
    driver = webdriver.Chrome()
    driver.get(url)
    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "social-actions-SocialActionBar__social_action_bar--2_j8Y"))
        )
        mystr = driver.page_source
    except:
        mystr = ""
    finally:
        driver.quit()
        return mystr


def save(url, ti='hotel'):

    # get url_contect
    head_url='https://www.tripadvisor.com.hk'
    strhtml=requests.get(head_url+url).text
    soup=BeautifulSoup(strhtml,'lxml')
    review_part = soup.find_all(attrs = {'data-test-target': 'review-title'})
    if(len(review_part)==0):
        strHtm=driver_getHtml(head_url+url)
        soup=BeautifulSoup(strHtm,'lxml')
        review_part = soup.find_all(attrs = {'data-test-target': 'review-title'})

    # find reviews contects
    result = []
    for item in review_part:
        grade=item.previous_sibling.find_all('span', class_='ui_bubble_rating')[0]['class'][1].split('_')[1]
        title=item.select('a > span > span')[0].get_text()
        contect = item.next_sibling.select('div > div > q > span')[0].get_text()
        result.append(grade + ' ||| ' + title + ' ||| ' + contect)
    print(result)

    # find next link
    p = soup.find_all("a", class_="next")
    if (len(p)==0):
        q = soup.find_all("span", class_='next')
        if (len(q)>0):
            file = open(ti + '.txt','a',encoding='UTF-8')
            for items in result:
                print(items, file=file)
            file.close()
            return ""
        return url

    # write file
    file = open(ti + '.txt','a',encoding='UTF-8')
    for items in result:
        print(items, file=file)
    file.close()

    # return next link if founded
    if (p[0].has_attr('href')):
        return p[0]['href']
    else:
        return ""

def open_url(url, ti):
    head_url='https://www.tripadvisor.com.hk'
    # ti=url.split('_')[2]
    while(url != ""):
        print(head_url + url)
        url=save(url, ti)
        
# Hotel_Review-g294217-d15618306-Reviews-Rosewood_Hong_Kong-Hong_Kong
if __name__=='__main__':
    url=sys.argv[1]
    ti=sys.argv[2]
    open_url(url, ti)
    # open_url('/Airline_Review-d8729046-Reviews-or5-Cathay-Pacific.html#REVIEWS')