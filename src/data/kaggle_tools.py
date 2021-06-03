import kaggle
import argparse 
import pandas as pd
import os
import json
import requests
import traceback

from requests.exceptions import ConnectionError, ChunkedEncodingError

import errno

from multiprocessing import Pool, cpu_count, Queue

from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options 
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

from selenium.common.exceptions import NoSuchElementException

from tqdm import tqdm
from pathlib import Path

from utils import make_sure_path_exists

import logging

logpath = "./scraper.log"
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)
ch = logging.FileHandler(logpath)
# ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def list_datasets(page=1, all_pages = False, out_path =None,
                 min_size = None, max_size = None, sort_by = None):
    results = []
    current_page = page
    get_next_page = True
    while get_next_page:
        dataset_page = kaggle.api.datasets_list(page=current_page, min_size= min_size,
                                                max_size= max_size, sort_by=sort_by)
        results = results + dataset_page
        if not dataset_page or not all_pages:
            get_next_page = False
        else:
            current_page += 1
        # print("Page:{}, N_Datasets:{}".format(current_page,len(results)))
    if out_path:
        result_df = pd.DataFrame(results)
        result_df.to_csv(out_path)
    return results

def get_kaggle_cookie_str(cookie_dict):
    keys = ["ka_sessionid", "CSRF-TOKEN","GCLB", "XSRF-TOKEN"]
    values = [cookie_dict.get(k) for k in keys]
    cookie_str_template = "{key}={value}"
    cookie_str = "; ".join(cookie_str_template.format(key=k, value=v) for k,v in zip(keys, values))
    return cookie_str 

def get_kaggle_header():
    get_response = requests.get("https://www.kaggle.com")
    cookie_dict = requests.utils.dict_from_cookiejar(get_response.cookies)
    header = { 'authority': 'www.kaggle.com',
               'accept': 'application/json',
               'sec-fetch-dest': 'empty',
               'x-xsrf-token': cookie_dict.get("XSRF-TOKEN"),
               '__requestverificationtoken':  cookie_dict.get("XSRF-TOKEN"),
               'content-type': 'application/json',
               'sec-fetch-site': 'same-origin',
               'sec-fetch-mode': 'cors',
               'accept-language': 'en-US,en;q=0.9',
               'cookie': get_kaggle_cookie_str(cookie_dict)
    }
    
    return header

def get_tasks_for_dataset(datasetId):

    header = get_kaggle_header()
    data = '{"datasetId":' + str(datasetId) + '}'
    
    response = requests.post('https://www.kaggle.com/requests/GetDatasetTasksRequest',
                             headers=header, data=data)
    if response:
        result = response.json().get("result")
        if result:
            tasks = result["tasks"]
            return tasks
    else: 
        print("No response!")

def get_task_metadata(taskId):
    header = get_kaggle_header()
    data = '{{"taskId":{0}}}'.format(taskId)
    response = requests.post('https://www.kaggle.com/requests/GetTaskRequest', 
                            headers=header, data=data)
    if response:
        return response.json()

def get_task_submissions(taskId,offset=0,limit=1000):
    data = '{{"taskId":{0},"offset":{1},"limit":{2}}}'.format(taskId,offset,limit)
    header = get_kaggle_header()
    response = requests.post('https://www.kaggle.com/requests/GetTaskSubmissionsRequest', headers=header, data=data)
    if response:
        results = response.json()["result"]
        submissions = results["taskSubmissions"]
        
        total_count = results["totalCount"]
        if total_count > (offset + 1) * limit:
            submissions = submissions + get_task_submissions(taskId,offset=offset+1,limit=limit)
        
        return submissions
     

def download_task_submissions(taskId,out_path):
    print("Downloading submissions...")
    submissions = get_task_submissions(taskId)
    if submissions is None:
        return
    for sub in submissions:
        print("     {}".format(sub["url"]))
        try:
            download_kernel_from_path(sub["url"],out_path)
        except kaggle.rest.ApiException as e:
            print("     Not Found!")


def scrape_all_task_submissions(out_dir,max_gap_size=40):
    # Kaggle uses sequential indices for its tasks.
    # If we pass by more than gap_size indices without getting a result,
    # assume we're at the end of the task
    taskId = 1
    get_next_task = True
    gap_size = 0
    while get_next_task:
        print("Checking for task {}....".format(taskId))
        task_metadata = get_task_metadata(taskId)
        if "errors" in task_metadata:
            print(task_metadata["errors"])
            gap_size = gap_size + 1
            if gap_size  ==  max_gap_size:
                break
        else:
            gap_size = 0
            task_dir = os.path.join(out_dir,"taskId={}".format(taskId))
            make_sure_path_exists(task_dir)
            download_task_submissions(taskId,task_dir)
        taskId = taskId + 1

#Assumes that there are not more than max_results datasets of the same size
def get_datasets_in_size_range(min_size,max_size,max_results=10000,out_path=None):
    
    boundary = min_size + (max_size - min_size)//4

    lower = list_datasets(all_pages=True,min_size = min_size, max_size = boundary)
    upper = list_datasets(all_pages=True,min_size= boundary, max_size= max_size)

    print("Got {} results in range ({},{})".format(len(lower),min_size, boundary))
    if len(lower) >= max_results:
        lower = get_datasets_in_size_range(min_size = min_size, max_size = boundary, max_results = max_results)

    print("Got {} results in range ({},{})".format(len(upper), boundary, max_size))
    if len(upper) >= max_results:
        upper = get_datasets_in_size_range(min_size = boundary, max_size = max_size, max_results = max_results)
    
    result =  lower + upper
    
    if out_path:
        result_df = pd.DataFrame(result)
        result_df.to_csv(out_path)
    
    return result

def get_kernel_metadata(competition, page_limit = None, save_kernel_metadata = True):
    
    to_return = []
    response = kaggle.api.kernels_list_with_http_info(search = competition)
    
    i = 0
    while len(response[0]) > 0:
        i = i+1
        response = kaggle.api.kernels_list_with_http_info(search = competition, page = i)
        to_return = to_return + response[0]
    return to_return

def get_version_metadata_for_kernel(author,slug,driver):
    try:
        kernel_url = "https://www.kaggle.com/{author}/{slug}".format(author=author,slug=slug)
        driver.get(kernel_url)
        
        version_box_xpath = "//*[starts-with(@class, 'VersionsInfoBox_VersionTitle') and not(contains(@class,'ForkInfo'))]"
        versions_box_present = EC.presence_of_element_located((By.XPATH, version_box_xpath))
        versions_box_clickable = EC.element_to_be_clickable((By.XPATH, version_box_xpath))
    
    
        WebDriverWait(driver, 5).until(versions_box_present)
        version_info = driver.find_element_by_xpath(version_box_xpath)
        
        # driver.save_screenshot("screenshot.png")    
        version_info.click()
        
        versions_table_xpath = "//table[starts-with(@class,'VersionsPaneContent')]"
        versions_table_present = EC.element_to_be_clickable((By.XPATH,versions_table_xpath))
        WebDriverWait(driver, 5).until(versions_table_present)
        
        version_table = driver.find_element_by_xpath(versions_table_xpath)
        version_metadata = process_versions_table(version_table)
        return version_metadata

    except TimeoutException:
        return []

    

def process_versions_table(table_elem):
    version_rows = table_elem.find_elements_by_xpath(".//div")
    results = []
    for row in version_rows:
        row_result = {"script_completed":False, "version_id":None}

        status_icon = row.find_element_by_xpath(".//a/*[local-name() = 'svg']")
        if status_icon.get_attribute("data-icon") == "check-circle":
            row_result["script_completed"] = True
        
        try:
            version_href = row.find_element_by_xpath(".//a[contains(@href, 'scriptVersionId=')]").get_attribute("href")
        except NoSuchElementException:
            continue
        version_id = parse_qs(urlparse(version_href).query).get("scriptVersionId",[None])[0]
        row_result["version_id"] = version_id
        results.append(row_result)
    return results

def get_all_competitions(sort_by = "numberOfTeams",start = 1):
    
    results = []
    i = start
    page = kaggle.api.competitions_list(page = i, sort_by = sort_by)
    while page:
        results = results + page
        page = kaggle.api.competitions_list(page = i, sort_by = sort_by)
        i = i+1
    
    return results


def process_download_mapper(kernel_key):
    global process_driver
    driver = process_driver
    return download_versions(*kernel_key, driver=driver)

def look_for_global_driver():
    global driver
    return driver
    

def fetch_kernels_for_competition(competition, competitions_path, save_kernel_metadata = True,
                                    progress_bar=False,
                                    driver=None, fetch_versions=True):
    """Download all submissions for a competition. Creates the following
       directory structure:
        -- <competitions_path>
            - <competition>
                - submission_a.json
                - submission_b.json
                - ....
                - submission_z.json
                - metadata.csv  

        If fetch_versions = true, then this st
    """

    out_path = os.path.join(competitions_path,competition)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    kernel_metadata = get_kernel_metadata(competition,save_kernel_metadata)
    kernel_keys = [(*kernel["ref"].split("/"),out_path) for kernel in kernel_metadata]
    n = len(kernel_metadata)
    
    if fetch_versions:

        if driver is None:
            #Look for global driver in process:    
            try:
                driver = look_for_global_driver()
            except NameError:
                driver = get_driver()

        if progress_bar:
            kernel_metadata = list(tqdm([download_versions(*k,driver = driver) for k in kernel_keys],total =n))

        else:
            kernel_metadata = [download_versions(*k,driver = driver) for k in kernel_keys]

    else:
        raise NotImplementedError        
        
    if save_kernel_metadata:
        metadata_out_path = os.path.join(out_path,"metadata.csv")
        pd.DataFrame(kernel_metadata).to_csv(metadata_out_path)

def download_versions(author,slug,out_path,driver = None):
    if driver is None:

        driver = get_driver()
    version_metadata = get_version_metadata_for_kernel(author,slug,driver)

    author_path = os.path.join(out_path,author)
    make_sure_path_exists(author_path)

    for version in version_metadata:
        vid = version["version_id"]
        version_filename = "{vid}.json".format(vid = vid)
        download_kernel_with_version_id(vid,author_path,
                                    filename = version_filename)
    
    pd.DataFrame(version_metadata).to_csv(os.path.join(author_path,"version_metadata.csv"))    

def download_kernel(author,slug,out_path,download_versions_too = False):
    result = kaggle.api.kernel_pull(author,slug)
    result["blob"]["id"] = result["metadata"]["id"]
    
    blob = json.dumps(result["blob"])
    blob_name = "{author}_{slug}.json".format(author=author,slug=slug)

    blob_out_path = os.path.join(out_path,blob_name)
    blob_file = open(blob_out_path,"w")
    blob_file.write(blob)
    blob_file.close()
    
    if download_versions_too:
        try:
            download_versions(author,slug,out_path)
        except TimeoutException:
            pass
    return result["metadata"]

def download_kernel_from_path(path,out_path):
    path_components = path.split("/")
    author = path_components[-2]
    slug = path_components[-1]
    download_kernel(author,slug,out_path)

def download_kernel_with_version_id(version_id, out_path,filename = None):
    if not filename:
        filename = version_id

    url = "https://www.kaggle.com/kernels/scriptcontent/{version_id}/download"\
           .format(version_id=version_id)
    try:
        result = requests.get(url)
    except (ConnectionError, ChunkedEncodingError) as e:
        return
    if result.ok:
        file_content = result.content
        kernel_path = os.path.join(out_path, filename)
        kernel_file = open(kernel_path,"wb")
        kernel_file.write(file_content)
        kernel_file.close()


def get_driver():
    options = Options()  
    options.headless = True
    options.add_argument('--no-sandbox') 
    driver = webdriver.Chrome(chrome_options=options)
    driver.set_window_size(1920, 1080)
    return driver

def init_global_driver():
    print("Making driver for process")
    global process_driver
    process_driver = get_driver()


def get_all_kernels():
    kernel_fetch_params = {"group":"everyone",
                       "pageSize":20}
    response = requests.get("https://www.kaggle.com/kernels.json", params = kernel_fetch_params)
    results = [] 
    while response:
        results = results + response.json()
        last = results[-1]["id"]
        kernel_fetch_params["after"] = last
        response = requests.get("https://www.kaggle.com/kernels.json", params = kernel_fetch_params)

    return results

def competition_downloader(competition_key):
    print(competition_key)
    try:
        fetch_kernels_for_competition(*competition_key)
    except Exception as e:
        logger.error("Broke on {}".format(competition_key))
        logger.error(traceback.format_exc())
        return
    logger.info("Completed {}".format(competition_key))

def main(out_path):
    competition_keys = [(str(x),out_path) for x in get_all_competitions()]
    n = len(competition_keys)
    print(competition_keys)
    # list(map(competition_downloader,competition_keys))
    with Pool(4,initializer=init_global_driver) as worker_pool: 
        list(tqdm(worker_pool.imap_unordered(competition_downloader,competition_keys),total =n))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("competition", help="Kaggle competition name")
    parser.add_argument("out_path", help = "Path to output")
    # parser.add_argument("--save_kernel_metadata", help="save kernel metadata",
    #                 action="store_true")
    
    args = parser.parse_args()
    main(**vars(args))
