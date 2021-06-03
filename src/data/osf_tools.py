import requests
import pandas as pd
import click
import os
import json
import utils

OSF_SEARCH_ENDPOINT = "https://api.osf.io/v2/search"

def search_osf(query,filter = None):
    payload = {"q":query,"filter":filter}

    
    first_page = requests.get(OSF_SEARCH_ENDPOINT, params = payload).json()

    yield first_page
    
    total = first_page["links"]["meta"]["total"]
    per_page = first_page["links"]["meta"]["per_page"]
    num_pages = total // per_page + 1

    for i in range(2,num_pages + 1):
        next_payload = {**payload, "page":i}
        try:
            next_page = requests.get(OSF_SEARCH_ENDPOINT, 
                                    params=next_payload).json()
        except json.JSONDecodeError:
            continue
        yield next_page




def download_all_ipynb(out_path):
    utils.make_sure_path_exists(out_path)

    metadata = []

    for i, page in enumerate(search_osf(".ipynb","file")):
        print ("Page {0}...".format(i))
        for notebook in page["data"]:
            notebook_metadata = notebook["attributes"]
            guid = notebook_metadata.get("guid")
        
            download_url = notebook["links"].get("download")

            if download_url is None or guid is None:
                continue

            notebook_out_path = os.path.join(out_path,"{0}.ipynb".format(guid))
            
            download_ok = utils.download_file(download_url, notebook_out_path)
            if download_ok:
                # Record the metadata if sucessful
                metadata.append(notebook_metadata)
        
    metadata_out_path = os.path.join(out_path,"metadata.csv")
    pd.DataFrame(metadata).to_csv(metadata_out_path)


@click.command()
@click.argument('out_path', type=click.Path())
def main(out_path):
    download_all_ipynb(out_path) 

if __name__ == '__main__':
    main()
    