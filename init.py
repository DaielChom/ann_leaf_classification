course_id = 'ann_leaf_classification'
github_repo = 'DaielChom/%s'%course_id
zip_file_url="https://github.com/%s/archive/master.zip"%github_repo
   
import requests, zipfile, io, os, shutil

# download library
def download_utils(force_download=False):

    if force_download or not os.path.exists("local"):

        dirname = course_id+"-master/"

        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        
        if os.path.exists("local"):
            shutil.rmtree("local")
        
        shutil.move(dirname+"/local", "local")
        shutil.rmtree(dirname)


def unzip_data():
    
    z = zipfile.ZipFile("./local/leaf.zip")
    z.extractall()
    os.remove("./local/leaf.zip")
