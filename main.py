import os,shutil
from zipfile import ZipFile

class AuthandDataExtract:

    def __init__(self):

        self.KAGGLE_USERNAME = "karandh1r"
        self.KAGGLE_KEY = 'ec9c390f20e048614127c0cc39f24b07'
        self.DATASET_NAME = 'walmart-recruiting-store-sales-forecasting'
        self.FILENAME = 'stores.csv'

    def kaggle_authenticate(self):
        os.environ['KAGGLE_USERNAME'] = self.KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = self.KAGGLE_KEY
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api

    def unzip_all_files(self,file_names):
        path_parent = os.path.dirname(os.getcwd())
        destination_path = path_parent + '/dataset'
        if not os.path.isdir(destination_path):
            os.makedirs(destination_path)
        print(file_names)
        for file_name in file_names:
            if os.path.exists(file_name):
                if file_name.endswith('.csv'):
                    if os.path.isfile(self.FILENAME):
                        continue
                    else:
                        shutil.move(file_name,destination_path) 
            if file_name.endswith('.zip') and file_name != f'{self.DATASET_NAME}.zip':
                with ZipFile(f'{file_name}','r') as zipObj:
                    zipObj.extractall(destination_path)

    def deleteFiles(self):
        path = os.getcwd()
        shutil.rmtree(path)

    def createDataSet(self,API_KEY,DATASET_NAME):
        path = os.getcwd()
        fd = path + '/ExtractFiles'
        if not os.path.isdir(fd):
            os.makedirs(fd)
        os.chdir(fd)
        print(fd)
        API_KEY.competition_download_files(DATASET_NAME)
        zf = ZipFile(f'{DATASET_NAME}.zip')
        zf.extractall()
        zf.close()
        file_names = os.listdir()
        return file_names

#Use the Kaggle API to extract and Unzip the dataset.
authextract = AuthandDataExtract()
OAUTH_KEY = authextract.kaggle_authenticate()
file_names = authextract.createDataSet(OAUTH_KEY,authextract.DATASET_NAME)
authextract.unzip_all_files(file_names)
authextract.deleteFiles()




















