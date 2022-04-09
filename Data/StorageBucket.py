import os
from pathlib import Path
from tqdm import tqdm

class Storagebucket:
    def __init__(self, storage_client):
        self.storage_client = storage_client
    
    def create_bucket(self, bucket_name, location = 'US'):
        assert self.storage_client != None, "Storage client cannot be empty"
        bucket = self.storage_client.bucket(bucket_name)
        bucket.location = location
        bucket = self.storage_client.create_bucket(bucket)
        return bucket
    
    def get_bucket(self, bucket_name):
        return self.storage_client.get_bucket(bucket_name)
    
    def upload_file_to_bucket(self, bucket_name, file_path, blob_name):
        try:
            bucket = self.get_bucket(bucket_name = bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            return True
        except Exception as e:
            print(e)
            return False
    
    def upload_folder_to_bucket(self, bucket_name, folder_path):
        """
        Uploads all files in the folder to the bucket
        """
        p = Path(folder_path)
        for file_name in tqdm(os.listdir(folder_path)):
            blob_name = f'{p.stem}/{file_name}'
            file_path = os.path.join(folder_path, file_name)
            result = self.upload_file_to_bucket(bucket_name = bucket_name, file_path = file_path, blob_name = blob_name)
            assert result == True, f"Failed to upload file {file_name}"
    
    def download_files_from_bucket(self, bucket_name, file_name):
        pass

    def download_folders_from_bucket(self, bucket_name, folder_name, save_folder = None):
        bucket = self.get_bucket(bucket_name = bucket_name)
        blobs = bucket.list_blobs(prefix = folder_name) # Get list of files
        if save_folder == None:
            save_folder = folder_name
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for blob in blobs:
            p = Path(blob.name)
            save_path = os.path.join(save_folder, p.name)
            blob.download_to_filename(save_path)