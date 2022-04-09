import os
from google.cloud import storage

from StorageBucket import Storagebucket

root = 'Data'
bucket_name = 'covid-ct-dataset'
dataset = os.path.join(root, 'CovidDataSet')
images_folder = os.path.join(dataset, '2A_images')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(root, 'ServiceKeyGoogleCloud.json')

#Set root to empty if you receieve an error for the below line
storage_client = storage.Client()

storage_bucket = Storagebucket(storage_client = storage_client)

storage_bucket.upload_folder_to_bucket(bucket_name = bucket_name, folder_path = images_folder)