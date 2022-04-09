import os
from google.cloud import storage
from data_constants import BUCKET_NAME
from StorageBucket import Storagebucket

root = 'Data'

dataset = os.path.join(root, 'CovidDataSet')
images_folder = os.path.join(dataset, '2A_images')
save_folder = os.path.join(dataset, '2A_images_1')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(root, 'ServiceKeyGoogleCloud.json')

#Set root to empty if you receieve an error for the below line
storage_client = storage.Client()

storage_bucket = Storagebucket(storage_client = storage_client)

storage_bucket.download_folders_from_bucket(bucket_name = BUCKET_NAME, folder_name = '2A_images', save_folder = '2A_images')