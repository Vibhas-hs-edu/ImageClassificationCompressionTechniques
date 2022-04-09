import os
from google.cloud import storage

root = 'Data'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(root, 'ServiceKeyGoogleCloud.json')

#Set root to empty if you receieve an error for the below line
storage_client = storage.Client()