import zipfile
import os

with zipfile.ZipFile('chest-xray-pneumonia.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')

print("Extracted!")
print(os.listdir('data/chest_xray'))