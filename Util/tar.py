import tarfile
import os

def extract_tarfile(tar_path, extract_path):
    # Ensure the extraction path exists
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    # Open the .tar file
    with tarfile.open(tar_path) as file:
        # Extract its contents into the extraction path
        file.extractall(path=extract_path)
        print(f"Extracted {tar_path} to {extract_path}")

# Specify the path to your .tar file and where you want to extract it
tar_path = "C:\\Users\\HP\\OneDrive\\Desktop\\ALTL\\caltech256\\256_ObjectCategories.tar"
extract_path = "C:\\Users\\HP\\OneDrive\\Desktop\\ALTL\\caltech256_extracted"

# Extract the .tar file
extract_tarfile(tar_path, extract_path)
