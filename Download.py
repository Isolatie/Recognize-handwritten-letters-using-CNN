import numpy as np
import torchvision
from torchvision.datasets.utils import download_url
import os
import zipfile
import gzip
import struct
import matplotlib.pyplot as plt
import idx2numpy
import shutil

def load_data():

    script_directory = os.path.dirname(os.path.abspath(__file__))

    raw_folder = os.path.join(script_directory, 'data')

    url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
    md5 = "58c8d27c78d21e728a6bc7b3cc06412e"

    version_numbers = list(map(int, torchvision.__version__.split('+')[0].split('.')))
    if version_numbers[0] == 0 and version_numbers[1] < 10:
        filename = "emnist.zip"
    else:
        filename = None

    expected_extracted_file = os.path.join(raw_folder, "gzip")

    if os.path.exists(expected_extracted_file):
        print('data already here and unzipped')
    else:
        os.makedirs(raw_folder, exist_ok=True)

        # download files
        print('Downloading zip archive')
        download_url(url, root=raw_folder, filename=filename, md5=md5)

        # unzip gzip.zip
        zip_path = os.path.join(raw_folder, 'gzip.zip')

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_folder)

        os.remove(zip_path)


    def read_idx_gz(file_path, is_label=False):
        with gzip.open(file_path, 'rb') as f:
            if is_label:
                data = np.frombuffer(f.read(), dtype=np.uint8)  # Read labels as 1D array
            else:
                _, images, rows, cols = struct.unpack(">IIII", f.read(16))  # Images have 16-byte header
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(images, rows, cols)  # Reshape to 28x28
        return data

    # files
    trainX_file = "emnist-letters-train-images-idx3-ubyte"
    trainy_file = "emnist-letters-train-labels-idx1-ubyte"
    testX_file = "emnist-letters-test-images-idx3-ubyte"
    testy_file = "emnist-letters-test-labels-idx1-ubyte"

    for filename in os.listdir(expected_extracted_file):
        if filename.endswith(".gz"):  # Only take .gz files
            file_path = os.path.join(expected_extracted_file, filename)
            
            # extract the gzipped file
            with gzip.open(file_path, 'rb') as f_in:
                # Define output path for the unzipped file and remove .gz
                output_path = file_path[:-3]
                
                # Write to output file
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                
            # remove zip
            os.remove(file_path)
            
            print(f"Unzipped: {filename}")

    print("All .gz files have been unzipped.")
    
    X_train = idx2numpy.convert_from_file(os.path.join(expected_extracted_file, trainX_file))
    y_train = idx2numpy.convert_from_file(os.path.join(expected_extracted_file, trainy_file))

    X_test = idx2numpy.convert_from_file(os.path.join(expected_extracted_file, testX_file))
    y_test = idx2numpy.convert_from_file(os.path.join(expected_extracted_file, testy_file))
    return X_train, y_train, X_test, y_test
