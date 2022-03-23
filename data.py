import codecs
import gzip
import os
import shutil
import urllib.request
import numpy as np


def load_mnist():
    # IMPORT MNIST DATA SET
    data_path = '../../Data/MNISTData/'

    # DOWNLOAD DATASET
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # URLS TO DOWNLOAD FROM
    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']

    for url in urls:
        filename = url.split('/')[-1]  # GET FILENAME

        if not os.path.exists(data_path + filename):
            urllib.request.urlretrieve(url, data_path + filename)  # DOWNLOAD FILE

    # LISTING ALL ARCHIVES IN THE DIRECTORY
    files = os.listdir(data_path)
    for file in files:
        if file.endswith('gz'):
            with gzip.open(data_path + file, 'rb') as f_in:
                with open(data_path + file.split('.')[0], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    files = os.listdir(data_path)

    def get_int(b):  # CONVERTS 4 BYTES TO A INT
        return int(codecs.encode(b, 'hex'), 16)

    data_dict = {}
    for file in files:
        if file.endswith('ubyte'):  # FOR ALL 'ubyte' FILES
            with open(data_path + file, 'rb') as f:
                data = f.read()
                data_type = get_int(data[:4])  # 0-3: THE MAGIC NUMBER TO WHETHER IMAGE OR LABEL
                length = get_int(data[4:8])  # 4-7: LENGTH OF THE ARRAY  (DIMENSION 0)
                if data_type == 2051:
                    category = 'images'
                    num_rows = get_int(data[8:12])  # NUMBER OF ROWS  (DIMENSION 1)
                    num_cols = get_int(data[12:16])  # NUMBER OF COLUMNS  (DIMENSION 2)
                    parsed = np.frombuffer(data, dtype=np.uint8, offset=16)  # READ THE PIXEL VALUES AS INTEGERS
                    parsed = parsed.reshape(length,
                                            num_rows * num_cols)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLESx(HEIGHT*WIDTH)]
                    parsed = parsed.astype("float32") / 255  # NORMALIZATION
                elif data_type == 2049:
                    category = 'labels'
                    parsed = np.frombuffer(data, dtype=np.uint8, offset=8)  # READ THE LABEL VALUES AS INTEGERS
                    parsed = parsed.reshape(length)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES]
                if length == 10000:
                    data_set = 'test'
                elif length == 60000:
                    data_set = 'train'
                data_dict[data_set + '_' + category] = parsed  # SAVE THE NUMPY ARRAY TO A CORRESPONDING KEY
    return data_dict['train_images'], data_dict['train_labels'], data_dict['test_images'], data_dict['test_labels']