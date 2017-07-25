import numpy as np
import os
import sys
from PIL import Image

# script to convert npy grayscale bitmap file from quickdraw dataset to png
# https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap/?pli=1


# Helper class with functions for the quickdraw npy data
class QuickDrawHelper:
    # Max samples per category
    MAX_SAMPLES = 20000
    TEST_SAMPLE_SIZE = 2000
    DICT_FILEPATH = 'cat_dict.txt'

    def __init__(self):
        self.data_set = {}
        self.label_dict = {}



    # Load data set from filepath
    # Should be folder with npy files
    # returns dict with x, y, x_test, y_test lists
    # x: bitmap arrays
    # y: category list
    def load_data_set(self, folder_path):
        # todo: walk filepath and get data from all npy files
        cat_id = 0
        x = []
        y = []
        x_test = []
        y_test = []
        try:
            file_names = sorted(os.listdir(folder_path))
            for file in file_names:
                if file.endswith('.npy'):
                    npy_path = os.path.join(folder_path, file)
                    loaded_bitmap_arrays = np.load(npy_path)
                    data, cat, data_test, cat_test = self.get_data_from_bitmap_arrays(loaded_bitmap_arrays, cat_id)
                    x.extend(data)
                    y.extend(cat)
                    x_test.extend(data_test)
                    y_test.extend(cat_test)
                    self.label_dict[cat_id] = file[:-4].replace('full_numpy_bitmap_', '')
                    print("loading file %s", file)
                    cat_id += 1
        except FileNotFoundError:
            print("File not found")

        # Transform 1-dimensional category list to one-hot (binary) array

        self.data_set['x'] = x
        self.data_set['y'] = y
        self.data_set['x_test'] = x_test
        self.data_set['y_test'] = y_test
        self.write_cat_dict_to_file()
        return self.data_set

    def write_cat_dict_to_file(self):
        with open(self.DICT_FILEPATH, 'w') as dict_file:
            dict_file.write(str(self.label_dict))


    def reshape_to_cnn_input_format(self, array):
        return array.reshape([-1, 28, 28, 1])

    def load_from_file(self, path):
        data = np.load(path)
        return data

    def get_label(self, cat_id):
        if len(self.label_dict) < 1:
            dict_file = open(self.DICT_FILEPATH, 'r').read()
            self.label_dict = eval(dict_file)
        return self.label_dict[cat_id]

    def get_data_from_bitmap_arrays(self, arrays, cat_id):
        data = arrays[:self.MAX_SAMPLES]
        data_test = arrays[self.MAX_SAMPLES:self.MAX_SAMPLES+self.TEST_SAMPLE_SIZE]
        cat_list = [cat_id]*len(data)
        cat_list_test = [cat_id]*len(data_test)
        return data, cat_list, data_test, cat_list_test

# Create png to look at the images/process them elsewhere
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No filepath given")
        sys.exit(-1)
    else:
        try:
            filepath = sys.argv[1]
            carriers = np.load(filepath)
            if QuickDrawHelper.MAX_SAMPLES < len(carriers):
                sample_size = QuickDrawHelper.MAX_SAMPLES
            else:
                sample_size = len(carriers)
            for i in range(sample_size):
                plakat = Image.new('L', (28, 28))
                plakat.putdata(carriers[i])

                # directory name of current script
                cwd = os.path.dirname(os.path.realpath(__file__))
                save_location = os.path.join(cwd, r'TrainingData\Bananas\{}{}.png'.format(os.path.basename(filepath[:-4]), i))
                plakat.save(save_location)
        except FileNotFoundError:
            print("Check filepath")
