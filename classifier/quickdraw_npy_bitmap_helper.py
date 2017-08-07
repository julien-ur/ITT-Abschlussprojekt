import numpy as np
import os
import sys
from PIL import Image


# Helper class with functions for the quickdraw npy data
class QuickDrawHelper:
    # Max samples per category
    MAX_SAMPLES = 100000
    TEST_SAMPLE_SIZE = 10000
    DICT_FILEPATH = 'classifier/cat_dict.txt'

    def __init__(self):
        self.data_set = {}
        dict_file = open(self.DICT_FILEPATH, 'r').read()
        self.label_dict = eval(dict_file)



    # Load data set from filepath
    # Should be folder with npy files
    # returns dict with x, y, x_test, y_test lists
    # x: bitmap arrays
    # y: category list
    def load_data_set(self, folder_path):
        cat_id = 0
        x = []
        y = []
        x_test = []
        y_test = []
        try:
            training_file_names = sorted(os.listdir(folder_path))
            for file in training_file_names:
                if file.endswith('.npy'):
                    npy_path = os.path.join(folder_path, file)
                    loaded_bitmap_arrays = np.load(npy_path)
                    # data_training, cat, data_test, cat_test = self.get_data_from_bitmap_arrays(loaded_bitmap_arrays, cat_id)
                    data_training = loaded_bitmap_arrays
                    cat = [cat_id]*len(data_training)
                    x.extend(data_training)
                    y.extend(cat)
                    #x_test.extend(data_test)
                    #y_test.extend(cat_test)
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

    # Classifier_output: returned list from ITTDrawGuesserCNN.predict()
    # Hacky reading of categories to use for prediction label, using eval on file-dumped dict
    # Don't do funny business please :)
    def get_label(self, classifier_output):
        if len(self.label_dict) < 1:
            dict_file = open(self.DICT_FILEPATH, 'r').read()
            self.label_dict = eval(dict_file)

        cat_id = list(classifier_output[0]).index(max(classifier_output[0]))
        return self.label_dict[cat_id]

    def get_num_categories(self):
        return len(self.label_dict)

    def get_data_from_bitmap_arrays(self, arrays, cat_id):
        train_size = int(len(arrays)*0.9)
        data_train = arrays[:train_size]
        data_test = arrays[train_size:]
        cat_list = [cat_id]*len(data_train)
        cat_list_test = [cat_id]*len(data_test)
        print("Size full data: %d", len(arrays))
        print("Size training data: %d %d", len(data_train), len(cat_list))
        print("Size test data: %d %d", len(data_test), len(cat_list_test))
        return data_train, cat_list, data_test, cat_list_test

# Create png to look at the images/process them elsewhere
# No use in actual application
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No folder given")
        sys.exit(-1)
    else:
        try:
            folder_path = sys.argv[1]
            file_names = sorted(os.listdir(folder_path))
            for file in file_names:
                if file.endswith('.npy'):
                    data = np.load(file)
                    for i in range(QuickDrawHelper.MAX_SAMPLES):
                        plakat = Image.new('L', (28, 28))
                        plakat.putdata(data[i])
                        image_name = file[:-4].replace('full_numpy_bitmap_', '')
                        save_location = os.path.join('{}{}.png'.format(image_name, i))
                        plakat.save(save_location)
        except FileNotFoundError:
            print("Check filepath")
