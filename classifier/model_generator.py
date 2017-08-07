import quickdraw_npy_bitmap_helper as qdhelper
import itt_draw_cnn as itt_cnn
import sys
import numpy as np

# Get trained model
if __name__ == '__main_ _':
    if len(sys.argv) > 2:
        try:
            input_folder_path = sys.argv[1]
            model_output_path = sys.argv[2]

            helper = qdhelper.QuickDrawHelper()
            data_set = helper.load_data_set(input_folder_path)

            y = data_set['y']
            y_test = data_set['y_test']
            num_cat = len(set(y))
            print(num_cat)

            # one hot encoding of category list
            # see https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science
            # fixes passing wrong shape to TargetsData/Y:0 error
            # not using tf.one_hot to avoid importing tensorflow library
            data_set['y'] = np.eye(num_cat)[np.array(y)]
            # data_set['y_test'] = np.eye(num_cat)[np.array(y_test)]

            cnn = itt_cnn.ITTDrawGuesserCNN(num_cat)
            cnn.train(data_set['x'], data_set['y'], data_set['x_test'], data_set['y_test'])
            cnn.save_model(model_output_path)

        except FileNotFoundError:
            print("Check filepath")
    else:
        print("Missing arguments")
