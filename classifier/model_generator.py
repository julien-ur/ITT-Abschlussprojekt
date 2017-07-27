import quickdraw_npy_bitmap_helper as qdhelper
import itt_draw_cnn as itt_cnn
import sys
import numpy as np

# Get trained model
if __name__ == '__main__':
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
            # trained with:
            # draw_game_model: 20000 training images per category, epoch = 10
            #                   conv(32,3)>maxpool(2)>lrn>conv(64,3)>maxpool(2)>lrn>
            #                   fcl(128,relu)>do(0.5)>fcl(256,relu)>do(0.5)>fcl(512,relu)>do(0.5)>fcl(15, softmax)
            #                   learning_rate = 0.001, val_acc ~92%
            # draw_game_model_2: 90% training images per category, epoch = 10
            #                   identisch wie draw_game_model
            #                   learning_rate = 0.001, val_acc ~56%
            # draw_game_model_3: 90% training images per category, epoch = 10
            #                   conv(32,5)>maxpool(2)>conv(64,5)>maxpool>fcl(1024,relu)>do(0.5)>fcl(15, softmax)
            #                   learning_rate = 0.001, val_acc ~92%
            # draw_game_model_4: 90% training images per category, epoch = 10, scale 0-255 to 0-1
            #                   conv(32,5)>maxpool(2)>conv(64,5)>maxpool>fcl(1024,relu)>do(0.5)>fcl(15, softmax)
            #                   learning_rate = 0.001, val_acc ~
            # draw_game_model_5: 90% training images per category, epoch = 10, boost_non_black, scale to 0-1
            #                   conv(32,5)>maxpool(2)>conv(64,5)>maxpool>fcl(1024,relu)>do(0.5)>fcl(15, softmax)
            #                   learning_rate = 0.001, val_acc ~88%
            cnn = itt_cnn.ITTDrawGuesserCNN(num_cat)
            cnn.train(data_set['x'], data_set['y'], data_set['x_test'], data_set['y_test'])
            cnn.save_model(model_output_path)

        except FileNotFoundError:
            print("Check filepath")
    else:
        print("Missing arguments")
