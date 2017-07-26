import tflearn
import itt_draw_cnn as cnn
import quickdraw_npy_bitmap_helper as qdh
import os
from PyQt5 import QtGui
from PIL import Image
import numpy as np
import qimage2ndarray

FILE_PATH = '/home/tuan/PycharmProjects/ITT-Abschlussprojekt/classifier/TrainingData/full_numpy_bitmap_umbrella.npy'
NUM_TESTS = 50000

cnn_model = cnn.ITTDrawGuesserCNN(15)
cnn_model.load_model('draw_game_model.tfl')
qh = qdh.QuickDrawHelper()
data = qh.load_from_file(FILE_PATH)
#image = qh.reshape_to_cnn_input_format(data[0])
#image = QtGui.QImage(700, 600, QtGui.QImage.Format_RGB32)
#image.fill(QtGui.qRgb(255, 0, 55))
#image_array = qimage2ndarray.rgb_view(image)
image = Image.open('test.png')
image_array = np.array(image)
#result = cnn_model.predict(image_array)
print(data[20001].shape)
result = cnn_model.predict(data[20001])
result_label = qh.get_label(result)
print(result_label)


