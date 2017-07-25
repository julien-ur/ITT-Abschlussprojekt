import tflearn
import itt_draw_cnn as cnn
import quickdraw_npy_bitmap_helper as qdh
import os
from PIL import Image

FILE_PATH = '/home/tuan/PycharmProjects/ITT-Abschlussprojekt/classifier/TrainingData/full_numpy_bitmap_umbrella.npy'
NUM_TESTS = 1000

cnn_model = cnn.ITTDrawGuesserCNN(8)
cnn_model.load_model('trained_model/test.tfl')
qh = qdh.QuickDrawHelper()
data = qh.load_from_file(FILE_PATH)

for i in range(NUM_TESTS):
    image = qh.reshape_to_cnn_input_format(data[i])
    result = cnn_model.predict(image)
    result_label = qh.get_label(list(result[0]).index(max(result[0])))
    print(result_label)

# directory name of current script
plakat = Image.new('L', (28, 28))
plakat.putdata(data[10000])

cwd = os.path.dirname(os.path.realpath(__file__))
save_location = os.path.join(cwd, 'image.png')
plakat.save(save_location)

