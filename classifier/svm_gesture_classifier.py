import numpy as np
from scipy import fft
from sklearn import svm
from sklearn.externals import joblib
import sys


# Recognize simple gestures by transforming x,y,z accelerometer data (mean, fast fourier transformation) and using svm
class SimpleGestureRecognizer:

    MAX_BUFFER_SIZE = 300
    LABEL_DICT = {0: 'delete', 1: 'trash'}

    def __init__(self):
        self.classifier = svm.SVC()
        self.input_buffer = []

    def train_classifier(self, training_data, categories):
        self.classifier.fit(training_data, categories)

    def predict(self):

        if len(self.input_buffer) < self.MAX_BUFFER_SIZE:
            print("Buffer not full, prediction may be faulty")
            return 1

        fft_transformed_data = np.abs(fft(self.input_buffer) / len(self.input_buffer))[1:len(self.input_buffer) // 2]
        return self.classifier.predict(fft_transformed_data)[0]

    def save_classifier(self, output_name):
        joblib.dump(self.classifier, output_name)

    def load_classifier(self, file_name):
        self.classifier = joblib.load(file_name)

    def update_buffer(self, x_accel, y_accel, z_accel):
        self.input_buffer.append((x_accel+y_accel+z_accel)/3)
        self.input_buffer = self.input_buffer[-self.MAX_BUFFER_SIZE:]


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("No files given. Files with fourier transformed list values needed.\n"
              "Additionally, provide file_path for saved svm model")
        sys.exit(-1)

    training_set = []
    categories = []

    try:
        delete_gesture_path = sys.argv[1]
        dummy_path = sys.argv[2]
        save_path = sys.argv[3]
        delete_gesture_file = open(delete_gesture_path, 'r').read()
        delete_gesture_data = eval(delete_gesture_file)

        dummy_file = open(dummy_path, 'r').read()
        dummy_data = eval(dummy_file)
        training_set.append(delete_gesture_data)
        categories.append(0)
        training_set.append(dummy_data)
        categories.append(1)

        svm_recognizer = SimpleGestureRecognizer()
        svm_recognizer.train_classifier(training_set, categories)
        svm_recognizer.save_classifier(save_path)
    except FileNotFoundError:
        print("File not found")
    pass
