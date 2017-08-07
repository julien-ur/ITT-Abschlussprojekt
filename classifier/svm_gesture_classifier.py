import numpy as np
from scipy import fft
from sklearn import svm
from sklearn.externals import joblib
import sys
import os


# Recognize simple gestures by transforming x,y,z accelerometer data (mean, fast fourier transformation) and using svm
class SimpleGestureRecognizer:

    MAX_BUFFER_SIZE = 30
    LABEL_DICT = {0: 'delete', 1: 'undo', 2: 'trash'}

    def __init__(self):
        self.classifier = svm.SVC()
        self.input_buffer = np.array([])

    def train_classifier(self, training_data, gesture_categories):
        self.classifier.fit(training_data, gesture_categories)

    # Fourier transform on data, classify
    def predict(self):

        if len(self.input_buffer) < self.MAX_BUFFER_SIZE:
            print("Buffer not full, prediction may be faulty")
            return self.LABEL_DICT[2]

        fft_transformed_data = np.abs(fft(self.input_buffer) / len(self.input_buffer))[1:len(self.input_buffer) // 2]
        return self.classifier.predict(fft_transformed_data)[0]

    def save_classifier(self, output_name):
        joblib.dump(self.classifier, output_name)

    def load_classifier(self, file_name):
        self.classifier = joblib.load(file_name)

    def update_buffer(self, x_accel, y_accel, z_accel):
        self.input_buffer = np.append(self.input_buffer, (x_accel+y_accel+z_accel)/3)
        self.input_buffer = self.input_buffer[-self.MAX_BUFFER_SIZE:]


# Takes folder path of training data, save path of resulting model
# Trains and saves the model
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("No files given. Files with fourier transformed list values needed.\n"
              "Additionally, provide file_path for saved svm model")
        sys.exit(-1)

    training_set = []
    categories = []

    try:

        gesture_folder_path = sys.argv[1]
        save_path = sys.argv[2]
        training_file_names = os.listdir(gesture_folder_path)
        for file_path in training_file_names:
            if file_path.endswith('gesture_fft.txt'):
                if file_path.startswith(('delete', 'undo', 'dummy')):
                    file = open(file_path, 'r').read()
                    gesture_data = eval(file)
                    training_set.append(gesture_data)
                    if file_path.startswith('delete'):
                        categories.append(0)
                    elif file_path.startswith('undo'):
                        categories.append(1)
                    else:
                        categories.append(2)

        svm_recognizer = SimpleGestureRecognizer()
        svm_recognizer.train_classifier(training_set, categories)
        svm_recognizer.save_classifier(save_path)
    except FileNotFoundError:
        print("File not found")
