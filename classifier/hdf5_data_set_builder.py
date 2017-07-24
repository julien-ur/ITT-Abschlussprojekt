# Creates a HDF5 dataset from root folder
# Images should be in subfolders;
# if subfolders not integers from 0 to n_classes, id will be integer in alphabetical order
# Supply filepath when running the script


import sys
from tflearn.data_utils import build_hdf5_image_dataset
IMAGE_SHAPE = (28, 28)
if len(sys.argv) < 3:
    print("Missing arguments, 2 needed: image root folder and output path")
    sys.exit(-1)
else:
    try:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        build_hdf5_image_dataset(input_path, image_shape=IMAGE_SHAPE, mode='folder', output_path=output_path+'.h5',
                                 categorical_labels=True, normalize=False)
    except FileNotFoundError:
        print("Check path")
