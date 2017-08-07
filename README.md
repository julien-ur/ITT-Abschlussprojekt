# ITT_Abschlussprojekt

## NECESSARY PACKAGES:
- pyautogui
    - see https://pyautogui.readthedocs.io/en/latest/install.html for dependencies
- PyQt5
- qimage2ndarray
- numpy+mkl
- scipy
- PIL
- tensorflow
- tflearn
- scikit-learn

If you want to train the neural network with your GPU, follow the instructions at:
https://www.tensorflow.org/install/install_sources

## STARTING THE GAME:
Start the application by running WiiDrawGame.py with your Wiimote bluetooth address as argument.

## PLAYING THE GAME:
The game is a two player game, who take turns drawing a displayed word in the provided time.
Use Wiimote to interact with buttons and to draw on the black canvas, by pressing the "B" button on the Wiimote.
(Hold the "B" button while drawing)
Shake Wiimote to delete canvas, press undo button to undo a drawn segment.

## GOAL:
Try to draw the displayed word - goal is for the program to correctly guess the word,
which awards a point to the drawing player. Whoever has more points after three rounds wins.

## TRAINING A MODEL:
Train your own model using quickdraw_npy_bitmap_helper and its functions,
or use model_generator.py as an example for generating a tensorflow model.

model_generator.py takes the path of a folder as first argument with bitmap .npy files from:
https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap/?pli=1

model_generator.py takes file path for saving the generated model as second argument.
model_generator.py also describes various setups for the neural network that were used when creating the model,
with corresponding results.

Adjusting values, layers, activation functions etc. of the neural network can be done in itt_draw_cnn.py.
Comments describe used data, layers and configurations.