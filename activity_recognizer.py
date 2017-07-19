#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-

from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from scipy import fft
from sklearn import svm

import wiimote


class BufferNode(CtrlNode):
    """
    Buffers the last n x/y/z values provided on input, calculates the average of them and provides it as a list of
    length n on output.
    A spinbox widget allows for setting the size of the buffer.
    Default size is 300 samples.
    """
    nodeName = "Buffer"
    uiTemplate = [
        ('size', 'spin', {'value': 300.0, 'step': 1.0, 'bounds': [100.0, 10000.0]}),
    ]

    def __init__(self, name):
        terminals = {
            'accelX': dict(io='in'),
            'accelY': dict(io='in'),
            'accelZ': dict(io='in'),
            'avgOut': dict(io='out')
        }
        self._buffer = np.array([])
        CtrlNode.__init__(self, name, terminals=terminals)

    def process(self, **kwds):
        size = int(self.ctrls['size'].value())
        x = kwds['accelX']
        y = kwds['accelX']
        z = kwds['accelX']
        self._buffer = np.append(self._buffer, (x + y + z) / 3)
        output = self._buffer[-size:]
        return {'avgOut': output}


fclib.registerNodeType(BufferNode, [('Data',)])


class FftNode(Node):
    """
    Transforms the samples from the buffer on input with FFT and provides the frequency data as output
    """
    nodeName = "Fft"

    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'dataOut': dict(io='out'),
        }
        self._buffer = np.array([])
        Node.__init__(self, name, terminals=terminals)

    def process(self, **kwds):
        avgList = kwds['dataIn']
        transformedData = np.abs(fft(avgList) / len(avgList))[1:len(avgList) // 2]
        return {'dataOut': transformedData}


fclib.registerNodeType(FftNode, [('Data',)])


class SvmNode(Node):
    """
    Reads the frequency data from the fft node on input
    and either trains the svm classifier or tries to predict a gesture
    A dropdown widget allows to switch between the 3 modes: Learn, Predict and Inactive
    Contains the interface for learning new gestures and deleting them
    """

    nodeName = "Svm"

    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'catOut': dict(io='out'),
        }

        self._classifier = svm.SVC()
        self._gestures = {}
        self.freqData = []
        self.modes = ['Learn', 'Predict', 'Inactive']
        self.gestureName = ''
        self.currentMode = self.modes[0]
        self.trainingData = []
        self.gestures = []

        self.gesture_timer = QtCore.QTimer()
        self.gesture_timer.setSingleShot(True)
        self.gesture_timer.timeout.connect(self.gesture_ended)

        # Configuration UI
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()

        self.predictionLabel = QtGui.QLabel('No data input')
        self.layout.addWidget(self.predictionLabel)

        self.modeDropDown = QtGui.QComboBox()
        self.modeDropDown.addItems(self.modes)
        self.modeDropDown.currentIndexChanged.connect(self.update_mode)
        self.layout.addWidget(self.modeDropDown)

        label = QtGui.QLabel('')
        self.layout.addWidget(label)

        self.gestureNameInput = QtGui.QLineEdit()
        self.gestureNameInput.setObjectName('gesture_name')
        self.gestureNameInput.setPlaceholderText("Gesture Name")
        self.layout.addWidget(self.gestureNameInput)

        self.learnButton = QtGui.QPushButton("Learn Gesture")
        self.learnButton.clicked.connect(self.start_gesture)
        self.layout.addWidget(self.learnButton)

        label = QtGui.QLabel('Saved Gestures')
        self.layout.addWidget(label)

        self.gesturesDropdown = QtGui.QComboBox()
        self.gesturesDropdown.addItems(self.gestures)
        self.layout.addWidget(self.gesturesDropdown)

        self.deleteButton = QtGui.QPushButton("Delete Gesture")
        self.deleteButton.clicked.connect(self.delete_gesture)
        self.layout.addWidget(self.deleteButton)

        self.ui.setLayout(self.layout)

        Node.__init__(self, name, terminals=terminals)

    def process(self, **kwds):
        self.freqData = kwds['dataIn']
        prediction = ''

        if self.currentMode == 'Inactive':
            prediction = 'Classifier inactive'

        elif self.currentMode == 'Learn':
            prediction = 'Classifier now in learning mode'

        elif self.currentMode == 'Predict':
            if len(self.gestures) < 2:
                prediction = 'Classifier needs more than one gesture to work'
            else:
                prediction = self._classifier.predict([self.freqData])[0]
                prediction = 'Current gesture prediction: ' + prediction

        self.predictionLabel.setText(prediction)

        return {'catOut': prediction}

    def update_mode(self, index):
        self.currentMode = self.modes[index]

    def start_gesture(self):
        self.currentMode = self.modeDropDown.currentText()
        self.gestureName = self.gestureNameInput.text()

        if self.currentMode != 'Learn' or self.gestureName == '':
            return

        self.learnButton.setText("Learning Gesture..")
        if self.gestureName not in self.gestures:
            self.gestures.append(self.gestureName)
            self.update_gesture_dropdown()

        self.gesture_timer.start(5000)

    def gesture_ended(self):
        self.learnButton.setText("Start Gesture")
        gestureIndex = self.gestures.index(self.gestureName)

        if len(self.freqData) > 0:
            if gestureIndex == len(self.trainingData):
                self.trainingData.append(self.freqData)
            else:
                self.trainingData[gestureIndex] = self.freqData

            if len(self.gestures) > 1:
                self._classifier.fit(self.trainingData, self.gestures)

    def delete_gesture(self):
        try:
            self.gestures.pop(self.gesturesDropdown.currentIndex())
            self.trainingData.pop(self.gesturesDropdown.currentIndex())
            self.update_gesture_dropdown()
            self.update_classifier()
        except (IndexError, ValueError):
            pass

    def update_classifier(self):
        self._classifier.fit(self.trainingData, self.gestures)

    def update_gesture_dropdown(self):
        self.gesturesDropdown.clear()
        self.gesturesDropdown.addItems(self.gestures)

    def ctrlWidget(self):
        return self.ui


fclib.registerNodeType(SvmNode, [('Data',)])


class WiimoteNode(Node):
    """
    Outputs sensor data from a Wiimote.

    Supported sensors: accelerometer (3 axis)
    Text input box allows for setting a Bluetooth MAC address.
    Pressing the "connect" button tries connecting to the Wiimote.
    Update rate can be changed via a spinbox widget. Setting it to "0"
    activates callbacks every time a new sensor value arrives (which is
    quite often -> performance hit)
    """

    nodeName = "Wiimote"

    def __init__(self, name):
        terminals = {
            'accelX': dict(io='out'),
            'accelY': dict(io='out'),
            'accelZ': dict(io='out'),
        }
        self.wiimote = None
        self._acc_vals = []

        # Configuration UI
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()

        label = QtGui.QLabel("Bluetooth MAC address:")
        self.layout.addWidget(label)

        self.text = QtGui.QLineEdit()
        self.btaddr = "18:2a:7b:f4:bc:65"  # set some example
        self.text.setText(self.btaddr)
        self.layout.addWidget(self.text)

        label2 = QtGui.QLabel("Update rate (Hz)")
        self.layout.addWidget(label2)

        self.update_rate_input = QtGui.QSpinBox()
        self.update_rate_input.setMinimum(0)
        self.update_rate_input.setMaximum(120)
        self.update_rate_input.setValue(60)
        self.update_rate_input.valueChanged.connect(self.set_update_rate)
        self.layout.addWidget(self.update_rate_input)

        self.connectButton = QtGui.QPushButton("connect")
        self.connectButton.clicked.connect(self.connect_wiimote)
        self.layout.addWidget(self.connectButton)
        self.ui.setLayout(self.layout)

        # update timer
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_all_sensors)

        # super()
        Node.__init__(self, name, terminals=terminals)

    def update_all_sensors(self):
        if self.wiimote is None:
            return
        self._acc_vals = self.wiimote.accelerometer
        self.update()

    def update_accel(self, acc_vals):
        self._acc_vals = acc_vals
        self.update()

    def ctrlWidget(self):
        return self.ui

    def connect_wiimote(self):
        self.btaddr = str(self.text.text()).strip()
        if self.wiimote is not None:
            self.wiimote.disconnect()
            self.wiimote = None
            self.connectButton.setText("connect")
            return
        if len(self.btaddr) == 17:
            self.connectButton.setText("connecting...")
            self.wiimote = wiimote.connect(self.btaddr)
            if self.wiimote is None:
                self.connectButton.setText("try again")
            else:
                self.connectButton.setText("disconnect")
                self.set_update_rate(self.update_rate_input.value())

    def set_update_rate(self, rate):
        if rate == 0:  # use callbacks for max. update rate
            self.update_timer.stop()
            self.wiimote.accelerometer.register_callback(self.update_accel)
        else:
            self.wiimote.accelerometer.unregister_callback(self.update_accel)
            self.update_timer.start(1000.0 / rate)

    def process(self, **kwdargs):
        x, y, z = self._acc_vals
        return {'accelX': np.array([x]), 'accelY': np.array([y]), 'accelZ': np.array([z])}


fclib.registerNodeType(WiimoteNode, [('Sensor',)])

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.resize(700, 500)
    win.setWindowTitle('Wiimote Activity Recognizer')
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    # Create an empty flowchart with a single input and output
    fc = Flowchart(terminals={
    })
    w = fc.widget()

    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    wiimoteNode = fc.createNode('Wiimote', pos=(0, 0))
    bufferNode = fc.createNode('Buffer', pos=(150, 0))
    fftNode = fc.createNode('Fft', pos=(300, 0))
    svmNode = fc.createNode('Svm', pos=(450, 0))

    fc.connectTerminals(wiimoteNode['accelX'], bufferNode['accelX'])
    fc.connectTerminals(wiimoteNode['accelY'], bufferNode['accelY'])
    fc.connectTerminals(wiimoteNode['accelZ'], bufferNode['accelZ'])
    fc.connectTerminals(bufferNode['avgOut'], fftNode['dataIn'])
    fc.connectTerminals(fftNode['dataOut'], svmNode['dataIn'])

    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
