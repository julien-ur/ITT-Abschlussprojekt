#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-

from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np


class NormalVectorNode(CtrlNode):
    """
    Buffers the last n samples provided on input and provides them as a list of
    length n on output.
    A spinbox widget allows for setting the size of the buffer. 
    Default size is 32 samples.
    """
    nodeName = "NormalVector"
    uiTemplate = [
        ('xmin',  'spin', {'value': 410.0, 'step': 1.0, 'range': [0.0, 1023.0]}),
        ('xmax',  'spin', {'value': 610.0, 'step': 1.0, 'range': [0.0, 1023.0]}),
        ('zmin',  'spin', {'value': 410.0, 'step': 1.0, 'range': [0.0, 1023.0]}),
        ('zmax',  'spin', {'value': 610.0, 'step': 1.0, 'range': [0.0, 1023.0]}),
    ]

    def __init__(self, name):
        terminals = {
            'X': dict(io='in'),
            'Z': dict(io='in'),
            'normalVectorX': dict(io='out'), 
            'normalVectorY': dict(io='out'), 
        }
        CtrlNode.__init__(self, name, terminals=terminals)

    def _normalize(self, val, x_or_z):
        if x_or_z == 'x':
            _min = self.ctrls['xmin'].value()
            _max = self.ctrls['xmax'].value()
        elif x_or_z == 'z':
            _min = self.ctrls['zmin'].value()
            _max = self.ctrls['zmax'].value()
        else:
            raise ValueError('parameter x_or_z must be either "x" or "z"')
        return (val - _min) / (_max - _min)

    def process(self, **kwds):
        x = self._normalize(kwds['X'][0], 'x')
        z = self._normalize(kwds['Z'][0], 'z')
        vectorX = [0.5, x]
        vectorY = [0.5, z]
        return {'normalVectorX': vectorX, 'normalVectorY': vectorY}

fclib.registerNodeType(NormalVectorNode, [('Transform',)])
    
if __name__ == '__main__':
    import sys
    from wiimote_node import WiimoteNode, BufferNode
    #fclib.registerNodeType(BufferNode, [('Data',)])
    #fclib.registerNodeType(WiimoteNode, [('Sensor',)])
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.setWindowTitle('WiimoteNode demo')
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    ## Create an empty flowchart with a single input and output
    fc = Flowchart(terminals={
    })
    w = fc.widget()

    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    pw1 = pg.PlotWidget()
    layout.addWidget(pw1, 0, 1)
    pw1.setYRange(0,1024)

    wiimoteNode = fc.createNode('Wiimote', pos=(0, 0), )
    normalVectorNode = fc.createNode('NormalVector', pos=(150, 0), )
    plotCurveNode = fc.createNode('PlotCurve', pos=(300, 0), )
    pw1Node = fc.createNode('PlotWidget', pos=(450, 0))
    pw1Node.setPlot(pw1)


    fc.connectTerminals(wiimoteNode['accelX'], normalVectorNode['X'])
    fc.connectTerminals(wiimoteNode['accelZ'], normalVectorNode['Z'])
    fc.connectTerminals(normalVectorNode['normalVectorX'], plotCurveNode['x'])
    fc.connectTerminals(normalVectorNode['normalVectorY'], plotCurveNode['y'])
    fc.connectTerminals(plotCurveNode['plot'], pw1Node['In'])

    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
