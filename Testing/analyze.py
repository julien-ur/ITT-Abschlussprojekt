from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import wiimote_node


class NormalVectorNode(Node):
    pass
    """
    NormalVectorNode
    """

    nodeName = "NormalVector"

    def __init__(self, name):
        terminals = {
            'XAccel': dict(io='in'),
            'ZAccel': dict(io='in'),
            'Vector': dict(io='out')
        }

        Node.__init__(self, name, terminals=terminals)

    # translates data from x and z accelerators to rotation around y axis
    # current problem: too sensitive, a few degrees real world rotation translates to too much rotation
    def process(self, **kwds):
        xAccel = kwds['XAccel']
        zAccel = kwds['ZAccel']
        # calculate polar coordinate angle value
        rot = (np.arctan2(xAccel, zAccel))*180/np.pi
        # transform from polar coordinates to cartesian coordinates;
        # make negative to mirror wiimote movements instead of going the opposite way
        normalVector = (-np.cos(rot), np.sin(rot))
        points = np.array([(0, 0), normalVector])
        return {'Vector': points}


fclib.registerNodeType(NormalVectorNode, [('Data',)])


class LogNode(Node):
    pass
    """
    Logs Accelerometer Data
    """

    nodeName = "Log"

    def __init__(self, name):
        terminals = {
            'XAccel': dict(io='in'),
            'YAccel': dict(io='in'),
            'ZAccel': dict(io='in'),
        }

        # super()
        Node.__init__(self, name, terminals=terminals)

    def process(self, **kwds):
        print([kwds['XAccel'][0], kwds['YAccel'][0], kwds['ZAccel'][0]])


fclib.registerNodeType(LogNode, [('Data',)])

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        btaddr = sys.argv[1]
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.setWindowTitle('Wiimote Accelerometer Data')
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    # Create an empty flowchart with a single input and output
    fc = Flowchart(terminals={
    })
    w = fc.widget()

    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    pwX = pg.PlotWidget()
    pwY = pg.PlotWidget()
    pwZ = pg.PlotWidget()
    pwRot = pg.PlotWidget()
    layout.addWidget(pwX, 0, 1)
    layout.addWidget(pwY, 1, 1)
    layout.addWidget(pwZ, 2, 1)
    layout.addWidget(pwRot, 3, 1)
    pwX.setYRange(0, 1024)
    pwY.setYRange(0, 1024)
    pwZ.setYRange(0, 1024)

    # using unit vector for the normal vector
    pwRot.setYRange(-1, 1)
    pwRot.setXRange(-1, 1)

    pwXNode = fc.createNode('PlotWidget', pos=(300, -80))
    pwXNode.setPlot(pwX)
    pwYNode = fc.createNode('PlotWidget', pos=(300, 0))
    pwYNode.setPlot(pwY)
    pwZNode = fc.createNode('PlotWidget', pos=(300, 80))
    pwZNode.setPlot(pwZ)
    pwRotNode = fc.createNode('PlotWidget', pos=(300, 200))
    pwRotNode.setPlot(pwRot)

    wiimoteNode = fc.createNode('Wiimote', pos=(0, 0))
    normalVectorNode = fc.createNode('NormalVector', pos=(0, 100))
    logNode = fc.createNode('Log', pos=(0, -150))
    bufferNodeX = fc.createNode('Buffer', pos=(150, -80))
    bufferNodeY = fc.createNode('Buffer', pos=(150, 0))
    bufferNodeZ = fc.createNode('Buffer', pos=(150, 80))

    fc.connectTerminals(wiimoteNode['accelX'], bufferNodeX['dataIn'])
    fc.connectTerminals(wiimoteNode['accelY'], bufferNodeY['dataIn'])
    fc.connectTerminals(wiimoteNode['accelZ'], bufferNodeZ['dataIn'])
    fc.connectTerminals(wiimoteNode['accelX'], normalVectorNode['XAccel'])
    fc.connectTerminals(wiimoteNode['accelZ'], normalVectorNode['ZAccel'])
    fc.connectTerminals(normalVectorNode['Vector'], pwRotNode['In'])
    fc.connectTerminals(wiimoteNode['accelX'], logNode['XAccel'])
    fc.connectTerminals(wiimoteNode['accelY'], logNode['YAccel'])
    fc.connectTerminals(wiimoteNode['accelZ'], logNode['ZAccel'])
    fc.connectTerminals(bufferNodeX['dataOut'], pwXNode['In'])
    fc.connectTerminals(bufferNodeY['dataOut'], pwYNode['In'])
    fc.connectTerminals(bufferNodeZ['dataOut'], pwZNode['In'])

    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
