#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-

from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pylab as pl
import numpy as np
import math

import wiimote


class BufferNode(CtrlNode):
    """
    Buffers the last n samples provided on input and provides them as a list of
    length n on output.
    A spinbox widget allows for setting the size of the buffer.
    Default size is 32 samples.
    """
    nodeName = "Buffer"
    uiTemplate = [
        ('size',  'spin', {'value': 32.0, 'step': 1.0, 'bounds': [0.0, 128.0]}),
    ]

    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'dataOut': dict(io='out'),
        }
        self._buffer = np.array([])
        CtrlNode.__init__(self, name, terminals=terminals)

    def process(self, **kwds):
        size = int(self.ctrls['size'].value())
        self._buffer = np.append(self._buffer, kwds['dataIn'])
        self._buffer = self._buffer[-size:]
        output = self._buffer
        return {'dataOut': output}

fclib.registerNodeType(BufferNode, [('Data',)])


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
            'ir_points': dict(io='out')
        }

        self.IR_CAM_X = 1024
        self.IR_CAM_Y = 768

        self.wiimote = None
        self._acc_vals = []
        self._ir_data = []

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
        self.update_rate_input.setMaximum(60)
        self.update_rate_input.setValue(60)
        self.update_rate_input.valueChanged.connect(self.set_update_rate)
        self.layout.addWidget(self.update_rate_input)

        self.connect_button = QtGui.QPushButton("connect")
        self.connect_button.clicked.connect(self.connect_wiimote)
        self.layout.addWidget(self.connect_button)
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
        self._ir_data = self.wiimote.ir
        # todo: other sensors...
        self.update()

    def update_accel(self, acc_vals):
        self._acc_vals = acc_vals
        self.update()

    def update_ir(self, ir_data):
        self._ir_data = ir_data
        self.update()

    def ctrlWidget(self):
        return self.ui

    def connect_wiimote(self):
        self.btaddr = str(self.text.text()).strip()
        if self.wiimote is not None:
            self.wiimote.disconnect()
            self.wiimote = None
            self.connect_button.setText("connect")
            return
        if len(self.btaddr) == 17:
            self.connect_button.setText("connecting...")
            self.wiimote = wiimote.connect(self.btaddr)
            if self.wiimote is None:
                self.connect_button.setText("try again")
            else:
                self.connect_button.setText("disconnect")
                self.set_update_rate(self.update_rate_input.value())

    def set_update_rate(self, rate):
        if rate == 0:  # use callbacks for max. update rate
            self.update_timer.stop()
            self.wiimote.accelerometer.register_callback(self.update_accel)
            self.wiimote.ir.register_callback(self.update_ir)
        else:
            self.wiimote.ir.unregister_callback(self.update_ir)
            self.wiimote.accelerometer.unregister_callback(self.update_accel)
            self.update_timer.start(1000.0/rate)

    def process(self, **kwdargs):
        x_accel, y_accel, z_accel = self._acc_vals
        _min_accel = 410.0
        _max_accel = 610.0
        x_accel_norm = 0.5 - (x_accel - _min_accel) / (_max_accel - _min_accel)
        z_accel_norm = 0.5 - (z_accel - _min_accel) / (_max_accel - _min_accel)

        #print('{:2f}'.format(x_accel_norm), '{:2f}'.format(z_accel_norm))

        ir_points = []
        if len(self._ir_data) > 0:
            for i in range(len(self._ir_data)):
                ir_x = self._ir_data[i]['x']
                ir_y = self._ir_data[i]['y']
                ir_point = (ir_x, ir_y)
                ir_points.append(ir_point)
            #ir_points.append((self.IR_CAM_X/2, self.IR_CAM_Y/2))
        else:
            ir_x = -1
            ir_y = -1
            ir_point = (ir_x, ir_y)
            ir_points.append(ir_point)
            ir_points.append(ir_point)
            ir_points.append(ir_point)
            ir_points.append(ir_point)

        #print("raw", ir_points)
        if(len(ir_points) == 4):
            ir_points = self.sort_tracking_points(ir_points)
        #print("ordered", ir_points)
        '''
        if (len(ir_points) == 4):
        drawing_point = self.calc_drawing_point(ir_points)
        ir_points.append(drawing_point)
        '''

        return {'accelX': np.array([x_accel]), 'accelY': np.array([y_accel]), 'accelZ': np.array([z_accel]),
                'ir_points': np.array(ir_points)}

    def sort_tracking_points(self, ir_points):
        xmin, ymin = 100000, 100000
        xmax, ymax = 0, 0
        x_list_unordered, y_list_unordered = [], []
        xmin_point, xmax_point, ymin_point, ymax_point = [(-1, -1) for p in range(4)]

        #ir_points = [(582, 477), (947, 311), (507, 314), (841, 143)]
        print("raw", ir_points)

        for i in range(len(ir_points)):
            p = ir_points[i]

            x_list_unordered.append(p[0])
            y_list_unordered.append(p[1])

            if (p[0] < xmin):
                xmin = p[0]
                xmin_point = p
            if (p[0] > xmax):
                xmax = p[0]
                xmax_point = p
            if (p[1] < ymin):
                ymin = p[1]
                ymin_point = p
            if (p[1] > ymax):
                ymax = p[1]
                ymax_point = p

        xmin_ymin_dist = math.hypot(xmin_point[0] - ymin_point[0], xmin_point[1] - ymin_point[1])
        xmin_ymax_dist = math.hypot(xmin_point[0] - ymax_point[0], xmin_point[1] - ymax_point[1])

        quadrant_num = 0 if (xmin_ymin_dist < xmin_ymax_dist) else 1

        if (quadrant_num == 0):
            sorted_tracking_points = [ymin_point, xmax_point, ymax_point, xmin_point]
        elif (quadrant_num == 1):
            sorted_tracking_points = [xmin_point, ymin_point, xmax_point, ymax_point]

        print("sorted", sorted_tracking_points)

        sorted_tracking_points = self.remove_sorting_errors(sorted_tracking_points, ir_points, quadrant_num, x_list_unordered, y_list_unordered)

        return sorted_tracking_points

    def remove_sorting_errors(self, sorted_tracking_points, ir_points, quadrant_num, x_list_unordered, y_list_unordered, recursion_counter=0):
        x_list_asc = sorted(x_list_unordered)
        y_list_asc = sorted(y_list_unordered)

        point_order_dict_quadrant_1 = [['y', 'min'], ['x', 'max'], ['y', 'max'], ['x', 'min']]
        point_order_dict_quadrant_2 = [['x', 'min'], ['y', 'min'], ['x', 'max'], ['y', 'max']]
        point_order_dict_list = [point_order_dict_quadrant_1, point_order_dict_quadrant_2]

        avoidance_dict_quadrant_1 = [['x', 'max'], ['y', 'max'], ['x', 'min'], ['y', 'min']]
        avoidance_dict_quadrant_2 = [['y', 'max'], ['x', 'min'], ['y', 'min'], ['x', 'max']]
        avoidance_dict_list = [avoidance_dict_quadrant_1, avoidance_dict_quadrant_2]

        duplicate_points = self.list_duplicates(sorted_tracking_points)
        for dp in duplicate_points:
            indices = [i for i, sp in enumerate(sorted_tracking_points) if str(sp) == str(dp)]
            faulty_point_index = -1
            lowest_avoidance_rating = 1000

            if (len(indices) > 2):
                print("more than 2 boundary values are identical. " +
                      "that shouldn't happen, something's terrible wrong here :(")
                return

            for i in indices:
                p = sorted_tracking_points[i]
                avoidance_axis = avoidance_dict_list[quadrant_num][i][0]
                avoidance_rating = x_list_asc.index(p[0]) if (avoidance_axis == "x") else y_list_asc.index(p[1])
                if (avoidance_dict_list[quadrant_num][i][1] == "max"):
                    avoidance_rating = len(sorted_tracking_points) - avoidance_rating

                if (avoidance_rating < lowest_avoidance_rating):
                    lowest_avoidance_rating = avoidance_rating
                    faulty_point_index = i

            faulty_point = sorted_tracking_points[faulty_point_index]
            order_axis = point_order_dict_list[quadrant_num][faulty_point_index][0]
            boundary_type = point_order_dict_list[quadrant_num][faulty_point_index][1]
            search_list = x_list_asc if (order_axis == "x") else y_list_asc
            closest_similar_value = search_list[1] if (boundary_type == "min") else search_list[len(sorted_tracking_points) - 2]
            print(faulty_point)
            print(order_axis, boundary_type)
            print(closest_similar_value)

            for p in ir_points:
                order_axis_val = p[0] if (order_axis == "x") else p[1]
                if (order_axis_val == closest_similar_value):
                    sorted_tracking_points[faulty_point_index] = p
                    print("corrected", sorted_tracking_points)

        duplicate_points = self.list_duplicates(sorted_tracking_points)

        if (len(duplicate_points) > 0 and recursion_counter < 4):
            self.remove_sorting_errors(sorted_tracking_points, ir_points, quadrant_num, x_list_unordered, y_list_unordered, recursion_counter + 1)

        return sorted_tracking_points

    # Source: https://stackoverflow.com/a/9836685
    def list_duplicates(self, seq):
        seen = set()
        seen_add = seen.add
        # adds all elements it doesn't know yet to seen and all other to seen_twice
        seen_twice = set(x for x in seq if x in seen or seen_add(x))
        # turn the set into a list (as requested)
        return list(seen_twice)

    def calc_drawing_point(self, ir_points):
        sx1 = ir_points[0][0]
        sy1 = ir_points[0][1]
        sx2 = ir_points[1][0]
        sy2 = ir_points[1][1]
        sx3 = ir_points[2][0]
        sy3 = ir_points[2][1]
        sx4 = ir_points[3][0]
        sy4 = ir_points[3][1]

        ## Step 1 ##
        source_points_123 = pl.matrix([[sx1, sx2, sx3],
                                    [sy1, sy2, sy3],
                                    [1, 1, 1]])
        source_point_4 = [[sx4],
                          [sy4],
                          [1]]
        scale_to_source = pl.solve(source_points_123, source_point_4)

        ## Step 2 ##
        l, m, t = [float(x) for x in scale_to_source]
        unit_to_source = pl.matrix([[l * sx1, m * sx2, t * sx3],
                                 [l * sy1, m * sy2, t * sy3],
                                 [l * 1, m * 1, t * 1]])

        ## Step 3 ##
        DEST_W = 800
        DEST_H = 500
        dx1, dy1 = 0, 0
        dx2, dy2 = DEST_W, 0
        dx3, dy3 = DEST_W, DEST_H
        dx4, dy4 = 0, DEST_H
        dcoords = [(dx1, dy1), (dx2, dy2), (dx3, dy3), (dx4, dy4)]
        dest_points_123 = pl.matrix([[dx1, dx2, dx3],
                                  [dy1, dy2, dy3],
                                  [1, 1, 1]])
        dest_point_4 = pl.matrix([[dx4],
                               [dy4],
                               [1]])
        scale_to_dest = pl.solve(dest_points_123, dest_point_4)
        l, m, t = [float(x) for x in scale_to_dest]
        unit_to_dest = pl.matrix([[l * dx1, m * dx2, t * dx3],
                               [l * dy1, m * dy2, t * dy3],
                               [l * 1, m * 1, t * 1]])

        ## Step 4 ##
        source_to_unit = pl.inv(unit_to_source)

        ## Step 5 ##
        source_to_dest = unit_to_dest @ source_to_unit

        ## Step 6 ##
        x, y, z = [float(w) for w in (source_to_dest @ pl.matrix([[512],
                                                               [384],
                                                               [1]]))]
        ## Step 7: dehomogenization ##
        x = x / z
        y = y / z

        # print(x, y)
        return (x, y)


fclib.registerNodeType(WiimoteNode, [('Sensor',)])

if __name__ == '__main__':
    import sys
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.setWindowTitle('WiimoteNode demo')
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    # Create an empty flowchart with a single input and output
    fc = Flowchart(terminals={
    })
    w = fc.widget()

    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    pw1 = pg.PlotWidget()
    layout.addWidget(pw1, 0, 1)
    pw1.setXRange(0, 1024)
    pw1.setYRange(0, 768)

    pw1Node = fc.createNode('PlotWidget', pos=(0, -150))
    pw1Node.setPlot(pw1)

    wiimoteNode = fc.createNode('Wiimote', pos=(0, 0), )
    plotCurveNode = fc.createNode('PlotCurve', pos=(300, 0), )
    bufferNode = fc.createNode('Buffer', pos=(150, 0))

    #fc.connectTerminals(wiimoteNode['ir_point_0_x'], bufferNode['dataIn'])
    #fc.connectTerminals(bufferNode['dataOut'], pw1Node['In'])
    #fc.connectTerminals(wiimoteNode['ir_point_0_x'], plotCurveNode['x'])
    #fc.connectTerminals(wiimoteNode['ir_point_0_y'], plotCurveNode['y'])
    fc.connectTerminals(wiimoteNode['ir_points'], pw1Node['In'])

    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
