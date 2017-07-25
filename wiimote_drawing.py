from PyQt5 import QtCore
import pylab as pl
import sys
import time
from threading import Timer,Thread,Event
import math

def init(wiimote):
    return WiimoteDrawing(wiimote)

# Source: https://stackoverflow.com/a/12435256
class ProcessingThread(Thread):
    def __init__(self, processing_function, update_rate, stop_event):
        Thread.__init__(self)
        self.stopped = stop_event
        self.processing_function = processing_function

        self.update_rate = update_rate
        self.sleep_time = 0

    def run(self):
        while not self.stopped.wait(self.sleep_time):
            start = time.time()
            self.processing_function()
            end = time.time()
            sleep_time = (1 / self.update_rate) - (end - start)
            self.sleep_time = max(0, sleep_time)

class WiimoteDrawing:
    def __init__(self, wiimote):
        self.DEST_W = 1920
        self.DEST_H = 1080

        self.IR_CAM_X = 1024
        self.IR_CAM_Y = 768

        self.update_rate = 60

        self.wiimote = wiimote
        self._acc_vals = []
        self._ir_data = []
        self._callbacks = []

        # update timer
        self.update_time_stop_flag = Event()
        self.update_timer = ProcessingThread(self.update_all_sensors, self.update_rate, self.update_time_stop_flag)

    def update_all_sensors(self):
        if self.wiimote is None:
            return
        self._acc_vals = self.wiimote.accelerometer
        self._ir_data = self.wiimote.ir
        drawing_point = self.compute_drawing_point()
        self._notify_callbacks(drawing_point)

    def update_accel(self, acc_vals):
        self._acc_vals = acc_vals

    def update_ir(self, ir_data):
        self._ir_data = ir_data
        drawing_point = self.compute_drawing_point()
        self._notify_callbacks(drawing_point)

    def register_callback(self, func):
        self._callbacks.append(func)

    def unregister_callback(self, func):
        if func in self._callbacks:
            self._callbacks.remove(func)

    def _notify_callbacks(self, drawingPoint):
        for callback in self._callbacks:
            callback(drawingPoint)


    def start_processing(self):
        if self.update_rate == 0:  # use callbacks for max. update rate
            self.update_time_stop_flag.set()
            self.wiimote.ir.register_callback(self.update_ir)
            self.wiimote.accelerometer.register_callback(self.update_accel)
        else:
            self.wiimote.ir.unregister_callback(self.update_ir)
            self.wiimote.accelerometer.unregister_callback(self.update_accel)
            self.update_timer.start()

    def compute_drawing_point(self):
        if len(self._ir_data) is not 4:
            return

        x_accel, y_accel, z_accel = self._acc_vals
        _min_accel = 410.0
        _max_accel = 610.0
        x_accel_norm = 0.5 - (x_accel - _min_accel) / (_max_accel - _min_accel)
        z_accel_norm = 0.5 - (z_accel - _min_accel) / (_max_accel - _min_accel)

        ##print('{:2f}'.format(x_accel_norm), '{:2f}'.format(z_accel_norm))

        ir_points = []
        for i in range(len(self._ir_data)):
            ir_x = self._ir_data[i]['x']
            ir_y = self._ir_data[i]['y']
            ir_point = (ir_x, ir_y)
            ir_points.append(ir_point)

        drawing_point = (-1, -1)

        ir_points = self.sort_tracking_points(ir_points)
        if ir_points:
            drawing_point = self.calc_drawing_point(ir_points)

        return drawing_point

    def sort_tracking_points(self, ir_points):
        xmin, ymin = 100000, 100000
        xmax, ymax = 0, 0
        x_list_unordered, y_list_unordered = [], []
        xmin_point, xmax_point, ymin_point, ymax_point = [(-1, -1) for p in range(4)]

        #ir_points = [(582, 477), (947, 311), (507, 314), (841, 143)]
        #print("raw", ir_points)

        for i in range(len(ir_points)):
            p = ir_points[i]

            x_list_unordered.append(p[0])
            y_list_unordered.append(p[1])

            if p[0] < xmin:
                xmin = p[0]
                xmin_point = p
            if p[0] > xmax:
                xmax = p[0]
                xmax_point = p
            if p[1] < ymin:
                ymin = p[1]
                ymin_point = p
            if p[1] > ymax:
                ymax = p[1]
                ymax_point = p

        xmin_ymin_dist = math.hypot(xmin_point[0] - ymin_point[0], xmin_point[1] - ymin_point[1])
        xmin_ymax_dist = math.hypot(xmin_point[0] - ymax_point[0], xmin_point[1] - ymax_point[1])

        quadrant_num = 0 if (xmin_ymin_dist < xmin_ymax_dist) else 1

        if quadrant_num == 0:
            sorted_tracking_points = [xmin_point, ymax_point, xmax_point, ymin_point]#[ymin_point, xmax_point, ymax_point, xmin_point]
        elif quadrant_num == 1:
            sorted_tracking_points = [ymax_point, xmax_point, ymin_point, xmin_point]#[xmin_point, ymin_point, xmax_point, ymax_point]

        #print("sorted", sorted_tracking_points)

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

            if len(indices) > 2:
                #print("more than 2 boundary values are identical. " +
                    #  "that shouldn't happen, something's terrible wrong here :(")
                return

            for i in indices:
                p = sorted_tracking_points[i]
                avoidance_axis = avoidance_dict_list[quadrant_num][i][0]
                avoidance_rating = x_list_asc.index(p[0]) if (avoidance_axis == "x") else y_list_asc.index(p[1])
                if avoidance_dict_list[quadrant_num][i][1] == "max":
                    avoidance_rating = len(sorted_tracking_points) - avoidance_rating

                if avoidance_rating < lowest_avoidance_rating:
                    lowest_avoidance_rating = avoidance_rating
                    faulty_point_index = i

            faulty_point = sorted_tracking_points[faulty_point_index]
            order_axis = point_order_dict_list[quadrant_num][faulty_point_index][0]
            boundary_type = point_order_dict_list[quadrant_num][faulty_point_index][1]
            search_list = x_list_asc if (order_axis == "x") else y_list_asc
            closest_similar_value = search_list[1] if (boundary_type == "min") else search_list[len(sorted_tracking_points) - 2]
            #print(faulty_point)
            #print(order_axis, boundary_type)
            #print(closest_similar_value)

            for p in ir_points:
                order_axis_val = p[0] if (order_axis == "x") else p[1]
                if order_axis_val == closest_similar_value:
                    sorted_tracking_points[faulty_point_index] = p
                    #print("corrected", sorted_tracking_points)

        duplicate_points = self.list_duplicates(sorted_tracking_points)

        if len(duplicate_points) > 0 and recursion_counter < 4:
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
        sx1, sy1 = ir_points[0]
        sx2, sy2 = ir_points[1]
        sx3, sy3 = ir_points[2]
        sx4, sy4 = ir_points[3]

        # Step 1
        source_points_123 = pl.matrix([[sx1, sx2, sx3],
                                       [sy1, sy2, sy3],
                                       [1, 1, 1]])
        source_point_4 = [[sx4],
                          [sy4],
                          [1]]
        scale_to_source = pl.solve(source_points_123, source_point_4)

        # Step 2
        l, m, t = [float(x) for x in scale_to_source]
        unit_to_source = pl.matrix([[l * sx1, m * sx2, t * sx3],
                                    [l * sy1, m * sy2, t * sy3],
                                    [l * 1, m * 1, t * 1]])

        # Step 3
        # we adjust the destination rectangle in order to use the whole ir sensor area for drawing
        rectangle_long_side_dist = math.hypot(sx2 - sx1, sy2 - sy1)
        rectangle_short_side_dist = math.hypot(sx3 - sx2, sy3 - sy2)

        rectangle_to_sensor_diff_x = (self.IR_CAM_X - rectangle_long_side_dist)
        rectangle_to_sensor_diff_y = (self.IR_CAM_Y - rectangle_short_side_dist)

        origin_x = rectangle_to_sensor_diff_x/2 - rectangle_long_side_dist/2
        origin_y = rectangle_to_sensor_diff_y/2 - rectangle_short_side_dist/2

        max_x = self.DEST_W - origin_x
        max_y = self.DEST_H - origin_y
        '''
        #print("IR", self.IR_CAM_X, self.IR_CAM_Y)
        #print("rectangle", rectangle_long_side_dist, rectangle_short_side_dist)
        #print("diff", rectangle_to_sensor_diff_x, rectangle_to_sensor_diff_y)
        #print("origin", origin_x, origin_y)
        #print("dest", self.DEST_W, self.DEST_H)
        '''
        dx1, dy1 = origin_x, origin_y
        dx2, dy2 = max_x, origin_y
        dx3, dy3 = max_x, max_y
        dx4, dy4 = origin_x, max_y

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

        # Step 4
        source_to_unit = pl.inv(unit_to_source)

        # Step 5
        source_to_dest = unit_to_dest @ source_to_unit

        # Step 6
        x, y, z = [float(w) for w in (source_to_dest @ pl.matrix([[self.IR_CAM_X/2],
                                                                  [self.IR_CAM_Y/2],
                                                                  [1]]))]
        # Step 7: dehomogenization
        x = x / z
        y = y / z

        #print("drawing point", x, y)
        return x, y
