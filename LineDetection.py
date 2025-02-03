import cv2
import numpy as np
import math
import time

class LineDetection:
    def __init__(self, pixel_to_meter=0.01):
        """
        Initializes the LaneDetection class with necessary parameters.

        :param pixel_to_meter: Conversion factor from pixels to meters (default: 0.01)
        """
        self.pixel_to_meter = pixel_to_meter
        self.previous_left = None
        self.previous_right = None
        self.previous_offset = None
        self.previous_time = time.time()
        self.polygon_points = []
        self.drawing = False

    def mouse_callback(self, event, x, y, flags, param):
        """
        Callback function to capture mouse events for selecting the Region of Interest (ROI).

        :param event: Mouse event
        :param x: x-coordinate of the mouse click
        :param y: y-coordinate of the mouse click
        :param flags: Mouse event flags
        :param param: Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygon_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True

    def filter_lines(self, lines):
        """
        Filters the detected lines into left and right lines based on their slope.

        :param lines: Detected lines from Hough Transform
        :return: Left and right lines as two separate lists
        """
        left_lines, right_lines = [], []

        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 1e-6:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:
                continue
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

        return left_lines, right_lines

    def average_lines(self, frame, lines, previous_line, alpha=0.7):
        """
        Averages the current lines with the previous ones to reduce noise and fluctuations.

        :param frame: Current video frame
        :param lines: Detected lines
        :param previous_line: Previous line to be averaged with
        :param alpha: Weight factor for averaging
        :return: Averaged line
        """
        if not lines:
            return previous_line

        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        if len(x_coords) == 0:
            return previous_line

        poly_fit = np.polyfit(x_coords, y_coords, 1)
        y1_new = frame.shape[0]
        y2_new = int(frame.shape[0] * 0.6)
        x1_new = int((y1_new - poly_fit[1]) / poly_fit[0])
        x2_new = int((y2_new - poly_fit[1]) / poly_fit[0])
        new_line = (x1_new, y1_new, x2_new, y2_new)

        if previous_line is not None:
            x1_new = int(previous_line[0] * alpha + new_line[0] * (1 - alpha))
            y1_new = int(previous_line[1] * alpha + new_line[1] * (1 - alpha))
            x2_new = int(previous_line[2] * alpha + new_line[2] * (1 - alpha))
            y2_new = int(previous_line[3] * alpha + new_line[3] * (1 - alpha))
            new_line = (x1_new, y1_new, x2_new, y2_new)

        return new_line

    def angle_between_lines(self, line1, line2):
        """
        Calculates the angle between two lines.

        :param line1: First line (x1, y1, x2, y2)
        :param line2: Second line (x1, y1, x2, y2)
        :return: Angle between the two lines in degrees
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        angle1 = math.atan2(y2 - y1, x2 - x1)
        angle2 = math.atan2(y4 - y3, x4 - x3)
        angle = math.degrees(angle1 - angle2)
        return angle

    def process_frame(self, frame, polygon, prev_left, prev_right, prev_offset, prev_time):
        """
        Processes each frame to detect lane markings, calculate vehicle position, and estimate steering angle.

        :param frame: The current video frame
        :param polygon: The Region of Interest (ROI) polygon
        :param prev_left: Previous left lane line
        :param prev_right: Previous right lane line
        :param prev_offset: Previous lane offset
        :param prev_time: Previous timestamp for offset rate calculation
        :return: Processed frame with annotations, state dictionary, and previous state values
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        car_center = width // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 50, 150)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=30)
        left_lines, right_lines = self.filter_lines(lines)

        left_lane = self.average_lines(frame, left_lines, prev_left)
        right_lane = self.average_lines(frame, right_lines, prev_right)

        line_image = np.zeros_like(frame)
        if left_lane is not None:
            cv2.line(line_image, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (255, 0, 0), 5)
        if right_lane is not None:
            cv2.line(line_image, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (255, 0, 0), 5)

        left_x_bottom = left_lane[0] if left_lane is not None else 0
        right_x_bottom = right_lane[0] if right_lane is not None else width

        cv2.circle(line_image, (left_x_bottom, height), 8, (0, 255, 0), -1)
        cv2.circle(line_image, (right_x_bottom, height), 8, (0, 255, 0), -1)
        cv2.circle(line_image, (car_center, height), 8, (0, 0, 255), -1)

        lane_center = (left_x_bottom + right_x_bottom) // 2

        offset_pixels = lane_center - car_center
        offset = offset_pixels * self.pixel_to_meter

        left_distance_pixels = car_center - left_x_bottom
        right_distance_pixels = right_x_bottom - car_center
        left_distance = left_distance_pixels * self.pixel_to_meter
        right_distance = right_distance_pixels * self.pixel_to_meter

        lane_width_pixels = right_x_bottom - left_x_bottom
        lane_width = lane_width_pixels * self.pixel_to_meter

        current_time = time.time()
        dt = current_time - prev_time if current_time - prev_time > 0 else 1e-6
        if prev_offset is not None:
            offset_rate = (offset - prev_offset) / dt
        else:
            offset_rate = 0

        if left_lane is not None and right_lane is not None:
            left_angle = math.degrees(math.atan2((left_lane[1] - left_lane[3]), (left_lane[0] - left_lane[2])))
            right_angle = math.degrees(math.atan2((right_lane[1] - right_lane[3]), (right_lane[0] - right_lane[2])))
            steering_angle = (left_angle + right_angle) / 2
        else:
            steering_angle = 0

        output = cv2.addWeighted(output_frame, 0.8, line_image, 1, 0)
        cv2.putText(output, f"Offset: {offset:.2f} m", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(output, f"Left Dist: {left_distance:.2f} m", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(output, f"Right Dist: {right_distance:.2f} m", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(output, f"Lane Width: {lane_width:.2f} m", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(output, f"Steering Angle: {steering_angle:.2f} degrees", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        state = {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'offset': offset,
            'offset_rate': offset_rate,
            'lane_width': lane_width,
            'left_distance': left_distance,
            'right_distance': right_distance,
            'steering_angle': steering_angle
        }

        return output, state, left_lane, right_lane, offset, current_time
