"""Competition driver.

- Paved road: PID lane following.
- Red crosswalk line: stop, wait for the pedestrian to finish crossing, go.
- Dark object close ahead (the truck is mostly black): stop until it has moved out of the way.
- Purple line: move to the next section. Each section is 'pid' or 'teleop' (param `sections`).
- 'teleop' sections: you drive, and every frame + your steering is saved to ~/il_data/<time>/.

Teleop MUST be remapped so it doesn't fight this node on /cmd_vel:
  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/teleop_cmd
"""
import csv
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

# OpenCV HSV ranges (H 0-180). Track texture: purple = RGB(255,0,255), red = RGB(244,0,0)
PURPLE = [((140, 120, 80), (165, 255, 255))]
RED = [((0, 120, 80), (8, 255, 255)), ((172, 120, 80), (180, 255, 255))]


def colour_mask(hsv, ranges):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, lo, hi)
    return m


class Driver(Node):
    def __init__(self):
        super().__init__('driver', parameter_overrides=[Parameter('use_sim_time', value=True)])
        defaults = {
            'speed': 0.3, 'kp': 1.2, 'kd': 0.4, 'max_turn': 1.5,   # PID
            'sat_max': 40, 'val_min': 50, 'roi': 0.6,              # road mask
            'strip': 0.85,        # line detection only looks below this fraction of the image height
            'line_frac': 0.05,    # fraction of that strip that must be line-coloured to count
            'sections': ['pid', 'teleop'],  # mode before 1st purple line, after 1st, after 2nd, ...
            'start_section': 0,   # use this if you respawn further along the course
            'red_strip': 0.85,    # red crosswalk line is looked for below this fraction of the image (earlier than purple)
            'red_frac': 0.02,     # fraction of that strip that must be red to stop
            'cross_speed': 0.8,  # speed while crossing the crosswalk after the pedestrian has passed
            'cross_time': 3.0,    # seconds to drive at cross_speed (no stopping) after 'go'
            'max_wait': 15.0,     # give up waiting for the pedestrian after this many seconds
            'road_v_max': 120,    # asphalt is dark grey; brighter greys (cars) are NOT road
            'veh_v_max': 35,      # 'very dark' pixels = vehicle (truck body/tyres are near-black)
            'veh_frac': 0.03,     # fraction of the look-ahead box that must be very dark to stop
            'veh_top': 0.6,       # top of the look-ahead box (fraction of image height); keep below the horizon/tree line
            'veh_clear': 1.0,
            'veh_max_wait': 10.0, # never sit in WAIT_VEHICLE longer than this
     # box must stay clear this long (s) before driving on
        }
        for k, v in defaults.items():
            self.declare_parameter(k, v)

        self.br = CvBridge()
        self.section = self.p('start_section')
        self.state = 'DRIVE'
        self.prev_err = 0.0
        self.teleop = Twist()
        self.purple_frames = 0
        self.ignore_purple_until = 0.0
        self.ignore_red_until = 0.0
        self.wait_start = self.last_motion = 0.0
        self.veh_clear_since = None
        self.red_frac = 0.0
        self.veh_wait_start = 0.0
        self.seen_motion = False
        self.prev_gray = None
        self.writer, self.out, self.n = None, None, 0

        self.cmd = self.create_publisher(Twist, '/cmd_vel', 1)
        self.create_subscription(Twist, '/teleop_cmd', self.on_teleop, 1)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, 1)
        self.get_logger().info(f'starting in section {self.section}: {self.mode()}')

    # ---------- helpers ----------
    def p(self, k):
        return self.get_parameter(k).value

    def now(self):
        # Wall-clock time. The NPCs' own pauses (e.g. the pedestrian's 2 s wait) use wall time too,
        # and this keeps working even if /clock isn't coming through (sim time would be stuck at 0).
        return time.monotonic()

    def mode(self):
        s = self.p('sections')
        return s[min(self.section, len(s) - 1)]

    def on_teleop(self, msg):
        self.teleop = msg

    def colour_frac(self, hsv, ranges, start):
        strip = hsv[int(hsv.shape[0] * start):]
        return cv2.countNonZero(colour_mask(strip, ranges)) / float(strip.shape[0] * strip.shape[1])

    # ---------- main loop: one step per camera frame ----------
    def on_image(self, msg):
        t = self.now()
        raw = self.br.imgmsg_to_cv2(msg, 'bgr8')
        frame = cv2.resize(raw, (320, 180))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        purple = self.colour_frac(hsv, PURPLE, self.p('strip')) > self.p('line_frac')
        self.red_frac = self.colour_frac(hsv, RED, self.p('red_strip'))
        red = self.red_frac > self.p('red_frac')
        crossing = t < self.ignore_red_until  # just got the go-ahead at the crosswalk
        # Any red in the lower half = crosswalk ahead. The pedestrian there is handled by WAIT_PED,
        # so the vehicle check is switched off (otherwise the pedestrian looks like a 'vehicle').
        near_cross = self.colour_frac(hsv, RED, 0.5) > 0.003

        # Purple line: needs 3 frames in a row, then ignored for 3 s so one line counts once.
        if t > self.ignore_purple_until:
            self.purple_frames = self.purple_frames + 1 if purple else 0
            if self.purple_frames >= 3:
                self.section += 1
                self.purple_frames, self.ignore_purple_until = 0, t + 3.0
                self.get_logger().info(f'purple line -> section {self.section}: {self.mode()}')

        # Red crosswalk line (only while PID is driving).
        # (also overrides WAIT_VEHICLE: the pedestrian's dark clothes can trip the vehicle check first)
        if self.state in ('DRIVE', 'WAIT_VEHICLE') and self.mode() == 'pid' and red and not crossing:
            self.state = 'WAIT_PED'
            self.wait_start = self.last_motion = t
            self.seen_motion, self.prev_gray = False, None
            self.get_logger().info('crosswalk: waiting for pedestrian')

        # Vehicle close ahead (only while PID is driving, and not already waiting for the pedestrian).
        vehicle = self.vehicle_ahead(hsv, frame) and not crossing and not near_cross
        if self.state == 'DRIVE' and self.mode() == 'pid' and vehicle:
            self.state, self.veh_clear_since, self.veh_wait_start = 'WAIT_VEHICLE', None, t
            self.get_logger().info('vehicle ahead: stopping')

        if self.state == 'WAIT_VEHICLE':
            tw = Twist()
            if vehicle:
                self.veh_clear_since = None
            elif self.veh_clear_since is None:
                self.veh_clear_since = t
            elif t - self.veh_clear_since > self.p('veh_clear'):
                self.state = 'DRIVE'
                self.get_logger().info('vehicle gone: go')
            if self.state == 'WAIT_VEHICLE' and t - self.veh_wait_start > self.p('veh_max_wait'):
                self.state = 'DRIVE'
                self.get_logger().warn('vehicle wait timed out, going anyway')
        elif self.state == 'WAIT_PED':
            tw = Twist()  # stand still
            if self.pedestrian_clear(frame, t):
                self.state = 'DRIVE'
                self.ignore_red_until = t + self.p('cross_time')  # drive over both red lines without stopping
                self.get_logger().info('crosswalk clear: go')
        elif self.mode() == 'pid':
            tw = self.pid(hsv, frame)
            if crossing and tw.linear.x > 0:
                tw.linear.x = self.p('cross_speed')
        else:  # 'teleop': you drive, we record
            tw = self.teleop
            self.record(raw, tw)

        self.cmd.publish(tw)
        self.show(frame, purple, red)

    # ---------- behaviours ----------
    def pid(self, hsv, frame):
        h, w = hsv.shape[:2]
        y0 = int(h * self.p('roi'))
        # road (grey) + white lines have ~zero saturation; grass and sand are saturated
        # Brighter greys (~150, e.g. car paint) are excluded so parked/moving cars don't look like road.
        sub = hsv[y0:]
        low_sat = sub[..., 1] <= self.p('sat_max')
        v = sub[..., 2]
        road = (v >= self.p('val_min')) & (v <= self.p('road_v_max'))
        mask = ((low_sat & (road | (v >= 200))) * 255).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        frame[y0:][mask > 0] = (0.5 * frame[y0:][mask > 0] + (0, 127, 0)).astype(np.uint8)

        tw = Twist()
        if cv2.countNonZero(mask) > 0.05 * mask.size:
            M = cv2.moments(mask, binaryImage=True)
            cx = M['m10'] / M['m00']
            err = (cx - w / 2) / (w / 2)  # -1 road is left ... +1 road is right
            turn = -(self.p('kp') * err + self.p('kd') * (err - self.prev_err))
            tw.linear.x = self.p('speed')
            tw.angular.z = float(np.clip(turn, -self.p('max_turn'), self.p('max_turn')))
            self.prev_err = err
            cv2.circle(frame, (int(cx), (y0 + h) // 2), 6, (0, 0, 255), -1)
        else:  # lost the road: stop and turn toward where it was
            tw.angular.z = -0.5 * float(np.sign(self.prev_err) or 1.0)
        return tw

    def vehicle_ahead(self, hsv, frame):
        """Look-ahead box in the middle of the road. The truck is mostly near-black and
        low saturation; road (~77), grass and sand are all brighter or saturated."""
        h, w = hsv.shape[:2]
        y1, y2, x1, x2 = int(self.p('veh_top') * h), int(self.p('strip') * h), int(0.2 * w), int(0.8 * w)
        box = hsv[y1:y2, x1:x2]
        dark = (box[..., 2] <= self.p('veh_v_max')) & (box[..., 1] <= 80)
        frame[y1:y2, x1:x2][dark] = (0, 0, 255)  # debug: dark pixels painted red
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        return dark.mean() > self.p('veh_frac')

    def pedestrian_clear(self, frame, t):
        """Robot is stopped, so anything changing between frames is an NPC moving.
        The pedestrian walks across, pauses ~2 s on each side, walks back.
        Go right after we've seen it move and then stop (start of a pause)."""
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            moving = cv2.countNonZero((diff > 25).astype(np.uint8)) > 0.003 * diff.size
            if moving:
                self.seen_motion, self.last_motion = True, t
        self.prev_gray = gray
        waited = t - self.wait_start
        if waited > self.p('max_wait'):
            self.get_logger().warn('pedestrian wait timed out, going anyway')
            return True
        return waited > 0.5 and self.seen_motion and t - self.last_motion > 0.3

    def record(self, raw, tw):
        if tw.linear.x <= 0:  # only learn from driving forward
            return
        if self.writer is None:
            self.out = os.path.expanduser(f'~/il_data/{int(time.time())}')
            os.makedirs(self.out)
            self.file = open(f'{self.out}/labels.csv', 'w')
            self.writer = csv.writer(self.file)
            self.get_logger().info(f'recording to {self.out}')
        name = f'{self.n:06d}.png'
        cv2.imwrite(f'{self.out}/{name}', cv2.resize(raw, (160, 90)))
        self.writer.writerow([name, tw.angular.z])
        self.file.flush()
        self.n += 1
        if self.n % 100 == 0:
            self.get_logger().info(f'{self.n} frames saved')

    def show(self, frame, purple, red):
        h = frame.shape[0]
        cv2.line(frame, (0, int(h * self.p('strip'))), (frame.shape[1], int(h * self.p('strip'))), (255, 255, 0), 1)
        cv2.line(frame, (0, int(h * self.p('red_strip'))), (frame.shape[1], int(h * self.p('red_strip'))), (0, 0, 255), 1)
        label = f'{self.state} | sec {self.section}: {self.mode()}'
        label += ' | PURPLE' if purple else ''
        label += f' | red {self.red_frac * 100:.1f}%' + (' STOP' if red else '')
        if self.writer is not None and self.mode() == 'teleop':
            label += f' | REC {self.n}'
        cv2.putText(frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow('driver', cv2.resize(frame, (640, 360)))
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.writer is not None:
            node.file.close()
