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
            'sat_max': 40, 'val_min': 50, 'roi': 0.6,              # road mask (roi = top of the part of the image PID looks at; bigger = look closer)
            'lane_bias': 0.0,     # + = hold the robot right of the road centre (e.g. 0.15), - = left
            'strip': 0.85,        # line detection only looks below this fraction of the image height
            'line_frac': 0.05,    # fraction of that strip that must be line-coloured to count
            'sections': ['pid', 'teleop'],  # mode before 1st purple line, after 1st, after 2nd, ...
            'start_section': 0,   # use this if you respawn further along the course
            'red_strip': 0.85,    # red crosswalk line is looked for below this fraction of the image (earlier than purple)
            'red_frac': 0.02,     # fraction of that strip that must be red to stop
            'cross_speed': 0.8,  # speed while crossing the crosswalk after the pedestrian has passed
            'approach_speed': 0.3,  # slow to this when red is in view (crosswalk coming) so the stop at the line is accurate
            'cross_time': 3.0,    # seconds to drive at cross_speed (no stopping) after 'go'
            'max_wait': 15.0,     # give up waiting for the pedestrian after this many seconds
            'road_v_max': 120,    # asphalt is dark grey; brighter greys (cars) are NOT road
            'veh_v_max': 35,      # 'very dark' pixels = vehicle (truck body/tyres are near-black)
            'veh_frac': 0.03,     # fraction of the look-ahead box that must be very dark to stop
            'veh_top': 0.6,       # top of the look-ahead box (fraction of image height); keep below the horizon/tree line
            'veh_min_h': 0.25,    # biggest dark blob must be at least this fraction of the box height (trees are a thin strip)
            'veh_clear': 1.0,     # box must stay clear this long (s) before driving on
            'veh_max_wait': 10.0, # never sit in WAIT_VEHICLE longer than this
            # pedestrian (robot is stopped at the crosswalk, so moving pixels = pedestrian)
            'ped_min_area': 0.002,  # moving blob must cover this fraction of the image to count as the pedestrian
            'ped_center': 0.5,      # where 'in front of the robot' is, as a fraction of image width (0 = left edge, 0.5 = centre)
            'ped_side': 0.2,        # 'middle' band half-width (fraction of image width): inside = in front of the robot
            # clue boards (blue frame) as landmarks
            'board_frac': 0.02,     # blue must cover this fraction of a SIDE camera image for a board to count as 'passing'
            'side_cams': True,      # count boards with the left/right cameras (False = old front-camera counting)
            'board_on': 2,          # frames in a row the board must be seen (fewer = still counts at high speed)
            'board_gap': 1.5,       # ignore any new board for this many seconds after counting one (stops double counts)
            'board_alternate': False,  # True = next board must be on the OTHER side from the last one
            # truck: wait at the T junction, then follow it
            'truck_after_board': 3, # start waiting for the truck once this many boards have gone by (0 = never)
            'truck_delay': 0.0,     # seconds of PID after that board before stopping (creep up to the T)
            'truck_wait_max': 25.0, # give up waiting for the truck after this (one truck lap is ~14 s)
            # SHIFT: sidestep right after board 3, before stopping to wait for the truck.
            # Turn out for shift_turn s, straight for shift_straight s, turn back for shift_turn s.
            'shift_ang': 1.0,       # + = shift LEFT, - = shift RIGHT (0 = no shift)
            'shift_turn': 0.5,
            'shift_straight': 0.5,
            'shift_speed': 0.3,
            'shift_rot': 0.3,       # then spin in place this many seconds (+ = LEFT, - = right) to face the truck loop
            'rot_speed': 1.0,       #   spin rate (rad/s)
            'truck_go_delay': 1.0,  # after first seeing the truck, wait this long (let it get ahead) before turning
            # Board avoidance (front camera): a big blue board frame low in the image = about to hit it.
            'avoid_frac': 0.04,     # blue blob must cover this fraction of the front image
            'avoid_gain': 1.2,      # extra turn rate (rad/s) away from the board
            'avoid_slow': 0.2,      # cap forward speed at this while avoiding
            'turn_speed': 0.3,      # scripted turn through the T: forward speed
            'turn_straight': 0.5,   #   seconds straight ahead first (into the junction)
            'turn_time': 1.5,       #   then seconds turning
            'turn_ang': 1.0,        #   turn rate (rad/s): + = LEFT, - = RIGHT
            'follow_speed': 0.4,    # after the turn: PID steering, capped at this speed (truck does 0.5)
            'follow_time': 25.0,    # fallback: give up looking for the exit after this many s of following
            # EXIT: leave the truck loop. The exit branch opens on the LEFT: road appears in the front image's lower-left box.
            'exit_min_time': 5.0,   # only look for the exit this many s after FOLLOW_TRUCK starts
            'exit_road': 0.4,       # road must fill this fraction of that box
            'exit_box': [0.55, 0.85, 0.0, 0.3],  # the box: top, bottom, left, right (fractions of the front image)
            'exit_straight': 0.6,   # then seconds straight (front cam sees the opening early)
            'exit_time': 1.5,       #   then seconds turning
            'exit_ang': 1.0,        #   turn rate (rad/s): + = LEFT
            'exit_speed': 0.3,
            'truck_close': 0.8,     # stop if the truck's bottom edge is below this fraction of the image (too close)
            'truck_lost': 1.5,      # truck out of sight this long (s) -> back to PID
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
        self.ped_x, self.ped_status, self.ped_crossed = None, '', False
        self.boards = 0
        # one board counter per camera: [frames seen, frames not seen, currently visible]
        self.counters = {'front': [0, 0, False], 'left': [0, 0, False], 'right': [0, 0, False]}
        self.side_view = {'left': None, 'right': None}  # latest small side images for the debug window
        self.last_board_t, self.last_board_cam = -1e9, None
        self.truck_trigger_t, self.truck_done = None, False
        self.truck_last_seen, self.truck_info = 0.0, ''
        self.truck_seen_t, self.turn_start, self.follow_start = None, 0.0, 0.0
        self.shift_start = 0.0
        self.exit_start = 0.0
        self.writer, self.out, self.n = None, None, 0

        self.cmd = self.create_publisher(Twist, '/cmd_vel', 1)
        self.create_subscription(Twist, '/teleop_cmd', self.on_teleop, 1)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, 1)
        # Side cameras (custom robot). Need the ros_gz_bridge command running for these topics.
        self.create_subscription(Image, '/left_camera/image_raw', lambda m: self.on_side(m, 'left'), 1)
        self.create_subscription(Image, '/right_camera/image_raw', lambda m: self.on_side(m, 'right'), 1)
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

        if not self.p('side_cams'):
            self.count_boards(hsv, 'front')

        # Truck: after board N, creep for truck_delay s, then stop at the T and wait for the truck.
        if (not self.truck_done and self.truck_trigger_t is None and self.p('truck_after_board') > 0
                and self.boards >= self.p('truck_after_board')):
            self.truck_trigger_t = t
            self.get_logger().info(f'board {self.boards} passed: truck section')
        if (self.state == 'DRIVE' and self.mode() == 'pid' and self.truck_trigger_t is not None
                and not self.truck_done and t - self.truck_trigger_t > self.p('truck_delay')):
            self.state, self.shift_start = 'SHIFT', t
            self.get_logger().info('board 3 passed: shifting over before the T')

        # Red crosswalk line (only while PID is driving).
        # (also overrides WAIT_VEHICLE: the pedestrian's dark clothes can trip the vehicle check first)
        if self.state in ('DRIVE', 'WAIT_VEHICLE') and self.mode() == 'pid' and red and not crossing:
            self.state = 'WAIT_PED'
            self.wait_start = self.last_motion = t
            self.seen_motion, self.prev_gray, self.ped_x, self.ped_crossed = False, None, None, False
            self.get_logger().info('crosswalk: waiting for pedestrian')

        # Vehicle close ahead (only while PID is driving, and not already waiting for the pedestrian).
        vehicle = self.vehicle_ahead(hsv, frame) and not crossing and not near_cross
        if self.state == 'DRIVE' and self.mode() == 'pid' and vehicle:
            self.state, self.veh_clear_since, self.veh_wait_start = 'WAIT_VEHICLE', None, t
            self.get_logger().info('vehicle ahead: stopping')

        if self.state in ('SHIFT', 'WAIT_TRUCK', 'T_TURN', 'FOLLOW_TRUCK', 'EXIT'):
            tw = self.truck_step(hsv, frame, t)
        elif self.state == 'WAIT_VEHICLE':
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
            elif near_cross and tw.linear.x > 0:  # crosswalk coming up: slow down before the red line
                tw.linear.x = min(tw.linear.x, self.p('approach_speed'))
        else:  # 'teleop': you drive, we record
            tw = self.teleop
            self.record(raw, tw)

        if self.mode() == 'pid' and tw.linear.x > 0:
            tw = self.avoid_board(hsv, frame, tw)
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
            err = (cx - w / 2) / (w / 2) + self.p('lane_bias')  # -1 road is left ... +1 road is right
            turn = -(self.p('kp') * err + self.p('kd') * (err - self.prev_err))
            tw.linear.x = self.p('speed')
            tw.angular.z = float(np.clip(turn, -self.p('max_turn'), self.p('max_turn')))
            self.prev_err = err
            cv2.circle(frame, (int(cx), (y0 + h) // 2), 6, (0, 0, 255), -1)
        else:  # lost the road: stop and turn toward where it was
            tw.angular.z = -0.5 * float(np.sign(self.prev_err) or 1.0)
        return tw

    def vehicle_ahead(self, hsv, frame):
        """Truck = one big, solid, near-black blob in the look-ahead box.
        Background trees only make small specks / a thin strip along the top of the box,
        so keep just the biggest connected blob and require it to be big AND tall."""
        h, w = hsv.shape[:2]
        y1, y2, x1, x2 = int(self.p('veh_top') * h), int(self.p('strip') * h), int(0.2 * w), int(0.8 * w)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        box = hsv[y1:y2, x1:x2]
        dark = ((box[..., 2] <= self.p('veh_v_max')) & (box[..., 1] <= 80)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # wipe out single specks
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        if n <= 1:  # label 0 is the background, so no blobs at all
            return False
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # biggest blob
        frame[y1:y2, x1:x2][labels == i] = (0, 0, 255)       # debug: biggest blob painted red
        big = stats[i, cv2.CC_STAT_AREA] > self.p('veh_frac') * dark.size
        tall = stats[i, cv2.CC_STAT_HEIGHT] > self.p('veh_min_h') * dark.shape[0]
        return bool(big and tall)

    def pedestrian_clear(self, frame, t):
        """Robot is stopped, so pixels that change between frames = the pedestrian moving.
        Wait until he has walked ACROSS in front of us (seen in the middle band), then go the
        moment he leaves the middle on the far side. That starts the crossing with his whole
        'walk to the side + 2 s pause + walk back' window ahead of us."""
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        h, w = gray.shape
        if self.prev_gray is not None:
            motion = (cv2.absdiff(gray, self.prev_gray) > 25).astype(np.uint8)
            motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, labels, stats, cents = cv2.connectedComponentsWithStats(motion, connectivity=8)
            if n > 1:
                i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                if stats[i, cv2.CC_STAT_AREA] > self.p('ped_min_area') * motion.size:
                    self.seen_motion, self.last_motion, self.ped_x = True, t, float(cents[i][0])
                    x, y, bw, bh = (int(v) for v in stats[i, :4])
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 255), 2)  # debug: pedestrian box
        self.prev_gray = gray

        band = self.p('ped_side') * w
        mid = self.p('ped_center') * w
        middle = self.ped_x is not None and abs(self.ped_x - mid) <= band
        if middle:
            self.ped_crossed = True  # he is walking in front of us right now
        cv2.line(frame, (int(mid - band), 0), (int(mid - band), h), (255, 0, 255), 1)  # middle band edges
        cv2.line(frame, (int(mid + band), 0), (int(mid + band), h), (255, 0, 255), 1)
        if not self.seen_motion:
            self.ped_status = 'ped: not seen'
        elif middle:
            self.ped_status = 'ped: IN FRONT'
        else:
            self.ped_status = 'ped: side, passed' if self.ped_crossed else 'ped: side, waiting for him to cross'

        waited = t - self.wait_start
        if waited > self.p('max_wait'):
            self.get_logger().warn('pedestrian wait timed out, going anyway')
            return True
        return self.ped_crossed and self.ped_x is not None and not middle

    def on_side(self, msg, side):
        """Side cameras only count clue boards (and feed the debug window)."""
        small = cv2.resize(self.br.imgmsg_to_cv2(msg, 'bgr8'), (320, 180))
        if self.p('side_cams'):
            self.count_boards(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), side)
        self.side_view[side] = small

    def count_boards(self, hsv, cam):
        """Clue boards have a saturated blue frame. Count one each time a big blue patch
        has been in this camera's view (board_on+ frames) and then leaves it (5+ frames)."""
        blue = cv2.inRange(hsv, (105, 150, 50), (130, 255, 255))
        now = cv2.countNonZero(blue) > self.p('board_frac') * blue.size
        c = self.counters[cam]
        c[0] = c[0] + 1 if now else 0
        c[1] = c[1] + 1 if not now else 0
        if not c[2] and c[0] >= self.p('board_on'):
            c[2] = True
        elif c[2] and c[1] >= 5:
            c[2] = False
            t = self.now()
            if t - self.last_board_t < self.p('board_gap'):
                self.get_logger().info(f'{cam}: board ignored (only {t - self.last_board_t:.1f}s after the last one)')
            elif self.p('board_alternate') and cam == self.last_board_cam:
                self.get_logger().info(f'{cam}: board ignored (same side as the last one)')
            else:
                self.boards += 1
                self.last_board_t, self.last_board_cam = t, cam
                self.get_logger().info(f'passed clue board #{self.boards} ({cam} camera)')

    def avoid_board(self, hsv, frame, tw):
        """Safety layer on top of EVERY driving state (PID, sidestep, T turn, follow, exit):
        if a clue board's blue frame is big and low in the front view, steer away from it and slow down."""
        h, w = hsv.shape[:2]
        blue = cv2.inRange(hsv, (105, 150, 50), (130, 255, 255))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(blue, connectivity=8)
        if n <= 1:
            return tw
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < self.p('avoid_frac') * blue.size or y + bh < 0.5 * h:
            return tw
        side = 1.0 if cents[i][0] < w / 2 else -1.0       # board on the left -> turn right (negative)
        tw.angular.z = float(np.clip(tw.angular.z - side * self.p('avoid_gain'), -self.p('max_turn'), self.p('max_turn')))
        tw.linear.x = min(tw.linear.x, self.p('avoid_slow'))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 0), 2)
        cv2.putText(frame, 'AVOID BOARD', (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        return tw

    def find_truck(self, hsv, frame):
        """Biggest near-black blob below the horizon. Returns (centre_x, bottom_y, height) or None."""
        h, w = hsv.shape[:2]
        y1, y2 = int(0.35 * h), int(self.p('strip') * h)
        reg = hsv[y1:y2]
        dark = ((reg[..., 2] <= self.p('veh_v_max')) & (reg[..., 1] <= 80)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
        best = None
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            bottom = y1 + y + bh
            # trees/backdrop: a thin strip that ends near the horizon. Truck: sits lower, on the road.
            if area < 0.004 * dark.size or bottom < 0.55 * h or bh < 0.08 * h:
                continue
            if best is None or area > best[3]:
                best = (float(cents[i][0]), bottom, bh, area, (x, y1 + y, bw, bh))
        if best is None:
            return None
        x, y, bw, bh = best[4]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 255), 2)  # debug: truck box
        return best[:3]

    def left_branch(self, hsv, frame):
        """Fraction of the front image's lower-left box that is road (grey/white) = a road opening on the left."""
        h, w = hsv.shape[:2]
        t0, b0, l0, r0 = self.p('exit_box')
        y1, y2, x1, x2 = int(t0 * h), int(b0 * h), int(l0 * w), int(r0 * w)
        box = hsv[y1:y2, x1:x2]
        v = box[..., 2]
        road = (box[..., 1] <= self.p('sat_max')) & (((v >= self.p('val_min')) & (v <= self.p('road_v_max'))) | (v >= 200))
        frac = float(np.count_nonzero(road)) / road.size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if frac > self.p('exit_road') else (0, 128, 255), 1)
        return frac

    def truck_step(self, hsv, frame, t):
        """WAIT_TRUCK: stand still at the T until the truck shows up, then give it truck_go_delay s.
        T_TURN: scripted turn through the junction (no camera steering, so no chasing the truck into the sign).
        FOLLOW_TRUCK: normal PID steering, speed capped below the truck's, stop if it's too close.
        EXIT: once road opens up on the left of the front image, scripted left turn out of the loop."""
        h, w = hsv.shape[:2]
        truck = self.find_truck(hsv, frame)
        tw = Twist()
        if truck is not None:
            self.truck_last_seen = t
        if self.state == 'SHIFT':
            e, a = t - self.shift_start, self.p('shift_ang')
            t1, t2 = self.p('shift_turn'), self.p('shift_straight')
            tr = self.p('shift_rot')
            if a == 0:
                t1 = t2 = 0.0
            if e >= 2 * t1 + t2 + abs(tr):
                self.state, self.wait_start = 'WAIT_TRUCK', t
                self.get_logger().info('at the T: waiting for the truck')
                return tw
            if e >= 2 * t1 + t2:              # spin in place toward the loop
                tw.angular.z = float(np.sign(tr) * self.p('rot_speed'))
                self.truck_info = f'rotate {e - 2 * t1 - t2:.1f}s'
                return tw
            tw.linear.x = self.p('shift_speed')
            if e < t1:
                tw.angular.z = float(a)       # turn out
            elif e >= t1 + t2:
                tw.angular.z = -float(a)      # turn back to the original heading
            self.truck_info = f'shift {"left" if a > 0 else "right"} {e:.1f}s'
            return tw
        if self.state == 'WAIT_TRUCK':
            if truck is not None and self.truck_seen_t is None:
                self.truck_seen_t = t
                self.get_logger().info('truck spotted')
            if self.truck_seen_t is not None:
                self.truck_info = f'truck: seen, going in {self.p("truck_go_delay") - (t - self.truck_seen_t):.1f}s'
                if t - self.truck_seen_t > self.p('truck_go_delay'):
                    self.state, self.turn_start = 'T_TURN', t
                    self.get_logger().info('turning through the T')
            else:
                self.truck_info = f'truck: waiting {t - self.wait_start:.0f}s'
                if t - self.wait_start > self.p('truck_wait_max'):
                    self.state, self.truck_done = 'DRIVE', True
                    self.get_logger().warn('truck wait timed out, back to PID')
            return tw
        if self.state == 'T_TURN':
            e = t - self.turn_start
            if e < self.p('turn_straight'):
                tw.linear.x = self.p('turn_speed')
            elif e < self.p('turn_straight') + self.p('turn_time'):
                tw.linear.x, tw.angular.z = self.p('turn_speed'), float(self.p('turn_ang'))
            else:
                self.state, self.follow_start = 'FOLLOW_TRUCK', t
                self.get_logger().info('turn done: following truck with PID')
            self.truck_info = f'truck: scripted turn {e:.1f}s'
            return tw
        if self.state == 'EXIT':
            e = t - self.exit_start
            if e < self.p('exit_straight'):
                tw.linear.x = self.p('exit_speed')
            elif e < self.p('exit_straight') + self.p('exit_time'):
                tw.linear.x, tw.angular.z = self.p('exit_speed'), float(self.p('exit_ang'))
            else:
                self.state, self.truck_done = 'DRIVE', True
                self.get_logger().info('out of the loop: normal driving')
            self.truck_info = f'exit turn {e:.1f}s'
            return tw
        # FOLLOW_TRUCK
        followed = t - self.follow_start
        branch = self.left_branch(hsv, frame)
        if followed > self.p('exit_min_time') and branch > self.p('exit_road'):
            self.state, self.exit_start = 'EXIT', t
            self.get_logger().info(f'road opening on the left ({branch * 100:.0f}%): turning out of the loop')
            return tw
        if followed > self.p('follow_time'):
            self.state, self.truck_done = 'DRIVE', True
            self.get_logger().info('truck section done: normal driving')
            return self.pid(hsv, frame)
        if truck is not None and truck[1] > self.p('truck_close') * h:
            self.truck_info = 'truck: TOO CLOSE'
            return tw
        tw = self.pid(hsv, frame)
        tw.linear.x = min(tw.linear.x, self.p('follow_speed'))
        self.truck_info = f'truck: following (PID) | left road {branch * 100:.0f}%' + (' armed' if followed > self.p('exit_min_time') else '')
        return tw

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
        label += f' | boards {self.boards}'
        if self.state == 'WAIT_PED':
            label += ' | ' + self.ped_status
        if self.state in ('SHIFT', 'WAIT_TRUCK', 'T_TURN', 'FOLLOW_TRUCK', 'EXIT'):
            label += ' | ' + self.truck_info
        if self.writer is not None and self.mode() == 'teleop':
            label += f' | REC {self.n}'
        cv2.putText(frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        view = cv2.resize(frame, (640, 360))
        if self.p('side_cams'):  # left | right camera strip under the main view, board-seen in green
            tiles = []
            for side in ('left', 'right'):
                img = self.side_view[side]
                tile = np.zeros((180, 320, 3), np.uint8) if img is None else img.copy()
                seen = self.counters[side][2]
                txt = f'{side}: ' + ('no image (bridge running?)' if img is None else ('BOARD' if seen else '-'))
                cv2.putText(tile, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if seen else (0, 255, 255), 1)
                tiles.append(tile)
            view = np.vstack([view, np.hstack(tiles)])
        cv2.imshow('driver', view)
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