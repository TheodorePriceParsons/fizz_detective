import rclpy, cv2, csv, os, time
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class Logger(Node):
    def __init__(self):
        super().__init__('il_logger')
        self.out = os.path.expanduser(f'~/il_data/{int(time.time())}')
        os.makedirs(self.out)
        self.f = open(f'{self.out}/labels.csv', 'w'); self.w = csv.writer(self.f)
        self.br, self.cmd, self.n = CvBridge(), None, 0
        self.create_subscription(Twist, '/cmd_vel', lambda m: setattr(self, 'cmd', m), 1)
        self.create_subscription(Image, '/camera/image_raw', self.img, 1)

    def img(self, msg):
        if self.cmd is None or self.cmd.linear.x <= 0:   # only record while driving forward
            return
        f = cv2.resize(self.br.imgmsg_to_cv2(msg, 'bgr8'), (160, 90))
        name = f'{self.n:06d}.png'; cv2.imwrite(f'{self.out}/{name}', f)
        self.w.writerow([name, self.cmd.angular.z]); self.n += 1

def main():
    rclpy.init(); rclpy.spin(Logger())