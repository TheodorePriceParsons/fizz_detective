"""Time-trial pass: start timer, drive > 1 m forward, stop timer. Uses sim time (/clock)."""
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist

from controller_pkg.score_client import ScoreClient


class TimeTrial(Node):
    def __init__(self):
        super().__init__('time_trial',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.declare_parameter('team', 'TeamName')
        self.declare_parameter('password', 'pass')
        self.declare_parameter('speed', 0.4)       # m/s
        self.declare_parameter('distance', 1.6)    # m (> 1 m required)
        p = lambda n: self.get_parameter(n).value
        self.score = ScoreClient(self, p('team'), p('password'))
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 1)
        self.speed, self.drive_s = p('speed'), p('distance') / p('speed')
        self.state, self.t0 = 'WAIT', None
        self.create_timer(0.1, self.tick)  # timers follow sim time

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def tick(self):
        t = self.now()
        if t == 0.0:            # /clock not received yet
            return
        if self.t0 is None:
            self.t0 = t
        dt = t - self.t0
        if self.state == 'WAIT' and dt > 1.0:      # let pubs register (notes 3.4.3)
            self.score.start()
            self.state, self.t0 = 'DRIVE', t
        elif self.state == 'DRIVE':
            tw = Twist()
            tw.linear.x = self.speed if dt < self.drive_s else 0.0
            self.cmd.publish(tw)
            if dt > self.drive_s + 0.5:
                self.state, self.t0 = 'STOP', t
        elif self.state == 'STOP' and dt > 0.3:
            self.score.stop()
            self.state = 'DONE'
            self.create_timer(0.5, lambda: rclpy.try_shutdown())


def main():
    rclpy.init()
    node = TimeTrial()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
