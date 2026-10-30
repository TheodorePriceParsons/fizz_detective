"""Thin wrapper around /score_tracker. Only allowed outputs: /cmd_vel and /score_tracker."""
from std_msgs.msg import String

# Clue location IDs, in the order of clues.csv (row index + 1).
CLUE_IDS = {'SIZE': 1, 'VICTIM': 2, 'CRIME': 3, 'TIME': 4,
            'PLACE': 5, 'MOTIVE': 6, 'WEAPON': 7, 'BANDIT': 8}


class ScoreClient:
    def __init__(self, node, team, password):
        assert len(team) <= 8 and len(password) <= 8 and ' ' not in team + password
        self.node, self.team, self.pw = node, team, password
        self.pub = node.create_publisher(String, '/score_tracker', 1)
        self.submitted = {}

    def _send(self, loc, value):
        msg = String(data=f'{self.team},{self.pw},{loc},{value}')
        self.pub.publish(msg)
        self.node.get_logger().info(f'score_tracker <- {msg.data}')

    def start(self):
        self._send(0, 'NA')

    def stop(self):
        self._send(-1, 'NA')

    def submit(self, loc, value):
        value = value.replace(' ', '').upper()  # spaces are optional; queue=1 so don't spam
        if self.submitted.get(loc) == value:
            return
        self.submitted[loc] = value
        self._send(loc, value)
