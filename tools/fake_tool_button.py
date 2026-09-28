"""Publish a fake Virtuoso joint state so the tool-button trigger can be tested
without the robot.

Mimics /ves/right/joint/measured_jp: five joints, of which "tool" carries the
cut button as 0.0 (released) or 1.0 (held). Publishes continuously at 50 Hz,
as the real driver does, so the node sees a realistic stream rather than single
messages.

    # hold for 20 s, release, exit  (one episode)
    python3 tools/fake_tool_button.py --hold 20

    # released, then press when you hit Enter, release on the next Enter
    python3 tools/fake_tool_button.py --manual

    # press/release twice, 15 s each  (two episodes)
    python3 tools/fake_tool_button.py --hold 15 --cycles 2

    # simulate a one-sample dropout mid-cut: the episode must NOT split
    python3 tools/fake_tool_button.py --hold 20 --glitch
"""
import argparse
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

JOINTS = ["inner_rotation", "outer_rotation", "tool", "roll", "insertion"]
RATE_HZ = 50.0


class FakeButton(Node):

    def __init__(self, topic):
        super().__init__("fake_tool_button")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(JointState, topic, qos)
        self.value = 0.0
        self.create_timer(1.0 / RATE_HZ, self._tick)
        self.get_logger().info(f"Publishing {topic} at {RATE_HZ:.0f} Hz "
                               f"(joint 'tool').")

    def _tick(self):
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(JOINTS)
        m.position = [0.0, 0.0, float(self.value), 0.0, 0.0]
        self.pub.publish(m)

    def set(self, v, label):
        self.value = float(v)
        self.get_logger().info(f"tool = {self.value:.1f}   ({label})")


def scripted(node, hold, gap, cycles, glitch):
    time.sleep(2.0)                      # let the node connect first
    for c in range(cycles):
        node.set(1.0, f"PRESS  (episode {c + 1} starts)")
        if glitch:
            time.sleep(hold / 2.0)
            node.set(0.0, "one-sample dropout -- episode must NOT end")
            time.sleep(1.0 / RATE_HZ)
            node.set(1.0, "back, still cutting")
            time.sleep(hold / 2.0)
        else:
            time.sleep(hold)
        node.set(0.0, f"RELEASE (episode {c + 1} saves)")
        if c < cycles - 1:
            time.sleep(gap)
    time.sleep(3.0)
    node.get_logger().info("Done. Ctrl-C to exit.")


def manual(node):
    time.sleep(1.0)
    held = False
    while True:
        try:
            input("  press Enter to " + ("RELEASE" if held else "PRESS") + " ... ")
        except EOFError:
            return
        held = not held
        node.set(1.0 if held else 0.0, "held" if held else "released")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/ves/right/joint/measured_jp")
    ap.add_argument("--hold", type=float, default=20.0,
                    help="seconds to hold the button per cycle")
    ap.add_argument("--gap", type=float, default=5.0,
                    help="seconds released between cycles")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--glitch", action="store_true",
                    help="drop the signal for one sample mid-cut")
    ap.add_argument("--manual", action="store_true",
                    help="toggle the button with Enter instead of a script")
    args = ap.parse_args()

    rclpy.init()
    node = FakeButton(args.topic)
    driver = (threading.Thread(target=manual, args=(node,), daemon=True)
              if args.manual else
              threading.Thread(target=scripted,
                               args=(node, args.hold, args.gap,
                                     args.cycles, args.glitch), daemon=True))
    driver.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
