#!/usr/bin/env python3
"""Export a recorded bag's camera frames at the processing size, for offline
policies and for replay through the node.

    python3 tools/export_bag_frames.py ~/toolhead_bags/session_.../bag OUT_DIR
        [--topic /ves_camera/image_rect] [--size 540] [--native] [--fps 30]

Writes
    OUT_DIR/frames/NNNNNN.png   every frame in the bag, resized exactly as the
                                node does (cv2.resize INTER_AREA, full frame,
                                no crop); --native keeps camera resolution
    OUT_DIR/frames.csv          idx, image header stamp, bag receive time,
                                estimated frames missing before this one
    OUT_DIR/frames.mp4          the same frames as a video (lossy; for viewing)

Needs a sourced ROS 2 environment (rosbag2_py, rclpy, rosidl_runtime_py);
nothing is published and nothing is commanded.
"""
import argparse
import csv
import os

import cv2
import numpy as np


def to_bgr(msg):
    """Same conversion as ros_toolhead_pose_node._to_bgr."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if enc == "rgb8" else img.copy()
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    if enc in ("bgra8", "rgba8"):
        img = buf.reshape(msg.height, msg.width, 4)[:, :, :3]
        return img[:, :, ::-1].copy() if enc == "rgba8" else img.copy()
    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("out")
    ap.add_argument("--topic", default="/ves_camera/image_rect")
    ap.add_argument("--size", type=int, default=540)
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--png-level", type=int, default=1)
    a = ap.parse_args()

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if a.topic not in types:
        raise SystemExit(f"{a.topic} not in bag; topics: {sorted(types)}")
    msg_type = get_message(types[a.topic])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[a.topic]))

    fdir = os.path.join(a.out, "frames")
    os.makedirs(fdir, exist_ok=True)
    vw = None
    rows = []
    prev = None
    periods = []
    i = 0
    while reader.has_next():
        _topic, data, t_bag = reader.read_next()
        msg = deserialize_message(data, msg_type)
        img = to_bgr(msg)
        if not a.native and img.shape[:2] != (a.size, a.size):
            img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(fdir, f"{i:06d}.png"), img,
                    [cv2.IMWRITE_PNG_COMPRESSION, a.png_level])
        if vw is None:
            h, w = img.shape[:2]
            vw = cv2.VideoWriter(os.path.join(a.out, "frames.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
        vw.write(img)
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if prev is not None:
            periods.append(st - prev)
        rows.append([i, msg.header.stamp.sec, msg.header.stamp.nanosec,
                     f"{st:.9f}", t_bag, st - prev if prev is not None else ""])
        prev = st
        i += 1
    if vw is not None:
        vw.release()

    med = float(np.median([p for p in periods if p > 0])) if periods else 0.0
    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "stamp_sec", "stamp_nsec", "stamp", "bag_time_ns",
                    "missing_before"])
        for r in rows:
            dt = r.pop()
            miss = (max(int(round(dt / med)) - 1, 0)
                    if med > 0 and dt != "" else "")
            w.writerow(r + [miss])
    print(f"{i} frames -> {fdir}  (period ~{med * 1e3:.1f} ms)")


if __name__ == "__main__":
    main()
