from __future__ import annotations

import argparse
import signal
import threading

from robojudo.tools.g1_trajectory_recorder import G1TrajectoryRecorder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record G1 state, torso IMU, and commanded motor targets without publishing DDS messages."
    )
    parser.add_argument("--net-if", required=True, help="Network interface used for Unitree DDS")
    parser.add_argument("--output", help="Output .msgpack path; defaults to logs/g1_trajectory_<time>.msgpack")
    return parser.parse_args()


def main():
    args = parse_args()
    stop_event = threading.Event()

    def request_stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    recorder = G1TrajectoryRecorder(net_if=args.net_if, output_path=args.output)
    print(f"Recording G1 trajectory to {recorder.output_path}. Press Ctrl+C to stop.")
    try:
        while not stop_event.wait(0.25):
            recorder.raise_if_failed()
    finally:
        summary = recorder.close()
    print(f"Trajectory saved: {recorder.output_path} {summary}")


if __name__ == "__main__":
    main()
