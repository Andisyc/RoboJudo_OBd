from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path

from robojudo.tools.g1_trajectory_recorder import G1TrajectoryRecorder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record G1 state, torso IMU, and commanded motor targets without publishing DDS messages."
    )
    parser.add_argument("--net-if", required=True, help="Network interface used for Unitree DDS")
    parser.add_argument("--output", help="Output .msgpack path; defaults to logs/g1_trajectory_<time>.msgpack")
    parser.add_argument(
        "--start-trigger-file",
        help="Arm immediately, but do not write until this file exists",
    )
    parser.add_argument(
        "--ready-file",
        help="Create this file after DDS subscriptions are initialized",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stop_event = threading.Event()

    def request_stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    trigger_path = Path(args.start_trigger_file) if args.start_trigger_file else None
    recorder = G1TrajectoryRecorder(
        net_if=args.net_if,
        output_path=args.output,
        armed=trigger_path is not None,
    )
    if args.ready_file:
        Path(args.ready_file).touch(exist_ok=True)
    if trigger_path is None:
        print(f"Recording G1 trajectory to {recorder.output_path}. Press Ctrl+C to stop.")
    else:
        print("Trajectory recorder armed; waiting for the first forward command.")
    try:
        while not stop_event.wait(0.02 if not recorder.has_started else 0.25):
            if trigger_path is not None and not recorder.has_started and trigger_path.exists():
                recorder.start()
                print(f"Forward command detected; recording to {recorder.output_path}.")
            recorder.raise_if_failed()
    finally:
        has_recorded = recorder.has_started
        summary = recorder.close()
    if has_recorded:
        print(f"Trajectory saved: {recorder.output_path} {summary}")
    else:
        print("No forward command detected; no trajectory file was created.")


if __name__ == "__main__":
    main()
