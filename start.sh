#!/usr/bin/env bash

# 设置为 True 时随主程序自动记录轨迹；设置为 False 时只启动主程序。
RECORD_TRAJECTORY=True
UNITREE_NET_IF="${UNITREE_NET_IF:-eth0}"

# 神奇的xml文件用于解决连不上G1 low-state问题
export CYCLONEDDS_URI="file://$(pwd)/cyclonedds.xml"

# 手柄需插在台式服务器，而不是远端笔记本
export SDL_JOYSTICK_DEVICE=/dev/input/js0

recorder_pid=""
trajectory_trigger_dir=""
trajectory_trigger_file=""

recording_enabled() {
    case "${RECORD_TRAJECTORY}" in
        True|true|TRUE)
            return 0
            ;;
        False|false|FALSE)
            return 1
            ;;
        *)
            echo "Invalid RECORD_TRAJECTORY=${RECORD_TRAJECTORY}; use True or False." >&2
            exit 2
            ;;
    esac
}

start_trajectory_recorder() {
    if ! recording_enabled; then
        unset ROBOJUDO_TRAJECTORY_START_FILE
        echo "Trajectory recording disabled."
        return
    fi

    if ! trajectory_trigger_dir="$(mktemp -d "${TMPDIR:-/tmp}/robojudo-trajectory.XXXXXX")"; then
        echo "Could not create trajectory recording trigger directory." >&2
        exit 1
    fi
    trajectory_trigger_file="${trajectory_trigger_dir}/start"
    export ROBOJUDO_TRAJECTORY_START_FILE="${trajectory_trigger_file}"

    python scripts/record_g1_trajectory.py \
        --net-if "${UNITREE_NET_IF}" \
        --start-trigger-file "${trajectory_trigger_file}" &
    recorder_pid=$!
    echo "Trajectory recorder armed (pid=${recorder_pid})."
}

stop_trajectory_recorder() {
    if [[ -n "${recorder_pid}" ]]; then
        if kill -0 "${recorder_pid}" 2>/dev/null; then
            kill -TERM "${recorder_pid}" 2>/dev/null || true
        fi

        if ! wait "${recorder_pid}"; then
            echo "Trajectory recorder exited with an error." >&2
        fi
        recorder_pid=""
    fi

    if [[ -n "${trajectory_trigger_file}" ]]; then
        rm -f -- "${trajectory_trigger_file}"
        trajectory_trigger_file=""
    fi
    if [[ -n "${trajectory_trigger_dir}" ]]; then
        rmdir -- "${trajectory_trigger_dir}" 2>/dev/null || true
        trajectory_trigger_dir=""
    fi
    unset ROBOJUDO_TRAJECTORY_START_FILE
}

on_exit() {
    local controller_status=$?
    trap - EXIT INT TERM
    stop_trajectory_recorder
    exit "${controller_status}"
}

handle_signal() {
    exit "$1"
}

run_controller() {
    # 启动命令，请自己修改run_pipeline.py内的参数
    python scripts/run_pipeline.py
}

trap on_exit EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

start_trajectory_recorder
run_controller
controller_status=$?
exit "${controller_status}"
