import logging
import time
import torch
import numpy as np
from box import Box

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.policy import Policy, PolicyCfg
from robojudo.tools.dof import DoFAdapter
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


class PolicyWrapper: # A wrapper for Policy to handle observation and action adaptation.
    def __init__(self, cfg_policy: PolicyCfg, env_dof_cfg: DoFConfig, device: str):
        self.env_dof_cfg = env_dof_cfg

        policy_type = cfg_policy.policy_type
        policy_name = policy_type
        if hasattr(cfg_policy, "policy_name"):
            policy_name += "@" + cfg_policy.policy_name  # type: ignore
        # while policy_name in self.policies.keys():
        #     policy_name += "_new"
        self.name = policy_name

        policy_class: type[Policy] = getattr(robojudo.policy, policy_type)
        self.policy: Policy = policy_class(cfg_policy=cfg_policy, device=device)
        self.obs_adapter = DoFAdapter(env_dof_cfg.joint_names, self.policy.cfg_obs_dof.joint_names)
        self.actions_adapter = DoFAdapter(self.policy.cfg_action_dof.joint_names, env_dof_cfg.joint_names)

    def get_observation(self, env_data: Box, ctrl_data: Box):
        env_data_adapted = env_data.copy()
        env_data_adapted.dof_pos = self.obs_adapter.fit(env_data_adapted.dof_pos)
        env_data_adapted.dof_vel = self.obs_adapter.fit(env_data_adapted.dof_vel)
        return self.policy.get_observation(env_data_adapted, ctrl_data)

    def get_action(self, obs):
        action = self.policy.get_action(obs)
        return self.actions_adapter.fit(action)

    def get_pd_target(self, obs):
        action = self.policy.get_action(obs)
        pd_target = action + self.policy.default_pos
        return self.actions_adapter.fit(pd_target, template=self.env_dof_cfg.default_pos)

    def get_init_dof_pos(self):
        return self.actions_adapter.fit(self.policy.get_init_dof_pos(), template=self.env_dof_cfg.default_pos)

    def __getattr__(self, name): # Fallback: delegate other func to the wrapped policy.
        return getattr(self.policy, name)

"""
class PolicyWrapper: # Optimized PolicyWrapper for End-to-End GPU execution
    def __init__(self, cfg_policy: PolicyCfg, env_dof_cfg: DoFConfig, device: str):
        self.device = torch.device(device)
        self.env_dof_cfg = env_dof_cfg

        # 1. 初始化 Policy
        policy_type = cfg_policy.policy_type
        policy_name = policy_type
        if hasattr(cfg_policy, "policy_name"):
            policy_name += "@" + cfg_policy.policy_name
        self.name = policy_name

        policy_class: type[Policy] = getattr(robojudo.policy, policy_type)
        self.policy: Policy = policy_class(cfg_policy=cfg_policy, device=device)

        # 2. [GPU优化] 预计算索引映射
        env_names = env_dof_cfg.joint_names
        policy_obs_names = self.policy.cfg_obs_dof.joint_names
        policy_act_names = self.policy.cfg_action_dof.joint_names

        # Obs: Env -> Policy
        obs_indices = [env_names.index(name) for name in policy_obs_names]
        self.obs_dof_indices = torch.tensor(obs_indices, device=self.device, dtype=torch.long)

        # Action: Policy -> Env
        action_indices = [policy_act_names.index(name) for name in env_names]
        self.action_dof_indices = torch.tensor(action_indices, device=self.device, dtype=torch.long)

        # 3. 预加载常量
        self.policy_default_pos = torch.tensor(
            self.policy.default_pos, device=self.device, dtype=torch.float32)
        
        # [保留] 用于 get_init_dof_pos 的 CPU 适配器 (非性能关键)
        self.obs_adapter = DoFAdapter(env_names, policy_obs_names)
        self.actions_adapter = DoFAdapter(policy_act_names, env_names)

    def get_observation(self, env_data: dict, ctrl_data: dict):
        
        # # 零拷贝 GPU 处理
        # # 假设 env_data 里的 dof_pos/vel 已经是 numpy, 这里转 Tensor
        # # (如果在 Env 里已经优化成 Tensor 更好，这里为了兼容假设是 Numpy)
        # if isinstance(env_data["dof_pos"], np.ndarray):
        #     dof_pos = torch.from_numpy(env_data["dof_pos"]).to(self.device, non_blocking=True)
        #     dof_vel = torch.from_numpy(env_data["dof_vel"]).to(self.device, non_blocking=True)
        #     base_ang_vel = torch.from_numpy(env_data["base_ang_vel"]).to(self.device, non_blocking=True)
        #     base_quat = torch.from_numpy(env_data["base_quat"]).to(self.device, non_blocking=True)
        # else:
        #     # 已经是 Tensor
        #     dof_pos = env_data["dof_pos"]
        #     dof_vel = env_data["dof_vel"]
        #     base_ang_vel = env_data["base_ang_vel"]
        #     base_quat = env_data["base_quat"]

        # # Adapter (Index Select)
        # dof_pos_adapted = dof_pos[self.obs_dof_indices]
        # dof_vel_adapted = dof_vel[self.obs_dof_indices]

        # # 构造轻量级 Tensor 字典传给 Policy
        # env_data_tensor = {
        #     "dof_pos": dof_pos_adapted,
        #     "dof_vel": dof_vel_adapted,
        #     "base_ang_vel": base_ang_vel,
        #     "base_quat": base_quat,}
        
        # # [新增] 转发 BeyondMimicPolicy 需要的额外字段
        # # 自动将 numpy 转为 tensor, 如果 key 存在的话
        # extra_keys = ["base_lin_vel", "base_pos", "torso_pos", "torso_quat"]
        # for key in extra_keys:
        #     if key in env_data:
        #         val = env_data[key]
        #         if isinstance(val, np.ndarray):
        #             env_data_tensor[key] = torch.from_numpy(val).to(self.device, non_blocking=True)
        #         else:
        #             env_data_tensor[key] = val
        
        # return self.policy.get_observation(env_data_tensor, ctrl_data)
        
        return self.policy.get_observation(env_data, ctrl_data)

    def get_action(self, obs):
        # 输出纯 Tensor action
        action = self.policy.get_action(obs)

        # 如果 action 是 Tensor，转回 Numpy (为了后续兼容)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # 执行 Adapter (CPU)
        return self.actions_adapter.fit(action)
        # return action[self.action_dof_indices]

    def get_pd_target(self, obs):
        # 核心控制流：全 GPU 计算
        action = self.policy.get_action(obs)
        pd_target = action + self.policy_default_pos
        
        # 重排回 Env 顺序
        pd_target_env = pd_target[self.action_dof_indices]
        
        # 唯一的一次 GPU->CPU 回传 (给电机)
        return pd_target_env.cpu().numpy()

    def get_init_dof_pos(self):
        # 保持原样，只用于初始化
        return self.actions_adapter.fit(self.policy.get_init_dof_pos(), template=self.env_dof_cfg.default_pos)

    def __getattr__(self, name):
        return getattr(self.policy, name)
"""

@pipeline_registry.register
class RlPipeline(Pipeline):
    cfg: RlPipelineCfg

    def __init__(self, cfg: RlPipelineCfg):
        super().__init__(cfg=cfg)

        # load in environment (unitree or dummy)
        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        # load in controller (keyboard or joystick)
        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        # load in policy (for obs & action)
        self.policy = PolicyWrapper(
            cfg_policy=self.cfg.policy,
            env_dof_cfg=self.env.dof_cfg,
            device=self.device,)
        
        # load in dof_cfg & mujoco
        # (dummy & unitree visualizer=None)
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer

        # load in freq & cycle
        self.freq = self.cfg.policy.freq
        self.dt = 1.0 / self.freq

        self.self_check()
        self.reset()

    def self_check(self):
        self.env.self_check()
        for _ in range(10):
            self.step(dry_run=True)

    def reset(self):
        logger.info("Pipeline reset")
        self.timestep = 0

        self.env.reset()
        # self.env.reborn(init_qpos=[0.2, 0.2, 0.8] + [ 0.707, 0, 0, 0.707]) # FOR SIM DEBUG
        self.policy.reset()
        self.ctrl_manager.reset()

    def safety_check(self):
        if not self.do_safety_check:
            return
        gravity_ori = get_gravity_orientation(self.env.base_quat)
        angle = np.arccos(np.clip(-gravity_ori[2], -1.0, 1.0))
        if abs(angle) > 1.0:  # more than ~57 degrees
            logger.error("Robot fallen! Shutdown for safety.")
            if hasattr(self.env, "reborn"):
                self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
            else:
                self.env.shutdown()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            # match command:
                # case "[SHUTDOWN]":
            if command == "[SHUTDOWN]":
                logger.warning("Emergency shutdown!")
                self.env.shutdown()
                # case "[SIM_REBORN]":
            elif command == "[SIM_REBORN]":
                if hasattr(self.env, "reborn"):
                    logger.warning("Simulation Env reborn!")
                    self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,)

    def step(self, dry_run=False):
        policy_state = self.policy.snapshot_state() if dry_run and hasattr(self.policy, "snapshot_state") else None

        # update [dof, odo, FK, con]
        self.env.update()

        # get proprioception
        env_data = self.env.get_data()

        # get control command
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
        commands = ctrl_data.get("COMMANDS", [])
        if dry_run:
            commands = list(commands) + ["[UNILAB_FREEZE_PHASE]"]
            ctrl_data["COMMANDS"] = commands
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        # get obs for policy & ext for mujoco
        obs, extras = self.policy.get_observation(env_data, ctrl_data)

        # forward propagation for PD signal
        pd_target = self.policy.get_pd_target(obs)

        # if not dummy_env update obs info
        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))

        # output callback info to terminal
        self.post_step_callback(env_data, ctrl_data, extras, pd_target)
        if policy_state is not None:
            self.policy.restore_state(policy_state)

    def prepare(self, init_motor_angle=None):
        # get init dof pos from policy
        if init_motor_angle is not None:
            desired_motor_angle = init_motor_angle
        else:
            desired_motor_angle = self.policy.get_init_dof_pos()

        # logger.info(f"{desired_motor_angle=}")

        # self.env.dof_pos is an interface func
        current_motor_angle = np.array(self.env.dof_pos)

        # logger.info(f"{current_motor_angle=}")

        traj_len = 1000 # total 1k step
        last_step_time = time.time()
        logger.warning("prepare_init")
        pbar = ProgressBar("Prepare", traj_len)

        # iterate compute & update joint angle
        for t in range(traj_len):
            t0 = time.perf_counter()

            # get current joint angle from motor
            current_motor_angle = np.array(self.env.dof_pos)

            # 更新电机角度耗时
            t1 = time.perf_counter()

            # inside 300 step current mixup with desire angle
            # after 300 step domanite by desire joint angle
            blend_ratio = np.minimum(t / 300, 1)

            # compute mixup action with current and desire angle
            action = (1 - blend_ratio) * current_motor_angle + blend_ratio * desired_motor_angle

            t2 = time.perf_counter()

            # forward propagation but abandon action
            # fill in history buffer, warm up network
            self.step(dry_run=True)

            t3 = time.perf_counter()

            # print("\n")
            # print(f"action.device: {action.device}") # np.ndarry
            # print("\n")

            # send motor mixup action
            self.env.step(action)

            t4 = time.perf_counter()

            # compute period, sleep if too fast, error if too slow
            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error(f"Warning: frame drop: time_diff: {time_diff}") # dt = 0.02, 0.02 x 1000 = 20ms # self.freq: {self.freq}, self.dt: {self.dt}, 
            last_step_time = time.time()
            if t % 50 == 0: pbar.update(50)

            t5 = time.perf_counter()
            """
            # === [新增] 打印详细耗时分析 ===
            total_ms = (t5 - t0) * 1000
            # 只有当总耗时超过 15ms 时才打印，避免刷屏 (目标是 20ms)
            if total_ms > 15.0:
                print(
                f"rl_pipeline Total: {total_ms:.2f}ms | "   #  G1      | 5090gpu | 5090cpu
                f"ReadEnv: {(t1-t0)*1000:.2f}ms | "   # 0.03 ~ 0.07 ms | 0.01 ms | 0.01 ms
                f"infer: {(t2-t1)*1000:.2f}ms | "     # 0.07 ~ 0.11 ms | 0.02 ms | 0.02 ms
                f"envstep: {(t3-t2)*1000:.2f}ms | "   # 47.8 ~ 100 ms  | 2.30 ms | 2.30 ms
                f"sendmotor: {(t4-t3)*1000:.2f}ms | " # 3.37 ~ 6.04 ms | 0.79 ms | 0.79 ms
                f"post: {(t5-t4)*1000:.2f}ms")        # 26.6 ~ 57.3 ms | 17.1 ms | 17.1 ms
            print("\n")
            """
            # reset obs, policy, ctrl buffer at 900 steps
            # since at 900 steps robot already reach init state
            if t == 0.9 * traj_len:
                logger.info(f"{'=' * 10} RESET ZERO POSITION {'=' * 10}")
                self.reset()

        time.sleep(0.01)
        pbar.close()
        logger.warning("prepare_done")


if __name__ == "__main__":
    pass
