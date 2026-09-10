from __future__ import annotations
import logging
from collections.abc import Callable
from enum import Enum, auto

import numpy as np

import robojudo.environment
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlLocoMimicPipelineCfg
from robojudo.pipeline.rl_multi_policy_pipeline import PolicyManager, RlMultiPolicyPipeline
from robojudo.pipeline.rl_pipeline import PolicyWrapper
from robojudo.policy import PolicyCfg
from robojudo.utils.progress import ProgressBar

logger = logging.getLogger(__name__)

# policy interpolate manager (handcraft transition)
class PolicyInterpManager(PolicyManager):
    # define states of FSM
    class InterpState(Enum):
        IDLE = auto()
        START = auto()
        IN_PROGRESS = auto()
        END = auto()

    # interpolate frames: loco -> mimic
    DURATIONS_LOCO_MIMIC = [0, 75, 25]  # [start, in-progress, end] in steps

    # interpolate frames: mimic -> loco
    DURATIONS_MIMIC_LOCO = [25, 75, 0]  # [start, in-progress, end] in steps

    def __init__(self, cfg_policy_loco: PolicyCfg, cfg_policies: list[PolicyCfg], env: Environment, 
                 loco_dof_pos: np.ndarray | None = None, device: str = "cpu"):
        # set loco policy as 0th
        cfg_policies_all = [cfg_policy_loco] + cfg_policies
        super().__init__(cfg_policies_all, env, device)

        # set current 0th to loco
        self.policy_loco_id = 0

        self.policy_mimic_num = len(cfg_policies)
        assert self.policy_mimic_num > 0, "At least one mimic policy is required for switching."
        self.policy_mimic_ids = list(range(1, self.policy_mimic_num + 1))
        self.policy_mimic_idx = 0 # init chosen mimic policy index (from the top)

        # Interpolation variables
        self.interp_state = self.InterpState.IDLE
        self.interp_timestep = 0
        self.interp_durations = [20, 40, 20]  # [start, in-progress, end] in steps
        self.interp_pbar = None
        self.interp_callback_start = None
        self.interp_callback_end = None

        # buffer loco policy joint angle after mimic
        self.loco_dof_pos = loco_dof_pos if loco_dof_pos is not None else self.env.default_pos.copy()
        self.override_dof_pos = self.loco_dof_pos.copy()

    def _interpolate_init(self, get_target_pos: Callable[[], np.ndarray], durations: list[int],
                          callback_start=None, callback_end=None):
        self.interp_get_target_pos = get_target_pos # the end joint angle of transition
        self.interp_durations = durations # transition steps
        self.interp_callback_start = callback_start
        self.interp_callback_end = callback_end
        self.interp_pbar = ProgressBar("Interpolation", durations[1])

        self.interp_state = self.InterpState.START

        # Starting Tasks: after durations[0] step execute start
        self.timer.add(self._interpolate_start, delay_steps=durations[0])

        # Ending Tasks: after durations[3] step execute end
        self.timer.add(self._interpolate_end, delay_steps=sum(durations) + 1)

    def _interpolate_start(self):
        if self.interp_state != self.InterpState.START:
            return
        
        # output interp start callback to terminal
        if self.interp_callback_start is not None:
            self.interp_callback_start()
            self.interp_callback_start = None

        # buffer start pos: current joint angle
        self.interp_start_pos = self.env.dof_pos.copy()

        # buffer end pos: target joint angle
        self.interp_target_pos = self.interp_get_target_pos()

        # set up time step
        self.interp_timestep = 0

        self.interp_state = self.InterpState.IN_PROGRESS

        # logger.debug("Interpolation started.")

    def _interpolate_end(self):
        if self.interp_state != self.InterpState.END:
            return
        
        # force target pos as the interp end pos
        self.override_dof_pos = self.interp_target_pos.copy()
        if self.interp_pbar:
            self.interp_pbar.close()
            self.interp_pbar = None
        
        # output interp end callback to terminal
        if self.interp_callback_end is not None:
            self.interp_callback_end()
            self.interp_callback_end = None
        
        self.interp_state = self.InterpState.IDLE

        # logger.debug("Interpolation ended.")

    def _interpolate_step(self):
        if self.interp_state != self.InterpState.IN_PROGRESS:
            return

        if self.interp_pbar:
            self.interp_pbar.set(self.interp_timestep)

        # compute progress (0 -> 1)
        progress = self.interp_timestep / self.interp_durations[1]

        alpha = min(progress, 1.0) # compute current mixup pos with progress & mixup coef
        self.override_dof_pos = (1 - alpha) * self.interp_start_pos + alpha * self.interp_target_pos

        # update interp time step
        if self.interp_timestep < self.interp_durations[1]:
            self.interp_timestep += 1
        else:
            self.interp_state = self.InterpState.END

    def toggle_mimic_policy(self, delta: int): # loco -> mimic
        # if current policy isn't loco then directly return
        # current policy must be loco, then switch to mimic
        if self.current_policy_id != self.policy_loco_id:
            logger.warning("Cannot switch mimic policy when policy is mimic.")
            return

        self.policy_mimic_idx = (self.policy_mimic_idx + delta) % self.policy_mimic_num
        policy_id = self.policy_mimic_ids[self.policy_mimic_idx]
        policy_name = self.policy_by_id(policy_id).name
        logger.info(f"Switch mimic policy to {self.policy_mimic_idx}: {policy_name}")

    def switch_to_loco(self): # mimic -> loco
        # if current policy is loco then directly return
        # current policy must be mimic, then switch to loco
        if self.current_policy_id == self.policy_loco_id and self.interp_state == self.InterpState.IDLE:
            logger.warning("Already in locomotion policy.")
            return

        if self.current_policy_id != self.policy_loco_id:
            self.policy_by_id(self.policy_loco_id).reset()
            self.warmup_policy_indices.add(self.policy_loco_id)
        
        self._interpolate_init(
            get_target_pos=lambda: self.loco_dof_pos,
            durations=self.DURATIONS_MIMIC_LOCO,
            callback_start=lambda: self.set_policy(self.policy_loco_id),)

    def switch_to_mimic(self): # loco -> mimic
        # if current policy isn't loco then directly return
        # current policy must be loco, then switch to mimic
        if self.current_policy_id != self.policy_loco_id:
            logger.warning("Already in mimic policy.")
            return

        policy_mimic_id = self.policy_mimic_ids[self.policy_mimic_idx]
        self.policy_by_id(policy_mimic_id).reset()
        self.warmup_policy_indices.add(policy_mimic_id)

        self._interpolate_init(
            get_target_pos=lambda: self.policy_by_id(policy_mimic_id).get_init_dof_pos(),
            durations=self.DURATIONS_LOCO_MIMIC,
            callback_end=lambda: self.set_policy(policy_mimic_id),)

    def step(self, env_data, ctrl_data):
        super().step(env_data, ctrl_data)
        self._interpolate_step()


@pipeline_registry.register
class RlLocoMimicPipeline(RlMultiPolicyPipeline):
    cfg: RlLocoMimicPipelineCfg

    @property
    def policy(self) -> PolicyWrapper:
        return self.policy_manager.policy

    def __init__(self, cfg: RlLocoMimicPipelineCfg):
        # Skip RlMultiPolicyPipeline initialization
        Pipeline.__init__(self, cfg=cfg)

        # load in environment (unitree or dummy)
        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        # print("\n")
        # print(f"RlLocoMimicPipeline: {self.device}")
        # print("\n")
        # print(f"self.cfg.env: {self.cfg.env}")
        # print("\n")
        # print(f"self.env: {self.env}")
        # print("\n")
        # temp = 1
        # assert temp == 2

        # load in controller (keyboard or joystick)
        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        # print("\n")
        # print(f"self.cfg.ctrl: {self.cfg.ctrl}")
        # print("\n")
        # print(f"self.ctrl_manager: {self.ctrl_manager}")
        # print("\n")
        # print(f"self.device: {self.device}")
        # print("\n")
        # temp = 1
        # assert temp == 2

        # upper body override
        self.num_upper_body_dof = self.cfg.upper_dof_num
        if upper_dof_pos_default := self.cfg.upper_dof_pos_default:
            loco_dof_pos = self.env.default_pos.copy()
            loco_dof_pos[-self.num_upper_body_dof :] = upper_dof_pos_default
            self.loco_dof_pos = loco_dof_pos
        else:
            self.loco_dof_pos = self.env.default_pos
        if override_dof_indices := self.cfg.upper_dof_override_indices:
            self.override_dof_indices = override_dof_indices
        else:
            self.override_dof_indices = list(range(-self.num_upper_body_dof, 0))

        # load in policy (for obs & action)
        self.policy_manager = PolicyInterpManager(
            cfg_policy_loco=self.cfg.loco_policy,
            cfg_policies=self.cfg.mimic_policies,
            env=self.env,
            loco_dof_pos=self.loco_dof_pos,
            device=self.device,)
        
        # print("\n")
        # print(f"self.policy: {self.policy}")
        # print("\n")
        # temp = 1
        # assert temp == 2
        
        # load in dof_cfg & mujoco
        # (dummy & unitree visualizer=None)
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer

        # load in freq & cycle
        self.freq = self.cfg.loco_policy.freq
        self.dt = 1.0 / self.freq

        self.policy_locomotion_mimic_flag = 0 # 0: locomotion, 1: mimic

        self.self_check()
        self.reset()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1

        commands = ctrl_data.get("COMMANDS", [])

        # Handle policy CALLBACK
        for callback in extras.get("CALLBACK", []):
            # match callback:
                # case "[MOTION_DONE]":
            if commands == "[MOTION_DONE]":
                if self.policy_locomotion_mimic_flag == 1:
                    commands.append("[POLICY_LOCO]")
                    logger.info("Mimic motion done, switch to locomotion policy.")

        for command in commands:
            """
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Emergency shutdown!")
                    self.env.shutdown()
                case "[SIM_REBORN]":
                    if hasattr(self.env, "reborn"):
                        logger.warning("Simulation Env reborn!")
                        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                case cmd if cmd.startswith("[POLICY_SWITCH]"):
                    switch_target = cmd.split(",")[1]
                    if switch_target == "NEXT":
                        self.policy_manager.toggle_mimic_policy(1)
                    elif switch_target == "LAST":
                        self.policy_manager.toggle_mimic_policy(-1)
                case "[POLICY_LOCO]":
                    self.policy_locomotion_mimic_flag = 0
                    self.policy_manager.switch_to_loco()
                case "[POLICY_MIMIC]":
                    self.policy_locomotion_mimic_flag = 1
                    self.policy_manager.switch_to_mimic()
            """

            if command == "[SHUTDOWN]":
                logger.warning("Emergency shutdown!")
                self.env.shutdown()
            elif command == "[SIM_REBORN]":
                if hasattr(self.env, "reborn"):
                    logger.warning("Simulation Env reborn!")
                    self.env.reborn()
            elif command.startswith("[POLICY_SWITCH]"):
                switch_target = command.split(",")[1]
                if switch_target == "NEXT":
                    self.policy_manager.toggle_mimic_policy(1)
                elif switch_target == "LAST":
                    self.policy_manager.toggle_mimic_policy(-1)
            elif command == "[POLICY_LOCO]":
                self.policy_locomotion_mimic_flag = 0
                self.policy_manager.switch_to_loco()
            elif command == "[POLICY_MIMIC]":
                self.policy_locomotion_mimic_flag = 1
                self.policy_manager.switch_to_mimic()

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        # # Handle policy switch after step to avoid mid-step change
        self.policy_manager.step(env_data, ctrl_data)

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,)

    def step(self, dry_run=False):
        import time
        t0 = time.perf_counter()

        # update [dof, odo, FK, con]
        self.env.update()

        # 读取硬件状态耗时
        t1 = time.perf_counter()

        # get proprioception
        env_data = self.env.get_data()

        # get control command
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        # 读取手柄/键盘输入耗时
        t2 = time.perf_counter()

        # if current policy is loco
        if self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id:
            ctrl_data["ref_dof_pos"] = self.policy.obs_adapter.fit(self.policy_manager.override_dof_pos)

        # get obs for policy & ext for mujoco
        obs, extras = self.policy.get_observation(env_data, ctrl_data)

        # 构建观测量耗时
        t3 = time.perf_counter()

        # forward propagation for PD signal
        pd_target = self.policy.get_pd_target(obs)

        # 推理耗时
        t4 = time.perf_counter()

        # if current policy is loco
        if self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id:
            pd_target[self.override_dof_indices] = self.policy_manager.override_dof_pos[self.override_dof_indices]

        # if not dummy_env update obs info
        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))
            # logger.debug(pd_target)
        
        # 发送给电机耗时
        t5 = time.perf_counter()

        # output callback info to terminal
        self.post_step_callback(env_data, ctrl_data, extras, pd_target)

        # 打印到控制台耗时
        t6 = time.perf_counter()

        """
        # === [新增] 打印详细耗时分析 ===
        total_ms = (t6 - t0) * 1000
        # 只有当总耗时超过 15ms 时才打印，避免刷屏 (目标是 20ms)
        if total_ms > 15.0:
            print(f"rl_loco_mimic Total: {total_ms:.2f}ms | " # 60 ~ 106 ms    | 33 ~ 70 ms
                  f"ReadEnv: {(t1-t0)*1000:.2f}ms | "         # 4 ~ 57 ms      | 7 ~ 42 ms
                  f"GetCtrl: {(t2-t1)*1000:.2f}ms | "         # 5 ~ 11 ms      | 2 ~ 8 ms
                  f"MakeObs: {(t3-t2)*1000:.2f}ms | "         # 11 ~ 32 ms     | 
                  f"Infer: {(t4-t3)*1000:.2f}ms | "           # 16 ~ 28 ms     | 
                  f"WriteEnv: {(t5-t4)*1000:.2f}ms | "        # 0.02 ~ 0.05 ms | 
                  f"Post: {(t6-t5)*1000:.2f}ms")              # 0.22 ~ 0.35 ms | 
        """

    # invoke prepare() from RlPipeline
    def prepare(self):
        init_motor_angle = self.loco_dof_pos.copy()
        super().prepare(init_motor_angle=init_motor_angle)


if __name__ == "__main__":
    pass
