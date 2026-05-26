#!/usr/bin/env python

from gym.envs.registration import register



register(
    id='ContinuousVecGridEnv-v0',
    entry_point='envs.vectorized_grid:ContinuousGridEnv',
)

register(
    id='GoalGrid-v0',
    entry_point='envs.goal_grid:GoalContinuousGrid',
)

register(
    id='ReacherDraw-v0',
    entry_point='envs.reacher_trace:ReacherTraceEnv',
)

register(
    id='HopperFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Hopper-v2'
    )
)

register(
    id='Walker2dFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Walker2d-v2'
    )
)

register(
    id='HalfCheetahFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='HalfCheetah-v2'
    )
)

register(
    id='AntFH-v0',
    entry_point='envs.mujocoFH:MujocoFH',
    kwargs=dict(
        env_name='Ant-v2'
    )
)


register(id='PointMazeRight-v0', entry_point='envs.point_maze_env:PointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 1})
register(id='PointMazeLeft-v0', entry_point='envs.point_maze_env:PointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 0})

# A modified ant which flips over less and learns faster via TRPO
register(id='CustomAnt-v0', entry_point='envs.ant_env:CustomAntEnv',
         kwargs={'gear': 30, 'disabled': False})
register(id='DisabledAnt-v0', entry_point='envs.ant_env:CustomAntEnv',
         kwargs={'gear': 30, 'disabled': True})


# ---------------------------------------------------------------------------
# CartPole adapted for SAC-based (continuous-action) IL
# ---------------------------------------------------------------------------
# SOAR-IL's SAC backbone needs a continuous Box action space. CartPole-v1 has
# Discrete(2). We wrap it: continuous action in [-1, 1] is binarised by sign
# before being passed to the underlying env. The physics underneath is
# unchanged; this is a standard adapter used to compare SAC vs. discrete IL.
import gymnasium as _gymnasium
from gymnasium import spaces as _gymnasium_spaces
import numpy as _np


class CartPoleContinuousWrapper(_gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = _gymnasium_spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=_np.float32
        )

    def step(self, action):
        a = int(_np.asarray(action).reshape(-1)[0] > 0)  # +ve → right (1), -ve → left (0)
        return self.env.step(a)


def make_cartpole_continuous():
    base = _gymnasium.make("CartPole-v1")
    return CartPoleContinuousWrapper(base)


_gymnasium.envs.registration.register(
    id="CartPoleContinuous-v0",
    entry_point="envs:make_cartpole_continuous",
    max_episode_steps=500,
)
