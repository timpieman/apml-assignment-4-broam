from typing import List, Tuple

from mazelab import VonNeumannMotion
import numpy as np
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color
import gymnasium as gym
from gym.spaces import Box
from gym.spaces import Discrete
from mazelab.generators import random_maze, morris_water_maze
from PIL import Image as PImage
from gym.utils import seeding
import random


class Maze(BaseMaze):
    def __init__(self, x):
        self.x = x
        super().__init__()

    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis=1))
        goal = Object('goal', 3, color.goal, False, [])
        agent = Object('agent', 2, color.agent, False, [])
        return free, obstacle, agent, goal


class TaskEnv(gym.Env):
    env_id = 'RandomMaze-v0'
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 3}
    reward_range = (-float('inf'), float('inf'))

    def __init__(self,
                 size: int = 15,
                 time_out: int = 100,
                 timeout_reward=-1,
                 goal_reward=1,
                 invalid_reward=-1,
                 time_reward_multiplicator=.01):
        """Contructor for the TaskEnvironment

        Args:
            size (int, optional): The size of the maze. Defaults to 15.
            time_out (int, optional): Time to explore the maze before the game is over. Defaults to 100.
        """
        super().__init__()
        self.start_idx = [[1, 1]]
        self.goal_idx = [[int(size * .75), int(size * .75)]]
        self.maze = Maze(random_maze(width=size, height=size, complexity=.7, density=.9))
        self.motions = VonNeumannMotion()
        self.viewer = None
        self.time_out = time_out
        self.timer = 0
        self.timeout_reward = timeout_reward
        self.goal_reward = goal_reward
        self.invalid_reward = invalid_reward
        self.time_reward_multiplicator = time_reward_multiplicator
        self.seed()
        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        self.episode_actions = []

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, object]:
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        self.episode_actions.append((action, "VALID" if valid else "INVALID"))
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_timeout():
            reward = self.timeout_reward
            done = True
        elif self._is_goal(new_position):
            reward = self.goal_reward
            done = True
        elif not valid:
            reward = self.invalid_reward
            done = False
        else:
            reward = -(self.timer * self.time_reward_multiplicator)
            done = False
        self.timer += 1
        return self.maze.objects.agent.positions[0], reward, done, {}

    def reset(self) -> Tuple[int, int]:
        """Resets the environment. The agent will be transferred to a random location on the map. The goal stays the same and the timer is set to 0.

        Returns:
            Tuple[int, int]: The initial position of the agent.
        """
        self.maze.objects.goal.positions = self.goal_idx
        free = self.maze.objects.free.positions
        self.timer = 0
        gx, gy = self.goal_idx[0]
        dist = 1
        is_valid = False
        start_tmp = [1, 1]
        while not is_valid:
            start_tmp = random.sample(list(free), 1)[0]
            left_boundary, right_boundary = gx - dist, gx + dist
            upper_boundary, lower_boundary = gy - dist, gy + dist
            v_cond = start_tmp[0] not in range(left_boundary, right_boundary + 1)
            h_cond = start_tmp[1] not in range(upper_boundary, lower_boundary + 1)
            is_valid = v_cond or h_cond

        self.maze.objects.agent.positions = [start_tmp]
        return self.maze.objects.agent.positions[0]

    def _is_valid(self, position: np.ndarray) -> bool:
        """Checks if the position belongs to a wall or other disturbance

        Args:
            position (np.ndarray): The position whose validity to check

        Returns:
            bool: Validity is True for positions that are free and False for impassable positions
        """
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position: np.ndarray) -> bool:
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def _is_timeout(self) -> bool:
        """Checks whether the environment has reached its timeout.

        Returns:
            bool: True for timeout is exceeded and false if not.
        """
        return self.timer >= self.time_out

    def get_image(self) -> np.ndarray:
        """Helper for render function that returns an image of the current environment.

        Returns:
            np.ndarray: An array with the shape [height, width, rgb] image with values from 0 to 255
        """
        return self.maze.to_rgb()

    def seed(self, seed: int = None) -> List[int]:
        """Ensures reproductability

        Args:
            seed (int, optional): A seed number. Defaults to None.

        Returns:
            List[int]: The seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        """Renders the environment and returns either the img array or starts a live viewer.

        Args:
            mode (str, optional): Either "rgb_array" or "human". Defaults to 'rgb_array'.

        Returns:
            np.ndarray: The image
        """
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8)
        img_height, img_width = img.shape[:2]
        ratio = self.observation_space.shape[0] / img_width
        img = PImage.fromarray(img).resize([int(ratio * img_width), int(ratio * img_height)])
        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# def register_environment(cls):
#     env_dict = gym.envs.registration.registry.env_specs.copy()
#     for env in env_dict:
#         if cls.env_id in env:
#             print("Remove {} from registry".format(env))
#             del gym.envs.registration.registry.env_specs[env]
#     gym.envs.register(id=cls.env_id, entry_point=cls, max_episode_steps=200)

# register_environment(TaskEnv)
