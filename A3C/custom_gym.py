import gym
import numpy as np
import random
import cv2


class CustomGym:
    def __init__(self, skip_actions=4, num_frames=4, w=84, h=84):
        self.game_name = "SpaceInvaders-v4"
        self.env = gym.make(self.game_name)
        self.num_frames = num_frames
        self.skip_actions = skip_actions
        self.w = w
        self.h = h

        self.action_space = range(self.env.action_space.n)

        self.action_size = len(self.action_space)
        self.observation_shape = self.env.observation_space.shape

        self.state = None

    def preprocess(self, obs, is_start=False):
        grayscale = obs.astype('float32').mean(2)
        s = cv2.resize(grayscale, (self.w, self.h)).astype('float32') * (1.0/255.0)
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        if is_start or self.state is None:
            self.state = np.repeat(s, self.num_frames, axis=3)
        else:
            self.state = np.append(s, self.state[:,:,:,:self.num_frames-1], axis=3)
        return self.state

    def render(self):
        self.env.render()

    def reset(self):
        return self.preprocess(self.env.reset(), is_start=True)

    def step(self, action_idx):
        action = self.action_space[action_idx]
        accum_reward = 0
        prev_s = None
        for _ in range(self.skip_actions):
            s, r, term, info = self.env.step(action)
            accum_reward += r
            if term:
                break
            prev_s = s
        # Takes maximum value for each pixel value over the current and previous
        # frame. Used to get round Atari sprites flickering (Mnih et al. (2015))
        if prev_s is not None:
            s = np.maximum.reduce([s, prev_s])
        return self.preprocess(s), accum_reward, term, info
