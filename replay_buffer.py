import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr = 0
        self.size = 0
        self.size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size
        self.size = min(self.size + 1, self.size)
    
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {'s': self.obs1_buf[idxs],
                's2': self.obs2_buf[idxs],
                'a': self.acts_buf[idxs],
                'r': self.rews_buf[idxs],
                'd': self.done_buf[idxs]}
        