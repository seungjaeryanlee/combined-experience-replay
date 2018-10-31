import gym

class TimeoutWrapper(gym.Wrapper):
    def __init__(self, env, max_steps):
        gym.Wrapper.__init__(self, env)
        self.step_count = 0
        self.max_steps = max_steps
    
    def step(self, action):
        self.step_count += 1
        obs, rew, done, info = self.env.step(action)
        if not done and self.step_count >= self.max_steps:
            done = True
            info['timeout'] = True
        else:
            info['timeout'] = False

        return obs, rew, done, info

def get_lunar_lander(max_steps=None, seed=None):
    env = gym.make('LunarLander-v2')
    if max_steps:
        env = TimeoutWrapper(env, max_steps)
    if seed:
        env.seed(seed)

    return env
