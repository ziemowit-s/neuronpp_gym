import time
import gym

with gym.make('Pong-v0') as env:
    env.reset()

    print('action_space', env.action_space)
    print('action_space', env.observation_space)
    print('reward_range', env.reward_range)

    for _ in range(1000):
        env.render(mode="human")
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        time.sleep(0.01)
