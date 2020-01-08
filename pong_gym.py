import time
import gym

from agents.ebner2019_ach_da_agent import Ebner2019AChDaAgent

with gym.make('Pong-v0') as env:
    env.reset()

    agent = Ebner2019AChDaAgent(observation_space=env.observation_space,
                                action_space=env.action_space,
                                reward_space=env.reward_range)

    print('action_space', env.action_space)
    print('action_space', env.observation_space)
    print('reward_range', env.reward_range)

    for _ in range(1000):
        env.render(mode="human")
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        time.sleep(0.01)
        #action = agent.step(observation=observation, reward=reward)