import gym
import gym.wrappers
import time

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'tmp/cartpole-experiment-1', force=True)

for i_episode in range(200):

    observation = env.reset()

    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            time.sleep(1)
            break
