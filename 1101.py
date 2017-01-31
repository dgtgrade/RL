# multithread gym test

import gym
import gym.wrappers
import time
import threading


def worker(no):
    while True:
        print('worker{}'.format(no))
        time.sleep(1)


def gym_worker(no):

    print("gym-worker-{} is staring".format(no))

    env = gym.make('Breakout-v0')

    while True:

        observation = env.reset()

        while True:
            # TODO: XIO error when try to render 5 gyms
            # env.render()
            time.sleep(1)
            action = env.action_space.sample()
            observation, t_reward, done, info = env.step(action)
            print(info)
            if done:
                break

threads = []

for i in range(5):
    t = threading.Thread(target=gym_worker, args=(i, ))
    threads.append(t)
    t.start()
