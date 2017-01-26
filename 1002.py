# Follows a random search idea from:
# http://kvfrans.com/simple-algoritms-for-solving-cartpole/

# import gym
import gym.wrappers
import numpy as np
import time

#
float_formatter = lambda x: "%.6f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

#
# https://github.com/openai/gym/wiki/CartPole-v0
#
env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'tmp/cartpole-experiment-1', force=True)

#
EP_MAX = 1000
T_MAX = 200
#
# Solved Requirements:
# Considered solved when the average reward is
# greater than or equal to 195.0 over 100 consecutive trials.
CONSECUTIVE_EPS_TO_SOLVE = 100
T_TO_SOLVE = 195

#
t_best = 0
params_best = np.zeros(4)
consecutive_success = 0

for ep in range(EP_MAX):

    # noinspection PyRedeclaration
    observation = env.reset()

    if consecutive_success == 0:
        params = np.random.random(4) * 2 - 1

    print("ep: {}".format(ep))
    print("params: {}".format(params))

    t = 0

    while True:

        t += 1

        env.render()

        # print("observation: {}".format(observation))
        action = 1 if np.dot(observation, params) > 0 else 0

        #
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t))
            time.sleep(1)
            break

    if t > t_best:
        print("Good! Episode ran more than previous best ({}) timesteps".format(
            t_best))
        t_best = t

    if t > T_TO_SOLVE:
        consecutive_success += 1
    else:
        consecutive_success = 0

    print("Consecutive Successes: {}".format(consecutive_success))

    if consecutive_success == CONSECUTIVE_EPS_TO_SOLVE:
        print("WOW! {} consecutive success! problem solved".format(
            consecutive_success))
        break
