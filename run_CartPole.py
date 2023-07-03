"""
gym: 0.26.2
gym-notices 0.0.8
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

RENDER = False  # 在屏幕上显示模拟窗口会拖慢运行速度 我们等计算机学得差不多了再显示模拟
DISPLAY_REWARD_THRESHOLD = 10000  # 当回合总reward大于400时显示模拟窗口

env = gym.make('CartPole-v1')
# env = gym.make('CartPole-v1', render_mode='human')  # CartPole模拟
env.reset(seed=1)  # 普通的Policy Gradient方法，使得回合的variance比较大，所以我们选了
env = env.unwrapped  # 取消限制


print(env.action_space)  # 显示可用action
print(env.observation_space)  # 显示可用state的observation
print(env.observation_space.high)  # 显示observation最高值
print(env.observation_space.low)  # 显示observation最低值


# 定义
RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    n_neuron=20,
    learning_rate=0.02,
    reward_decay=0.99,  # gamma
)

for i_episode in range(424):

    observation, info = env.reset()

    while True:
        if RENDER:
            env_test = gym.make('CartPole-v1', render_mode='human')
            env_test.reset()

            # env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, truncated, info = env.step(action)

        RL.store_transition(observation, action, reward)  # 存储这一回合的transition

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True  # 判断是否显示

            print('episode:', i_episode, ' reward:', int(running_reward))

            vt = RL.learn()  # 学习，输出vt，我们下节课讲这个vt的作用

            if i_episode == 423:
                plt.plot(vt)  # plot这个回合的vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
