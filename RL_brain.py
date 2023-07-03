import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.random.seed(1)
torch.manual_seed(1)


class NetWork(nn.Module):
    """
    神经网络结构
    # 全连接1
    # 全连接2
    # ReLU
    """
    def __init__(self,
                 n_actions,
                 n_features,
                 n_neuron=10):
        super(NetWork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_features,
                      out_features=n_neuron,
                      bias=True),
            nn.Linear(in_features=n_neuron,
                      out_features=n_actions,
                      bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class PolicyGradient:
    """
    PolicyGradient算法
    """
    # 初始化
    def __init__(self,
                 n_actions,
                 n_features,
                 n_neuron=10,
                 learning_rate=0.01,
                 reward_decay=0.95):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_neuron = n_neuron
        self.lr = learning_rate
        self.gamma = reward_decay

        # 之前Q-learning算法定义一个memory共同存储observation action reward
        # 这里定义三个memory分别存储observation action reward
        # self.ep_obs存储observation
        # self.ep_as存储action
        # self.ep_rs存储reward
        # learn网络的时候用memory里边全部内容学习，没有batch_size学习，相对简单些
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.net = NetWork(n_actions=self.n_actions,
                           n_features=self.n_features,
                           n_neuron=self.n_neuron)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(),
                                      lr=self.lr)

    # 选行为（和Q-learning算法相比有改变）
    def choose_action(self, observation):
        s = torch.FloatTensor(observation)
        out = self.net(s)  # 给net一个输入
        prob_weights = F.softmax(out, dim=0)
        prob_weights = prob_weights.detach().numpy()
        # 根据概率抽样得到一个action
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)
        return action

    # 存储回合 transition（和Q-learning算法相比有改变）
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)  # 这句是什么意思
        self.ep_rs.append(r)

    # 学习更新参数（和Q-learning算法相比有改变）
    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # 转换成torch.tensor数据类型
        s = torch.FloatTensor(np.vstack(self.ep_obs))
        action = torch.LongTensor(np.stack(self.ep_as))

        discounted_ep_rs_norm = torch.FloatTensor(discounted_ep_rs_norm)

        # net输出
        out = self.net(s)

        # train on episode
        # loss = nn.CrossEntropyLoss(reduction='none')(out, action) 没有带weight
        neg_log_prob = nn.CrossEntropyLoss(reduction='none')(out, action)
        loss = torch.mean(neg_log_prob * discounted_ep_rs_norm)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # empty episode data
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm.detach().numpy()

    # 衰减回合的reward（和Q-learning算法相比有新内容）
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs