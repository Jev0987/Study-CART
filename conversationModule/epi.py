import random

import numpy as np
import torch
from conversationModule import env
from conversationModule import agent
from conversationModule import message
from config import global_config as cfg
from torch.autograd import Variable

def auxiliary_reward():
    """

    :return: 额外reward字典，表示每个poi通过相应的操作后应该有的reward值
    """
    import pandas as pd
    auxiliary_reward_dict = dict()
    df = pd.read_csv("../data/reward.csv")
    max_index = df.shape[0]
    for index, row in df.iterrows():
        #如果当前的poi在reward名单,+1,不在则保持默认值1
        if str(int(row['Item_id'])) not in auxiliary_reward_dict:
            auxiliary_reward_dict[str(int(row['Item_id']))] = 1
        else:
            auxiliary_reward_dict[str(int(row['Item_id']))] += 1

    for key in auxiliary_reward_dict.keys():
        auxiliary_reward_dict[key] = round(auxiliary_reward_dict[key] / max_index, 4)

    return auxiliary_reward_dict

def choose_start_facet(busi_id):
    """

    :param busi_id:
    :return: 随机选择几个feature
    """
    choose_pool = list()
    if cfg.item_dict[str(busi_id)]['starts'] is not None:
        choose_pool.append('starts')
    if cfg.item_dict[str(busi_id)]['clusters'] is not None:
        choose_pool.append('clusters')
    if cfg.item_dict[str(busi_id)]['POI_Type'] is not None:
        choose_pool.append('POI_Type')
    print('choose_pool is: {}'.format(choose_pool))

    THE_FEATURE = random.choice(choose_pool)

    return THE_FEATURE

def get_reward(history_list, gamma, trick):
    """

    :param history_list:
    :param gamma:
    :param trick:
    :return:
    """
    prev_reward = - 0.01

    # -2: reach maximum turn, end.
    # -1: recommend unsuccessful
    # 0: ask attribute, unsuccessful
    # 1: ask attribute, successful
    # 2: recommend successful!

    r_dict = {
        2: 1 + prev_reward,
        1: 0.1 + prev_reward,
        0: 0 + prev_reward,
        -1: 0 - 0.1,
        -2: -0.3
    }

    reward_list = [r_dict[item] for item in history_list]
    print('gamma: {}'.format(gamma))

    rewards = []
    R = 0
    for r in reward_list[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)

    if trick == 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    return rewards

''' recommend procedure(推荐过程)'''
def run_one_episode(transE_model, user_id, busi_id, MAX_TURN, do_random, write_fp, strategy, TopKTaxo,
                    PN_model, gamma, trick, mini, optimizer1_transE, optimizer2_transE, alwaysupdate, start_facet, mask, sample_dict, choose_pool,features, items):
    success = None

    the_user = env.user(user_id, busi_id)

    numpy_list = list()
    log_prob_list, reward_list = Variable(torch.Tensor()), list()
    action_tracker, candidate_length_tracker = list(), list()

    the_agent = agent.agent(transE_model, user_id, busi_id, do_random, write_fp, strategy, TopKTaxo, numpy_list, PN_model, log_prob_list, action_tracker, candidate_length_tracker, mini, optimizer1_transE, optimizer2_transE, alwaysupdate, mask, sample_dict, choose_pool, features, items)

    data = dict()
    data['facet'] = start_facet
    start_signal = message(cfg.AGENT, cfg.USER, cfg.EPISODE_START, data)