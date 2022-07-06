import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from collections import defaultdict
from config import global_config as cfg
import random

from util_fea_sim import feature_distance
from util_sense import rank_items

random.seed(0)

def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var

class agent():
    def __init__(self, transE_model, user_id, busi_id, do_random, write_fp, strategy, TopKTaxo, numpy_list, PN_model, log_prob_list, action_tracker, candidate_length_tracker, mini, optimizer1_fm, optimizer2_fm, alwaysupdate, do_mask, sample_dict, choose_pool, features, items):
        # _______ input parameters_______
        self.user_id = user_id
        self.busi_id = busi_id
        self.transE_model = transE_model

        self.turn_count = 0
        self.F_dict = defaultdict(lambda : defaultdict())
        self.recent_candidate_list = [int(k) for k, v in cfg.item_dict.item()]
        self.recent_candidate_list_ranked = self.recent_candidate_list

        self.asked_feature = list() #record asked facets
        self.do_random = do_random
        self.rejected_item_list_ = list()

        self.history_list = list()

        self.write_fp = write_fp
        self.strategy = strategy
        self.TopKTaxo = TopKTaxo
        self.entropy_dict_10 = None
        self.entropy_dict_50 = None
        self.entropy_dict = None
        self.distance_dict = None
        self.distance_dict2 = None
        self.PN_model = PN_model

        self.known_feature = list() # category id list
        self.known_facet = list()

        self.residual_feature_big = None
        self.change = None
        self.skip_big_feature = list()
        self.numpy_list = numpy_list

        self.log_prob_list = log_prob_list
        self.action_tracker = action_tracker
        self.candidate_length_tracker = candidate_length_tracker
        self.mini_update_already = False
        self.mini = mini
        self.optimizer1_fm = optimizer1_fm
        self.optimizer2_fm = optimizer2_fm
        self.alwaysupdate = alwaysupdate
        self.previous_dict = None
        self.rejected_time = 0
        self.do_mask = do_mask
        self.big_feature_length = 11
        self.feature_length = 289
        self.sample_dict = sample_dict
        self.choose_pool = choose_pool

        self.features = features
        self.items = items
        self.known_feature_category = []
        self.known_feature_cluster =[]
        self.known_feature_type =[]
        self.known_feature_total =[]

    def get_batch_data(self, pos_neg_pairs, bs, iter_):
        PAD_IDX1 = len(cfg.user_list) + len(cfg.item_dict)
        PAD_IDX2 = cfg.feature_count

        left = iter_ * bs
        right = min((iter_ + 1) * bs, len(pos_neg_pairs))
        pos_list, pos_list2, neg_list, neg_list2 = list(), list(), list(), list()
        for instance in pos_neg_pairs[left: right]:
            pos_list.append(torch.LongTensor([self.user_id, instance[0] + len(cfg.user_list)]))
            neg_list.append(torch.LongTensor([self.user_id, instance[1] + len(cfg.user_list)]))
        preference_list = torch.LongTensor(self.known_feature).expand(len(pos_list), len(self.known_feature))

        pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
        pos_list2 = preference_list

        neg_list = pad_sequence(neg_list, batch_first=True, padding_value=PAD_IDX1)
        neg_list2 = preference_list

        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2)

    def mini_update_transE(self):
        device = torch.device('cuda')
        self.transE_model.to(device)
        self.transE_model.train()
        optimizer = optim.SGD(self.transE_model.parameters(),lr=0.01)

        items = self.items
        features = self.features
        item_features_list = features.strip().split(' ')# strip()移除字符串头尾指定的字符（默认为空格或换行符）或字符序列

        target_time, target_category, target_cluster, target_poi_type = item_features_list[-1].split(',')#目标 item的features

        userID = torch.LongTensor([self.user_id])
        userID = userID.to(device)
        target_time = torch.LongTensor([int(target_time)])
        target_time = target_time.to(device)

        np_array = np.zeros([1, 50], dtype=np.int)
        asked_cluster = None
        asked_poi_type = None
        feature_list = []

        if len(self.known_feature_cluster) > 0:
            asked_cluster = self.known_feature_cluster[0]
        if len(self.known_feature_type) > 0:
            asked_poi_type = self.known_feature_type[0]
        if len(self.known_feature_category) > 0:
            feature_list = self.known_feature_category

        if asked_cluster != None:
            asked_cluster = torch.LongTensor([asked_cluster])
            asked_cluster = asked_cluster.to(device)
            asked_cluster_embedding = self.transE_model.relations_emb_clusters(asked_cluster)
        else:
            asked_cluster = torch.LongTensor([0])
            asked_cluster = asked_cluster.to(device)
            asked_cluster_embedding = torch.LongTensor(np_array).to(device)

        if asked_poi_type != None:
            asked_poi_type = torch.LongTensor([asked_poi_type])
            asked_poi_type = asked_poi_type.to(device)
            asked_poi_type_embedding = self.transE_model.relations_emb_poi_type(asked_poi_type)
        else:
            asked_poi_type = torch.LongTensor([0])
            asked_poi_type = asked_poi_type.to(device)
            asked_poi_type_embedding = torch.LongTensor(np_array).to(device)

        item_features_list = features.strip().split(' ')
        target_time, target_category, target_cluster, target_poi_type = item_features_list[-1].split(
            ',')  # target item features
        target_time = torch.LongTensor([int(target_time)])
        target_time = target_time.to(device)

        target_category = torch.LongTensor([int(target_category)])
        target_category = target_category.to(device)

        user_item_list = items.strip().split(' ')
        target_item_id = user_item_list[-1]
        target_item_id = torch.LongTensor([int(target_item_id)])
        target_item_id = target_item_id.to(device)
        positive_item_triples = torch.stack(
            (userID, target_time, target_category, asked_cluster, asked_poi_type, target_item_id), dim=1)

        for reject_item in self.rejected_item_list_:
            reject_item = torch.LongTensor([int(reject_item)])
            reject_item = reject_item.to(device)
            negative_item_triples = torch.stack(
                (userID, target_time, target_category, asked_cluster, asked_poi_type, reject_item), dim=1)

            optimizer.zero_grad()
            lsigmoid = nn.LogSigmoid()
            diff, _, _ = self.transE_model.forward(positive_item_triples, negative_item_triples)
            loss = - lsigmoid(diff).sum(dim=0)
            loss.backward()
            optimizer.step()

    def vectorize(self):

        list4 = [v for k,v in self.distance_dict2.items()]

        list5 = self.history_list + [0] * (10 - len(self.history_list))

        list6 = [0] * 8

        if len(self.recent_candidate_list) <= 5:
            list6[0] = 1
        if len(self.recent_candidate_list) > 5 and len(self.recent_candidate_list) <= 10:
            list6[1] = 1
        if len(self.recent_candidate_list) > 10 and len(self.recent_candidate_list) <= 15:
            list6[2] = 1
        if len(self.recent_candidate_list) > 15 and len(self.recent_candidate_list) <= 20:
            list6[3] = 1
        if len(self.recent_candidate_list) > 20 and len(self.recent_candidate_list) <= 25:
            list6[4] = 1
        if len(self.recent_candidate_list) > 25 and len(self.recent_candidate_list) <= 30:
            list6[5] = 1
        if len(self.recent_candidate_list) > 30 and len(self.recent_candidate_list) <= 35:
            list6[6] = 1
        if len(self.recent_candidate_list) > 35:
            list6[7] = 1

        list4 = [float(i)/sum(list4) for i in list4]
        list_cat = list4 + list5 + list6
        list_cat = np.array(list_cat)

        assert len(list_cat) == 29
        return list_cat

    def update_upon_featue_inform(self, input_message):
        """

        :param input_message:
        """
        assert input_message.message_type == cfg.INFORM_FACET

        facet = input_message.data['facet']

        if facet is None:
            print('?')
        self.asked_feature.append(facet)
        value = input_message.data['value']

        if facet in ['cluster', 'POI_Type']:
            if value is not None and value[0] is not None:#value is in list
                self.recent_candidate_list = [k for k in self.recent_candidate_list if cfg.item_dict[str(k)][facet] in value]

                self.recent_candidate_list = list(set(self.recent_candidate_list) - set([self.busi_id])) + [self.busi_id]
                self.known_facet.append(facet)
                fresh = True
                if facet == 'clusters':
                    if int(value[0]) not in self.known_feature_cluster:
                        self.known_feature_cluster.append(int(value[0]))
                    else:
                        fresh = False
                if facet == 'POI_Type':
                    if int(value[0]) not in self.known_feature_type:
                        self.known_feature_type.append(int(value[0]))
                    else:
                        fresh = False

                self.known_feature = list(set(self.known_feature)) #feature = values

                if cfg.play_by != 'ADD' and cfg.play_by != 'ADD_valid':
                    self.known_feature_total.clear()
                    self.known_feature_total.append(self.known_feature_cluster)
                    self.known_feature_total.append(self.known_feature_type)
                    self.known_feature_total.append(self.known_feature_category)

                    self.distance_dict = feature_distance(self.known_feature_total, self.user_id, self.TopKTaxo,
                                                          self.features)
                    self.distance_dict2 = self.distance_dict.copy()

                    self.recent_candidate_list_ranked = rank_items(self.known_feature_total, self.items, self.features,
                                                                   self.transE_model, self.recent_candidate_list,
                                                                   self.rejected_item_list_)


