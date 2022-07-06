import sys
import json

import torch
import pandas as pd
import generate_data
import time

import models
import models as model
#判断cuda是否可用
def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var

#删除重复的元素
def get_unique_column_values(column):
    column_list = []
    for i in column:
        for j in i:
            if j not in column_list:
                column_list.append(j)
    return column_list

#Conifg类
class _Config():
    def __init__(self):
        self.init_data()
        self.init_basic()
        self.init_type()

        self.init_misc()
        self.init_test()
        self.init_FM_related()

    #初始化数据
    def init_data(self):
        #获取数据（训练集，验证集，测试集）
        train_dict, valid_dict, test_dict = generate_data.get_data()
        print("********** generate data *********")
        sequence_list = []
        f1 = open("train_list_item.txt", "w")#记录poi itemid
        f2 = open("train_list_features.txt", "w")#记录poi 时间信息，类别信息，大类别（cluster）信息，POI种类信息
        f3 = open("train_list_location.txt", "w")#poi 地点信息
        for key in train_dict.keys():
            num_total_sequences = len(train_dict[key][0])
            for i in range(num_total_sequences):
                sequence_length = len(train_dict[key][0][i]) #单条sequence的长度
                sequence_list.append(sequence_length)
                f1_row = str(key) + ''
                f2_row = ''
                f3_row = ''
                for j in range(sequence_length):
                    f1_row += str(train_dict[key][1][i][j]) + ' '
                    f2_row += str(train_dict[key][8][i][j]) + ',' + str(train_dict[key][3][i][j]) + ',' \
                              + str(train_dict[key][4][i][j]) + ',' + str(train_dict[key][5][i][j]) + ' '
                f3_row += str(train_dict[key][6][i][sequence_length-1])
                f1_row += '\n'
                f2_row += '\n'
                f3_row += '\n'

                f1.write(f1_row)
                f2.write(f2_row)
                f3.write(f3_row)
        f1.close()
        f2.close()
        f3.close()

        sequence_list = []
        f4 = open("valid_list_item.txt", "w")
        f5 = open("valid_list_features.txt", "w")
        f6 = open("valid_list_location.txt", "w")

        for key in valid_dict.keys():
            num_total_sequences = len(valid_dict[key][0])
            for i in range(num_total_sequences):
                sequence_length = len(valid_dict[key][0][i])
                sequence_list.append(sequence_length)
                f4_row = str(key) + ' '
                f5_row = ''
                f6_row = ''
                for j in range(sequence_length):
                    f4_row += str(valid_dict[key][1][i][j]) + ' '
                    f5_row += str(valid_dict[key][8][i][j]) + ',' + str(valid_dict[key][3][i][j]) + ',' + \
                              str(valid_dict[key][4][i][j]) + ',' + str(valid_dict[key][5][i][j]) + ' '
                f6_row = str(valid_dict[key][6][i][sequence_length - 1])

                f4_row += '\n'
                f5_row += '\n'
                f6_row += '\n'
                f4.write(f4_row)
                f5.write(f5_row)
                f6.write(f6_row)
        f4.close()
        f5.close()
        f6.close()

        sequence_list = []
        f7 = open("test_list_item.txt", "w")
        f8 = open("test_list_features.txt", "w")
        f9 = open("test_list_location.txt", "w")
        for key in test_dict.keys():
            num_total_sequences = len(test_dict[key][0])
            for i in range(num_total_sequences):
                sequence_length = len(test_dict[key][0][i])
                sequence_list.append(sequence_length)
                f7_row = str(key) + ' '
                f8_row = ''
                f9_row = ''
                for j in range(sequence_length):
                    f7_row += str(test_dict[key][1][i][j]) + ' '
                    f8_row += str(test_dict[key][8][i][j]) + ',' + str(test_dict[key][3][i][j]) + ',' + \
                              str(test_dict[key][4][i][j]) + ',' + str(test_dict[key][5][i][j]) + ' '
                f9_row = str(test_dict[key][6][i][sequence_length - 1])

                f7_row += '\n'
                f8_row += '\n'
                f9_row += '\n'
                f7.write(f7_row)
                f8.write(f8_row)
                f9.write(f9_row)
        f7.close()
        f8.close()
        f9.close()

        train_list_item = []
        f = open("train_list_item.txt","r")
        for x in f:
            train_list_item.append(x)
        f.close()

        train_list_features = []
        f = open("train_list_features.txt", "r")
        for x in f:
            train_list_features.append(x)
        f.close()

        train_list_location = []
        f = open("train_list_location.txt", "r")
        for x in f:
            train_list_location.append(x)
        f.close()

        valid_list_item = []
        f = open("valid_list_item.txt", "r")
        for x in f:
            valid_list_item.append(x)
        f.close()

        valid_list_features = []
        f = open("valid_list_features.txt", "r")
        for x in f:
            valid_list_features.append(x)
        f.close()

        valid_list_location = []
        f = open("valid_list_location.txt", "r")
        for x in f:
            valid_list_location.append(x)
        f.close()

        test_list_item = []
        f = open("test_list_item.txt", "r")
        for x in f:
            test_list_item.append(x)
        f.close()

        test_list_features = []
        f = open("test_list_features.txt", "r")
        for x in f:
            test_list_features.append(x)
        f.close()

        test_list_location = []
        f = open("test_list_location.txt", "r")
        for x in f:
            test_list_location.append(x)
        f.close()
        self.train_list_item = train_list_item
        self.train_list_features = train_list_features
        self.train_list_location = train_list_location

        self.valid_list_item = valid_list_item
        self.valid_list_features = valid_list_features
        self.valid_list_location = valid_list_location

        self.test_list_item = test_list_item
        self.test_list_features = test_list_features
        self.test_list_location = test_list_location

    def init_basic(self):
        df = pd.read_csv("./data/new_transE_3.csv")
        # 实体计数
        self.user_length = df['User_id'].max() + 1
        self.item_length = df['Item_id'].max() + 1

        self.entity_count_list = []
        self.entity_count_list.append(int(self.user_length))
        self.entity_count_list.append(int(self.item_length))

        # 类别数
        # 关系计数
        time_length = df['new_time'].max()+1 #new_time表示的是小时（几点）
        category_length = df['L2_Category_name'].max()+1
        cluster_length = df['clusters'].max()+1 #经纬度非常接近的地点簇
        poi_type_length = df['POI_Type'].max()+1

        self.type_count = poi_type_length
        self.relation_count_list = []
        self.relation_count_list.append(int(time_length))
        self.relation_count_list.append(int(category_length))
        self.relation_count_list.append(int(cluster_length))
        self.relation_count_list.append(int(poi_type_length))

        self.vector_length = 50
        self.margin = 1.0
        self.device = torch.device('cuda')
        self.norm = 1
        self.learning_rate = 0.01

        user_list = df[['User_id']].values.tolist()
        self.user_list = get_unique_column_values(user_list)

        busi_list = df[['Item_id']].values.tolist()
        self.busi_list = get_unique_column_values(busi_list)

        #L2类别和大的类别名称关系
        with open('./data/L2.json', 'r') as f:
            self.taxo_dict = json.load(f)
        #所有的certain poi（包括信息：评分星级，地点id，L2类别）
        with open('./data/poi.json', 'r') as f:
            self.poi_dict = json.load(f)


        #每个poi的一些特征信息
        df3 = pd.read_csv("./data/dict.csv")
        item_dict=dict()
        star_list =[]
        #遍历行数据
        for index, row in df3.iterrows():
            '''
            0
            Item_id             0.0
            POI_Type            0.0
            L2_Category_name    0.0
            stars               4.0
            clusters            0.0
            Name: 0, dtype: float64
            '''
            if str(int(row['Item_id'])) not in item_dict:
                star_list.clear()
                row_dict = dict()
                star_list.append(row['stars'])
                row_dict['stars'] = row['stars']
                row_dict['clusters'] = int(row['clusters'])
                row_dict['L2_Category_name'] = [int(row['L2_Category_name'])]
                row_dict['POI_Type'] = int(row['POI_Type'])
                row_dict['feature_index'] = [int(row['L2_Category_name'])]
                row_dict['feature_index'].append(int(row['clusters']) + category_length)
                row_dict['feature_index'].append(int(row['POI_Type']) + category_length + cluster_length)
                row_dict['feature_index'].append(
                    2 * int(row['stars']) - 2 + category_length + cluster_length + poi_type_length)
                item_dict[str(int(row['Item_id']))] = row_dict
            else:
                star_list.append(row['stars'])
                item_dict[str(int(row['Item_id']))]['stars'] = (sum(star_list)) / len(star_list)
                item_dict[str(int(row['Item_id']))]['L2_Category_name'].append(int(row['L2_Category_name']))
                item_dict[str(int(row['Item_id']))]['feature_index'].append(int(row['L2_Category_name']))
        self.item_dict = item_dict
        print(item_dict)
    def init_type(self):
        self.INFORM_FACET = 'INFORM_FACET'
        self.ACCEPT_REC = 'ACCEPT_REC'
        self.REJECT_REC = 'REJECT_REC'

        self.ASK_FACET = 'ASK_FACET'
        self.MAKE_REC = 'MAKE_REC'
        self.FINISH_REC_ACP = 'FINISH_REC_ACP'
        self.FINISH_REC_REJ = 'FINISH_REC_REJ'
        self.EPISODE_START = 'EPISODE_START'

        self.USER = 'USER'
        self.AGENT = 'AGENT'

    def init_misc(self):
        self.FACET_POOL = ['clusters',  'POI_Type']#cluster 表示为某cluster区域内的poi，poi type表示为是否为certain poi
        self.FACET_POOL += self.taxo_dict.keys()
        print('Total feature length is: {}, Top 10 namely: {}'.format(len(self.FACET_POOL), self.FACET_POOL[: 10]))
        self.REC_NUM = 10
        self.MAX_TURN = 10
        self.play_by = None
        self.calculate_all = None

    def init_FM_related(self):
        clusters_max = 0
        category_max = 0
        feature_max = 0
        for k, v in self.item_dict.items():
            if v['clusters'] > clusters_max:
                clusters_max = v['clusters']
            if max(v['L2_Category_name']) > category_max:
                category_max = max(v['L2_Category_name'])
            if max(v['feature_index']) > feature_max:
                feature_max = max(v['feature_index'])

        stars_list = [3.5, 2.5, 1.5, 3.0, 4.5, 1.0, 4.0, 5.0, 2.0]#星级评价
        poi_list = [0,1]#是否为certain poi

        self.star_count, self.poi_count = len(stars_list), len(poi_list)
        self.clusters_count, self.category_count, self.feature_count = clusters_max + 1, category_max + 1, feature_max + 1

        self.clusters_span = (0, self.clusters_count)
        self.poi_span = (self.clusters_count, self.clusters_count + self.poi_count)
        self.star_span = (self.clusters_count + self.star_count, self.clusters_count + self.star_count + self.poi_count)

        self.spans = [self.clusters_span, self.star_span, self.poi_span]

        print('clusters max: {}, category max: {}, feature max: {}'.format(self.clusters_count, self.category_count, self.feature_count))

        #两个模型，一个是对item的attention，一个是对relation的attention
        model_item = models.Attention_item(input_dim=2*self.vector_length, dim1=64, output_dim=1)
        model_relation = models.Attention_relation(input_dim=2*self.vector_length, dim1=64, output_dim=1)

        device = torch.device('cuda')
        model = models.TransE(self.relation_count_list, self.entity_count_list, device, dim=self.vector_length,
                              margin=self.margin, norm=self.norm, item_att_model=model_item, relation_att_model=model_relation)


        model.load_state_dict(torch.load("./weight/new_transE.pt"))#pt文件里是保存好训练好的模型。
        self.entities_emb_head = model.entities_emb_head
        self.entities_emb_tail = model.entities_emb_tail
        self.relations_emb_time = model.relations_emb_time
        self.relations_emb_category = model.relations_emb_category
        self.relations_emb_clusters = model.relations_emb_clusters
        self.relations_emb_poi_type = model.relations_emb_poi_type
        self.transE_model = model.to(device)


    def init_test(self):
        pass

    def change_param(self, playby, eval, update_count, update_reg, purpose, mod, mask):
        self.play_by = playby
        self.eval = eval
        self.update_count = update_count
        self.update_reg = update_reg
        self.purpose = purpose
        self.mod = mod
        self.mask = mask

start = time.time()
global_config = _Config()
print('Config takes: {}'.format(time.time() - start))

print('___Config Done!!___')