import sys
import json

import torch

import generate_data

#判断cuda是否可用
def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var

#删除重复的元素
def get_unique_colum_values(column):
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
        print("********** generate data done *********")
        sequence_list = []
        f1 = open("train_list_item.txt", "w")
        f2 = open("train_list_features.txt", "w")
        f3 = open("train_list_location.txt", "w")
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

        # def init_basic(self):
    #
    # def init_type(self):
    #
    # def init_misc(self):
    #
    # def init_FM_related(self):
    #
    # def init_test(self):
    #
    # #def change_param(self):