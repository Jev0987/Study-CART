# -*- coding: utf-8 -*-
import pandas as pd
import copy

'''
生成数据
'''

def get_data():
    df = pd.read_csv("./data/new_transE_3.csv")
    '''
    多列分组 groupby()
    记录次数 groupby().size()
    生成新的表格df2，里面记录根据User_id和Date的访问次数
    eg:
    User_id    Date       
      0       2012422  12
              2012423   3
              2012424   5
              2013129   4
              2013708   5
    '''
    new = df.groupby(['User_id','Date']).size()
    df2 = pd.DataFrame(new)
    df2.rename(columns={0:'counts'},inplace=True)

    count_list = df2['counts'].tolist() #所有用户不同时间访问的次数列表
    current_user = df.loc[0]['User_id'] #第0行的userid

    # 创建空字典
    total_dict = dict()
    #根据不同的列标签创建字典
    #{'user_list': [], 'item_list': [], 'L1_category_list': [], 'L2_category_list': [], 'cluster_list': [], 'type_list': [], 'location_list': [], 'star_list': [], 'time_list': []}
    count_dict = dict()
    count_dict['user_list'] = []
    count_dict['item_list'] = []
    count_dict['L1_category_list'] = []
    count_dict['L2_category_list'] = []
    count_dict['cluster_list'] = []
    count_dict['type_list'] = []
    count_dict['location_list'] = []
    count_dict['star_list'] = []
    count_dict['time_list'] = []

    begin_row_index = 0
    previous_user = 0
    for count in count_list:
        #记录当前行userid
        current_user = df.loc[begin_row_index]['User_id']

        #判断上一行的userid是否等于当前的userid，
        #如果不同，将之前的count_dict记录到total_dict中，并清空count_dict重新记录
        #如果相同，则只记录count_dict
        if previous_user != current_user:
            total_dict[str(previous_user)] = copy.deepcopy(count_dict)
            count_dict.clear()
            count_dict['user_list'] = []
            count_dict['item_list'] = []
            count_dict['L1_category_list'] = []
            count_dict['L2_category_list'] = []
            count_dict['cluster_list'] = []
            count_dict['type_list'] = []
            count_dict['location_list'] = []
            count_dict['star_list'] = []
            count_dict['time_list'] = []
            previous_user = current_user
        else:
            total_dict[str(current_user)] = copy.deepcopy(count_dict)
        '''
        记录用户具体的check-in数据
        生成的count_dict中包含多个sequences(最短长度为2)，eg：{'user_list':[[0,0],[0,0,0],[0,0,0,0]],'item_list':[[0,1],[0,1,2],[0,1,2,3]],....}
        total_list中保存的是每个用户对应的所有count_dict
        '''
        for end_row_index in range(2, count+1):
            df3 = df[begin_row_index : begin_row_index + end_row_index]
            count_dict['user_list'].append(df3['User_id'].tolist())
            count_dict['item_list'].append(df3['Item_id'].tolist())
            count_dict['L1_category_list'].append(df3['L1_Category_name'].tolist())
            count_dict['L2_category_list'].append(df3['L2_Category_name'].tolist())
            count_dict['cluster_list'].append(df3['clusters'].tolist())
            count_dict['type_list'].append(df3['POI_Type'].tolist())
            count_dict['location_list'].append(df3['Location_id'].tolist())
            count_dict['star_list'].append(df3['stars'].tolist())
            count_dict['time_list'].append(df3['new_time'].tolist())
        begin_row_index += count

    #生成训练集，验证集，测试集（对应不同的用户生成的sequences）
    train_dict = dict()
    valid_dict = dict()
    test_dict = dict()
    sub_dict = dict()
    sub_dict['user_list'] = []
    sub_dict['item_list'] = []
    sub_dict['L1_category_list'] = []
    sub_dict['L2_category_list'] = []
    sub_dict['cluster_list'] = []
    sub_dict['type_list'] = []
    sub_dict['location_list'] = []
    sub_dict['star_list'] = []
    sub_dict['time_list'] = []
    for key in total_dict.keys():
        length = len(total_dict[key]['user_list'])#不同用户的sequences数量
        train_list_group = []
        valid_list_group = []
        test_list_group = []

        '''
        [[0,0],[0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0,],[0,0,0,0,,0],[0,0,0,0,0,0,0,0],[0,0,0,0]]
        测试集为sequences的前几条 (用下面的分片计算的话，第1，2条为测试集)
        验证集为sequences的中间几条（第2，3，4为验证集）
        训练集为sequences后面的所有条目（第4，5，6，7，8，9，10为训练集）
        '''
        test_list_group.append(total_dict[key]['user_list'][: int(0.1 * (length) + 1)])
        valid_list_group.append(total_dict[key]['user_list'][int(0.1 * (length) + 1): int(0.3 * (length) + 1)])
        train_list_group.append(total_dict[key]['user_list'][int(0.3 * (length) + 1):])

        test_list_group.append(total_dict[key]['item_list'][: int(0.1 * (length) + 1)])
        valid_list_group.append(total_dict[key]['item_list'][int(0.1 * (length) + 1): int(0.3 * (length) + 1)])
        train_list_group.append(total_dict[key]['item_list'][int(0.3 * (length) + 1):])

        test_list_group.append(total_dict[key]['L1_category_list'][: int(0.1 * (length) + 1)])
        valid_list_group.append(total_dict[key]['L1_category_list'][int(0.1 * (length) + 1): int(0.3 * (length) + 1)])
        train_list_group.append(total_dict[key]['L1_category_list'][int(0.3 * (length) + 1):])

        test_list_group.append(total_dict[key]['L2_category_list'][: int(0.1 * (length) + 1)])
        valid_list_group.append(total_dict[key]['L2_category_list'][int(0.1 * (length) + 1): int(0.3 * (length) + 1)])
        train_list_group.append(total_dict[key]['L2_category_list'][int(0.3 * (length) + 1):])

        test_list_group.append(total_dict[key]['cluster_list'][: int(0.1 * (length) + 1)])
        valid_list_group.append(total_dict[key]['cluster_list'][int(0.1 * (length) + 1): int(0.3 * (length) + 1)])
        train_list_group.append(total_dict[key]['cluster_list'][int(0.3 * (length) + 1):])

        test_list_group.append(total_dict[key]['type_list'][: int(0.1 * (length) + 1)])
        valid_list_group.append(total_dict[key]['type_list'][int(0.1 * (length) + 1): int(0.3 * (length) + 1)])
        train_list_group.append(total_dict[key]['type_list'][int(0.3 * (length) + 1):])

        test_list_group.append(total_dict[key]['location_list'][: int(0.1 * (length) + 1)])
        valid_list_group.append(total_dict[key]['location_list'][int(0.1 * (length) + 1): int(0.3 * (length) + 1)])
        train_list_group.append(total_dict[key]['location_list'][int(0.3 * (length) + 1):])

        test_list_group.append(total_dict[key]['star_list'][: int(0.1 * (length) + 1)])
        valid_list_group.append(total_dict[key]['star_list'][int(0.1 * (length) + 1): int(0.3 * (length) + 1)])
        train_list_group.append(total_dict[key]['star_list'][int(0.3 * (length) + 1):])

        test_list_group.append(total_dict[key]['time_list'][: int(0.1 * (length) + 1)])
        valid_list_group.append(total_dict[key]['time_list'][int(0.1 * (length) + 1): int(0.3 * (length) + 1)])
        train_list_group.append(total_dict[key]['time_list'][int(0.3 * (length) + 1):])

        test_dict[key] = test_list_group
        valid_dict[key] = valid_list_group
        train_dict[key] = train_list_group
    return train_dict,valid_dict,test_dict