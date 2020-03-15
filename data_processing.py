#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import re
from pandas import DataFrame


'''
以下几个函数为一个整体，去除数据中的杂乱部分，预处理掉类似这种内容：
1. ';}Ie_+='';Kc_(sy_,Ie_);}}function ht_(){var Sm_=aJ_();for (sy_=0; sy_< wF_.length;aT_++){var nv_=oX_(wF_[aT_],'_');Ie_+='';}Ie_+='';Kc_(sy_,Ie_);}else{var wF_= oX_(nK_[sy_],',');var Ie_='';for (aT_=0; aT_< wF_.length;aT_++){Ie_+=Mg_(wF_[
2. 緈冨_吶庅羙 2017/02/04 18:43:53 发表在 19楼
'''
pat_case1 = r'[\'a-zA-Z_+=();}{,0-9<\s\[\]\.]{50,}'
pat_case2 = r'发表在.{0,5}(楼|板凳)'

def remove_str(instr, pat):
    patc = re.compile(pat, re.M | re.I)
    obj = patc.search(instr)
    if obj is None:
        return instr
    #     print(obj.group(0))
    pos = instr.find(obj.group(0))
    outstr = instr.replace(obj.group(0), '')
    return outstr

def process_case1(instr):
    return remove_str(instr, pat_case1)

def process_case2(instr):
    pat = re.compile(pat_case2, re.M | re.I)
    obj = pat.search(instr)
    if obj is None:
        return instr
    return instr[instr.find(obj.group(0)) + len(obj.group(0)):]

def process_str(instr):
    o_str = instr
    o_str = process_case1(o_str)
    o_str = process_case2(o_str)
    return o_str

def clean_text(id_flag_test_file, output_clean_file):
    id_flag_test_df = pd.read_csv(id_flag_test_file, encoding='utf-8', sep=',', header=0)
    id_flag_test_df['text'].apply(lambda x: process_str(x))
    id_flag_test_df.to_csv(output_clean_file, encoding='utf-8', sep=',', header=0)

# clean_text('data/train_flag_text_eda_remove_blank.csv', 'data/train_flag_text_eda_remove_blank_clean.csv')
# clean_text('data/train_id_text.csv', 'data/train_id_text_clean.csv')
# clean_text('data/test_id_text.csv', 'data/test_id_text_clean.csv')
# exit()


def convert_to_submit(result_csv='bert_output_robert_large_2_eda_no/test_results.tsv',
                      result_file='1.csv' ,submit_file='bert_output_robert_large_2_eda_no.csv', rule=1, is_test=False):
    '''
    将bert输出文件转化为比赛所需的提交格式，is_tset区分测试与验证
    :param result_csv:bert模型的logits文件
    :param result_file:含有测试文件的所有信息+预测的标签
    :param submit_file:result_file中删除某些列
    :param rule:规则1：正常 规则2：调整阈值 规则3：其它场外信息（如18个空格）
    :param is_test:是否是无标签的测试文件
    :return:
    '''
    # result_csv='albert_large_output_checkpoints_14_2/test_results.tsv',
    import os
    print(os.path.isfile(result_csv))
    result_df = pd.read_csv(result_csv, encoding='utf-8', sep='\t', header=None)
    print(result_df.shape)
    if rule == 1:
        label_index = list(result_df.idxmax(axis=1))
    elif rule == 2:
        # 调整 0 1 类别的划分阈值
        label_index = []
        for index,row in result_df.iterrows():
            if row[0]-row[1] > 0.015:
                label_index.append(0)
            else:
                label_index.append(1)
    # print(label_index)

    test_data_df = pd.read_csv('data/test.csv', encoding='utf-8', sep=',')
    if is_test:
        test_data_df.columns = ['id', 'title', 'context']
    else:
        test_data_df.columns = ['id', 'flag', 'title', 'context']
    test_data_df['pre_label'] = label_index
    test_data_df.to_csv(result_file, encoding='utf-8', sep=',', index=None)

    submit_df = test_data_df[['id', 'pre_label']]  # pre_label需要自己手动改为flag
    submit_df.to_csv(submit_file, encoding='utf-8', sep=',', index=None)
    return

# v1
# convert_to_submit(result_csv='output_robert_large_2_eda_no_epoch6/test_results.tsv',
#                       result_file='large_robert_epoch6_glue1_all_rows.csv' ,submit_file='large_robert_epoch6_glue1.csv', rule=1, is_test=True)
# convert_to_submit(result_csv='output_robert_large_2_eda_no_epoch4/test_results.tsv',
#                       result_file='large_robert_epoch4_glue1_all_rows.csv' ,submit_file='large_robert_epoch4_glue1.csv', rule=1, is_test=True)
# convert_to_submit(result_csv='output_robert_large_2_eda_no_epoch3/test_results.tsv',
#                       result_file='large_robert_epoch3_glue1_all_rows.csv' ,submit_file='large_robert_epoch3_glue1.csv', rule=1,is_test=True)
# convert_to_submit(result_csv='lager_robert_epoch9_again_output/test_results.tsv',
#                       result_file='large_robert_epoch9_again_glue1_all_rows.csv' ,submit_file='large_robert_epoch9_again_glue1.csv', rule=1,is_test=True)

# 结果为89.1
# convert_to_submit(result_csv='bert_output_robert_89/test_results.tsv',
#                       result_file='large_robert_epoch9_glue1_all_rows.csv' ,submit_file='large_robert_epoch9_glue1.csv', rule=1,is_test=True)

# convert_to_submit(result_csv='bert_output_robert_final/test_results.tsv',
#                       result_file='new_data_large_robert_epoch9_glue1_all_rows.csv' ,submit_file='new_data_large_robert_epoch9_glue1.csv', rule=1,is_test=True)
# exit()


def gold_rule(id_flag_title_content_test_file, corrected_all_cols_file, corrected_submit_file):
    '''
    一定正确的规则
    :param id_flag_title_content_test_file:含有测试文件的所有信息+预测的标签 的测试结果文件
    :return:
    '''
    id_flag_title_content_test_df = pd.read_csv(id_flag_title_content_test_file, encoding='utf-8', sep=',', header=0)
    # AttributeError: ("'Series' object has no attribute 'context'", 'occurred at index id')
    # def correct_flag(x):
    #     if str(x.context).startswith('                  '):
    #         return 1
    #     else:
    #         return x.flag
    #
    # id_flag_title_content_test_df['correct_flag'] = id_flag_title_content_test_df.apply(lambda x:correct_flag(x))
    # 先标记为2，试一下这个代码是否有效果？
    # 报错
    # id_flag_title_content_test_df.loc[str(id_flag_title_content_test_df.context).startswith("                  "), 'flag'] = 2

    correct_flag_list = []
    different_num = 0
    for index, row in id_flag_title_content_test_df.iterrows():
        if str(row['context']).startswith("                  ") and str(row['context'])[18]!= ' ':
            if row['pre_label'] != 1:
                different_num += 1
            correct_flag_list.append(1)
            print(row)
        else:
            correct_flag_list.append(row['pre_label'])
    print('has {}/{}={} deifferent flags'.format(different_num, id_flag_title_content_test_df.shape[0], different_num / id_flag_title_content_test_df.shape[0]))

    id_flag_title_content_test_df['correct_flag'] = correct_flag_list
    id_flag_title_content_test_df.to_csv(corrected_all_cols_file, encoding='utf-8', index=None)
    id_flag_title_content_test_df[['id', 'correct_flag']].to_csv(corrected_submit_file, encoding='utf-8', index=None)
    return

# id_flag_title_content_test_file = 'large_robert_epoch399_glue1_all_cols.csv'
# corrected_all_cols_file = 'corrected_all_cols.csv'
# corrected_submit_file = 'corrected_submit.csv'
# gold_rule(id_flag_title_content_test_file, corrected_all_cols_file, corrected_submit_file)

# 纠正89.1
# id_flag_title_content_test_file = 'large_robert_epoch9_glue1_all_rows.csv'
# corrected_all_cols_file = 'epoch9_glue1_corrected_all_cols.csv'
# corrected_submit_file = 'epoch9_glue1_corrected_submit.csv'
# gold_rule(id_flag_title_content_test_file, corrected_all_cols_file, corrected_submit_file)

# 纠正新数据
# id_flag_title_content_test_file = 'new_data_large_robert_epoch9_glue1_all_rows.csv'
# corrected_all_cols_file = 'new_data_epoch9_glue1_corrected_all_cols.csv'
# corrected_submit_file = 'new_data_epoch9_glue1_corrected_submit.csv'
# gold_rule(id_flag_title_content_test_file, corrected_all_cols_file, corrected_submit_file)
# exit()


def compare_file(file1, file2):
    '''
    对比两个文件内容是否一样
    :param file1:
    :param file2:
    :return:
    '''
    temp_file = open('data/different.csv', 'w+', encoding='utf-8')
    f1_df = pd.read_csv(file1, encoding='utf-8', sep=',')
    f2_df = pd.read_csv(file2, encoding='utf-8', sep=',')
    different_num = 0
    for row1, row2 in zip(f1_df.itertuples(), f2_df.itertuples()):
        if getattr(row1, "correct_flag") != getattr(row2, "correct_flag"):
            different_num += 1
            # print(getattr(row1, "pre_label"), getattr(row2, "pre_label"))
            # print(getattr(row1, 'title'), getattr(row1, 'context'))
            temp_file.write(str(getattr(row1, "id")) + "," +
                            str(getattr(row1, "pre_label")) + "," +
                            str(getattr(row1, 'title')) +
                            str(getattr(row1, 'context')) +
                            str(getattr(row2, "pre_label"))+ '\n')
            print(row1)

    temp_file.close()

    print('has {}/{}={} deifferent flags'.format(different_num, f1_df.shape[0], different_num/f1_df.shape[0]))

# file1 = 'large_robert_epoch9_again_glue1_all_rows.csv'
# file2 = 'large_robert_epoch9_glue1_all_rows.csv'
# compare_file(file1, file2)

file1 = 'new_data_epoch9_glue1_corrected_all_cols.csv'
file2 = 'corrected_all_cols.csv'
compare_file(file1, file2)


def result_logits_merge(input_file1, input_file2, input_file3):
    '''合并3（多）个文件的标签'''
    input1_df = pd.read_csv(input_file1, encoding='utf-8', sep='\t', header=None)  # 注意不是/t
    input2_df = pd.read_csv(input_file2, encoding='utf-8', sep='\t', header=None)
    input3_df = pd.read_csv(input_file3, encoding='utf-8', sep='\t', header=None)
    label_logits0_merge = []
    label_logits1_merge = []
    for i1, i2, i3 in zip(input1_df.values, input2_df.values, input3_df.values):
        label_logits0_merge.append(i1[0] + i2[0] + i3[0])
        label_logits1_merge.append(i1[1] + i2[1] + i3[1])

    c = {"label_logits0_merge": label_logits0_merge,
         "label_logits1_merge": label_logits1_merge}  # 将列表a，b转换成字典
    result_logits_merge_df = DataFrame(c)  # 将字典转换成为数据框
    result_logits_merge_df.to_csv("final_merge_logits_test_results.tsv", encoding='utf-8', header=None, index=None, sep='\t')

    convert_to_submit(result_csv='final_merge_logits_test_results.tsv',
                      result_file='final_large_robert_epoch399_glue1_all_cols.csv', submit_file='final_large_robert_epoch399_glue1.csv', rule=1, is_test=True)

# epoch为6的可能有些问题，所以替代为9的
# result_logits_merge('output_robert_large_2_eda_no_epoch3/test_results.tsv',
#                     'lager_robert_epoch9_again_output/test_results.tsv',
#                     'bert_output_robert_89/test_results.tsv',)

# result_logits_merge('output_robert_large_2_eda_no_epoch3/test_results.tsv',
#                     'lager_robert_epoch9_again_output/test_results.tsv',
#                     'bert_output_robert_89/test_results.tsv')

# result_logits_merge('output_robert_large_2_eda_no_epoch3/test_results.tsv',
#                     'bert_output_robert_final/test_results.tsv',
#                     'bert_output_robert_89/test_results.tsv')
# exit()


def merge_train_test_file(train_file, has_label_test_file, new_train_file):
    '''
    合并原始训练集与带有预测标签的测试集
    :param train_file:
    :param has_label_test_file:
    :return:
    '''
    train_df = pd.read_csv(train_file, encoding='utf-8', sep=',')
    has_label_test = pd.read_csv(has_label_test_file, encoding='utf-8', sep=',')

    # 删除pre_label
    has_label_test.drop(labels=['pre_label'], axis=1, inplace=True)

    # 调整列的顺序
    flag = has_label_test['flag']
    has_label_test.drop(labels=['flag'], axis=1, inplace=True)
    has_label_test.insert(1, 'flag', flag)
    print(has_label_test.head())

    new_train_df = pd.concat([train_df, has_label_test], axis=0)

    from sklearn.utils import shuffle
    new_train_df = shuffle(new_train_df)

    new_train_df.to_csv(new_train_file, encoding='utf-8', sep=',', index=None)
    return

# train_file = 'data/train.csv'
# has_label_test_file = 'epoch9_glue1_corrected_all_cols.csv'
# new_train_file = 'new_train_file.csv'
# merge_train_test_file(train_file, has_label_test_file, new_train_file)

