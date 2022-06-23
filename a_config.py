'''
简写解释
cm 混淆矩阵
lr 学习率
prefix 命名前缀

'''

import os

# base setting
## 项目根目录路径，一般不需要修改
project_root = os.path.split(__file__)[0]
print(project_root)

## 指定运行在哪个设备
device = 'cuda:0'
## 训练参数，分别是学习率，训练轮数，每批量大小
train_lr = 1e-3
# epoch = 150
epoch = 150
# batch_size = 24
batch_size = 64
# batch_count = 10
# batch_count = 2
batch_count = 300
# net_in_hw = (64, 64)
# net_out_hw = (64, 64)
net_in_hw = (64, 64)
net_out_hw = (64, 64)

# 0到第一个数字为只训练检测，第一个数字到第二个数字为只训练分类检测，第二个数字后为同时启动训练
process_control = [50, 100,]
# process_control = [3, 6,]
assert len(process_control) == 2


# 特别处理1，让分类分支也只是训练检测，不分类
make_cla_is_det = False
# 特别处理2，是否使用排斥编码
use_repel_code = True

match_distance_thresh_list = [6, 9]

b1_loss_type = 'softmax'
b2_loss_type = 'aloss'
b3_loss_type = 'aloss'
ce_loss = ('lovasz_softmax', 'softmax')
reg_loss = ('lovasz_hinge', 'aloss')
assert b1_loss_type in ce_loss + reg_loss
assert b2_loss_type in ce_loss + reg_loss
assert b3_loss_type in ce_loss + reg_loss

b1_is_ce_loss = b1_loss_type in ce_loss
b2_is_ce_loss = b2_loss_type in ce_loss
b3_is_ce_loss = b3_loss_type in ce_loss

# 是否从最近检查点开始训练
is_train_from_recent_checkpoint = True
# eval是否使用best检查点
eval_which_checkpoint = 'best'
assert eval_which_checkpoint in ('last', 'best', 'minloss')


# dataset setting
## 数据集位置
# dataset_path = project_root + '/data_2'
dataset_path = 'D:\\数据集\\fold1_2_3'
#
# # net setting


# save_postfix
# save_postfix = '_m9_cla_local_bn_mask'
# save_postfix = '_type3_m11_k1_bl'
save_postfix = '_type3_m11_k1_bl_tdssgray'
# save_postfix = '_tmp'


#
#
# ## 指定模型保存位置和日志保存位置
net_save_dir = project_root + '/save' + save_postfix
net_train_logs_dir = project_root + '/logs' + save_postfix
## 测试输出
## 用于预测新的图像
### 将新图像放入此文件夹
new_img_test_in_dir = project_root + '/test_in'
### 对新图像分析后的结果将存入此文件夹
new_img_test_out_dir = project_root + '/test_out'
# ### 用于预测ndpi图像的文件夹输入
# new_ndpi_test_in_dir = project_root + '/test_ndpi_in' + dataset_type
# ### 用于预测ndpi图像的文件夹输出，输出为xml文件
# new_ndpi_test_out_dir = project_root + '/test_ndpi_out' + dataset_type
