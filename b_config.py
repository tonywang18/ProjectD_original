'''
这个是用于结肠炎症检测的配置文件
'''
from a_config import is_ce_loss

color_tuple_to_class_id = {
    (0, 0, 0):          0,  # 黑色    # bg
    (255, 0, 0):        1,  # 黄色    # others
    (0, 255, 0):        2,  # 红色    # epithelial
    (0, 0, 255):        3,  # 红色    # fibroblast
    (0, 255, 255):      4,  # 红色    # inflammatory
}

class_id_to_color_tuple = dict(zip(color_tuple_to_class_id.values(), color_tuple_to_class_id.keys()))


# # 预测时，将哪类与哪类合并
# pred_class_merge_table = {
# }


# # 难样本区，被指定的颜色区域将会被较高频采样，但不会有该分类出现
# hardsample_zone_classes = []


all_class_ids = (0, 1, 2, 3, 4)
all_class_name = ('bg', 'others', 'epithelial', 'fibroblast', 'inflammatory')

# 要预测哪几个类别，同时这里也是网络输出类别数量

pred_class_ids = (1,)
pred_class_num = len(pred_class_ids)
pred_class_weight = (1,)
if is_ce_loss:
    pred_class_ids = (0,) + pred_class_ids
    pred_class_num += 1
    pred_class_weight = (1,) + pred_class_weight

assert len(pred_class_ids) == len(pred_class_weight), 'Error, pred_class_weight must be have same len with pred_class_ids.'

# 网络输出通道编号，与 pred_class_ids 一一对应
net_out_class_ids = list(range(pred_class_num))
assert len(net_out_class_ids) == len(pred_class_ids)

