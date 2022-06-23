import csv
import os
import yaml


wait_check_dir = [

]

dir_type = [

]

assert len(wait_check_dir) == len(dir_type)


table = [['type', 'thresh', 'det_f1', 'det_prec', 'det_recall', 'avg_cla_f1', 'avg_cla_prec', 'avg_cla_recall']]


for i in range(5):
    table[0].append(f'cla{i}_f1')
    table[0].append(f'cla{i}_prec')
    table[0].append(f'cla{i}_recall')


for i, dir in enumerate(wait_check_dir):

    ths = os.listdir(dir)
    for th in ths:
        item = [dir_type[i]]
        sub_dir = f'{dir}/{th}'
        if not os.path.isdir(sub_dir):
            continue
        th = float(th)
        item.append(th)
        det_cfg = f'{sub_dir}/all_det_a2.txt'
        det_cfg = yaml.safe_load(open(det_cfg, 'r'))
        a = det_cfg[6]
        item.extend([f'{a["f1"]:.3f}', f'{a["prec"]:.3f}', f'{a["recall"]:.3f}'])

        cla_cfg = f'{sub_dir}/all_cla.txt'
        cla_cfg = yaml.safe_load(open(cla_cfg, 'r'))
        a = cla_cfg[6]
        item.extend([f'{a["f1"]:.3f}', f'{a["prec"]:.3f}', f'{a["recall"]:.3f}'])
        for k in range(5):
            item.extend([f'{a[k]["f1"]:.3f}', f'{a[k]["prec"]:.3f}', f'{a[k]["recall"]:.3f}'])
        table.append(item)

writer = csv.writer(open('o1.csv', 'w'))
writer.writerows(table)
