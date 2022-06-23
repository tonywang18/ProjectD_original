'''
软件 ASAP 的 XML文件 读取和写入工具
目前支持 轮廓，方框，椭圆，箭头 的读入和写入操作
注意，除了标签数据，其他非必要的信息目前不提供支持

读取后将会返回 区域列表和颜色元组列表
存档时需要的也是 区域列表和颜色元组列表

如何使用？
请看本文件最下面的测试样例

注意：
这里读取返回和储存时需要的坐标点格式不是 xy，而是 yx。使用时请务必注意。

数据格式：
CONTOUR：一串坐标点。[point1, point2, point3, ...]
BOX: 有两种方式，可通过 use_box_y1x1y2x2 参数选择。方式1：[左上角坐标，右下角坐标]，方式2：[左上角坐标，右上角坐标，右下角坐标，左下角坐标]
ELLIPSE：未知，等待补充
ARROW：读取时，有两种方式：可通过 keep_arrow_tail 参数选择，如果为真，格式为[point_head, point_tail]，否则格式为 point_head
       存储时，若只有 point_head，没有 point_tail，需设定 auto_tail 为真，将自动生成 point_tail，否则会报错。

'''
# TODO 未完成


import lxml.etree as etree
import numpy as np
from typing import Tuple
import math
from matplotlib import colors as plt_colors


TYPE_CONTOUR =  'Polygon'   # 单个格式：[pt1_yx, pt2_yx, pt3_yx, ...]
TYPE_BOX =      'Rectangle' # 单个格式：[pt_tl_yx, pt_tr_yx, pt_br_yx, pt_bl_yx]
TYPE_POINT =    'Dot'       # 单个格式：[y1, x1]
TYPE_SPLINE =   'Spline'    # 单个格式：[pt1_yx, pt2_yx, pt3_yx, ...]
TYPE_POINTSET = 'PointSet'  # 单个格式：[pt1_yx, pt2_yx, pt3_yx, ...]


# def color_int_to_tuple(color_int):
#     '''
#     将RGB颜色元组转换为颜色整数
#     :param color_int:
#     :return:
#     '''
#     color_str = hex(color_int)[2:]
#     assert len(color_str) <= 6, 'Found unknow color!'
#     pad_count = 6 - len(color_str)
#     color_str = ''.join(['0'] * pad_count) + color_str
#     b, g, r = int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16)
#     return r, g, b
#
#
# def color_tuple_to_int(color_tuple):
#     '''
#     将RGB颜色元组转换为颜色整数
#     :param color_tuple:
#     :return:
#     '''
#     assert len(color_tuple) == 3, 'Found unknow color tuple!'
#     r, g, b = color_tuple
#     color_int = r + (g << 8) + (b << 16)
#     return color_int


def color_str_to_tuple(t):
    c = tuple(int(c*255+0.5) for c in  plt_colors.to_rgb(t))
    return c


def color_tuple_to_str(t):
    # t='(255,0,0)'
    t: str
    a = []
    for c in t:
        assert 0 <= c <= 255
        a.append(f'{c:0>2x}')
    s = '#' + ''.join(a)
    return s


# def read_asap_xml(in_xml):
#     """
#     :param in_xml: xml file
#     :return:
#     """
#     coords_list = []
#     meta_list = []
#
#     root = etree.parse(in_xml)
#
#     for ann in root.findall('//Annotations/Annotation'):
#         name = str(ann.attrib['Name'])
#         dtype = str(ann.attrib['Type'])
#         color = str(ann.attrib['Color'])
#
#         color = color_str_to_tuple(color)
#         color = color[::-1]
#
#         coords = []
#
#         for v in ann.findall('Coordinates/Coordinate'):
#             x = float(v.attrib['X'])
#             y = float(v.attrib['Y'])
#             coords.append([y, x])
#
#         meta = {'name': name, 'type': dtype, 'color': color}
#         coords = np.float32(coords)
#
#         coords_list.append(coords)
#         meta_list.append(meta)
#
#     return coords_list, meta_list


class AsapXmlReader:
    def __init__(self, file=None, use_box_y1x1y2x2=True):
        '''

        :param file:            读取文件路径
        :param use_box_y1x1y2x2:读取方盒标签时是否使用y1x1y2x2坐标，若设为False则使用[左上，右上，右下，左下]坐标
        '''
        self.use_box_y1x1y2x2 = use_box_y1x1y2x2
        self.item_list = []
        if file is not None:
            self.read(file)

    def read(self, file):
        tree = etree.parse(file)
        for ann in tree.findall('//Annotations/Annotation'):

            dtype = str(ann.attrib['Type'])

            if dtype not in (TYPE_CONTOUR, TYPE_BOX, TYPE_POINT, TYPE_SPLINE, TYPE_POINTSET):
                print(f'Warning! Found unknow type "{dtype}". Will ignore.')
                continue

            color_tuple = color_str_to_tuple(ann.attrib['Color'])
            # # BGR to RGB
            # color_tuple = color_tuple[::-1]

            name = str(ann.attrib['Name'])
            group = str(ann.attrib['PartOfGroup'])

            coords = []
            for v in ann.findall('Coordinates/Coordinate'):
                x = float(v.attrib['X'])
                y = float(v.attrib['Y'])
                coords.append([y, x])

            coords = np.float32(coords)

            if dtype == TYPE_BOX and self.use_box_y1x1y2x2:
                yx_min = np.min(coords, axis=0)
                yx_max = np.max(coords, axis=0)
                coords = np.concatenate([yx_min, yx_max], axis=0)

            item = {'coords': coords, 'name': name, 'dtype': dtype, 'color': color_tuple, 'group': group}
            self.item_list.append(item)

    def _get_type(self, dtype):
        coords, colors, names, groups = [], [], [], []
        for item in self.item_list:
            if item['dtype'] == dtype:
                coords.append(item['coords'])
                colors.append(item['color'])
                names.append(item['name'])
                groups.append(item['group'])
        return coords, colors, names, groups

    def get_contours(self):
        return self._get_type(TYPE_CONTOUR)
    
    def get_boxes(self):
        return self._get_type(TYPE_BOX)

    def get_points(self):
        return self._get_type(TYPE_POINT)

    def get_splines(self):
        return self._get_type(TYPE_SPLINE)

    def get_pointsets(self):
        return self._get_type(TYPE_POINTSET)


class AsapXmlWriter:

    def __init__(self, contour_default_is_closure=True, use_box_y1x1y2x2=True):
        '''
        :param contour_default_is_closure:  默认输入的轮廓是否是闭合的
        :param allow_box_y1x1y2x2:          是否允许方框坐标为 y1x1y2x2，若设为False，则需要手动保证方框输入坐标为 [左上，右上，右下，左下] 格式坐标
        '''
        self.contour_default_is_closure = contour_default_is_closure
        self.use_box_y1x1y2x2 = use_box_y1x1y2x2
        # 每个类别的存储处，存储方式：item = {coords: [yx,...], name: 'abc', color: (R,G,B), group: 'None'}
        self.item_list = []

    def _add_items(self, coords, colors, dtypes, names=None, groups=None, is_closures=None):
        assert len(coords) == len(colors) == len(dtypes)

        if is_closures is None:
            is_closures = [self.contour_default_is_closure] * len(coords)
        else:
            assert len(is_closures) == len(coords)

        if names is None:
            names = [''] * len(coords)
        else:
            assert len(names) == len(coords)

        if groups is None:
            print('Warning! The group is ignore now.')
        groups = ['None'] * len(coords)

        # color_set = set(colors)
        for coord, color, dtype, name, group, is_closure in zip(coords, colors, dtypes, names, groups, is_closures):
            if is_closure and np.any(coord[0] != coord[-1]):
                assert dtype in (TYPE_CONTOUR, TYPE_SPLINE)
                coord = np.concatenate([coord, coord[-1:]], 0)

            item = {'coords': coord, 'color': color, 'dtype': dtype, 'name': name, 'group': group}
            self.item_list.append(item)

    def add_contours(self, contours, colors, names=None, groups=None, is_closures=None):
        self._add_items(contours, colors, [TYPE_CONTOUR]*len(contours), names, groups, is_closures)

    def add_boxes(self, boxes, colors, names=None, groups=None):
        boxes = np.asarray(boxes, np.float32)
        if self.use_box_y1x1y2x2:
            assert boxes.ndim == 2 and boxes.shape[1] == 4
        else:
            assert boxes.ndim == 3 and boxes.shape[1] == 4 and boxes.shape[2] == 2
        self._add_items(boxes, colors, [TYPE_BOX]*len(boxes), names, groups, [False]*len(boxes))

    def add_points(self, points, colors, names=None, groups=None):
        points = np.asarray(points, np.float32)
        assert points.ndim == 2 and points.shape[1] == 2
        self._add_items(points, colors, [TYPE_POINT]*len(points), names, groups, [False]*len(points))

    def add_splines(self, splines, colors, names=None, groups=None, is_closures=None):
        self._add_items(splines, colors, [TYPE_SPLINE]*len(splines), names, groups, is_closures)

    def add_pointsets(self, pointsets, colors, names=None, groups=None):
        self._add_items(pointsets, colors, [TYPE_POINTSET]*len(pointsets), names, groups, [False]*len(pointsets))

    def write(self, file):
        raise NotImplemented

        # Annotations = etree.Element('Annotations', {'MicronsPerPixel': '0'})
        # ann_id = 0
        # for color_regs, type_id in zip([self.contour_color_regs, self.box_color_regs, self.arrow_color_regs, self.ellipse_color_regs],
        #                                [TYPE_CONTOUR, TYPE_BOX, TYPE_ARROW, TYPE_ELLIPSE]):
        #     for color in color_regs.keys():
        #         ann_id += 1
        #         LineColor = str(color_tuple_to_int(color))
        #         Annotation = etree.SubElement(Annotations, 'Annotation',
        #                                       {'Id': str(ann_id), 'Name': '', 'ReadOnly': '0', 'NameReadOnly': '0',
        #                                        'LineColorReadOnly': '0', 'Incremental': '0', 'Type': '4',
        #                                        'LineColor': LineColor, 'Visible': '1', 'Selected': '0',
        #                                        'MarkupImagePath': '', 'MacroName': ''})
        #
        #         Attributes = etree.SubElement(Annotation, 'Attributes')
        #         etree.SubElement(Attributes, 'Attribute', {'Name': '', 'Id': '0', 'Value': ''})
        #         Regions = etree.SubElement(Annotation, 'Regions')
        #         RegionAttributeHeaders = etree.SubElement(Regions, 'RegionAttributeHeaders')
        #         etree.SubElement(RegionAttributeHeaders, 'AttributeHeader',
        #                          {'Id': "9999", 'Name': 'Region', 'ColumnWidth': '-1'})
        #         etree.SubElement(RegionAttributeHeaders, 'AttributeHeader',
        #                          {'Id': "9997", 'Name': 'Length', 'ColumnWidth': '-1'})
        #         etree.SubElement(RegionAttributeHeaders, 'AttributeHeader',
        #                          {'Id': "9996", 'Name': 'Area', 'ColumnWidth': '-1'})
        #         etree.SubElement(RegionAttributeHeaders, 'AttributeHeader',
        #                          {'Id': "9998", 'Name': 'Text', 'ColumnWidth': '-1'})
        #
        #         for contour_id, contour in enumerate(color_regs[color]):
        #             Region = etree.SubElement(Regions, 'Region',
        #                                       {'Id': str(contour_id), 'Type': str(type_id), 'Zoom': '1', 'Selected': '0',
        #                                        'ImageLocation': '', 'ImageFocus': '-1', 'Length': '0', 'Area': '0',
        #                                        'LengthMicrons': '0', 'AreaMicrons': '0', 'Text': '', 'NegativeROA': '0',
        #                                        'InputRegionId': '0', 'Analyze': '1', 'DisplayId': str(contour_id)})
        #             etree.SubElement(Region, 'Attributes')
        #             Vertices = etree.SubElement(Region, 'Vertices')
        #             for v_yx in contour:
        #                 etree.SubElement(Vertices, 'Vertex', {'X': str(v_yx[1]), 'Y': str(v_yx[0]), 'Z': '0'})
        #
        #         etree.SubElement(Annotation, 'Plots')
        #
        # doc = etree.ElementTree(Annotations)
        # doc.write(open(file, "wb"), pretty_print=True)


if __name__ == '__main__':
    print('Testing')
    reader = AsapXmlReader("e:/a.xml")
    a1 = reader.get_contours()
    a2 = reader.get_boxes()
    a3 = reader.get_points()
    a4 = reader.get_splines()
    a5 = reader.get_pointsets()

    print(a1)
    print(a2)
    print(a3)
    print(a4)
    print(a5)

    # writer = ImageScopeXmlWriter()
    # writer.add_arrows(arrows, arrow_colors)
    # writer.add_boxes(boxes, box_colors)
    # writer.add_contours(contours, contour_colors)
    # writer.add_ellipses(ellipses, ellipse_colors)
    # writer.write('test2.xml')
