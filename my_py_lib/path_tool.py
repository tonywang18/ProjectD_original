'''
路径工具
为了处理麻烦的路径名称问题
'''
import glob
import os
import sys
from typing import Union


def split_file_path(p, base_dir=None):
    '''
    分离完整路径为 文件夹路径，文件基本名，文件后缀名
    如果 base_dir 被指定，那么将分离为 基础文件夹路径，中间文件夹路径，文件基本名，文件后缀名

    例子：
    o = split_file_path('C:/dir/123.txt')
    print(o)
    ('C:/dir', '123', '.txt')

    o = split_file_path('C:/dir/123.txt', 'C:')
    print(o)
    ('C:', '/dir', '123', '.txt')

    :param p:
    :return:
    '''
    dir_path: str
    dir_path, name = os.path.split(p)
    basename, extname = os.path.splitext(name)

    if base_dir is None:
        if dir_path == '':
            dir_path = '.'
        return dir_path, basename, extname

    assert dir_path.startswith(base_dir), f'Error! dir_path:{dir_path} must be startwith base_dir:{base_dir}'
    dir_path = dir_path.removeprefix(base_dir)
    if dir_path == '':
        dir_path = '.'
    return base_dir, dir_path, basename, extname


def insert_basename_end(p, s):
    '''
    在基本名后面插入字符串

    例子：
    s = insert_basename_end('C:/12.txt', '_a')
    print(s)
    C:/12_a.txt

    :param p: 原始文件名
    :param s: 要附加的字符串
    :return:
    '''
    dir_path, basename, extname = split_file_path(p)
    o = dir_path + basename + s + extname
    return o


def replace_extname(p, s):
    '''
    替换掉后缀名

    例子：
    s = replace_extname('C:/12.txt', '.jpg')
    print(s)
    C:/12.jpg

    :param p: 原始文件名
    :param s: 要附加的字符串
    :return:
    '''
    dir_path, basename, extname = split_file_path(p)
    o = dir_path + '/' + basename + s
    return o


def get_home_dir():
    '''
    获得当前用户家目录，支持windows，linux和macosx
    :return:
    '''
    if sys.platform == 'win32':
        homedir = os.environ['USERPROFILE']
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        homedir = os.environ['HOME']
    else:
        raise NotImplemented(f'Error! Not this system. {sys.platform}')
    return homedir


def find_file_by_exts(dirs: Union[list[str], str], exts: Union[list[str], str], recursive=True, replace_backslash=False):
    '''
    检索目录内所有符合要求后缀名的文件
    :param dirs:                可以输入一个或一组文件夹
    :param exts:                可以输入一个或一组后缀名
    :param recursive:           是否检索子文件夹
    :param replace_backslash:   是否自动转换反斜杠到斜杠
    :return:
    '''
    if isinstance(dirs, str):
        dirs = [dirs]
    if isinstance(exts, str):
        exts = [exts]

    files = []
    for dir in dirs:
        for file in glob.glob(f'{dir}/**/*', recursive=recursive):
            if os.path.splitext(file)[1] in exts and os.path.isfile(file):
                files.append(file)

    if replace_backslash:
        files = backslash2slash(files)
    return files


def backslash2slash(s: Union[list[str], str]):
    '''
    转换字符串内所有反斜杠到正斜杠
    :param s:
    :return:
    '''
    if isinstance(s, str):
        s = s.replace('\\', '/')
    else:
        s = [a.replace('\\', '/') for a in s]
    return s


def open2(file, mode='r', *args, **kwargs):
    '''
    创建文件时自动创建相关文件夹
    :param file:
    :param mode:
    :param args:
    :param kwargs:
    :return:
    '''
    if mode.startswith('w'):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    return open(file, mode, *args, **kwargs)
