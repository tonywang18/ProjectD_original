'''
本库与 my_cpp_lib 的 io_utils.hpp 成配对关系
'''
import io
import numpy as np


def _type_size(T):
	'''
	获得类型的大小
	:param T:
	:return:
	'''
	if T in {np.int32, np.uint32, np.float32}:
		T_size = 4
	elif T in {np.int16, np.uint16, np.float16}:
		T_size = 2
	elif T in {np.int8, np.uint8, np.bool_}:
		T_size = 1
	elif T in {np.int64, np.uint64, np.float64}:
		T_size = 8
	else:
		raise RuntimeError('Error! Unsupported type.')
	return T_size


def Size(f: io.FileIO):
	'''
	获得文件大小
	:param f: 
	:return: 
	'''
	cur_pos = f.tell()
	f.seek(0, io.SEEK_END)
	size = f.tell()
	f.seek(cur_pos, io.SEEK_SET)
	return size


def RemainingSize(f: io.FileIO):
	'''
	获得文件剩余大小
	:param f: 
	:return: 
	'''
	pos_begin = f.tell()
	f.seek(0, io.SEEK_END)
	size = f.tell() - pos_begin
	f.seek(pos_begin)
	return size


def ReadAll(f: io.FileIO, size_limit=np.uint64(-1)):
	'''
	从文件流里读入一些字节
	:param f:
	:param size_limit:
	:return:
	'''
	size = RemainingSize(f)
	if size > size_limit:
		raise RuntimeError("ReadAll read size bigger than size_limit.")
	buf = f.read(size)
	if len(buf) != size:
		raise RuntimeError("Read data len small than size.")
	return buf


def Read(f: io.FileIO, size):
	'''
	从文件流里读入一些字节
	:param f:
	:param size:
	:return:
	'''
	buf = f.read(size)
	if len(buf) != size:
		raise RuntimeError("Read data len small than size.")
	return buf


def Write(f: io.FileIO, buf: bytes):
	'''
	往文件流里写入一些字节
	:param f:
	:param buf:
	:return:
	'''
	f.write(buf)


def ReadType(f: io.FileIO, T: np.dtype):
	'''
	读入一个T类的模版函数，注意T类必须为简单类型，不能包含指针或引用之类的东西，间接包含也不行。
	:param f:
	:param T:
	:return:
	'''
	T_size = _type_size(T)
	buf = Read(f, T_size)
	return np.frombuffer(buf, T, 1)


def WriteType(f: io.FileIO, data: np.ndarray):
	'''
	保存一个T类的模版函数，注意T类必须为简单类型，不能包含指针或引用之类的东西，间接包含也不行。
	:param f:
	:param data:
	:return:
	'''
	T_size = _type_size(data.dtype)
	buf = data.tobytes()
	assert len(buf) == T_size
	Write(f, buf)


def ReadString(f: io.FileIO, encoding='utf8'):
	'''
	读入一个u8string
	:param f:
	:param encoding:
	:return:
	'''
	size = ReadType(f, np.uint32)
	buf = Read(f, size)
	s = buf.decode(encoding=encoding)
	return s


def WriteString(f: io.FileIO, s: str, encoding='utf8'):
	'''
	保存一个string
	:param f:
	:param s:
	:param encoding:
	:return:
	'''
	size = np.uint32(len(s))
	WriteType(f, size)
	buf = s.encode(encoding=encoding)
	Write(f, buf)


def ReadVector(f: io.FileIO, T: np.dtype):
	'''
	读入一个vector的模版函数，注意T类必须为简单类型，不能包含指针或引用之类的东西，间接包含也不行。
	如果需要用此函数读入复杂类型，请特化此模板函数
	:param f:
	:param T:
	:return:
	'''
	n = ReadType(f, np.uint32)
	size = n * _type_size(T)
	buf = Read(f, size)
	return np.frombuffer(buf, T, n)


def WriteVector(f: io.FileIO, v: np.ndarray):
	'''
	保存一个vector的模版函数，注意T类必须为简单类型，不能包含指针或引用之类的东西，间接包含也不行。
	如果需要用此函数写入复杂类型，请特化此模板函数
	:param f:
	:param v:
	:return:
	'''
	n = len(v)
	size = np.uint32(n * _type_size(v.dtype))
	WriteType(f, size)
	Write(f, v.tobytes())


def ReadStringVector(f: io.FileIO, encoding='utf8'):
	'''
	特化vector<string>的模板
	ReadVector函数偏特化失败，使用新名字，其他以后再说
	:param f:
	:param v:
	:param encoding:
	:return:
	'''
	n = ReadType(f, np.uint32)
	v = []
	for _ in range(n):
		v.append(ReadString(f, encoding=encoding))
	return v


def WriteStringVector(f: io.FileIO, v: list, encoding='utf8'):
	'''
	特化vector<string>的模板
	WriteVector函数偏特化失败，使用新名字，其他以后再说
	:param f:
	:param v:
	:param encoding:
	:return:
	'''
	n = np.uint32(len(v))
	WriteType(f, n)
	for s in v:
		WriteString(f, s, encoding=encoding)


def ReadClassVector(f: io.FileIO, T):
	'''
	读入一个T类的vector的模版函数，要求T类必须实现和可见Read，Write函数
	:param f:
	:param v:
	:return:
	'''
	n = ReadType(f, np.uint32)
	v = []
	for _ in range(n):
		a = T()
		a.Read(f)
		v.append(a)
	return v


def WriteClassVector(f: io.FileIO, v: list):
	'''
	写入一个T类的vector的模版函数，要求T类必须实现和可见Read，Write函数
	:param f:
	:param v:
	:return:
	'''
	size = np.uint32(len(v))
	WriteType(f, size)
	for a in v:
		a.Write(f)
