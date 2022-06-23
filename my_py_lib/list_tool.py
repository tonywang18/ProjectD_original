from typing import Iterable
import math


def list_multi_get_with_ids(self: list, ids: Iterable):
    return [self[i] for i in ids]


def list_multi_get_with_bool(self: list, bools: Iterable):
    assert len(self) == len(bools)
    a = [self[i] for i, b in enumerate(bools) if b]
    return a


def list_multi_set_with_ids(self: list, ids: Iterable, items: Iterable):
    assert len(ids) == len(items)
    for _id, item in zip(ids, items):
        self[_id] = item


def list_multi_set_with_bool(self: list, bools: Iterable, items: Iterable):
    assert len(self) == len(bools)
    wait_set_ids = []
    for i, b in enumerate(bools):
        if b:
            wait_set_ids.append(i)
    assert len(wait_set_ids) == len(items)
    for i, item in zip(wait_set_ids, items):
        self[i] = item


def int_list(self: Iterable):
    return [int(i) for i in self]


def float_list(self: Iterable):
    return [float(i) for i in self]


def list_split_by_size(self: Iterable, size: int):
    self = list(self)
    g = []
    i = 0
    while True:
        s = self[i*size: (i+1)*size]
        i+=1
        if len(s) == size:
            g.append(s)
        elif len(s) > 0:
            g.append(s)
            break
        else:
            break
    return g


def list_split_by_group(self: Iterable, n_group: int):
    self = list(self)
    sizes = [int(len(self) / n_group)] * n_group
    for i in range(len(self) % n_group):
        sizes[i] += 1
    g = []
    i = 0
    for s in sizes:
        l = self[i: i+s]
        g.append(l)
        i += s
    return g


def list_group_by_classes(self: Iterable, classes: Iterable):
    cls_uq = list(set(classes))
    d = {}
    for c in cls_uq:
        d[c] = []

    for v, k in zip(self, classes):
        d[k].append(v)

    return d


def list_bools_to_ids(bools):
    ids = []
    for i, b in enumerate(bools):
        if b:
            ids.append(i)
    return ids


def list_del_items_with_ids(self: list, ids: Iterable, copy=False):
    if copy:
        self = self.copy()
    for i in sorted(ids, reverse=True):
        del self[i]
    return self


def list_del_items_with_bools(self: list, bools: Iterable, copy=False):
    assert len(self) == len(bools)
    ids = list_bools_to_ids(bools)
    list_del_items_with_ids(self, ids, copy)
    return self


def list_pop_items_with_ids(self: list, ids: Iterable):
    items = []
    for i in ids:
        items.append(self[i])

    list_del_items_with_ids(self, ids)
    return items


def list_pop_items_with_bools(self: list, bools: Iterable):
    assert len(self) == len(bools)
    ids = list_bools_to_ids(bools)
    items = list_pop_items_with_ids(self, ids)
    return items


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6]
    b = list_multi_get_with_ids(a, [0, 2, 4])
    assert a[0] == b[0] and a[2] == b[1] and a[4] == b[2]

    c = list_multi_get_with_bool(a, [True, False, True, False, False, False])
    assert a[0] == c[0] and a[2] == c[1]

    list_multi_set_with_ids(a, [0, 2], [3, 1])
    assert a[0] == 3 and a[2] == 1

    list_multi_set_with_bool(a, [True, False, True, False, False, False], [1, 3])
    assert a[0] == 1 and a[2] == 3

    b = list_split_by_size(a, 4)
    assert b == [[1, 2, 3, 4], [5, 6]]

    b = list_split_by_group(a, 2)
    assert b == [[1, 2, 3], [4, 5, 6]]

    b = list_group_by_classes(a, [1, 1, 2, 2, 3, 3])
    assert b == {1: [1, 2], 2: [3, 4], 3: [5, 6]}

    a2 = [False, True, False, True]
    b = list_bools_to_ids(a2)
    assert b == [1, 3]

    a2 = [1,2,3,4,5,6]
    b = list_del_items_with_ids(a2, [4,2,5])
    assert a2 == [1, 2, 4] == b

    a2 = [1,2,3,4,5,6]
    b = list_del_items_with_bools(a2, [False, False, True, False, True, False])
    assert a2 == [1, 2, 4, 6] == b

    a2 = [1,2,3,4,5,6]
    b = list_pop_items_with_ids(a2, [4,2,5])
    assert a2 == [1, 2, 4] and b == [5, 3, 6]

    a2 = [1,2,3,4,5,6]
    b = list_pop_items_with_bools(a2, [False, False, True, False, True, False])
    assert a2 == [1, 2, 4, 6] and b == [3, 5]
