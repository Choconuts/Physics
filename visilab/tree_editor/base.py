#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : base.py
@Author: Chen Yanzhen
@Date  : 2020/12/14 22:42
@Desc  : 
"""

from visilab.utils import *


class TreeNodeBase(AutoJsonCode):

    def __init__(self, label):
        self.label = label
        self.children = []

    def __getitem__(self, item):
        if isinstance(item, int):
            if item < len(self.children):
                return self.children[item]
            else:
                raise IndexError
        else:
            for c in self.children:
                if c.label == item:
                    return c
            return None

    def __setitem__(self, key, value):
        assert not isinstance(key, int)
        value.label = key
        self.add(value)

    def __delitem__(self, key):
        self.children.remove(self[key])

    def __len__(self):
        return len(self.children)

    def add(self, node):
        self._rename_child_for_distinction(node)
        self.children.append(node)

    def _rename_child_for_distinction(self, node):
        base = node.label
        idx = 0
        try:
            ss = base.split(" ")
            if self[base] is not None and len(ss) >= 2:
                idx = int(ss[-1])
                base = " ".join(ss[:-1])
        except Exception as e:
            pass
        while self[node.label] is not None and self[node.label] != node:
            idx += 1
            node.label = base + " %d" % idx

    def all_children(self):
        from itertools import chain
        res = chain(*[chain([c], c.all_children()) for c in self.children])
        return res
