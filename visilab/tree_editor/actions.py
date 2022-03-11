#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : actions.py
@Author: Chen Yanzhen
@Date  : 2020/12/14 22:39
@Desc  : 
"""


from visilab.render_api import Inputable
from .base import TreeNodeBase
from visilab.utils import JsonFree


class NodeAction(Inputable, JsonFree):

    def __init__(self, name, node: TreeNodeBase, need_param=False):
        self.name = name
        self.need_param = need_param
        self.node = node

    def gui(self, label, *args, **kwargs):
        raise NotImplementedError


