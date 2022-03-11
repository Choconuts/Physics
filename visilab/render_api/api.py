#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : api.py
@Author: Chen Yanzhen
@Date  : 2020/12/15 23:14
@Desc  : 
"""


class GeoLab:

    def __init__(self,
                 window_size: int = (800, 600),
                 window_title: str = "GeoLab",
                 ):
        pass

    def routine(self, runner):
        """ decorate a function as background coroutine and trace it """
        raise NotImplementedError

    def scatter(self, points,):
        pass

    def runner(self, lab):
        pass

