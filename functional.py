#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : functional.py
@Author: Chen Yanzhen
@Date  : 2022/3/11 13:46
@Desc  : 
"""

from visilab import *
from contextlib import contextmanager
import torch
import time
import numpy as np

WINDOW_SIZE = (1200, 600)

viewer = LabViewer(WINDOW_SIZE, title="Viewer")


VisibleField.Window_Name = "Inspector"
field = VisibleField("Field")


@TreeNode.as_builder
class LabInspector(TreeNode):

    def __init__(self, label="Config"):
        super().__init__(label)
        self.getters = {}
        self.setters = {}

        self.time_cost = {}
        self.cached_time_cost = {}
        self.counter = 0

    def inspect(self, label, *args, **kwargs):
        import imgui
        self.changed, new_value = ImguiTool.input_json("Config", self.value(), default_new_item=None)
        for k in self.setters:
            self.setters[k](new_value[k])

        self.calculate_time_cost()
        for k, v in self.cached_time_cost.items():
            imgui.text(f"{k}: {v:12.10f} s")

    def calculate_time_cost(self):
        if not viewer.lab_ui.pause.opened:
            self.counter += 1
            if self.counter >= 10:
                for k in self.time_cost:
                    self.cached_time_cost[k] = self.time_cost[k] / 10
                    self.time_cost[k] = 0
                self.counter = 0

    def watch(self, k, getter, setter):
        self.getters[k] = getter
        self.setters[k] = setter

    def value(self):
        return {
            k: self.getters[k]()
            for k in self.getters
        }


inspector = LabInspector()


def input_config(target, attrib=None):
    if attrib is None:
        for c in target.__class__.__mro__:
            for k in c.__dict__:
                if not k.startswith("__"):
                    if isinstance(c.__dict__[k], (int, float, bool, dict, list)):
                        input_config(target, k)
    else:
        inspector.watch(attrib, lambda: getattr(target, attrib), lambda v: setattr(target, attrib, v))


@contextmanager
def time_recorded(tag, verbose=False):
    start = time.time()
    yield
    end = time.time()

    if tag not in inspector.time_cost:
        inspector.time_cost[tag] = 0
    inspector.time_cost[tag] += end - start

    if verbose:
        print(f"[{tag}]", end - start)


def visualize_field(positions, scalars={}, vectors={}, graphs={}):
    """

    :param positions: 2D or 3D points (N x 2 or N x 3)
    :param scalars: Dict [ tag : a vector of scalars (N) ]
    :param vectors: Dict [ tag : 2D or 3D vectors (N x 2 or N x 3) ]
    :param graphs: Dict [ tag : pair of list of start/end indices ]
    :return:
    """

    def arr_2d_to_3d(arr):
        if len(arr.shape) == 2:
            if arr.shape[1] == 2:
                arr = np.concatenate([np.array(arr), np.ones_like(np.array(arr)[:, 0:1]) * 0.5], 1)
        return arr

    def prepare(arr, tag=None):
        if tag is not None:
            # change to dict
            if not isinstance(arr, dict):
                if isinstance(arr, torch.Tensor):
                    arr = arr.cpu().detach().numpy()
                else:
                    arr = np.array(arr)
                arr = arr_2d_to_3d(arr)
                arr = {"tag": np.array(arr)}

        elif isinstance(arr, torch.Tensor):
            arr = arr.cpu().detach().numpy()
            arr = arr_2d_to_3d(arr)
        else:
            arr = arr_2d_to_3d(np.array(arr))

        return arr

    positions = prepare(positions)
    scalars = prepare(scalars, "Scalar")
    vectors = prepare(vectors, "Vector")
    graphs = prepare(graphs, "Graph")

    field.update("", positions)
    field.scalar("Position", np.ones_like(positions[:, 0]))
    for k, v in scalars.items():
        field.scalar(k, v)
    for k, v in vectors.items():
        field.vector(k, v)
    for k, v in graphs.items():
        field.graph(k, v, None)


__init_func = None


def init_func(f):
    global __init_func
    __init_func = f
    return f


def step_func(f):
    viewer.lab_ui.pause.switch()

    if inspector is not None:
        viewer.inputting("Inspector", inspector)

    @viewer.coroutine
    def roll():
        __init_func()
        yield field.ui_window()
        while True:
            f()
            yield field.displayables()


class Simulatable:

    dt = 0.001

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs
        input_config(self)

    def step(self, dt):
        raise NotImplementedError

    def run(self):

        @init_func
        def init():
            self.__init__(*self.__args, **self.__kwargs)

        @step_func
        def step():
            self.step(self.dt)




