#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : viewer_lab.py
@Author: Chen Yanzhen
@Date  : 2020/12/16 17:50
@Desc  : 
"""


from visilab.rasterization.viewer import *
from collections.abc import Iterable
from visilab.tree_editor.scene import GeoNode
from enum import Enum
from functools import wraps
import json
import shutil


class Switcher(Inputable):

    def __init__(self, closed_label=None):
        self.opened = True
        self.closed_label = closed_label

    def switch(self, to=None):
        if to is None:
            self.opened = not self.opened
        else:
            self.opened = to

    def gui(self, label, *args, **kwargs):
        if imgui.button(label if not self.opened else self.closed_label, 120):
            self.changed = True
            self.switch()


class LabUI(Inputable):

    def __init__(self, tag="lab"):
        self.pause = Switcher("Start/Resume")
        self.show_windows = False
        self.show_camera = False
        self.restart = True
        self.cache_dir = "/tmp/" + tag
        self.use_cache = False
        self.color = [.98, .99, .99]
        self.speed = 10.
        self._timing = 0

    def should_step(self):
        if self.pause.opened:
            return False
        self._timing += self.speed
        if self._timing >= 10:
            self._timing -= 10
            return True
        else:
            return False

    def gui(self, label, *args, **kwargs):
        self.changed = self.pause.input("Pause")
        if imgui.button("Stop All", 120):
            self.changed = True
            self.restart = True
        if imgui.button("Show Windows", 120):
            self.changed = True
            self.show_windows = True
        if imgui.button("Show Camera", 120):
            self.changed = True
            self.show_camera = True
        self.changed, self.color = imgui.color_edit3("Background", *self.color)
        self.changed, self.speed = imgui.slider_float("speed", self.speed, 1., 10.,)
        self.changed, self.cache_dir = imgui.input_text("Cache", self.cache_dir, 100)
        self.changed, self.use_cache = imgui.checkbox("Use Cache", self.use_cache)
        if imgui.button("Clear Cache"):
            shutil.rmtree(os.path.join(self.cache_dir, "cache"))

    def get_cached_file(self, i):
        file = os.path.join(self.cache_dir, "cache", "%06d.json" % i)
        dirn = os.path.dirname(file)
        if not os.path.exists(dirn):
            os.makedirs(dirn)
        return file


class LabInstruction(EnumJsonCode):

    Pause = 0


class LabViewer(Viewer):

    def __init__(self,
                 size=(800, 600),
                 title="Viewer",
                 fov=80.,
                 view_center=[0.5, 0.5, 0.5],
                 camera_ui=True,
                 auto_run=True):
        super().__init__(size, title, fov, camera_ui)
        self.auto_run = auto_run
        self.view_center = view_center
        self.camera.translation -= np.array(self.view_center)
        # self.camera.scale *= 2
        self.camera.rot_after_trans = True
        self.camera.fov = 20
        self.camera.position[2] = 5
        self.lab_ui = LabUI(title)
        self.background_color = [0.9, .9, .9]

        axis = GeoNode("axis")
        axis.as_lines(np.zeros([3, 3]), np.eye(3), np.eye(3))
        self.displaying(axis)
        self.inputting("Lab", self.lab_ui)
        self.coroutine_runners = []
        self.coroutine_starters = []
        self.coroutine_obj_marks = []

    def coroutine(self, runner):
        runner = self.cached_runner(runner)
        self.coroutine_runners.append(runner().__iter__())
        self.coroutine_starters.append(runner)
        if self.auto_run:
            self.run()
        return runner

    def restart_all(self):
        self.coroutine_runners.clear()
        self.coroutine_runners = [runner().__iter__() for runner in self.coroutine_starters]

    def cached_runner(self, runner):

        class JsonTuple(AutoJsonCode):

            def __init__(self, data=None):
                self.data = data

        def encoder(x):
            assert isinstance(x, np.ndarray)
            return x.tolist()

        def decoder(d):
            return np.array(d)

        JsonCode.register_delegate(np.ndarray, encoder, decoder)

        @wraps(runner)
        def wrapper(*args, **kwargs):
            iterator = runner(*args, **kwargs).__iter__()
            i = 0
            while True:
                cache = self.lab_ui.get_cached_file(i)
                i += 1
                if not self.lab_ui.use_cache:
                    yield next(iterator)
                else:
                    if os.path.exists(cache):
                        with open(cache, "r") as fp:
                            obj = json.load(fp, object_hook=JsonCode.common_unserialize)
                            frame = obj.data
                    else:
                        frame = next(iterator)
                        obj = JsonTuple(frame)
                        try:
                            with open(cache, "w") as fp:
                                json.dump(obj, fp, default=obj.cls_serializer)
                        except Exception:
                            os.remove(cache)
                    yield frame

        return wrapper

    def auto_view_obj(self, obj):
        if isinstance(obj, (tuple, list)):
            if len(obj) >= 2 and isinstance(obj[1], str):
                self.view(obj[0], ui_window_name=obj[1])
            else:
                [self.auto_view_obj(o) for o in obj]
        elif isinstance(obj, Displayable):
            self.view(obj, ui_window_name=None)
            self.coroutine_obj_marks.append(obj)
        elif isinstance(obj, Inputable):
            self.view(obj, ui_window_name="Debug")
            self.coroutine_obj_marks.append(obj)
        elif isinstance(obj, Iterable):
            [self.auto_view_obj(o) for o in obj]
        elif isinstance(obj, LabInstruction):
            if obj == LabInstruction.Pause:
                self.lab_ui.pause.switch()
        else:
            raise NotImplementedError

    def step(self):
        super().step()

        if self.lab_ui.show_windows:
            self.open_all_ui_windows()
            self.lab_ui.show_windows = False
        if self.lab_ui.show_camera:
            self.camera_ui = True
            self.lab_ui.show_camera = False
        if self.lab_ui.restart:
            self.restart_all()
            self.lab_ui.restart = False
        self.background_color = self.lab_ui.color

        if self.lab_ui.should_step():
            for obj in self.coroutine_obj_marks:
                self.stop_viewing(obj)
            self.coroutine_obj_marks.clear()
            for frame in map(next, self.coroutine_runners):
                self.auto_view_obj(frame)

    def async_coroutine(self, runner=None):

        def decorate(runner):

            from threading import Thread

            tmp = []

            def task():
                for result in runner():
                    if tmp.__len__() == 0:
                        tmp.append(result)
                    else:
                        tmp[0] = result

            def func():
                thr = Thread(target=task)
                thr.start()
                while True:
                    yield tmp

            self.coroutine(func)
            return runner

        if runner is None:
            return decorate

        return decorate(runner)

