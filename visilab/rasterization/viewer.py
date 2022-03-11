#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : viewer.py
@Author: Chen Yanzhen
@Date  : 2020/12/15 17:57
@Desc  : 
"""

from .window import *
from .camera import *
from .pipeline import *
from .renderer import *
from typing import Union


class Viewer(GLFWWindow):

    def __init__(self, size=(800, 600), title="Viewer", fov=80., camera_ui=True):
        super().__init__(*size, title=title)
        self.camera = InputableCamera(fov, 0.01, 1000)
        self.camera_ui = camera_ui
        self.background_color = [0., 0.3, 0.5]
        self.render_layers = {}
        self.viewed_inputables = {}
        self.viewed_displayables = {}
        self.__window_opened = {}
        self.renderers = {}

    def register_render(self, priority=0):

        def decorator(f):
            if priority not in self.render_layers:
                self.render_layers[priority] = []
            self.render_layers[priority].append(f().__iter__())
            return f

        return decorator

    def step(self):
        for k in sorted(self.render_layers):
            ends = []
            for i, g in enumerate(self.render_layers[k]):
                try:
                    g.__next__()
                except StopIteration:
                    ends.append(i)
            for i in reversed(ends):
                self.render_layers.pop(i)

    def display(self):
        glClearColor(*self.background_color, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.aspect(self.w, self.h)
        self.step()

        for obj, tag in self.viewed_displayables.items():
            self.show_displayable(obj, tag)
        self.flush_renderer_cache()
        for obj, tag in self.viewed_inputables.items():
            self.show_inputable(obj, tag)

        if self.camera_ui:
            ex, op = imgui.begin("View", True)
            if ex and op:
                self.camera.input("Camera")
            if not op:
                self.camera_ui = False
            imgui.end()
        if not imgui.is_window_focused(imgui.HOVERED_ANY_WINDOW):
            self.camera.drag_update()

    def view(self, *objs: Union[Displayable, Inputable], ui_window_name=None):
        for obj in objs:
            if isinstance(obj, Inputable) and ui_window_name is not None:
                self.viewed_inputables[obj] = ui_window_name
            if isinstance(obj, Displayable) and ui_window_name is None:
                self.viewed_displayables[obj] = ui_window_name

    def inputting(self, label, *objs: Inputable):
        return self.view(*objs, ui_window_name=label)

    def displaying(self, *objs: Displayable, ui_window_name=None):
        return self.view(*objs)

    def stop_viewing(self, *objs: Union[Displayable, Inputable]):
        for obj in objs:
            if obj in self.viewed_inputables:
                self.viewed_inputables.__delitem__(obj)
            if obj in self.viewed_displayables:
                self.viewed_displayables.__delitem__(obj)

    def open_all_ui_windows(self):
        self.__window_opened.clear()

    def show_inputable(self, obj: Inputable, tag):
        if tag not in self.__window_opened or self.__window_opened[tag]:
            ex, self.__window_opened[tag] = imgui.begin(tag, True, )
            obj.input(tag)
            imgui.end()

    def show_displayable(self, obj: Displayable, tag):
        if isinstance(obj, MeshLikeDisplayable):
            if obj.can_render():
                self.display_mesh_obj(obj)

    def flush_renderer_cache(self):
        for renderer in self.renderers.values():
            cache = self.get_renderer_cache(renderer)
            renderer.render(*cache, camera=self.camera)
            cache.clear()

    @staticmethod
    def get_renderer_cache(renderer):
        if not hasattr(renderer, "cache__"):
            renderer.cache__ = []
        return renderer.cache__

    def display_mesh_obj(self, obj: MeshLikeDisplayable):
        sig = NaiveMeshRenderer.signature(obj)
        if sig not in self.renderers:
            self.renderers[sig] = NaiveMeshRenderer(obj)
        renderer = self.renderers[sig]
        self.get_renderer_cache(renderer).append(obj)

    def run(self):
        self.init()
        glEnable(GL_DEPTH_TEST)
        self.step()
        self.main_loop()

    UNIFORM_MVP = "transform"

    def set_uniform_mvp(self, shader: Shader, name=UNIFORM_MVP):
        mvp = self.camera.mvp()
        loc = glGetUniformLocation(shader.program, name)
        glUniformMatrix4fv(loc, 1, GL_FALSE, mvp)

