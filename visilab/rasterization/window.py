#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : window.py
@Author: Chen Yanzhen
@Date  : 2020/12/13 13:45
@Desc  : 
"""

import glfw
from OpenGL.GL import *
import imgui
from imgui.integrations.glfw import GlfwRenderer


class GLFWWindow:

    def __init__(self, *size, title="GL"):
        self.window = None
        self.w, self.h = size
        self.title = title
        self.imgui_impl = None

    @staticmethod
    def make_window(w, h, title):
        if not glfw.init():
            raise Exception("init glfw failed!")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        window = glfw.create_window(w, h, title, None, None)
        if not window:
            glfw.terminate()
            return None
        return window

    def display(self):
        raise NotImplementedError

    def init(self):
        imgui.create_context()
        self.window = self.make_window(self.w, self.h, self.title)
        glfw.make_context_current(self.window)
        self.imgui_impl = GlfwRenderer(self.window)

    def main_loop(self):
        while not glfw.window_should_close(self.window):
            self.w, self.h = glfw.get_window_size(self.window)
            glViewport(0, 0, self.w, self.h)
            glfw.poll_events()
            self.imgui_impl.process_inputs()
            imgui.new_frame()

            if self.display is not None:
                if self.display():
                    break

            imgui.render()
            self.imgui_impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

        self.imgui_impl.shutdown()
        glfw.terminate()

