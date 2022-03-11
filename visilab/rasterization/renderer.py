#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : renderer.py
@Author: Chen Yanzhen
@Date  : 2020/12/15 11:43
@Desc  : 
"""


from .window import *
from .camera import *
from .pipeline import *
from typing import Union


class Renderer:

    @classmethod
    def signature(cls, obj: Displayable) -> tuple:
        raise NotImplementedError

    def __init__(self, obj: Displayable):
        self.current_signature = self.signature(obj)

    def render(self, *objs: Displayable, camera: Camera):
        raise NotImplementedError


class NaiveMeshRenderer(Renderer):

    def __init__(self, obj: MeshLikeDisplayable):
        super().__init__(obj)
        self.geo = obj.render_geometry()
        self.lighting = obj.render_lighting()
        self.depth_test = obj.render_occlusion()
        if self.geo == GeoType.Point:
            self.point_width = obj.render_point_width()
        self.buffer = None
        self.shader = None
        self.make_shader()

    def make_shader(self):

        vs = f"""
            #version 330
            in vec3 position;
            in vec3 color;
            uniform mat4 transform;
            out vec3 newColor;
            void main()
            {{
                gl_Position = transform * vec4(position, 1.0f);
                newColor = color;
            }}
            """

        dis = """
            vec2 coord = gl_PointCoord - vec2(0.5);  //from [0,1] to [-0.5,0.5]
            if(length(coord) > 0.5)                  //outside of circle radius?
                discard;
        """

        fs = f"""
            #version 330
            in vec3 newColor;
            out vec4 outColor;
            void main()
            {{
                {dis if self.geo == GeoType.Point else ""}
                outColor = vec4(newColor, 1.0f);
            }}
            """
        self.buffer = Buffer()
        self.shader = Shader(vs, fs)
        self.buffer.layout(self.shader, position=3, color=3)

    def render(self, *objs: MeshLikeDisplayable, camera: Camera):
        if len(objs) == 0:
            return

        vas = None
        eas = None
        last_vn = None

        for obj in objs:
            ea = obj.render_indices()
            va = obj.render_positions()
            try:
                ca = obj.render_colors()
                if ca is None:
                    raise NotImplementedError
            except NotImplementedError:
                ca = np.ones_like(va) * np.array(obj.render_color())
            try:
                va = np.concatenate([va, ca], axis=1)
            except Exception as e:
                raise e

            if vas is None:
                vas = va.flatten()
                eas = ea.flatten()
                last_vn = len(va)
            else:
                eas = np.concatenate([eas, ea.flatten() + last_vn]).astype(np.uint32)
                vas = np.concatenate([vas, va.flatten()])
                last_vn += len(va)

        if self.geo == GeoType.Line:
            glEnable(GL_LINE_SMOOTH)
        if self.geo == GeoType.Point:
            glPointSize(max(self.point_width, 1))
        if self.depth_test:
            glEnable(GL_DEPTH_TEST)
        else:
            glDisable(GL_DEPTH_TEST)

        glUniformMatrix4fv(glGetUniformLocation(self.shader.program, "transform"), 1, GL_FALSE, camera.mvp())

        self.shader.use()
        self.buffer.draw(vas, eas, self.geo)

    @classmethod
    def signature(cls, obj: MeshLikeDisplayable) -> tuple:
        if obj.render_geometry() == GeoType.Point:
            return obj.render_geometry(), obj.render_lighting(), obj.render_occlusion(), obj.render_point_width()
        return obj.render_geometry(), obj.render_lighting(), obj.render_occlusion()



