#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : pipeline.py
@Author: Chen Yanzhen
@Date  : 2020/12/13 13:46
@Desc  : 
"""

from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from visilab.render_api import *
import numpy as np
import os


class Shader:

    def __init__(self, vs_or_path, fs_or_path):
        def load_shader(shader_file):
            with open(shader_file) as f:
                shader_source = f.read()
            f.close()
            return str.encode(shader_source)

        vert_shader = load_shader(vs_or_path) if os.path.exists(vs_or_path) else vs_or_path
        frag_shader = load_shader(fs_or_path) if os.path.exists(fs_or_path) else fs_or_path

        self.program = compileProgram(compileShader(vert_shader, GL_VERTEX_SHADER),
                                      compileShader(frag_shader, GL_FRAGMENT_SHADER))
        self.use()

    def use(self):
        glUseProgram(self.program)


class Buffer:

    def __init__(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)

    def bind(self):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)

    def draw(self, va=None, ea=None, ele=GeoType.Triangle):
        self.bind()
        if va is not None:
            va = va.astype(np.float32)
            glBufferData(GL_ARRAY_BUFFER, 4 * va.size, va, GL_DYNAMIC_DRAW)
        if ea is not None:
            ea = ea.astype(np.uint32)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * ea.size, ea, GL_DYNAMIC_DRAW)
        ele = {
            GeoType.Point: GL_POINTS,
            GeoType.Line: GL_LINES,
            GeoType.Triangle: GL_TRIANGLES,
        }[ele]
        glDrawElements(ele, len(ea), GL_UNSIGNED_INT, None)

    def layout(self, shader, **attribs):
        glBindVertexArray(self.vao)
        tot_size = sum(attribs.values())
        offset = 0
        for tag, size in attribs.items():
            loc = glGetAttribLocation(shader.program, tag)
            assert loc >= 0
            glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, tot_size * 4, ctypes.c_void_p(offset))
            glEnableVertexAttribArray(loc)
            offset += size * 4


class MeshFilter:

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = vertices
        self.faces = faces

    def id_array(self, geometry=GeoType.Triangle):
        mesh = self.mesh
        n = len(mesh.vertices)
        mesh.faces = np.array(mesh.faces)
        if geometry == GeoType.Point:
            return np.linspace(0, n - 1, n, dtype=np.int).flatten().astype(np.uint32)
        elif geometry == GeoType.Line:
            return mesh.faces[:, (0, 1, 1, 2, 2, 0)].reshape([-1, 2]).flatten().astype(np.uint32)
        return mesh.faces.flatten().astype(np.uint32)

    def vtx_array(self, *attribs):
        return np.hstack(attribs).flatten().astype(np.float32)

    def broadcast(self, attrib, like=None):
        if like is None:
            like = self.mesh.vertices
        return np.broadcast_to(np.array(attrib), like.shape)
