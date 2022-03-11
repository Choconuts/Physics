#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : mesh_obj.py
@Author: Chen Yanzhen
@Date  : 2020/12/13 16:03
@Desc  : 
"""

from OpenGL.GL import *
import numpy as np
import copy
import re
from .json_tools import JsonFree


class OBJ:
    def __init__(self, fdir, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        self.mtl=None

        material = None
        for line in open(fdir+filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                #v = map(float, values[1:4])
                v=[ float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                #v = map(float, values[1:4])
                v=[ float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                v = [float(x) for x in values[1:3]]

                self.texcoords.append(v)
            elif values[0] in ('usemtl', 'usemat') and len(values) > 1:
                material = values[1]
            elif values[0] == 'mtllib':
                #print(values[1])
                #self.mtl = MTL(fdir,values[1])
                self.mtl = [fdir,values[1]]
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

    def create_gl_list(self):
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        # glCullFace(GL_BACK)
        # glEnable(GL_CULL_FACE)

        for face in self.faces:
            vertices, normals, texture_coords, material = face

            # mtl = self.mtl[material]
            # if 'texture_Kd' in mtl:
            #     # use diffuse texmap
            #     glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
            # else:
            #     # just use diffuse colour
            #     # print(mtl['Kd'],"----")
            #     glColor(*mtl['Kd'])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()


class Mesh(JsonFree):

    def __init__(self, another_mesh=None):
        if another_mesh is not None:
            self.vertices = np.copy(another_mesh.vertices)
            self.faces = copy.deepcopy(another_mesh.faces)
            self.edges = copy.deepcopy(another_mesh.edges)
            self.bounds = copy.deepcopy(another_mesh.bounds)
        else:
            self.vertices = np.array([])
            self.faces = []
            self.edges = dict()
            self.bounds = dict()
        self.update()

    def load(self, obj_file_path):
        vertices = []
        bound_edges = dict()
        self.faces= []
        g = re.search('[a-zA-Z0-9$\\- ].obj$', obj_file_path)
        if g is None:
            return
        g = g.span(0)
        if obj_file_path is not None:
            obj = OBJ(obj_file_path[0:g[0]], obj_file_path[g[0]:g[1]])
            vertices.extend(obj.vertices)
            self.normal = np.array(obj.normals)
            for f in obj.faces:
                face = [f[0][0] - 1, f[0][1] - 1, f[0][2] - 1]
                self.faces.append(face)
                for i in range(3):
                    edge = tuple([face[i], face[(i + 1) % 3]])
                    self.add_edge(edge[0], edge[1])
                    self.add_edge(edge[1], edge[0])

                    # record bound edges
                    if edge not in bound_edges:
                        bound_edges[(edge[1], edge[0])] = 1
                    else:
                        bound_edges.pop(edge)
            for e in bound_edges:
                if e[0] not in self.bounds:
                    self.bounds[e[0]] = 1
                if e[1] not in self.bounds:
                    self.bounds[e[1]] = 1
        self.vertices = np.array(vertices)
        self.update()
        return self

    def from_vertices(self, vertices, faces, no_bounds_info=False):
        bound_edges = dict()
        for face in faces:
            self.faces.append(np.copy(face).tolist())
            for i in range(3):
                edge = tuple([face[i], face[(i + 1) % 3]])
                self.add_edge(edge[0], edge[1])
                self.add_edge(edge[1], edge[0])

                # record bound edges
                if edge not in bound_edges:
                    bound_edges[(edge[1], edge[0])] = 1
                else:
                    bound_edges.pop(edge)
        if not no_bounds_info:
            for e in bound_edges:
                if e[0] not in self.bounds:
                    self.bounds[e[0]] = 1
                if e[1] not in self.bounds:
                    self.bounds[e[1]] = 1
        self.vertices = np.array(vertices)
        self.update()
        return self

    def set_vertices(self, vertices):
        self.vertices = np.copy(vertices)

    def add_edge(self, a, b):
        if a not in self.edges:
            self.edges[a] = []
        if b not in self.edges[a]:
            self.edges[a].append(b)

    def save(self, path, save_normals=False):
        with open(path, 'w') as fp:
            for v in self.vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in np.array(self.faces) + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
            if self.normal is not None and save_normals:
                for v in self.normal:
                    fp.write('vn %f %f %f\n' % (v[0], v[1], v[2]))
        return self

    def to_vertex_buffer(self):
        buffer = []
        for face in self.faces:
            triangle = []
            norm = self.face_norm(face)
            for v in face:
                triangle.extend(self.vertices[v].tolist())
                # triangle.extend(norm.tolist())
                triangle.extend(self.normal[v])
            buffer.extend(triangle)
        return np.array(buffer, dtype="float32")

    def face_norm(self, face):

        def get(i):
            return self.vertices[face[i]] - self.vertices[face[i - 1]]

        n = np.cross(get(1), get(2))

        return n / (np.linalg.norm(n) + 0.0000001)

    def compute_vertex_normal(self):
        v_f = dict()
        self.vertex_face_map = dict()
        i = 0
        for f in self.faces:
            fn = self.face_norm(f)
            for v in f:
                if v not in v_f:
                    self.vertex_face_map[v] = []
                    v_f[v] = 0
                self.vertex_face_map[v].append(i)
                v_f[v] += fn
            i += 1
        for v in v_f:
            v_f[v] /= np.linalg.norm(v_f[v]) + 0.0000001
        # self.normal = v_f
        res = sorted(v_f.items(), key=lambda i: i[0])
        for r in range(len(res)):
            res[r] = res[r][1]
        self.normal = np.array(res)

    def update(self):
        self.compute_vertex_normal()

    def update_normal_only(self):

        v_f = dict()
        for f in self.faces:
            fn = self.face_norm(f)
            for v in f:
                if v not in v_f:
                    v_f[v] = 0
                v_f[v] += fn
        for v in v_f:
            v_f[v] /= np.linalg.norm(v_f[v]) + 0.0000001
        # self.normal = v_f
        res = sorted(v_f.items(), key=lambda i: i[0])
        for r in range(len(res)):
            res[r] = res[r][1]
        self.normal = np.array(res)

    def set_normal(self):
        res = sorted(self.normal.items(), key=lambda i:i[0])
        for r in range(len(res)):
            res[r] = res[r][1]
        res = np.array(res)
        self.normal = res


