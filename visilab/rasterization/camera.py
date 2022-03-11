#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : camera.py
@Author: Chen Yanzhen
@Date  : 2020/12/13 15:25
@Desc  : 
"""

import numpy as np
import pyrr
import imgui
from visilab.render_api import Inputable
from visilab.utils import *


class Camera:

    GRAVITY = np.array([0, -1, 0])

    def __init__(self, fov=60., near=0.001, far=1000.):
        self.position = np.array([0., 0, 3])
        self.target = np.array([0., 0, 0])
        self.up = np.array([0., 1, 0])
        self.aspect_ratio = 1.
        self.fov = fov
        self.near_clip = near
        self.far_clip = far

        self.translation = np.array([0., 0, 0])
        self.rotation = np.array([0., 0, 0])
        self.scale = np.array([1., 1, 1])
        self.rot_after_trans = False

    def aspect(self, w, h=1.):
        if h != 0:
            self.aspect_ratio = w / h
        return self

    def clip(self, near, far):
        self.near_clip = near
        self.far_clip = far
        return self

    def look_at(self, target, up=None, sense_gravity=True):
        self.target = target
        if up is not None:
            self.up = up
        else:
            # make camera balanced
            direction = self.target - self.position
            if sense_gravity is True:
                left = np.cross(direction, self.GRAVITY)
            else:
                left = np.cross(self.up, direction)
            up = np.cross(direction, left)
            up_length = np.linalg.norm(up, 2)
            if up_length == 0:
                self.up = np.array([1, 0, 0])
            else:
                self.up = up / up_length
        self.up = np.array([0., 1, 0])
        return self

    def mvp(self):
        projection_matrix = pyrr.matrix44.create_perspective_projection(
            self.fov,
            self.aspect_ratio,
            self.near_clip,
            self.far_clip
        )

        view_matrix = pyrr.matrix44.create_look_at(
            pyrr.vector3.create(*self.position),  # camera position
            pyrr.vector3.create(*self.target),  # camera target
            pyrr.vector3.create(*self.up)  # camera up vector
        )

        if self.rot_after_trans:
            model_matrix = \
                           pyrr.matrix44.create_from_translation(self.translation) @ \
                           pyrr.matrix44.create_from_z_rotation(self.rotation[2]) @ \
                           pyrr.matrix44.create_from_y_rotation(self.rotation[1]) @ \
                           pyrr.matrix44.create_from_x_rotation(self.rotation[0]) @ \
                           pyrr.matrix44.create_from_scale(self.scale)
        else:
            model_matrix = pyrr.matrix44.create_from_scale(self.scale) @ \
                           pyrr.matrix44.create_from_z_rotation(self.rotation[2]) @ \
                           pyrr.matrix44.create_from_y_rotation(self.rotation[1]) @ \
                           pyrr.matrix44.create_from_x_rotation(self.rotation[0]) @ \
                           pyrr.matrix44.create_from_translation(self.translation)

        return model_matrix @ view_matrix @ projection_matrix


class PYRRTransform(Inputable):

    def __init__(self, translation=[0., 0., 0.], rotation=[0., 0., 0.], scale=[1., 1., 1.]):
        self.translation = np.array(translation)
        self.rotation = np.array(rotation)
        self.scale = np.array(scale)

    def mat(self):
        model_matrix = pyrr.matrix44.create_from_scale(self.scale) @ \
                       pyrr.matrix44.create_from_z_rotation(self.rotation[2]) @ \
                       pyrr.matrix44.create_from_y_rotation(self.rotation[1]) @ \
                       pyrr.matrix44.create_from_x_rotation(self.rotation[0]) @ \
                       pyrr.matrix44.create_from_translation(self.translation)
        return model_matrix

    def apply(self, vertices):
        n = len(vertices)
        hoco = np.concatenate([vertices, np.ones([n, 1])], axis=1).reshape([n, 4, 1])
        hoco = self.mat().transpose() @ hoco
        hoco /= hoco[:, 3:]
        return hoco[:, :3, 0]

    def gui(self, label, *args, **kwargs):
        def vec3(tag, value):
            import itertools
            value = itertools.chain(value, [0.05])
            self.changed, values = imgui.drag_float3(tag, *value)
            return np.array(values)

        if imgui.tree_node(label, imgui.TREE_NODE_DEFAULT_OPEN):
            self.translation = vec3("position", self.translation)
            self.rotation = vec3("rotation", self.rotation)
            self.scale = vec3("scale", self.scale)
            imgui.tree_pop()

    ATTRIBS = ["translation", "rotation", "scale"]

    def from_dict(self, state):
        for k, v in state.items():
            if isinstance(v, (list, tuple)):
                v = np.array(v)
            self.__setattr__(k, v)
        return self

    def to_dict(self):
        out = {}
        for k in self.ATTRIBS:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                v = list(v)
            out[k] = v
        return out


class InputableCamera(Camera, Inputable):

    def __init__(self, fov=60., near=0.001, far=1000.):
        super().__init__(fov, near, far)
        self.anchor_rotation = self.rotation
        self.anchor_scale = self.scale
        self.anchor_position = self.position
        self.drag_speed = 0.01
        self.enable_drag = True

    def drag_update(self):
        if not self.enable_drag:
            return
        if imgui.is_mouse_dragging(0):
            delta = imgui.get_mouse_drag_delta(0,)
            self.rotation = self.anchor_rotation + np.array([delta[1], delta[0], 0]) * -self.drag_speed
        else:
            self.anchor_rotation = self.rotation
        if imgui.is_mouse_dragging(1):
            delta = imgui.get_mouse_drag_delta(1, )
            self.scale = self.anchor_scale * np.clip(np.exp(delta[0] * self.drag_speed * 0.5), 0.1, 10)
        else:
            self.anchor_scale = self.scale
        if imgui.is_mouse_dragging(2):
            delta = imgui.get_mouse_drag_delta(2, )
            self.translation = self.anchor_position + np.array([delta[0], -delta[1], 0]) * self.drag_speed * 0.5
        else:
            self.anchor_position = self.translation

    def gui(self, label, *args, **kwargs):
        with ImguiTool.id_scope_in(label):
            self.changed, self.enable_drag = imgui.checkbox("Enable Dragging", self.enable_drag)
            self.changed, self.rot_after_trans = imgui.checkbox("Trans then Rot", self.rot_after_trans)
            self.changed, self.drag_speed = imgui.input_float("Drag Speed", self.drag_speed)

            def vec3(tag, value):
                import itertools
                value = itertools.chain(value, [0.05])
                self.changed, values = imgui.drag_float3(tag, *value)
                return np.array(values)

            if imgui.tree_node("Transform", imgui.TREE_NODE_DEFAULT_OPEN):
                self.translation = vec3("position", self.translation)
                self.rotation = vec3("rotation", self.rotation)
                self.scale = vec3("scale", self.scale)
                imgui.tree_pop()

            if imgui.tree_node("Camera Parameters", imgui.TREE_NODE_DEFAULT_OPEN):
                self.position = vec3("position", self.position)
                self.look_at(vec3("target", self.target))
                self.up = vec3("up", self.up)
                changed, values = imgui.input_float2("Near/Far", self.near_clip, self.far_clip)
                changed, self.fov = imgui.slider_float("FOV", self.fov, 20, 120)
                if changed:
                    self.clip(*values)
                imgui.tree_pop()

