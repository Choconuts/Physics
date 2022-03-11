#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : imgui_tool.py
@Author: Chen Yanzhen
@Date  : 2020/12/13 15:27
@Desc  : 
"""


import imgui
from contextlib import contextmanager
from enum import Enum
import json


@contextmanager
def id_scope(nid, unsuccessful=False):
    if not unsuccessful:
        imgui.push_id(nid)
    yield None
    if not unsuccessful:
        imgui.pop_id()


@contextmanager
def id_out_scope(nid, successful=True):
    if successful:
        imgui.pop_id()
    yield None
    if successful:
        imgui.push_id(nid)


class HparamType(Enum):
    Int = 'int'
    Float = 'float'
    String = 'str'
    Bool = 'bool'
    Dict = 'dict'
    List = 'list'


def is_basetype(value):
    return isinstance(value, (int, str, float, bool))


def input_basetype(label, value):
    if type(value) == int:
        return imgui.input_int(label, value)
    elif isinstance(value, float):
        if True:
            v = str(value)
            c, new_s = imgui.input_text(label, v, 20)
            try:
                return c, float(new_s)
            except ValueError:
                return c, value
        else:
            return imgui.input_float(label, value, 0, 0, '%f')
    elif isinstance(value, bool):
        return imgui.checkbox(label, value)
    elif isinstance(value, str):
        return imgui.input_text(label, value, 100)
    else:
        raise TypeError(f'{label}: {value} is not basetype.')


def input_unknowntype(label):
    if imgui.button(label):
        imgui.open_popup('Choose Type' + label)

    d = {'int': int, 'float': float, 'bool':bool, 'str': str, 'list': list, 'dict': dict}
    t = None
    if imgui.begin_popup('Choose Type' + label):
        for e in HparamType:
            if imgui.selectable(e.name)[1]:
                t = d[e.value]()
        imgui.end_popup()

    return t is not None, t


def is_jsoncollection(value):
    if is_basetype(value) or value is None:
        return False
    try:
        json.dumps(value)
        return True
    except TypeError as e:
        return False


@contextmanager
def item_width(width):
    imgui.push_item_width(width)
    yield None
    imgui.pop_item_width()


def input_jsoncollection(label, value, default=None):
    if isinstance(value, list):
        new_value = []
        new_value.extend(value)
        if imgui.tree_node(label):
            c, length = imgui.input_int('len', len(value))
            vv = []
            for i in range(length):
                vi = default
                if i < len(value):
                    vi = value[i]
                if is_basetype(vi):
                    cc, vi = input_basetype(f'{i}', vi)
                elif is_jsoncollection(vi):
                    cc, vi = input_jsoncollection(f'{i}', vi, default)
                elif vi is None:
                    cc, vi = input_unknowntype(f'new item {i}')
                c = c or cc
                vv.append(vi)
            imgui.tree_pop()
            return c, vv

        imgui.same_line()
        imgui.text(f'({len(value)} items)')
        return False, new_value
    if isinstance(value, dict):
        if imgui.tree_node(label):
            new_value = {}
            c = False
            for i, (k, vi) in enumerate(value.items()):

                # key
                try:
                    with item_width(40):
                        cc, k = imgui.input_text(f'k{i}', k, 50)
                    c = c or cc
                except TypeError:
                    raise TypeError('Integer key is not allowed.')
                imgui.same_line()

                # value
                if is_basetype(vi):
                    cc, vi = input_basetype(f'v{i}', vi)
                elif is_jsoncollection(vi):
                    cc, vi = input_jsoncollection(f'v{i}', vi, default)
                elif vi is None:
                    cc, vi = input_unknowntype(f'new v{i}')
                c = c or cc
                imgui.same_line()
                with id_scope(k):
                    if not imgui.button('-'):
                        new_value[k] = vi
                    else:
                        c = True
            if imgui.button('+'):
                if '' not in new_value:
                    new_value[''] = None
                else:
                    for i in range(10000):
                        if str(i) not in new_value:
                            new_value[str(i)] = None
                            break
                c = True
            imgui.tree_pop()
            return c, new_value
        imgui.same_line()
        imgui.text(f'({len(value)} items)')
        return False, value
    else:
        raise TypeError(f'{label}: {value} is not json collection.')
    return False, None


def is_enum(value):
    return isinstance(value, Enum)


def input_enum(label, value):
    assert isinstance(value, Enum)
    v = value
    enum_class = type(v)
    if imgui.button(v.name, 200):
        imgui.open_popup(enum_class.__name__)
    imgui.same_line()
    imgui.text(label)
    newv = v
    if imgui.begin_popup(enum_class.__name__):
        for i, enumv in vars(enum_class).items():
            if not i.startswith('_'):
                if imgui.selectable(i)[0]:
                    newv = enumv
        imgui.end_popup()
        return newv != v, newv
    return False, newv


class ImguiTool:

    id_scope_in = id_scope
    id_scope_out = id_out_scope
    width_scope = item_width

    @staticmethod
    def input_json(label, value, default_new_item=None):
        if is_basetype(value):
            return input_basetype(label, value)
        elif is_jsoncollection(value):
            return input_jsoncollection(label, value, default_new_item)
        elif is_enum(value):
            return input_enum(label, value)
        else:
            raise NotImplementedError

    @staticmethod
    def input_enum(label, value):
        return input_enum(label, value)

    _any_window_focused = False

    @staticmethod
    def start_new_frame():
        ImguiTool._any_window_focused = False

    @staticmethod
    def check_any_window_focused():
        if imgui.is_window_focused():
            ImguiTool._any_window_focused = True

    @staticmethod
    def is_any_window_focused():
        return ImguiTool._any_window_focused

    @staticmethod
    def input_vars(obj, except_from=None, custom_input_prefix="input_"):
        d = {}
        d.update(vars(obj))
        if except_from is not None:
            for k in vars(except_from):
                if k in d:
                    del d[k]
        changed = False
        for k, v in list(d.items()):
            if k.startswith("_"):
                del d[k]
                continue
            else:
                custom = custom_input_prefix + k
                if hasattr(obj, custom):
                    c, v = getattr(obj, custom)(k, v)
                else:
                    try:
                        c, v = ImguiTool.input_json(k, v)
                    except NotImplementedError:
                        c = False
                if c:
                    setattr(obj, k, v)
                    changed = True
        return changed

    @staticmethod
    def input_color(label, color_as_list):
        if len(color_as_list) == 3:
            changed, values = imgui.color_edit3(label, *color_as_list)
        elif len(color_as_list) == 4:
            changed, values = imgui.color_edit3(label, *color_as_list)
        else:
            raise NotImplementedError
        for i, v in enumerate(values):
            color_as_list[i] = v
        return changed
