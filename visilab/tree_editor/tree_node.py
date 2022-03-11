#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : tree_node.py
@Author: Chen Yanzhen
@Date  : 2020/12/14 20:27
@Desc  : 
"""


from visilab.utils import *
from .actions import *
from .base import TreeNodeBase
import imgui


class InputAction(NodeAction):

    def __init__(self, name, node, default=None, description=None, button_width=120):
        super().__init__(name, node, True)
        self.description = description
        self.button_width = button_width
        self.default = default

    def input_value(self) -> bool:
        raise NotImplementedError

    def on_confirmed(self):
        raise NotImplementedError

    def gui(self, label, *args, **kwargs):
        imgui.text(self.description + "\n");
        imgui.separator()
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
        _ = self.input_value()
        imgui.pop_style_var()

        if imgui.button("Cancel", self.button_width, 0):
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Confirm", self.button_width, 0):
            imgui.close_current_popup()
            self.on_confirmed()
            self.changed = True
        imgui.set_item_default_focus()


class ChangeNameAction(InputAction):

    def __init__(self, node):
        super().__init__("Rename", node, node.label, "Input a new name")

    def input_value(self) -> bool:
        changed, self.default = imgui.input_text("name", self.default, 50)

    def on_confirmed(self):
        self.node.label = self.default


class NewNodeAction(NodeAction):

    def __init__(self, node):
        super().__init__("New", node, True)

    def gui(self, label, *args, **kwargs):
        builders = {
            "Empty": TreeNode
        }
        builders.update(TreeNode.factory_builders)
        for k in builders:
            if imgui.menu_item(k)[0]:
                self.node.add(builders[k](label=k))
                self.changed = True


class TreeNode(TreeNodeBase, Inputable):

    factory_builders = {}

    @staticmethod
    def as_builder(cls):
        if isinstance(cls, type):
            TreeNode.factory_builders[cls.__name__] = cls
        if isinstance(cls, str):
            def wrap(c):
                TreeNode.factory_builders[cls] = c
                return c
            return wrap
        return cls

    def __init__(self, label="Undefined"):
        super().__init__(label)

    def gui(self, label, *args, default_open=False, **kwargs):

        if self._gui_tree_node(label, default_open):
            self.inspect(label, *args, **kwargs)
            for c in self:
                k = c.label
                with ImguiTool.id_scope_in(k):
                    if self.deletable(k):
                        if imgui.button("x"):
                            del self[k]
                            self.changed = True
                            break
                        imgui.same_line()
                    self.changed = c.input(k, *args, **kwargs)
            imgui.tree_pop()

        if self.changed:
            for c in self.children:
                self._rename_child_for_distinction(c)

    def inspect(self, label, *args, **kwargs):
        pass

    def deletable(self, k):
        if k in self:
            return True
        return False

    def actions(self):
        return [ChangeNameAction(self), NewNodeAction(self)]

    def right_menu_popup(self):

        for action in self._node_actions__:
            if action.need_param:
                if imgui.begin_menu(action.name):
                    action.input(action.name)
                    imgui.end_menu()
            else:
                if imgui.menu_item(action.name)[0]:
                    action.input(action.name)
            imgui.separator()

    def _gui_tree_node(self, label, default_open):
        op = imgui.tree_node(label, imgui.TREE_NODE_DEFAULT_OPEN if default_open else 0)
        with ImguiTool.id_scope_in(label, op):
            if imgui.is_item_clicked(1):
                imgui.open_popup("right menu")
                self._node_actions__ = list(self.actions())
                self.__input_label = self.label

            if imgui.begin_popup("right menu"):
                self.right_menu_popup()
                imgui.end_popup()

        return op

    def __repr__(self):
        return self.label + "\n  ".join(map(repr, self.children))

