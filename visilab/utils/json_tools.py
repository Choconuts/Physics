#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : json_tools.py
@Author: Chen Yanzhen
@Date  : 2020/12/13 20:21
@Desc  : 
"""

from enum import Enum, EnumMeta
from typing import Union, Dict, List, Tuple, Iterable, Type
from functools import singledispatch


def identical(x):
    return x


class JsonFree:

    pass


class AutoRegister(type):

    def __new__(mcs, cls_name, bases, attr_dict):
        new_class = super(AutoRegister, mcs).__new__(mcs, cls_name, bases, attr_dict)
        JsonCode.decoder(new_class)
        return new_class


class EnumAutoRegister(EnumMeta):

    def __new__(mcs, cls_name, bases, attr_dict):
        new_class = super(EnumAutoRegister, mcs).__new__(mcs, cls_name, bases, attr_dict)
        JsonCode.decoder(new_class)
        return new_class


class JsonCode:

    __classname_key__ = "__classname__"
    __object_key__ = "__value__"

    _global_decoder_map = {}

    @staticmethod
    def decoder(cls):
        JsonCode._global_decoder_map[cls.__name__] = cls
        return cls

    encoder = singledispatch(identical)

    @staticmethod
    def build_decoder_map(*cls):
        return {c.__name__: c for c in cls}

    @classmethod
    def decoder_map(cls) -> Dict[str, Type]:
        return cls.build_decoder_map(*cls.decoder_list())

    @classmethod
    def decoder_list(cls) -> Union[List[Type], Tuple[Type], Iterable[Type]]:
        return []

    @classmethod
    def empty(cls):
        return cls()

    def serialize(obj):
        d = {}
        if isinstance(obj, JsonFree):
            return d
        d.update(vars(obj))
        for k, v in list(d.items()):
            if "__" in k or isinstance(v, JsonFree):
                del d[k]
                continue
            elif isinstance(v, JsonCode):
                d[k] = v.serialize()
        d[JsonCode.__classname_key__] = type(obj).__name__
        return d

    @classmethod
    def from_loaded_state(cls, state):
        if issubclass(cls, Enum):
            return cls(state["_value_"])
        obj = cls.empty()
        for key, value in state.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def common_unserialize(d):
        cls_map = JsonCode._global_decoder_map
        if JsonCode.__classname_key__ in d:
            cls = cls_map[d.pop(JsonCode.__classname_key__, None)]
            if isinstance(cls, type):
                if issubclass(cls, JsonCode):
                    return cls.from_loaded_state(d)
            else:
                if JsonCode.__object_key__ in d:
                    return cls(d[JsonCode.__object_key__])
                return cls(**d)
        else:
            return d

    @classmethod
    def unserialize(cls, d):
        cls_map = cls._global_decoder_map.copy()
        cls_map.update(cls.decoder_map())
        if JsonCode.__classname_key__ in d:
            del d[JsonCode.__classname_key__]

        if issubclass(cls, Enum):
            obj = cls(d["_value_"])
        else:
            obj = cls.empty()
        for key, value in d.items():
            if isinstance(value, dict) and JsonCode.__classname_key__ in value:
                sub_cls = cls_map[value.pop(JsonCode.__classname_key__, None)]
                assert issubclass(sub_cls, JsonCode)
                value = sub_cls.unserialize(value)
            setattr(obj, key, value)
        return obj

    @property
    def cls_serializer(self):
        def serialize(obj):
            enc = self.__class__.encoder
            obj = enc(obj)
            try:
                vars(obj)
            except TypeError as e:
                return obj
            return self.__class__.serialize(obj)
        return serialize

    @staticmethod
    def register_delegate(cls, encoder, decoder):

        def wrapped_encoder(x):
            return {
                JsonCode.__classname_key__: cls.__name__,
                JsonCode.__object_key__: encoder(x),
            }

        JsonCode.encoder.register(cls)(wrapped_encoder)
        JsonCode._global_decoder_map[cls.__name__] = decoder


class AutoJsonCode(JsonCode, metaclass=AutoRegister):

    pass


class EnumJsonCode(JsonCode, Enum, metaclass=EnumAutoRegister):

    pass

