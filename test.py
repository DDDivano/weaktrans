#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import paddle
from wt.yaml_loader import YamlLoader
from wt.weaktrans import WeakTrans
from wt.logger import Logger

# loading yaml
yml = YamlLoader("./test.yml")
# choose case
jit_case = WeakTrans(yml.get_case_info("conv2d"), logger=Logger)
# layer = jit_case.get_layer("paddle")
# loss = paddle.mean(layer)
# print(loss)


# @paddle.jit.to_static
# def func(data):
#     layer = eval(jit_case.get_jit("paddle")[0])(**data)
#     return layer
#
#
# func(jit_case.get_jit("paddle")[1])


