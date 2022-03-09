#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import paddle
from yaml_loader import YamlLoader
from weaktrans import WeakTrans

# loading yaml
yml = YamlLoader("./test.yml")
# choose case
jit_case = WeakTrans(yml.get_case_info("add"))
layer = jit_case.get_layer("paddle")
loss = paddle.mean(layer)
print(loss)
