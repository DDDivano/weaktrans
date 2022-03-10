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

cases = yml.get_all_case_name()

for i in cases:
    jit_case = WeakTrans(case=yml.get_case_info(i), logger=yml.logger)
    layer = jit_case.get_layer("paddle")
    loss = paddle.mean(layer)
    yml.logger.get_log().info(loss)
