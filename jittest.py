#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
main_test
"""
from wt.yaml_loader import YamlLoader
from wt.jittrans import JitTrans
from wt.weaktrans import WeakTrans


# loading yaml
def test():
    yml = YamlLoader("nn.yml")
    jit_case = JitTrans(yml.get_case_info("GRUCell"), logger=yml.logger)
    jit_case.jit_run()


yml = YamlLoader("nn.yml")
jit_case = JitTrans(yml.get_case_info("Conv1D"), logger=yml.logger)
jit_case.jit_run()

