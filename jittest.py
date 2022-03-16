#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
main_test
"""
from wt.yaml_loader import YamlLoader
from wt.jittrans import JitTrans


# loading yaml
def test():
    yml = YamlLoader("test.yml")
    jit_case = JitTrans(yml.get_case_info("conv2d"), logger=yml.logger)
    jit_case.jit_run()

