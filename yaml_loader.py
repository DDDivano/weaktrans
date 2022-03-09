#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import yaml
import logger


class YamlLoader(object):
    def __init__(self, yml):
        try:
            with open(yml) as f:
                self.yml = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            print(e)
        self.logger = logger.Logger(__class__.__name__)

    def __str__(self):
        return str(self.yml)

    def get_case_info(self, case_name):
        self.logger.get_log().info("get ->{}<- case profile".format(case_name))
        return self.yml.get(case_name)