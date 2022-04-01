#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import random

import paddle
import numpy as np
from wt.logger import logger
from inspect import isclass


class Framework(object):
    PADDLE = "paddle"
    TORCH = "pytorch"


class WeakTrans(object):
    def __init__(self, case, default_type=np.float32, seed=None, ):
        np.random.seed(seed)
        self.case = case
        self.default_type = default_type
        self.params = dict()
        self.logger = logger
        # desc
        self.logger.get_log().info(self.case.get("desc", "没有描述"))
        self.logger.get_log().info("default_type: {}".format(self.default_type))
        # 加载
        self._run()

    def __str__(self):
        # 打印方法
        return str(self.params)

    def _run(self):
        """
        test run
        :return:
        """
        self._generate_params(Framework.PADDLE)

    def get_layer(self, framework):

        # lazy func
        func = self.get_func(framework)
        params = self.get_input(framework)
        if isclass(eval(func)):
            data = params["data"]
            del(params["data"])
            obj = eval(func)(**params)
            return obj(data)
        else:
            return eval(func)(**params)

    def get_jit(self, framework):
        # lazy func
        func = self.get_func(framework)
        params = self.get_input(framework)
        return func, params


    def get_input(self, framework):
        # 获取参数输入
        return self.params[framework]

    def get_func(self, framework):
        # 获取测试方法
        return self.case[framework]["api_name"]

    def _randtool(self, dtype, low, high, shape):
        """
        np random tools
        """
        if dtype == "int":
            return np.random.randint(low, high, shape)
        elif dtype == "int32":
            return np.random.randint(low, high, shape)
        elif dtype == "int64":
            return np.random.randint(low, high, shape)
        elif dtype == "float":
            return low + (high - low) * np.random.random(shape)
        elif dtype == "float32":
            return low + (high - low) * np.random.random(shape)
        elif dtype == "float64":
            return low + (high - low) * np.random.random(shape)
        else:
            assert False, "dtype is not supported"

    def _generate_params(self, framework):
        """
        生成参数
        :return:
        """
        params = self.case[framework]["params"]
        kwargs = {}
        kwargs_for_log = {}
        for key, value in params.items():
            kwargs[key] = self._params_transform(key, value)
            kwargs_for_log[key] = kwargs[key]
            if isinstance(value, dict) and value.get("type") == "Tensor":
                del(kwargs_for_log[key])
                # tensor需要根据不同的框架转换
                if framework == Framework.PADDLE:
                    kwargs[key] = paddle.to_tensor(kwargs[key]).astype(value.get("dtype", self.default_type))
                elif framework == Framework.TORCH:
                    # 预留竞品转换
                    pass
        self.logger.get_log().info("Case非Tensor参数设置：{}".format(kwargs_for_log))
        self.params[framework] = kwargs

    def _params_transform(self, key, value):
        """
        参数转换函数，此处可以增加多种参数转换方式，和yaml结构对应
        :param value:
        :return:
        首先判断是否为dict，如果是，进行复杂类型转换。如果不是直接进行赋值操作
        在dict分支中，主要需要处理基本数据类型与tensor。
        分为random和非random两种情况。
        需要处理的数据类型包括：
        Number（数字）
        String（字符串）
        List（列表）
        Tuple（元组）
        Set（集合）
        Dictionary（字典）
        Tensor （向量）
        """
        data = None
        if isinstance(value, dict):
            # 参数可靠性校验
            self._param_check(key, value)
            if value.get("random", False):
                # 若开启random即进行自动数据生成,默认关闭
                if value.get("type") == "Tensor":
                    # Tensor自动生成
                    data_range = value.get("range", [-1, 1])
                    assert isinstance(data_range, list) and len(data_range) == 2
                    data = self._randtool(value.get("dtype", "float"),
                                          data_range[0],
                                          data_range[1],
                                          value.get("shape"))
                # elif
            else:
                data = value.get("value")
        else:
            data = value
        return data

    def _param_check(self, key, param_info):
        if param_info.get("random", False):
            # 开启random状态
            if param_info.get("type") == "Tensor":
                # 检测Tensor类型
                assert "shape" in param_info.keys(), "shape 参数必选"
                if "range" not in param_info.keys():
                    self.logger.get_log().warning("{} 未设置range参数，默认取值[-1, 1]".format(key))
            elif param_info.get("type") == "Number":
                # 检测Number类型
                if "range" not in param_info.keys():
                    self.logger.get_log().warning("{} 未设置range参数，默认取值[-1, 1]".format(key))
                if "dtype" not in param_info.keys():
                    self.logger.get_log().warn("{} 未设置dtype参数，默认类型int".format(key))
            elif param_info.get("type") == "String":
                # 检测String类型
                assert "range" in param_info.keys() and isinstance(param_info.get("range"), list), \
                    "range 参数必选且必须是List"
            elif param_info.get("type") == "List":
                pass
            elif param_info.get("type") == "Set":
                pass
            elif param_info.get("type") == "Dict":
                pass
        else:
            # 关闭random状态
            if "value" not in param_info.keys():
                self.logger.get_log().error("[{}]关闭配置自动生成时，需要设置value参数".format(key))
                assert False
