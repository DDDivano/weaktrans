#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
jittrans.py
"""
import os
import logging
import numpy as np
import paddle
import paddle.inference as paddle_infer
from paddle.static import InputSpec
from wt.weaktrans import WeakTrans
from inspect import isclass
import shutil


def naive_func(a, in_params, func):
    """用于动转静的方法"""
    layer = eval(func)(**a, **in_params)
    return layer


class BuildClass(paddle.nn.Layer):
    """
    用于动转静的nn.Layer
    """

    def __init__(self, in_params, func):
        super(BuildClass, self).__init__()
        self.func = eval(func)(**in_params)

    def forward(self, inputs):
        """
        forward
        """
        x = self.func(*inputs)
        return x


class BuildFunc(paddle.nn.Layer):
    """
    用于动转静的nn.Layer
    """

    def __init__(self, in_params, func):
        super(BuildFunc, self).__init__()
        self.func = eval(func)
        self._params = in_params

    def forward(self, inputs):
        """
        forward
        """
        x = self.func(**inputs, **self._params)
        return x


class BuildJitClass(paddle.nn.Layer):
    """
    用于动转静的nn.Layer
    """

    def __init__(self, in_params, func):
        super(BuildJitClass, self).__init__()
        self.func = eval(func)(**in_params)

    @paddle.jit.to_static
    def forward(self, inputs):
        """
        forward
        """
        x = self.func(*inputs)
        return x


class BuildJitFunc(paddle.nn.Layer):
    """
    用于动转静的nn.Layer
    """

    def __init__(self, in_params, func):
        super(BuildJitFunc, self).__init__()
        self.func = eval(func)
        self._params = in_params

    @paddle.jit.to_static
    def forward(self, inputs):
        """
        forward
        """
        x = self.func(**inputs, **self._params)
        return x


def sort_intensor(input_dict):
    """对输入进行排序，构建一个新的输入list"""
    inputs_key = sorted(input_dict.keys())
    inputs_value = []
    for k in inputs_key:
        inputs_value.append(input_dict[k])
    return inputs_value


def mk_dict_spec(input_dict):
    """根据输入的dict, 生成InputSpec"""
    input_spec_dict = {}
    for k, v in input_dict.items():
        v_shape = v.shape
        v_dtype = v.dtype
        if k in ['x', 'y', 'x1', 'x2', 'data', 'data0', 'theta']:
            v_shape[0] = None
        print('{} shape is {}'.format(k, v_shape))
        input_spec_dict[k] = InputSpec(shape=v_shape, dtype=v_dtype, name=k)
    return input_spec_dict


def mk_list_spec(input_dict):
    """根据输入的dict, 生成InputSpec"""
    input_spec_list = []
    for k, v in input_dict.items():
        v_shape = v.shape
        v_dtype = v.dtype
        if k in ['x', 'y', 'x1', 'x2', 'data', 'data0', 'theta']:
            v_shape[0] = None
        print('{} shape is {}'.format(k, v_shape))
        input_spec_list.append(InputSpec(shape=v_shape, dtype=v_dtype, name=k))
    return input_spec_list


class Framework(object):
    PADDLE = "paddle"
    TORCH = "pytorch"


class JitTrans(WeakTrans):
    def __init__(self, case, logger, default_type=np.float32, seed=None, ):
        super(JitTrans, self).__init__(case, logger, default_type=np.float32, seed=None, )
        self.atol = 1e-5
        self.rtol = 1e-6
        self.in_tensor, self.in_params, self.func = self.get_func_params("paddle")

        if isclass(eval(self.func)):
            self.func_type = 'class'
        else:
            self.func_type = 'func'

        self.jit_save_path = os.path.join(os.getcwd(), 'jit_save')

        if os.path.exists(self.jit_save_path):
            shutil.rmtree(self.jit_save_path)
        os.mkdir(self.jit_save_path)

    def get_func_params(self, framework):
        """获取输入tensor，api参数，api名称"""
        func = self.get_func(framework)
        params = self.get_input(framework)
        in_tensor = dict()
        in_params = dict()
        for k, v in params.items():
            if isinstance(v, paddle.Tensor):
                in_tensor[k] = v
            else:
                in_params[k] = v

        return in_tensor, in_params, func

    def init_test_object(self, method):
        if method == 'BuildClass' or method == 'BuildClassWithInputSpec':
            # 仅实例化一次，防止多次实例化后，因为随机种子不固定导致多个结果值res不相等
            obj = BuildClass(self.in_params, self.func)
            obj.eval()
        elif method == 'BuildFunc' or method == 'BuildFuncWithInputSpec':
            obj = BuildFunc(self.in_params, self.func)
            obj.eval()
        elif method == "naive_func":
            obj = naive_func

        return obj

    def mk_exp(self, obj, method):
        """获取动态图结果"""
        if method == 'BuildClass' or method == 'BuildClassWithInputSpec':
            inputs_value = sort_intensor(self.in_tensor)
            # print('inputs_value is: ', inputs_value)
            exp = obj(inputs_value)
        elif method == 'BuildFunc' or method == 'BuildFuncWithInputSpec':
            exp = obj(self.in_tensor)
        elif method == "naive_func":
            exp = obj(self.in_tensor, self.in_params, self.func)
            # print('net is: ', naive_func)
            # print(exp)
        return exp

    def mk_res(self, obj, method):
        """获取静态图结果"""
        if method == 'BuildClass':
            inputs_value = sort_intensor(self.in_tensor)
            jit_obj = paddle.jit.to_static(obj)
            res = jit_obj(inputs_value)
        elif method == 'BuildClassWithInputSpec':
            inputs_value = sort_intensor(self.in_tensor)
            input_spec = mk_list_spec(self.in_tensor)
            jit_obj = paddle.jit.to_static(obj, input_spec=[input_spec])
            res = jit_obj(inputs_value)
        elif method == 'BuildFunc':
            jit_obj = paddle.jit.to_static(obj)
            res = jit_obj(self.in_tensor)
        elif method == 'BuildFuncWithInputSpec':
            input_spec = mk_dict_spec(self.in_tensor)
            jit_obj = paddle.jit.to_static(obj, input_spec=[input_spec])
            res = jit_obj(self.in_tensor)
        elif method == "naive_func":
            jit_obj = paddle.jit.to_static(obj)
            res = jit_obj(self.in_tensor, self.in_params, self.func)
            # print('jit is: ', jit)
            # print(res)
        return res

    def jit_save(self, obj, method):
        """
        动转静保存模型，两种情况:
        1. 当self.func为class时，继承nn.Layer构建BuildClass类，
        此时paddle.jit.save将输出 api.pdiparams, api.pdiparams.info, api.pdmodel三个文件
        后续既会比对测试paddle.jit.load加载api.pdmodel的输出结果，也会比对预测库推理部署api.pdiparams和api.pdmodel的结果。
        2. 当self.func为func时，直接构建def函数方法，
        此时paddle.jit.save只输出 api.pdmodel一个文件
        后续只会比对测试paddle.jit.load加载api.pdmodel的输出结果
        """
        if method == 'BuildClass':
            inputs_value = sort_intensor(self.in_tensor)
            jit_obj = paddle.jit.to_static(obj)
            exp = jit_obj(inputs_value)  # 此行用于构建inputSpec,不可删除
            paddle.jit.save(jit_obj, path=os.path.join(self.jit_save_path, self.get_func("paddle")))
        elif method == 'BuildClassWithInputSpec':
            # inputs_value = sort_intensor(self.in_tensor)
            input_spec = mk_list_spec(self.in_tensor)
            paddle.jit.save(obj, path=os.path.join(self.jit_save_path, self.get_func("paddle")),
                            input_spec=[input_spec])
        elif method == 'BuildFunc':
            jit_obj = paddle.jit.to_static(obj)
            exp = jit_obj(self.in_tensor)  # 此行用于构建inputSpec,不可删除
            paddle.jit.save(jit_obj, path=os.path.join(self.jit_save_path, self.get_func("paddle")))
        elif method == 'BuildFuncWithInputSpec':
            input_spec = mk_dict_spec(self.in_tensor)
            # jit_obj = paddle.jit.to_static(obj, input_spec=[input_spec])
            paddle.jit.save(obj, path=os.path.join(self.jit_save_path, self.get_func("paddle")), input_spec=[input_spec])
        elif method == "naive_func":
            jit_obj = paddle.jit.to_static(obj)
            # print("jit save start-----")
            # print("in_tensor is: {}".format(self.in_tensor))
            # print("in_params is: {}".format(self.in_params))
            # print("func is: {}".format(self.func))
            exp = jit_obj(self.in_tensor, self.in_params, self.func)  # 此行用于构建inputSpec,不可删除
            paddle.jit.save(jit_obj, path=os.path.join(self.jit_save_path, self.get_func("paddle")))

    def jit_load(self):
        """paddle.jit.load加载"""
        jit = paddle.jit.load(os.path.join(self.jit_save_path, self.get_func("paddle")))
        inputs_value = sort_intensor(self.in_tensor)
        res = jit(*inputs_value)
        # print('jit load is: ', jit)
        # print(res)
        return res

    def infer_load(self):
        """paddle预测库加载，只会用于测试nn.Layer"""
        config = paddle_infer.Config(os.path.join(self.jit_save_path, self.get_func("paddle") + '.pdmodel'),
                                     os.path.join(self.jit_save_path, self.get_func("paddle") + '.pdiparams'))
        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        input_list = sort_intensor(self.in_tensor)

        for i, name in enumerate(input_names):
            input_handle = predictor.get_input_handle(name)
            input_handle.copy_from_cpu(input_list[i].numpy())

        predictor.run()
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        infer_res = output_handle.copy_to_cpu()

        # print('***' * 30)
        # print(infer_res)
        return infer_res

    def jit_run(self):
        """测试运行流程"""
        if self.func_type == 'class':
            self.test_method(method='BuildClass')
            # self.test_method(method='BuildClassWithInputSpec') # 需要进一步排查，涉及到某些有问题的api
        else:
            self.test_method(method='naive_func')
            self.test_method(method='BuildFunc')
            self.test_method(method='BuildFuncWithInputSpec')

    def test_method(self, method):
        obj = self.init_test_object(method)
        exp = self.mk_exp(obj=obj, method=method)
        print('exp is: ', exp)
        to_static_res = self.mk_res(obj=obj, method=method)
        print('to_static_res is: ', to_static_res)
        print('exp - to_static_res is: ', exp[0] - to_static_res[0])
        self.jit_save(obj=obj, method=method)
        load_res = self.jit_load()
        print('load_res is: ', load_res)
        print('load_res[0] is: ', load_res[0])
        # print('exp - load_res is: ', exp - load_res)
        compare(to_static_res, exp, self.atol, self.rtol)
        compare(load_res, exp, self.atol, self.rtol)
        # 若是nn.Layer组网且有参数pdiparams的情况，则需要进一步测试推理部署结果
        if self.func_type == 'class' and os.path.exists(
                os.path.join(self.jit_save_path, self.get_func("paddle") + '.pdiparams')):
            infer_res = self.infer_load()
            print('infer_res is: {}'.format(infer_res))
            # print('infer_exp is: {}'.format(exp[1].shape))
            if isinstance(exp, (list, tuple)):
                exp = exp[0]
            compare(infer_res, exp, self.atol, self.rtol)
        print('{} method {} test complete~~'.format(self.func, method))
        del obj


def compare(result, expect, delta=1e-10, rtol=1e-10):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :return:
    """
    if isinstance(expect, paddle.Tensor) or type(expect) == np.ndarray:
        # if type(expect) == list:
        #     expect = expect[0]
        if isinstance(result, paddle.Tensor):
            result = result.numpy()
        if isinstance(expect, paddle.Tensor):
            expect = expect.numpy()
        # print("-----" * 30)
        # print(result)
        # print("=====" * 30)
        # print(expect)
        res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # 出错打印错误数据
        if res is False:
            diff = abs(result - expect)
            logging.error("Output has diff! max diff: {}".format(np.amax(diff)))
        if result.dtype != expect.dtype:
            logging.error(
                "Different output data types! res type is: {}, and expect type is: {}".
                    format(result.dtype, expect.dtype))
        assert res
        assert result.shape == expect.shape
        assert result.dtype == expect.dtype
    elif type(expect) == list or type(expect) == tuple:
        for i in range(len(expect)):
            if isinstance(result, (np.generic, np.ndarray)) or isinstance(result, paddle.Tensor):
                if i > 0:
                    break
            # if isinstance(result[i], (np.generic, np.ndarray)) or isinstance(result, paddle.Tensor):
                compare(result, expect[i], delta, rtol)

            else:
                compare(result[i], expect[i], delta, rtol)
    else:
        raise Exception('expect is unknown data struction in compare_tool!!!')
