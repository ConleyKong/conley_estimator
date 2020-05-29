#-*- coding:utf-8 -*-
"""
-----------------------------------
    Project Name:    conley_estimator
    File Name   :    tfengine_mnist
    Author      :    Conley.K
    Create Date :    2020/5/29
    Description :    利用tfengine_core中的Estimator构建的mnist分类示例，
    仅需指定data数据源和network_fn网络结构构建方式即可实现模型的训练、测试、导出一条龙
--------------------------------------------
    Change Activity: 
        2020/5/29 18:29 : 
"""
import os
import sys
import warnings  #抑制numpy警告
import numpy as np
import tensorflow as tf

basedir = os.path.abspath(os.path.dirname(__file__) + "/../")
sys.path.append(basedir)
print("appended {} into system path".format(basedir))
warnings.simplefilter(action='ignore', category=FutureWarning)
from engines.core.tfengine_core import classify_estimator

def mnist_network_fn(features, params):
    """ 定义网络的具体结构，无关乎label，只涉及特征的前传网络结构，
    输出的logits不建议使用激活函数，若需要激活可以再logits后单独调用激活函数

    :param features: 网络接受的输入
    :param params: 网络定义参数，比如num_classes,hiden_units等，字典格式
    :return: 输出网络前传后的特征logits等，为提升健壮性，返回dict结构，以支持多字段
    """
    num_classes = params.get("num_classes",10)

    ######################################################
    #               Network Build Ops
    ######################################################
    input_node = tf.identity(features,"inputs")
    x = tf.reshape(input_node, (-1, 28, 28, 1))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
    logits = tf.keras.layers.Dense(num_classes)(x)

    network_fruit = {"logits":logits}
    return network_fruit

def serving_input_receiver_fn(raw_feature=True):
    """ 定义模型导出后，serving的输入值

    :param raw_feature: 是否给入的为原始特征数据
    :return:
    """

    if raw_feature:
        feature_spec = {
            "feature": tf.placeholder(dtype=tf.float32, shape=[None, 28, 28],name="inputs")
        }
        return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    else:  # Example格式输入
        # # 外界给到模型的参数名
        # input_str = tf.placeholder(tf.string, name='inputs')
        # # 对外界给入的参数进行简单解析获取到目标tensor
        # tt = tf.string_split(input_str, ',').values
        # SepalLength = tf.string_to_number([tt[0]], tf.float32)
        #
        # # 组装对外名称与对内名称接口
        # receiver_tensors = {'inputs': input_str}
        # features_spec = {
        #     'SepalLength': SepalLength,
        # }
        # return tf.estimator.export.ServingInputReceiver(features_spec, receiver_tensors)
        raise NotImplementedError


def mnist_estimator(run_type="train",**kwargs):
    """ 基于TF.Estimator构建MNist数据集上的Estimator，
    仅需要在函数中指定数据来源（data），模型名称（model_name）以及网络结构构建函数（network_fn）即可
    当运行模式为export时才需要指定serving_input_receiver_fn为serving提供输入格式

    :param run_type: 模型运行模式，支持train,test,predict,export等模式
    :param kwargs: 其他词典类型给入的参数信息，具体有：
                        predict_keys : list类型，["probs"]或["preds"]或["probs","preds"]，None的话全部输出
                        log_iter_step: int类型，多少步打印一次日志
                        ckpt_step    : 多少步输出一次ckpt
                        max_ckpt_num : 最多保存多少ckpt版本
                        learning_rate: float类型，模型学习初始化学习率
                        num_classes  : int，分类数量
                        min_train_step: int,最小训练步长，对于早停止的最小步数
                        max_train_step: int,最大训练步长
                        patient_step  : int,early stop模式时的容忍指标无进展步数
                        serving_input_receiver_fn: 可选，仅当使用export导出模型时需要配置serving的输入标记

    :return:
    """
    from gastation.MnistEstData import MnistDataset
    data = MnistDataset(basedir + "/../data/mnist/")
    model_name = "mnist-estimator"
    network_fn = mnist_network_fn

    return classify_estimator(data,network_fn,run_type,model_name,**kwargs)

def run():

    mnist_estimator("infer")
    # mnist_estimator("export",serving_input_receiver_fn=serving_input_receiver_fn)




if __name__ == "__main__":
    run()
