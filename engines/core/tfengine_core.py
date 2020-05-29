#-*- coding:utf-8 -*-
"""
-----------------------------------
    Project Name:    conley_estimator
    File Name   :    core
    Author      :    Conley.K
    Create Date :    2020/5/28
    Description :    构建深度学习的TF核心基础模型
--------------------------------------------
    Change Activity: 
        2020/5/28 10:23 : 基于https://www.jianshu.com/p/82cf413b899c 整理
        参考：https://juejin.im/post/5b9f805fe51d450e8c34ce8b构建keras+Estimator模型
        可参考 https://mikito.mythsman.com/post/5da44e775ed28235d7573581/进行多loss优化
=============================================================
     Estimator引擎优势：
    1. 学习流程：Estimator 封装了对机器学习不同阶段的控制，可以专注于对网络结构的控制，
        用户不再需要不断的为新机器学习任务重复编写训练、评估、预测的代码。
    2. 网络结构：Estimator 的网络结构是在 model_fn 中独立定义的，
        用户创建的任何网络结构都可以在 Estimator 的控制下进行机器学习。
        这可以允许用户很方便的使用别人定义好的 model_fn。
        ---------------------------------------------------------------
        ** model_fn模型函数必须要有features, mode两个参数，
        可自己选择加入labels（可以把label也放进features中）。
        最后要返回特定的tf.estimator.EstimatorSpec()。
        模型有三个阶段都共用的正向传播部分，和由mode值来控制返回不同tf.estimator.EstimatorSpec的三个分支。
        ---------------------------------------------------------------
    3. 数据导入：Estimator 的数据导入也是由 input_fn 独立定义的。
        例如，用户可以非常方便的只通过改变 input_fn 的定义，来使用相同的网络结构学习不同的数据。

    4. 预测分支，将想要预测的值写进字典里面：
        # 创建predictions字典，里面写进所有你想要在预测值输出的数值
        # 隐藏层的数值也可以，这里演示了输出所有隐藏层层结果。
        # 字典的key是模型，value给的是对应的tensor
        predictions = {
            "image": features['image'],
            "conv1_out": conv1,
            "pool1_out": pool1,
            "conv2_out": conv2,
            "pool2_out": pool2,
            "pool2_flat_out": pool2_flat,
            "dense1_out": dense1,
            "logits": logits,
            "classes": tf.argmax(input=logits, axis=1),
            "labels": features['label'],
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        # 当mode为tf.estimator.ModeKeys.PREDICT时，我们就让模型返回预测的操作
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    5. 训练分支需要loss和train_op操作符。
        # 训练和评估时都会用到loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=features['label'], logits=logits)
        # 训练分支
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            train_op = optimizer.minimize(
            loss=loss,
            # global_step用于记录训练了多少步
            global_step=tf.train.get_global_step())
            # 返回的tf.estimator.EstimatorSpec根据
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    6. 评估分支
        # 注意评估的时候，模型和训练时一样，是一个循环的loop，不断累积计算评估指标。
        # 其中有两个局部变量total和count来控制
        # 把网络中的某个tensor结果直接作为字典的value是不好用的
        # loss的值是始终做记录的，eval_metric_ops中是额外想要知道的评估指标
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=features['label'],
                                                           predictions=predictions["classes"])}
        # 不好用：eval_metric_ops = {"probabilities": predictions["probabilities"]}，可以在hook中打印结果
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    7. 各种hook
        * 日志输出hook，在训练或评估的循环中，每50次print出一次字典中的数值
            tensors_to_log = {"probabilities": "softmax_tensor"}
            logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)
        * early stopping Hook:
            early_stop_hook = tf.contrib.estimator.stop_if_no_increase_hook(
                estimator,
                metric_name,
                max_steps_without_increase,
                eval_dir=None,
                min_steps=0,
                run_every_secs=60,
                run_every_steps=None
            )
            metric_name: str类型，比如loss或者accuracy.
            max_steps_without_increase: int，如果没有增加的最大长是多少，如果超过了这个最大步长metric还是没有增加那么就会停止。
            eval_dir：默认是使用estimator.eval_dir目录，用于存放评估的summary file。
            min_steps：训练的最小步长，如果训练小于这个步长那么永远都不会停止。
            run_every_secs和run_every_steps：表示多长时间或多少步长调用一次should_stop_fn。


    8. 创建Estimator实例
        # model_dir 表示模型要存到哪里,训练后的模型参数会保存在model_dir中，随着训练在目录下生成拥有类似下面内容的checkpoint文件。
        mnist_classifier = tf.estimator.Estimator(model_fn=model_fn,
                                                  model_dir="mnist_model_cnn")

        # hooks：如果不送值，则训练过程中不会显示字典中的数值。
        # steps：指定了训练多少次，如果不送值，则训练到dataset API遍历完数据集为止。
        # max_steps：指定最大训练次数。
        mnist_classifier.train(input_fn=train_input_fn,
                               hooks=[logging_hook])

        # 评估训练集表现
        eval_results = mnist_classifier.evaluate(input_fn=train_eval_fn, checkpoint_path=None)
        print('train set')
        print(eval_results)
        # 评估测试集表现
        # checkpoint_path 是可以指定选择那个时刻保存的权重进行评估
        eval_results = mnist_classifier.evaluate(input_fn=test_input_fn, checkpoint_path=None)
        print('test set')
        print(eval_results)
        # predict 打印输出
        predicts =list(mnist_classifier.predict(input_fn=test_input_fn))
        predicts[0].keys()
        # 输出为：
        dict_keys(['image', 'conv1_out', 'pool1_out', 'conv2_out', 'pool2_out', 'pool2_flat_out', 'dense1_out', 'logits', 'classes', 'labels', 'probabilities'])
        # 打印第四个卷积特征图
        plt.figure(num=4,figsize=(28,28))
        for i in range(4):
            plt.subplot(1,4,i+1)
            plt.imshow(predicts[0]['conv1_out'][:,:,i],cmap = plt.cm.gray)
        plt.savefig('conv1_out.png')
=============================================================

"""
import functools
import os,sys,time
import warnings  #抑制numpy警告
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

basedir = os.path.abspath(os.path.dirname(__file__) + "/../")
sys.path.append(basedir)
print("appended {} into system path".format(basedir))
warnings.simplefilter(action='ignore', category=FutureWarning)
import utils.tf_metrics as tf_metrics

# 定义一个 sessionHook 之类用于对运行过程进行干预，可以控制的过程包括：
# “begin”,
#   “after_create_session”,
#       “before_run”,
#       “after_run”，
# “end”
class exampleHook(tf.train.SessionRunHook):

    def begin(self):
        """在创建会话之前调用
        调用begin()时，default graph会被创建，
        可在此处向default graph增加新op,begin()调用后，default graph不能再被修改
        """
        print("first")
        pass

    def after_create_session(self, session, coord):  # pylint: disable=unused-argument
        print("2nd")

        pass

    def before_run(self, run_context):  # pylint: disable=unused-argument
        print("3rd")

        return None

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):  # pylint: disable=unused-argument
        """调用在每个sess.run()之后
        参数run_values是befor_run()中要求的op/tensor的返回值；
        可以调用run_context.qeruest_stop()用于停止迭代
        sess.run抛出任何异常after_run不会被调用
        Args:
          run_context: A `SessionRunContext` object.
          run_values: A SessionRunValues object.
        """
        print("4th")
        pass

    def end(self, session):  # pylint: disable=unused-argument
        """在会话结束时调用
        end()常被用于Hook想要执行最后的操作，如保存最后一个checkpoint
        如果sess.run()抛出除了代表迭代结束的OutOfRange/StopIteration异常外，
        end()不会被调用
        Args:
          session: A TensorFlow Session that will be soon closed.
        """
        print("5th")
        pass

class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_time_start = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)




def classify_model_fn(features, labels, mode, params):
    """ 基于tf.Keras构建神经网络并使用Estimator管理模型流,主要用于分类模型，回归模型待完成

    :param features: This is the x-arg from the input_fn，输入的原始数据，可以为dict类型，把label也放在这里面
    :param labels: This is the y-arg from the input_fn,与feature对应的label结果，see e.g. train_input_fn for these two.
    :param mode: Estimator运行模式，Either TRAIN, EVAL, or PREDICT，train/eval/test共用此函数
    :param params: 其他参数，包含  mnist_network_fn：网络构建函数（*必选）,
                                label_num,      标签数量
                                learing_rate,   学习率
    :return:
    """
    if isinstance(features, dict) and features.get('feature') is not None:  # For serving
        features = features['feature']

    network_fn = params.get("network_fn")   # 网络构建函数

    num_classes = params.get("num_classes",10)
    learning_rate = params.get("learing_rate",0.01)
    average_type = params.get("average_type",'micro')  # metric均值类型
    positive_idx = list(range(1,num_classes))#[1, 2, 3]  # Class 0 is the 'negative' class,阳例标签

    # 更高级的可以使用feature_column进行输入组织
    """
    price = numeric_column('price')
    keywords_embedded = embedding_column(categorical_column_with_hash_bucket("keywords", 10K), dimensions=16)
    columns = [price, keywords_embedded, ...]
    features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
    dense_tensor = input_layer(features, columns)
    for units in [128, 64, 32]:
      dense_tensor = tf.compat.v1.layers.dense(dense_tensor, units, tf.nn.relu)
    prediction = tf.compat.v1.layers.dense(dense_tensor, 1)
    """

    netwrok_fruit = network_fn(features,params)
    logits = netwrok_fruit.get("logits")

    probs = tf.keras.layers.Softmax(name="softmax_tensor")(logits)
    print("probs tensor: ",probs)
    y_pred = tf.argmax(logits, 1)


    ######################################################
    #               EstimatorSpec Ops
    ######################################################
    if mode == tf.estimator.ModeKeys.PREDICT:   #预测
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.
        # TODO 此处可以修改predict要输出的值
        predictions = {"probs":probs,
                       "preds":y_pred
                       }
        # model export params
        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            tf.estimator.export.PredictOutput(predictions)
                          }
        # export_outputs = {
        #     'prediction': tf.estimator.export.PredictOutput(predictions)
        # }
        spec = tf.estimator.EstimatorSpec(mode, predictions=predictions,
                                          export_outputs=export_outputs
                                          )

    else:
        # Otherwise the estimator is supposed to be in either
        # training or evaluation-mode. Note that :
        # 1. all things about label should put in this block
        # 2. the loss-function is also required in Evaluation mode.

        ############################################################################################
        #               Metric Ops ()
        ############################################################################################
        # 千万要注意，所有关于label的操作都不要再pred模式下出现，否则会出现：Error: None values not supported.
        y_true = tf.argmax(labels, 1)

        # Define the evaluation metrics,
        # in this case the classification accuracy.
        # Tuple of (value, update_op)
        accuracy = tf.metrics.accuracy(labels=y_true,
                                       predictions=y_pred,
                                       name='acc_op')  # 计算精度
        precision = tf_metrics.precision(y_true, y_pred, num_classes, positive_idx,
                                         average=average_type)
        recall = tf_metrics.recall(y_true, y_pred, num_classes, positive_idx,
                                   average=average_type)
        f1 = tf_metrics.f1(y_true, y_pred, num_classes, positive_idx, average=average_type)
        f2 = tf_metrics.fbeta(y_true, y_pred, num_classes, positive_idx, average=average_type, beta=2)
        # use these metrics to monitor estimator
        metrics = {'precision': precision,
                   'recall': recall,
                   'f1': f1,
                   'f2': f2,
                   'accuracy': accuracy}
        # For Tensorboard Summary
        for metric_name, metric in metrics.items():
            scaler = metric[1]
            tf.summary.scalar(metric_name, scaler)
            scaler_watcher = tf.identity(scaler, name="{}_watcher".format(metric_name))
            print(metric_name, " ops : ", scaler_watcher)


        ######################################################
        #               Loss Define Ops
        ######################################################
        # Define the loss-function to be optimized, by first
        # calculating the cross-entropy between the output of
        # the neural network and the true labels for the input data.
        # This gives the cross-entropy for each image in the batch.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=probs)
        # Reduce the cross-entropy batch-tensor to a single number
        # which can be used in optimization of the neural network.
        loss = tf.reduce_mean(cross_entropy,name="loss")
        print("loss op: ",loss)


        ######################################################
        #               Train EstimatorSpec
        ######################################################
        if mode == tf.estimator.ModeKeys.TRAIN:   # 训练
            # Define the optimizer for improving the neural network.
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            # Get the TensorFlow op for doing a single optimization step.
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            # train_op = optimizer.minimize(loss)
            # Wrap all of this in an EstimatorSpec..此处也可以加上predictions=predictions
            spec = tf.estimator.EstimatorSpec(mode,
                                              loss=loss,
                                              train_op=train_op,
                                              eval_metric_ops=metrics)

        ######################################################
        #               Test EstimatorSpec
        ######################################################
        elif mode == tf.estimator.ModeKeys.EVAL:  # 测试
            # Wrap all of this in an EstimatorSpec.此处也可以加上predictions=predictions
            spec = tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)


        else:
            raise NotImplementedError("Currently only mode 'train'/'eval'/'test'/'predict' support!!")

    return spec


def classify_estimator(data,network_fn,run_type="train", model_name="mnist-estimator",**kwargs):
    """ 构建并运行对指定数据以及模型的训练Estimator

        :param data: 生成Dataset的类，需支持 get_train_dataset 和 get_test_dataset 分别返回训练以及测试数据
        :param network_fn: 模型结构构建函数，Estimator的核心
        :param run_type: Estimator的运行模式，支持train,train_simple,train_eval,eval,test,predict,infer,export
        :param model_name: 当前模型的名称，将用于模型ckpt等文件的保存到指定目录
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

        :return: None
        """
    predict_keys = kwargs.get("predic_keys", None)
    log_iter_step = kwargs.get("log_iter_step", 500)
    ckpt_step = kwargs.get("ckpt_step", 1000)
    learning_rate = kwargs.get("learning_rate", 0.01)
    num_classes = kwargs.get("num_classes", 10)
    max_ckpt_num = kwargs.get("max_ckpt_num", 3)
    patient_step = kwargs.get("patient_step", 1000)
    min_train_step = kwargs.get("min_train_step", patient_step)
    max_train_step = kwargs.get("max_train_step", 10000)
    serving_input_receiver_fn = kwargs.get("serving_input_receiver_fn")

    print("kwargs-predict_keys: ", predict_keys)

    ######################################################
    #                   Hooks Ops
    ######################################################
    tensors_to_log = {  # "loss":"loss",  #默认的log已经打印了loss
        "accuracy": "accuracy_watcher",
        "precision": "precision_watcher",
        "recall": "recall_watcher",
        "f1": "f1_watcher"
    }
    logging_hook = tf.estimator.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=log_iter_step)
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_secs=5)
    run_time_hook = TimeHistory()

    ######################################################
    #                 Estimator Creater
    ######################################################
    # classify_model_fn 使用到的参数
    params = {"network_fn":network_fn,
              "num_classes": num_classes,
              "learing_rate": learning_rate}
    estimator_config = tf.estimator.RunConfig().replace(  # estimator运行参数，比如GPU配置，ckpt配置等
        # session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': 1}),
        log_step_count_steps=log_iter_step,
        save_summary_steps=ckpt_step,
        # save_checkpoints_secs=5*60,     # 10*60,  # 每10分钟保存一次 checkpoints
        save_checkpoints_steps=ckpt_step,
        keep_checkpoint_max=max_ckpt_num,  # 保留最新的3个checkpoints
    )
    model_dir = os.path.abspath(basedir + "/../fruits/{}/".format(model_name))
    print("Model Save Dir: {}".format(model_dir))
    estimator = tf.estimator.Estimator(model_fn=classify_model_fn,
                                       params=params,
                                       model_dir=model_dir,
                                       config=estimator_config)

    if run_type == "train_simple":
        # Train the Model with no earlystop strategy and eval ops
        # train(input_fn,hooks=None,steps=None,max_steps=None,saving_listeners=None)
        estimator.train(data.get_train_dataset, steps=max_train_step, hooks=[logging_hook, run_time_hook])


    elif run_type == "train" or run_type == "train_eval":
        # Train and Eval the Model with earlystop strategy
        # 早停止控件，下面两个分别是 tf 1.x 和 tf 2.x 的写法
        # early_stop_hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, metric_name="accuracy",
        early_stop_hook = tf.estimator.experimental. \
            stop_if_no_increase_hook(estimator, metric_name="f1",
                                     max_steps_without_increase=patient_step,
                                     min_steps=min_train_step)
        train_spec = tf.estimator.TrainSpec(input_fn=data.get_train_dataset,
                                            hooks=[early_stop_hook, logging_hook, run_time_hook]
                                            , max_steps=max_train_step)
        eval_spec = tf.estimator.EvalSpec(input_fn=data.get_test_dataset,
                                          steps=None,
                                          start_delay_secs=100,
                                          throttle_secs=1)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    elif run_type == "eval" or run_type == "test":
        input_fn = data.get_test_dataset
        # evaluate(input_fn, steps=None, hooks=None, checkpoint_path=None,name=None)
        metrics_rlt = estimator.evaluate(input_fn)
        print(metrics_rlt)


    elif run_type == "infer" or run_type == "predict":
        input_fn = data.get_test_dataset
        # predict(input_fn,predict_keys=None,hooks=None,checkpoint_path=None,yield_single_examples=True)
        predictions = estimator.predict(input_fn,
                                        predict_keys=predict_keys,
                                        )
        for itm in predictions:
            print(itm)  # 输出为dict格式 {'preds': 1}

    elif run_type == "export":
        export_dir = os.path.abspath(model_dir + "/saved_model/")
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        # export_savedmodel(export_dir_base, serving_input_receiver_fn,assets_extra=None,as_text=False,
        # checkpoint_path=None,strip_default_attrs=False)
        estimator.export_saved_model(export_dir_base=export_dir,
                                     serving_input_receiver_fn=serving_input_receiver_fn())

    else:
        raise NotImplementedError


