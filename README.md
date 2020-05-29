# Easy Use of Tensorflow Estimator

## description
项目主要基于TensorFlow Estimator引擎，早就对Estimator的高效高伸缩性有所耳闻，
使用Estimator可以让我们将更多的精力集中于模型设计与数据输入设计上，
因此，花费一些精力根据网上诸多大牛的介绍和本人的一些体悟在Mnist数据集上实现了Estimator的模型评估器,具有以下特点：

-[x] Mnist数据的输入使用了tf.data.Dataset方式，位于gastation内，主要参考官方示例
-[x] 将网络模型的构建完全独立出来，每个独立的模型可以构建独立的一个文件，例如本示例就是以engine/core/tfengine_mnist中的network_fn来构建网络结构
-[x] 将可能会频繁重复使用的EstimatorSpec生成代码独立成完善的模块（engine/core/tfengine_core下的 classify_estimator ），目前仅支持分类模型，应该很容易就能改造成一个回归模型
-[x] Estimator增加了早停止hook，目前已内部集成了accuracy、precision、recall、f1、f2等多种评估
-[x] classify_estimator支持train_simple/train_eval/eval/test/export等多种运行模式
-[x] 模仿tfengine_mnist中的代码，只需编写对应的dataset生成器和model_fn即可实现模型的训练评估，如果要运行export则还需要实现serving_input_receiver_fn。
-[x] 更过特征请查看具体代码注释，其中 utils/tf_metrics 需要感谢https://github.com/guillaumegenthial/tf_metrics, 
tfengine_core着重感谢https://www.jianshu.com/p/82cf413b899c作者提供的核心思路

