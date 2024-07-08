# Skip-gram 简单实现过程

本文展示了 Skip-gram 模型的一个简单实现过程。我们使用 text8 语料库进行训练，并利用一组人工标注的数据进行评估。评估方法采用了斯皮尔曼秩相关系数。以下是我们的评估结果：

- **Simlex-999**:
  - **Statistic**: -0.04806096603269743
  - **p-value**: 0.22143454163727155

- **MEN**:
  - **Statistic**: 0.010008017771466165
  - **p-value**: 0.7118103757696796

- **WordSim353**:
  - **Statistic**: 0.0076863773521402565
  - **p-value**: 0.9029795856765185

注: 为了实验结果理想,斯皮尔曼秩相关系数（Statistic）的值应接近 1，表示高度正相关。p 值（p-value）应远小于 0.05，表示相关性显著。

尽管这些结果并不理想，但这只是一次简单的测试，旨在帮助理解 Skip-gram 模型的工作流程。结果的偏差可能是由于训练集规模过大所致。我们重新使用了原作者的训练数据，具体链接为：[原作者训练文件](https://github.com/yeatscircle/SkipGram/tree/master/data/actual)。

关于 Skip-gram 模型的原理，请参考以下链接：[Skip-gram 原理](https://github.com/yeatscircle/SkipGram/tree/master/Introduction)。

此外，text8 语料库经过简短 epoch 训练后的模型参数，请参考以下链接：[模型参数-2epoch](https://github.com/yeatscircle/SkipGram/tree/master/model)。
