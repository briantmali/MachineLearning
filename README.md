# MachineLearning
机器学习machine learning  有什么问题大家可以在这里讨论。
记不住这个链接的同学可以如下方式找到本工程。
<BR>1.打开下面的链接：
http://www.pano-tech.com/
<BR>2.点击社会活动
<BR>3.点击任意讲座图片

第一讲视频链接：https://www.youtube.com/watch?v=1qjufJ6z2tU
第三讲第四讲有演示代码


演示代码的环境搭建手顺如下：（第三讲testenvironment.txt文件内容）
实验代码与实验数据来自于下面的项目

https://github.com/mnielsen/neural-networks-and-deep-learning

【实验环境】
从零开始搭建环境的话推荐使用anaconda(对应python2.7版本)
支持在windows平台运行
https://www.continuum.io/downloads
因为示例代码的运行要用到多个机器学习相关库：
numpy
scikit-learn
scipy
Theano
※第三讲内容只用到numpy，之后的内容要用到其他库


【实验步骤】
进入示例代码解压缩后的目录
1.cd src
2.python
3.在python控制台程序中执行下述命令（每行一个命令）：
```python
  import mnist_loader
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  import network
  net = network.Network([784, 30, 10])
  net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```
