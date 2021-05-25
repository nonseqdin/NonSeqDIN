
This directory includes some scripts for preprocessing the Alipay and Taobao datasets.

* 由于我们使用内部训练框架，模型训练数据被导入阿里的maxcompute系统中. 读者请可以根据taobao_prepare_1.py和alipay_prepare_1.py的结果自行处理.
* alimama数据集的较大, 单机进行预处理有困难, 我们将下载的原始数据导入maxcompute系统后, 通过SQL实现处理。

