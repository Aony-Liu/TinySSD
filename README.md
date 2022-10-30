# TinySSD
人工智能实验 个人作业
## 环境配置：
python环境下的pytorch-gpu版本
## 训练流程：
1. 制作数据集：在background文件夹中保存需要用到的背景图片，在target文件夹中保存待检测的目标图片，随后运行create_train.py实现合成训练样本并将数据保存在新产生的sysu_train文件夹中；最后在test文件夹下保存最后用于测试代码运行效果的图片。
2. 读取训练数据：运行load_data.py实现对训练样本的数据读取
3. 运行main文件，观察检测效果。
