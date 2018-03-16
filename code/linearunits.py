# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 18:31:10 2018

@author: ZN
可以利用import的方法引入另一个文件的代码（继承）
1. 定义激活函数
2. 初始化线性单元，设置输入参数的个数
3. 训练数据单元
4. 运行，打印并训练
"""

from perceptron import Perceptron

f = lambda x : x

class LinearUnit(Perceptron):
    def __init__ (self, input_num):
        Perceptron.__init__(self, input_num, f)
        
def get_training_dataset():
    #输入向量列表，每一项是工作年限
    input_vecs = [[5],[3],[8],[1.4],[10.1]]
    #期望的输出列表，月薪，注意要与输入的一一对应
    labels = [5500,2300,7600,1800,11400]
    return input_vecs, labels

def train_linear_unit():
    #创建感知器，输入参数的特征数为1（工作年限）
    lu = LinearUnit(1)
    input_vecs, labels = get_training_dataset()
    #训练，迭代10轮，学习速率为 0.01
    lu.train(input_vecs , labels, 10, 0.01)
    return lu

if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print(linear_unit)
    #测试
    print('Work 3.4 years, monthly salary = %.2f'% linear_unit.predict([3.4]))