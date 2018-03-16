# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:24:56 2018

@author: ZN
步骤：
1. 初始化感知器,设置输入参数的个数以及激活函数，激活函数的类型自定（double -> double）
    注意：初始化权重项，偏置项为0
2. 设置str打印学习到的权重、偏置项return''%(self.weights,self.bias)
3.输入向量，输出结果
    先把输入向量和输出向量zip打包(x,w)
    利用map函数计算[x1*w1]
   (lambda是定义表达式)
4. 输入训练数据：向量，每个向量对应的实际值（label），训练轮数，学习率
5. 一次迭代
    输入输出打包成样本的列表[(input,label)]
    对每个样本，按照感知器规则计算输出并更新权重
6.更新权重    
    
"""
import functools as func
class Perceptron(object):
    
    def __init__(self,input_num,activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0
    
    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights,self.bias)
    
    def predict(self,input_vec):
        return self.activator(
                func.reduce(lambda a,b: a + b,
                       [ x * w for x, w in zip(input_vec,self.weights)]
                       ,0.0) + self.bias)
        
    def train(self,input_vecs,labels,iteration,rate):
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)
            
    def _one_iteration(self,input_vecs,labels,rate):
        samples = zip(input_vecs,labels)
        for(input_vec,label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label ,rate)
            
    def _update_weights(self, input_vec, output , label ,rate):
        delta = label - output
        self.weights = [w + rate * delta *x for x, w in zip(input_vec, self.weights)]
        self.bias += rate * delta
        
        
"""
实现or函数
1. 定义激活函数
2. 基于真值or表构建训练数据（赋值input_vecs,labels）
3. 利用真值训练感知器
       
def f(x):
    return 1 if x > 0 else 0
    
def get_training_dataset():
    input_vecs = [[1,1],[0,0],[1,0],[0,1]]
    labels = [1,0,1,1]
    return input_vecs, labels

def train_and_perceptron():
    #输入参数为2个，激活函数是f
    p=Perceptron(2, f)
    #训练，迭代10轮 ，学习速率为0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p
    
if __name__ == '__main__':
    and_perceptron = train_and_perceptron()
    print(and_perceptron)
    
#测试
print('1 or 0 = %d' % and_perceptron.predict([1,0]))

"""     
    
