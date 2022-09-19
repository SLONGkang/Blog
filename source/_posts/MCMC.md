---
title: MCMC
date: 2022-09-18 22:58:45
tags: "采样"
cover: https://th.bing.com/th/id/R.93252e6d15f6d90dcc25369299b843be?rik=Df2QCnmVKTGiNw&pid=ImgRaw&r=0
description: 蒙特卡洛采样和马尔科夫链的结合使用
copyright_author: 功夫康
copyright_author_href: https://kongfukang.com/
---
二元分布：
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

def get_q(x):#目标分布
    return np.sin(x[0]**2 + x[1]**2)+1

def get_p(x):#在这里建议分布在区域内取平均分布
    return 2.1

def plotContour():#画出分布的概率等高线
    delta = 0.5  #网格间距
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = get_q([X, Y])
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=1, fontsize=10)

def get_sample():#在区域内抽随机值
    return [np.random.random() * 6.0 - 3.0, np.random.random() * 6.0 - 3.0]

count = 10000

def testGetsample(count):#算是拒绝-接收采样，二元建议分布想不出来比较好的就用均匀分布
    last_x = [0,0] #X0
    sample = []
    x0 = []
    x1 = []
    for i in range(count):
        x = get_sample()
        acc = np.random.random()
        if(acc > get_q(x)/get_p(x)):
            x = last_x
        last_x = x
        x0.append(x[0])
        x1.append(x[1])
        sample.append(x)
    ##绘制采样的点
    plt.scatter(x0,x1,c = 'g',marker='.')
    plt.show()

testGetsample(count)
plotContour()
```

一维分布：
一维分布：

```python
mu1 = 3
sigma1 = 2
mu2 = -5
sigma2 = 3.2
def get_q(x):#目标分布
    return (0.3 * np.exp(-((x - mu1)**2)/(2*sigma1**2)) / (sigma1 * np.sqrt(2*np.pi)) 
            + 0.7 * np.exp(-((x - mu2)**2)/(2*sigma2**2)) / (sigma2 * np.sqrt(2*np.pi)))

def get_p(x, mu, sigma):#建议分布
    return np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

import scipy.optimize as optimize
def get_max(mu, sigma):#求正态建议分布的最值
    # mu = 0
    # sigma = 4
    e = lambda x: -(np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi)))
    x = np.arange(-10.0, 10.0, 0.01)
#     plt.plot(x, get_p(x, mu, sigma))
    x,y,z,t = optimize.fminbound(e, -10, 10, full_output=1)
#     print(x,y,z,t)
    return -y+0.05  #从建议分布取样使用拒绝采样

def get_ran():
    return np.random.random()*20.0 - 10.0

def get_from_p(mu, sigma):#从P正态分布按照概率函数采样
    while True:
        x = get_ran()
        n = np.random.random()
        if(n<get_p(x,mu,sigma)/get_max(mu, sigma)):
            return x

def testMetropolis(sigma):#Metropolis算法，这里建议分布概率函数对称，若不对称则为MH
#     sigma = 4
    samples = []
    iter = 10000
    samples.append(0) #x0
    res = []
    for i in range(20000):
        sample = get_from_p(samples[i], sigma)
        Acc = min(1,get_q(sample)/get_q(samples[i]))
        if(np.random.random() > Acc):
            sample = samples[i]
        samples.append(sample)
        if(i > iter):#假设10000轮后x已经收敛
            res.append(sample)
    return res

plt.hist(x, 20,density=True)
xx = np.arange(-10,10,0.01)
plt.plot(xx, get_p(xx,0,1)) #从建议分布上采样

plt.hist(testMetropolis(4), 50,density=True) #方差取2，4，6，10查看
x = np.arange(-10.0, 10.0, 0.01)
plt.plot(x, get_q(x))
plt.title("proposal distribution var is 4")
```


MH采样（使用不对称的建议分布）：

```python
mu1 = 3.79
sigma1 = 2.58
mu2 = -4.75
sigma2 = 2.24
def get_q(x):#目标分布，假设对外界隐藏
    return (0.7 * np.exp(-((x - mu1)**2)/(2*sigma1**2)) / (sigma1 * np.sqrt(2*np.pi)) 
            + 0.3 * np.exp(-((x - mu2)**2)/(2*sigma2**2)) / (sigma2 * np.sqrt(2*np.pi)))

#仅仅根据对图像的分析，有两个峰顶，因此建议分布可以采取两个封顶的建议分布
#画出图形感觉分布差不多
# sigma1 = 1.75
# sigma2 = 2#据观察猜测
def get_p(x, mu1 = -5, mu2 = 6, sigma1 = 1.75, sigma2 = 2, dou = 1):#mu1为左峰分布，mu2为右峰分布
    if dou:
        return 2 * (0.33 * np.exp(-((x - mu1)**2)/(2*sigma1**2)) / (sigma1 * np.sqrt(2*np.pi)) 
                + 0.64 * np.exp(-((x - mu2)**2)/(2*sigma2**2)) / (sigma2 * np.sqrt(2*np.pi)))
    else:
        return (0.33 * np.exp(-((x - mu1)**2)/(2*sigma1**2)) / (sigma1 * np.sqrt(2*np.pi)) 
            + 0.64 * np.exp(-((x - mu2)**2)/(2*sigma2**2)) / (sigma2 * np.sqrt(2*np.pi)))

import scipy.optimize as optimize
def get_max(mu1, mu2, sigma1, sigma2, low, up):#求正态建议分布的最值
    e = lambda x: -(0.33 * np.exp(-((x - mu1)**2)/(2*sigma1**2)) / (sigma1 * np.sqrt(2*np.pi)) 
                    + 0.64 * np.exp(-((x - mu2)**2)/(2*sigma2**2)) / (sigma2 * np.sqrt(2*np.pi)))
    x = np.arange(-10.0, 10.0, 0.01)
    x,y,z,t = optimize.fminbound(e, low, up, full_output=1)
    return -y+0.05

def get_ran():
    return np.random.random()*20.0 - 10.0

#在这里使用了两种不同的采样
#一种是对两个正态分布的中心值取样（分开取样，要不两个都向目标分布的最高峰收敛）
#另一种对最终样本取样，flag = 1
def get_from_p(mu1, mu2, sigma1, sigma2, flag = 1):
    if flag == 1:
        
        while True:
            x = get_ran()
            n = np.random.random()
            if(n<get_p(x, mu1, mu2, sigma1, sigma2)/get_max(mu1, mu2, sigma1, sigma2, -10, 10)):
                return x, flag
    else:
        while True:
            x = -1 * (np.random.random()* 10.0)
            n = np.random.random()
            if(n<get_p(x, mu1, mu2, sigma1, sigma2)/get_max(mu1, mu2, sigma1, sigma2, -10, 0)):
#                 print("x is {}".format(x))
                break
        while True:
            y = np.random.random()* 10.0
            n = np.random.random()
            if(n<get_p(y, mu1, mu2, sigma1, sigma2)/get_max(mu1, mu2, sigma1, sigma2, 0, 10)):
#                 print("y is {}".format(y))
                break
        return x, y

def testMetropolis_Hasting(sigma1, sigma2):
    samples = [get_ran()]
    x_list = [-(np.random.random()*10.0)]
    y_list = [np.random.random()*10.0]
    iter = 20000
		#单独对x，y进行区域收敛，在这基础上对样本sample进行采样
    for i in range(0, iter):
        x, y = get_from_p(x_list[i], y_list[i], sigma1, sigma2, flag = 0)
        
        Acc = min(1,get_q(x)/get_q(x_list[i])*
                  (get_p(x_list[i], x_list[i], y_list[i], sigma1, sigma2)/
                   get_p(x, x_list[i], y_list[i], sigma1, sigma2)))
        if(np.random.random() > Acc):
            x = x_list[i]
        x_list.append(x)
        Acc = min(1,get_q(y)/get_q(y_list[i])*
                  (get_p(y_list[i], x_list[i], y_list[i], sigma1, sigma2)/
                   get_p(y, x_list[i], y_list[i], sigma1, sigma2)))
        if(np.random.random() > Acc):
            y = y_list[i]
        y_list.append(y)
        
        sample,_ = get_from_p(x_list[i], y_list[i], sigma1, sigma2)
        Acc = min(1,get_q(sample)/get_q(samples[i])*
                  (get_p(samples[i], x_list[i], y_list[i], sigma1, sigma2)/
                   get_p(sample, x_list[i], y_list[i], sigma1, sigma2)))
        if(np.random.random() > Acc):
            sample = samples[i]
        samples.append(sample)
    return samples, x_list, y_list

x = np.arange(-10.0, 10.0, 0.01)
plt.plot(x, get_q(x),c = 'r')
plt.plot(x, get_p(x, -5, 5, 1.75, 2))

x = []
for i in range(10000):
    x1, x2= get_from_p(-5, 5, 1.75, 2, 1)
    x.append(x1)
plt.hist(x, 50,density=True)
xx = np.arange(-10,10,0.01)
plt.plot(xx, get_p(xx, -5, 5, 1.75, 2, 0))

#组合分别取1.75,2 & 2.5,1.5 & 2.25,1.75 & 3,1.5，四组
sigma1 = 1.75
sigma2 = 2
samples, x, y = testMetropolis_Hasting(sigma1, sigma2)
plt.hist(samples, 50,density=True)
xx = np.arange(-10.0, 10.0, 0.01)
plt.plot(xx, get_q(xx), c = 'b')
plt.plot(xx, get_p(xx, mu1 = sum(x[-50:])/len(x[-50:]), mu2 = sum(y[-50:])/len(y[-50:]),
                  sigma1 = sigma1, sigma2 = sigma2, dou = 0),c = 'r')
plt.title("proposal distribution var(s) are {}, {}".format(sigma1, sigma2))

每次都能正确收敛到两个对应值，但是这样分开收敛不知道对不对，另外感觉方差也可以加入马尔科夫链收敛过程中，但是接受概率怎么计算。。

---

Gibbs吉布斯采样：

```python
def get_q(x):#目标分布
    return np.sin(x[0]**2 + x[1]**2)+1

def plotContour():#画出分布的概率等高线
    delta = 0.5  #网格间距
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = get_q([X, Y])
#     Z = peaks(Z)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=1, fontsize=10)

def get_ran():
        return np.random.random() * 6.0 - 3.0

def gibbsSampler(x,dim):#吉布斯采样中的针对某一维度随机采样
    xes = []
    for t in range(10): #随机选择10个点
        xes.append(get_ran())
    tilde_ps = []
    for t in range(10): #计算这10个点的未归一化的概率密度值
        tmpx = x[:]
        tmpx[dim] = xes[t]
        tilde_ps.append(get_q(tmpx))
    #在这10个点上进行归一化操作，然后按照概率进行选择。
    norm_tilde_ps = np.asarray(tilde_ps)/sum(tilde_ps)
    u = np.random.random()
    sums = 0.0
    for t in range(10):
        sums += norm_tilde_ps[t]
        if sums>=u:
            return xes[t]

def gibbs(x):
    rst = np.asarray(x)[:]
    path = [(x[0],x[1])]
    for dim in range(2): 
        new_value = gibbsSampler(rst,dim)
        rst[dim] = new_value
        path.append([rst[0],rst[1]])
    return rst,path

def testGibbs():
    x0 = []
    x1 = []
    iter = 10000
    sample = [get_ran(), get_ran()]
#     samples.append(sample)#x0
    paths = [sample]
    for i in range(iter):
        samples.append([sample[0],sample[1]])
        sample,path = gibbs(sample)
        paths.extend(path)
        x0.append(sample[0])
        x1.append(sample[1])
        
    return x0, x1, paths

x0, x1, paths = testGibbs()

plt.scatter(x0, x1,c = 'g',marker='.')
#画出采样路径，可以看出都是依维度进行移动
path0 = []
path1 = []
for path in paths:
    path0.append(path[0])
    path1.append(path[1])
plt.plot(path0[:50],path1[:50],'k-',linewidth=0.5)

plotContour()
plt.show()
```


```
[代码实现](https://www.notion.so/5a8313f932724a2d9cb66173a8e9d4f4)
