# 机器学习公开课笔记一


#### 一、什么是机器学习(what is machine learning?)

###### 1.1 机器学习定义

主要有两种定义：

- Arthur Samuel (1959). Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.
- Tom Mitchell (1998) Well-posed Learning Problem: A computer program is said to learn from experience E with respect to some **task T** and some **performance measure P**, if its performance on T, as measured by P, improves with **experience E**. 

###### 1.2 机器学习算法

主要有两种机器学习的算法分类

- 监督学习
- 无监督学习

两者的区别为**是否需要人工参与数据结果的标注**。这两部分的内容占比很大，并且很重要，掌握好了可以在以后的应用中节省大把大把的时间~

还有一些算法也属于机器学习领域，诸如：

- 半监督学习: 介于监督学习于无监督学习之间
- 推荐算法: 没错，就是那些个买完某商品后还推荐同款的某购物网站所用的算法。
- 强化学习: 通过观察来学习如何做出动作，每个动作都会对环境有所影响，而环境的反馈又可以引导该学习算法。
- 迁移学习

###### 1.3 监督学习(Supervised Learning)

监督学习，即为教计算机如何去完成预测任务（有反馈），预先给一定数据量的输入**和对应的结果**即训练集，建模拟合，最后让计算机预测未知数据的结果。

监督学习一般有两种：

(1) 回归问题 (Regression)

​		回归问题即为预测一系列的**连续值**。

​		在房屋价格预测的例子中，给出了一系列的房屋面基数据，根据这些数据来预测任意面积的房屋价格。给出照片-年龄数据集，预测给定照片的年龄。

<center class="half">
    <img src="/images/ML_01.png" width="300"/>
</center>


(2) 分类问题 (Classification)

​		分类问题即为预测一系列的**离散值**。

​		即根据数据预测被预测对象属于哪个分类。

​		视频中举了癌症肿瘤这个例子，针对诊断结果，分别分类为良性或恶性。还例如垃圾邮件分类问题，也同样属于监督学习中的分类问题。

<center class="half">
    <img src="/images/ML_02.png" width="300"/>
</center>


视频中提到**支持向量机**这个算法，旨在解决当特征量很大的时候(特征即如癌症例子中的肿块大小，颜色，气味等各种特征)，计算机内存一定会不够用的情况。**支持向量机能让计算机处理无限多个特征。**

###### 1.4 无监督学习(Unsupervised Learning)

相对于监督学习，训练集不会有人为标注的结果（无反馈），我们**不会给出**结果或**无法得知**训练集的结果是什么样，而是单纯由计算机通过无监督学习算法自行分析，从而“得出结果”。计算机可能会把特定的数据集归为几个不同的簇，故叫做聚类算法。

无监督学习一般分为两种：

(1) 聚类 (Clustering)

- 新闻聚合
- DNA 个体聚类
- 天文数据分析
- 市场细分
- 社交网络分析

(2) 非聚类 (Non-clustering)

- 鸡尾酒问题

**新闻聚合**

在例如谷歌新闻这样的网站中，每天后台都会收集成千上万的新闻，然后将这些新闻分组成一个个的新闻专题，这样一个又一个聚类，就是应用了无监督学习的结果。

**鸡尾酒问题**

<center class="half">
    <img src="/images/ML_03.png" width="300"/>
</center>


#### 二、单变量线性回归 (Linear Regression with One Variable)

###### 2.1 模型表示 (Model Representation)

(1) 房价预测训练集

| Size in () | Price ($) in 1000's() |
| ---------- | --------------------- |
| 2104       | 460                   |
| 1416       | 232                   |
| 1534       | 315                   |
| 852        | 178                   |

房价预测训练集中，同时给出了输入  和输出结果 ，即给出了人为标注的**”正确结果“**，且预测的量是连续的，属于监督学习中的回归问题。

(2) 问题解决模型

<center class="half">
    <img src="/images/ML_04.png" width="300"/>
</center>

其中$h$代表结果函数，也称为**假设(hypothesis)** 。假设函数根据输入(房屋的面积)，给出预测结果输出(房屋的价格)，即是一个$X—>Y$的映射。

$h_\theta(x) = \theta_0 + \theta_1x$，为解决房价问题的一种可行表达式。

$x$: `特征/输入变量`

上式中，$\theta$为参数，$\theta$的变化才决定了输出结果，不同以往，这里的$x$被我们**视作已知**(不论是数据集还是预测时的输入)，所以怎样解得$\theta$以更好地拟合数据，成了求解该问题的最终问题。

单变量，即只有一个特征(如例子中房屋的面积这个特征)。

###### 2.2 代价函数 (Cost Function)

李航《统计学习方法》一书中，损失函数与代价函数两者为**同一概念**，未作细分区别，全书没有和《深度学习》一书一样混用，而是统一使用**损失函数**来指代这类类似概念。

吴恩达(Andrew Ng)老师在其公开课中对两者做了细分。**如果要听他的课做作业，不细分这两个概念是会被打小手扣分的**！这也可能是因为老师发现了业内混用的乱象，想要治一治吧。

**损失函数**(Loss/Error Function): 计算**单个**样本的误差。[link](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/yWaRd/logistic-regression-cost-function)

**代价函数**(Cost Function): 计算整个训练集**所有损失函数之和的平均值**

综合考虑，本笔记对两者概念进行细分，若有所谬误，欢迎指正。



我们的目的在于求解预测结果$\theta$最接近于实际结果$y$时$\theta$的取值，则问题可表达为**求解**$\sum_{i=0}^{m}{(h_\theta(x^{(i)}) - y^{i})}$的**最小值**。

> $m$: 训练集中的样本总数
>
> $y$: 目标变量/输出变量
>
> $(x, y)$: 训练集中的实例
>
> $(x^{i}, y{i})$: 训练集中的第$i$个样本实例

<center class="half">
    <img src="/images/ML_05.png" width="300"/>
</center>


上图展示了当$\theta$取不同值时，$h_\theta(x)$对数据集的拟合情况，蓝色虚线部分代表**建模误差**（预测结果与实际结果之间的误差），我们的目标就是最小化所有误差之和。

为了求解最小值，引入代价函数(Cost Function)概念，用于度量建模误差。考虑到要计算最小值，应用二次函数对求和式建模，即应用统计学中的平方损失函数（最小二乘法）：

$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m}(\hat{y_i} - {y_i})^2 = \frac{1}{2m} \sum_{i=1}^{m}(h_\theta(x_i) - {y_i})^2$

> $\hat{y_i}$: $y$的预测值
>
> 系数$\frac{1}{2}$存在与否都不会影响结果，这里是为了在应用梯度下降时便于求解，平方的导数会抵消掉$\frac{1}{2}$

讨论到这里，我们的问题就转化成了**求解**$J(\theta_0, \theta_1)$**的最小值**。

###### 2.3 代价函数 - 直观理解1 (Cost Function - Intuition I)

根据上节视频，列出如下定义：

- 假设函数(Hypothesis): $h_\theta(x) = \theta_0 + \theta_1x$
- 参数(Parameters): $\theta_0, \theta_1$
- 代价函数(Cost Function): $J(\theta_0, \theta_1) = \frac{1}{2m} \sum \limits_{i=1}^{m}(h_\theta(x_i) - {y_i})^2$
- 目标(Goal): ${\underset{\theta_0, \theta_1}{minimize} J(\theta_0, \theta_1)}$

为了直观理解代价函数到底是在做什么，先假设$\theta_1 = 0$，并假设训练集有三个数据，分别为$(1, 1), (2, 2), (3, 3)$，这样在平面坐标系中绘制出$h_\theta(x)$，并分析$J(\theta_0, \theta_1)$的变化。

<center class="half">
    <img src="/images/ML_06.png" width="300"/>
</center>


由图$J(\theta_0, \theta_1)$随着$\theta_1$的变化而变化，可见**当$\theta_1 = 1$**时，$J(\theta_0, \theta_1)=0$，**取得最小值**，对应于左图青色直线，即函数$h$拟合最好的情况。

