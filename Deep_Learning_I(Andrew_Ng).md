# Neural Networks and Deep Learning

Author: Daniel Liu
Contact me:  191240030@smail.nju.edu.cn
Course: Deeplearning.ai Part I - AndrewNg

## L1 - 深度学习概论 Introduction to Deep Learning

近年来，深度学习成为了最为流行的AI技术。
在小规模数据上，各种机器学习算法具有相似的表现，运行效果很大程度上取决于使用的技巧、组件以及算法实现的细节等。在大规模数据上，传统机器学习算法会在一定数据规模后遇到性能瓶颈，而深度学习会随着**数据规模**与**神经网络规模**的提升具有更好的表现。同时，**计算速度**的提升和**算法**的优化使得训练大规模的网络更加快捷。此外，传统的机器学习算法擅长处理**结构化数据**，即给定各种特征的相关数据进行训练，而深度学习在**非结构化数据**(图片、音频等)上也能有很好的效果。这些因素都使深度学习称为近些年人工智能的研究热点，并被应用在很多领域。
根据应用场景的不同，神经网络结构也有很多变种。房产估价、网络广告等应用场景下，使用常规的神经网络结构即可；计算机视觉常用**卷积神经网络 CNN**；语音识别和机器翻译等序列数据通常使用**循环神经网络 RNN**；智能驾驶等复杂的场景下，可能需要复杂的混合神经网络结构处理。

***

## L2 - 神经网络编程基础 Basics of Neural Network Programming

### 1 逻辑回归

**Notation:**
$X=\left[ x^{(1)}~x^{(2)}\cdots x^{(m)}  \right]$：神经网络一般约定将数据存放在一个$n_x\times m$的矩阵中
$y=[y^{(1)}~y^{(2)}\cdots y^{(m)}]$：标签存放在$1\times m$的矩阵中
$w\in\mathbb{R}^{n_x},b\in \mathbb{R}$：sigmoid函数的参数
$z^{(i)}=w^Tx^{(i)}+b$：用于sigmoid函数

**逻辑回归**中，输出$\hat{y}=P(y=1|x)$，即给定$x$，$y=1$的概率。$\hat{y}$表示为：
$$
\hat{y}=\sigma(w^Tx+b)=\sigma(z)=\frac{1}{1+e^{-z}}
$$
注意：神经网络中参数通常将$w,b$分开表示，而其他算法中可能用$\theta_0$表示$b$。

对于单个训练样本，定义**损失函数 Loss Function：**
$$
\mathcal{L}(y,\hat{y})=-\left[y\log\hat{y}+(1-y)\log(1-\hat{y})\right]
$$
对于整个逻辑回归模型，可以定义其**代价函数 Cost Function：**
$$
J(w,b)=-\frac{1}{m}\sum_{i=1}^m\left[ y^{(i)}\log\hat{y}^{(i)}+(1-y^{(i)})\log(1-\hat{y}^{(i)})  \right]
$$
### 2 梯度下降法

**梯度下降算法 Gradient Descendent：**
Repeat \{
	$w_i:=w_i-\alpha\frac{\part J(w,b)}{\part w_i}$
	$b:=b-\alpha\frac{\part J(w,b)}{\part b}$
\}

在梯度下降法中，关键在于计算出导数。通过**计算流程图 computation graph**可以理解，可以用**正向传播 forward propagation**计算代价函数，并用**反向传播 backward propagation**计算导数，后者结合了微积分的**链式法则**。尽管逻辑回归中用这种方式计算导数没有必要，但是鉴于逻辑回归与神经网络的联系，通过逻辑回归理解起来会容易些。

**逻辑回归导数计算：**
对于单个样本，逻辑回归正向的计算过程如下: 
$x_1,w_1,x_2,w_2b\rightarrow z=w_1x_2+w_2x_2+b\rightarrow a=\sigma(z)\rightarrow \mathcal{L}(a,y)$.
反向过程首先计算出$\mathrm{d}a=\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}a}=-\frac{y}{a}+\frac{1-y}{1-a}$，接着用链式法则计算$\mathrm{d}z=\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}z}=\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}a}\frac{\mathrm{d}a}{\mathrm{d}z}=a-y$，最后计算$\mathrm{d}w_1=x_1\mathrm{d}z$, $\mathrm{d}w_2=x_2\mathrm{d}z$, $\mathrm{d}b=\mathrm{d}z$.
(注意这里的"$\mathrm{d}a$"等符号都是简约记号)

计算整个模型的导数时，通过$J(w,b)=\frac{1}{m}\sum_{i=1}^m\mathcal{L}(w,b)$可以发现，只要对$\mathrm{d}w_i,\mathrm{d}b$在样本上累加并求平均即可，在每个样本上它们的反向计算方法和之前相同。

### 3 向量化

向量化用于取代程序中的for循环。在面对大量数据时，向量化计算要比for循环快很多。以$z=w^Tx+b$为例，Numpy中使用 z=np.dot(w, x)+b 会快很多。Numpy会充分利用CPU或GPU中的SIMD指令(并行化指令)进行并行计算，从而大大提高运算速度。GPU是图像处理单元，在并行计算方面性能比CPU还高。

**逻辑回归正向传播向量化：**
正向传播中需要计算$z^{(i)}=w^Tx^{(i)}+b,a^{(i)}=\sigma(z^{(i)})$。可以将其向量化实现：
$Z=\left[z^{(1)}~z^{(2)}\cdots z^{(m)}\right]=w^TX+[b~b\cdots b]\\A=\left[a^{(1)}~a^{(2)}\cdots a^{(m)}\right]=\sigma(Z)$

**逻辑回归反向传播向量化：**
反向传播计算$dz^{(i)}=a^{(i)}-y^{(i)},dw=\sum _{i=1}^mx^{(i)}_jdz^{(i)}, db=\sum_{i=1}^m dz^{(i)}$，向量化：
$dZ=A-Y\\dw=\frac{1}{m}X(dZ)^T\\db=\frac{1}{m}np.sum(dZ)$

**逻辑回归向量化代码：**

```python
for iter in range(1000):
    Z = np.dot(w.T, X) + b
	A = 1 / (1 + np.exp(-Z))
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m
    w = w - alpha * dw
    b = b - alpha * db
```

### 4 Python

Python编写神经网络中常用的**广播原则 broadcasting：**
矩阵与向量运算:
(m, n)  \[+-\*/\]  \[(m, 1) => (m, n)\]
(m, n)  \[+-\*/\]  \[(1, n) => (m, n)\]
矩阵/向量与常数运算：
(m, n) \[+-\*/\] [R => (m, n)]

**Python避错方法：**
a = np.random.randn(5)
上述代码得到的a类型为"rank 1 array"，这与我们常用的行向量/列向量都不同。a.shape结果为"(5,)"。这种类型的行为有时会与向量不同，因而导致代码运行效果不符合预期。为了创建行向量/列向量，应该使用如下代码：
a = np.random.randn(5, 1)
上述代码保证得到的是一个$5\times 1$的矩阵，即为行向量，其运算规则等都符合我们通常的预期。为了保证得到的a类型正确，可以用assert检查：
assert(a.shape == (5, 1))
assert对执行效率几乎没有影响。也可以用reshape来保证类型正确：
a = a.reshape(5, 1)

