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
$a=\hat{y}=\sigma(z)$：逻辑回归输出

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
**PS. 逻辑回归损失函数解释：**
逻辑回归中，$y=1$时$p(y|x)=\hat{y}$，$y=0$时$p(y|x) = 1-\hat{y}$。因此，可以简写为$p(y|x)=\hat{y}^y(1-\hat{y})^{(1-y)}$。用log的形式也可以确保递增，可得$\log p(y|x)=y\log\hat{y}+(1-y)\log(1-\hat{y})$。逻辑回归模型希望将这个函数最大化，但是梯度下降法是将函数最小化，因而我们添加负号，得到$\mathcal{L}(y,\hat(y))$.

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

## L3 - 浅层神经网络 One Hidden Layer Neural Network

### 1 神经网络表示与输出

神经网络由**输入层**，**输出层**和**隐藏层**组成。输入层接收参数$x$，输出层输出$\hat{y}$，每层都会进行计算。在符号标记中，用$[i]$上标表示层数，约定考虑层数时不考虑输入层，将其视作第0层，因此只有一个隐藏层的神经网络被称为2 Layer NN。除了输入层外，神经网络每层都有参数$W^{[i]}, b^{[i]}$。

**Notation:**
$W^{[i]},b^{[i]}$：第$i$层参数，$W^{[i]}$是$n^{[i]}\times n^{[i-1]}$的矩阵
$a^{[i]}：$第$i$层输出，神经网络输入表示为$x=a^{[0]}$
$n^{[i]}$：第$i$层神经元个数，$n^{[0]}=n_x$

**神经网络输出计算：**
假设一个2层神经网络每层神经元个数分别为3, 4, 1，则第一层的计算可以表示为：
$z^{[1]}_1=w_1^{[1]T}x+b_1^{[1]},a^{[1]}=\sigma(z^{(1)}_1)$
$\cdots\cdots~\cdots\cdots$
$z^{[1]}_4=w_4^{[1]T}x+b_4^{[1]},a^{[1]}=\sigma(z^{(1)}_4)$
整个第一层**向量化**可以表示为：
$z^{[1]}=\left[\begin{matrix} w_1^{[1]T} \\  w_2^{[1]T}\\ w_3^{[1]T}\\ w_4^{[1]T}  \end{matrix}\right]\left[\begin{matrix} x_1\\x_2\\x_3 \end{matrix}\right]+\left[\begin{matrix} b_1^{[1]}\\b_2^{[1]} \\b_3^{[1]} \\b_4^{[1]} \end{matrix}\right]=W^{[1]}x+b^{[1]}=\left[\begin{matrix} z_1^{[1]}\\z_2^{[1]} \\z_3^{[1]} \\z_4^{[1]} \end{matrix}\right]$
$a^{[1]}=\sigma(z^{[1]})$
这个神经网络的计算过程如下:
$z^{[1]}=W^{[1]}a^{[0]}+b^{[1]}$
$a^{[1]}=\sigma(z^{[1]})\\z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}\\a^{[2]}=\sigma(z^{[2]})$

**向量化：**
对于整个训练样本$X=[x^{[1]}~x^{[2]}\cdots x^{[m]}]$，可以将上述计算过程进一步向量化：
$Z^{[1]}=W^{[1]}X+b^{[1]}\\A^{[1]}=\sigma(Z^{[1]})\\Z^{[2]}=W^{[2]}A^{[1]}+b^{[2]}\\A^{[1]}=\sigma(Z^{[2]})$
其中$Z^{[1]}=[z^{[1](1)}~z^{[1](2)}\cdots z^{[1](m)}]$横向是对每个样本的遍历，纵向是对第1层每个神经元的遍历，$A^{[1]}$等同理.

### 2 激活函数

目前我们一直在用sigmoid函数$a=\sigma(z)=\frac{1}{1+e^{-z}}$作为**激活函数 activation function**，实际上可以激活函数$a=g(z)$还有很多其他更好的选择。在神经网络中也可以在不同层使用不同的激活函数$g^{[i]}$。

**Sigmoid函数：**$a=\frac{1}{1+a^{-z}}$
二元分类问题中需要输出0-1范围内的数值，可以在神经网络输出层使用Sigmoid函数。除此之外，不要用Sigmoid，因为tanh函数的表现总是比它好。

**tanh函数：**$a=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$
tanh函数的范围在-1~1之间，因而可以使得输出结果的均值接近0，从而给下一层的计算带来便利。tanh函数几乎处处比Sigmoid函数更优，因而实践中尽量避免使用Sigmoid。不过由于tanh和Sigmoid在$z$较大的时候导数很小，梯度下降法运行会比较慢。

**ReLU函数：**$a=\max(0,z)$
ReLU函数 Rectified Linear Unit在$z<0$时为0，$z\geq0$时为$z$，从而保证$z>0$是导数是1，与0相差很多。因而，使用ReLU函数的神经网络训练起来很快，目前它也是应用最广泛的激活函数。尽管$z<0$时导数为0，但是网络中大约一半的神经元导数为1已经能确保速度很快了。另外。数学上ReLU函数在$z=0$时不可微，但是实际中$z=0.00000\cdots$的概率很低，我们也可以在$z=0$处给其导数一个定义。

**Leaky ReLU函数：**$a=\max(0.01z, z)$
Leaky ReLU函数使得$z<0$时导数也不为0，因而比ReLU更优。但是实际应用中这样的优化不是很必要，因而大多数时候还是用ReLU函数。

**非线性激活函数与线性激活函数：**
如果激活函数是线性的，我们称之为**线性激活函数**。如果在神经网络中应用线性激活函数，那么输出$\hat{y}$将仅仅是输入$x$的线性组合，神经网络无法得到一些有趣的函数。现实中仅在回归问题中可能使用线性激活函数：输出$y\in\mathbb{R}$，我们在输出层设置线性激活函数，隐藏层设置非线性激活函数。

### 3 梯度下降法

**激活函数的导数：**
Sigmoid: $g'(z)=a(1-a)$
tanh: $g'(z)=1-a^2$
ReLU: $g'(z)=\left\{ \begin{matrix} 1~\textbf{if}~z\geq 0\\0 ~\textbf{if}~z < 0\end{matrix} \right.$
Leaky ReLU: $g'(z)=\left\{ \begin{matrix} 1~~~~~~\textbf{if}~z\geq 0\\0.01 ~\textbf{if}~z < 0\end{matrix} \right.$

**神经网络梯度下降法：**
**参数：**$W^{[1]},b^{[1]},W^{[2]},b^{[2]}$
**代价函数：**
$$
J(W^{[1]},b^{[1]},W^{[2]},b^{[2]})=\frac{1}{m}\sum_{i=1}^n\mathcal{L}(\hat{y},y)
$$
**梯度下降算法：**
Repeat \{
	Compute predicts $\hat{y}^{{(i)}},i=1,\cdots,m$   # Forward Propagation
	$\mathrm{d}w^{[1]}=\frac{\part J}{\part w^{[1]}}, \mathrm{d}b^{(1)}=\frac{\part J}{\part b^{[1]}},\cdots$ 			 # Backward Propagation
	$w^{[1]}:=w^{[1]}-\alpha\mathrm{d}w^{[1]}$
	$b^{[1]}:=b^{[1]}-\alpha\mathrm{d}b^{[1]}$
\}

神经网络的梯度下降法思路和逻辑回归很像，但其核心在于计算$\hat{y}^{(i)}$和导数的过程：**正向传播**和**反向传播**。正向传播前面已经介绍过，重点是反向传播。假设2层神经网络各层节点数为3, 4, 1，输出层激活函数为Sigmoid，隐藏层为$g$，下面展示**向量化的反向传播过程**：
$dZ^{[2]} = A^{[2]}-Y$ 			# sigmoid函数推导出的结果
$dW^{[2]}=\frac{1}{m} dZ^{[2]}A^T$
$db^{(2)}=\frac{1}{m}$np.sum($dZ^{[2]}$, axis = 1, keepdims = True) # 水平方向，得(n, 1)矩阵
$dZ^{[1]}=W^{[2]T}dZ^{[2]}*g^{[1]}{'}(Z^{[1]})$
$dw^{[1]}=\frac{1}{m}dZ^{[1]}X^T$
$db^{(1)}=\frac{1}{m}$np.sum($dZ^{[1]}$, axis = 1, keepdims = True)







### 4 随机初始化



