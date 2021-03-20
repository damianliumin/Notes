# Improving Deep Neural Networks

Author: Daniel Liu
Contact me:  191240030@smail.nju.edu.cn
Course: Deeplearning.ai Part II - AndrewNg

## L1 - 实用深度学习 Practical Aspects of Deep Learning

### 1 偏差与方差

**数据划分：**数据集可划分为训练集、验证集和测试集，在数据较少的时候7:3和6:2:2的划分很常见，但是大数据时代通常之用很小一部分(有时不到$1\%$)来验证和测试。

**偏差与方差：**欠拟合即高偏差，过拟合即高方差。在高偏差时可考虑使用**规模更大的网络**或**训练更长时间**(后者通常作用不大)，高方差时可考虑**增加数据**或**正则化**。传统的机器学习中常常会考虑偏差与方差间的trade-off，但是深度学习中很多时候改善方差、偏差中的一个不会使另一个增加另一个，这也是深度学习的优势之一。

### 2 正则化

**逻辑回归正则化：**逻辑回归的正则化代价函数如下：
$$
J(w, b)=\frac{1}{m}\sum_{i=1}^m\mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}\left\| w \right\|_2^2
$$
其中$\lambda$是**正则化参数**，$\left\|w \right\|^2 = \sum_{j=1}^{n_x}w_j^2=w^Tw$是$w$的$l_2-$范数，因而这种正则化被称为$L2$正则化。最后可以加上$\frac{\lambda}{2m}b^2$，不过这样没有必要，$w$已经覆盖了大多数参数。如果添加的是$\frac{\lambda}{2m}\left\| w \right\|_1=\frac{\lambda}{2m}\sum_{j=1}^{n_x}|w_j|$，即为$L1$正则化，这种方法得到的$w$通常比较稀疏，有较多的0。现在通常使用$L2$正则化。

**深度学习L2正则化：**深度学习的正则化代价函数如下：
$$
J(W^{[1]},b^{[1]},\cdots,W^{[L]},b^{[L]})=\frac{1}{m}\sum_{i=1}^m\mathcal{L}(\hat{y}^{(i)},y^{(i)})+\frac{\lambda}{2m}\sum_{l=1}^{L}\left\| W^{[l]} \right\|^2_F
$$
其中$\left\| W^{[l]} \right\|_F^2 = \sum_{i=1}^{n^{[l-1]}}\sum_{j=1}^{n^{[l]}}(W_{ij}^{[l]})^2$是**弗罗贝尼乌斯范数 Frobenius norm**，这也是$L2$正则化。在反向传播的过程中，正则化会产生如下影响：
$$
dW^{[l]} = (dz^{[l]}a^{[l-1]}) + \frac{\lambda}{m}W^{[l]}\\
W^{[l]} := W^{[l]} - \alpha\mathrm{d}W^{[l]} = (1-\frac{\alpha\lambda}{m})W^{[l]} - \alpha(dz^{[l]}a^{[l-1]})
$$
可以发现$W^{[l]}$前多了一个参数$1-\frac{\alpha\lambda}{m}$，这使得$W^{[l]}$每次都会减小一点，因而$L2$正则化有时也被称为**权重衰减 weight decay**。

**L2正则化作用原理的直观感受：**在代价函数$J$中添加正则项可以有效阻止$W$变得太大，通常更小的$W$会使得神经网络变得更加简单。另一方面，$W$减小后$z=Wa+b$会更小，而$\tanh$这样的激活函数在$z$较小的范围内又近似是线性的，因而神经网络计算出的函数会更为简单。通过$\lambda$的选取，我们可以在过拟合与欠拟合之间找到一个合适的状态。

**Dropout正则化：**
**Dropout正则化(随机失活)**会在每个样本训练时，在每层随机选取一些神经元使其失活，也即暂时删除这些单元与其他单元间的连线。实现方法如下：

```python
# for layer 3, keep-prop = 0.8
d3 = np.random.rand(a3.shape[0],a3.shape[1]) #convert to bool
a3 = np.multiply(a3, d3) # element-wise mult
a3 /= keep-prop
```

上例中用的是Inverted-dropout，通过a3 /= keep-prop使得$z^{[4]} = W^{[4]}a^{[3]}+b^{[4]}$的期望不变，从而省去了平均值改变带来的复杂问题以及保留额外参数的麻烦。

**Dropout正则化的理解：**考虑单个神经元，由于每次训练它的各个输入可能会被随即删除，最终得到的参数$w$会尽量均摊给各个输入，从而达到近似于L2正则化的效果。这两种正则化的效果很相似，在使用方法上有所不同。Dropout正则化的缺点在于会引入更多的超参数(每层选取的keep-prop可以不一样，容易过拟合的复杂层可以keep-prop低一点，简单层可以设为1，即不进行dropout)。另外，Dropout会使得代价函数$J$没有固定的形式，因而很难调试程序，通常的做法是先关闭dropout函数确保程序正确性，在使用dropout进行训练。
Dropout通常应用在CV领域，因为CV的图像输入往往有很大的特征数，因而对模型来说数据通常是不够的。Dropout常被用来减少过拟合的情况。

**其他正则化方法：**

+ **数据扩增：**例如将图像反转、旋转、裁剪、扭曲以得到新数据
+ **Early Stopping**：通常随着训练过程的推进，$W$会从初始化的小值变得越来越大，因而在训练过程中提前终止可能得到较为合适的$W$。绘制出的$J_{train}$越来越小，而$J_{test}$通常先减后增，拐点就是一个较为合适的值。

注意，Early Stopping通常可以达到和L2正则化一样的效果，同时免去了寻找正则化参数的过程，计算代价小得多。但是通常设计机器学习系统时，我们将最小化$J$和防止过拟合作为两个独立的任务处理，通过**正交化原理**协调任务，但是Early Stopping将这两个任务融合起来，导致需要考虑的问题变得复杂得多，因而算力充足的情况下一般使用L2正则化。

### 3 归一化输入

**归一化 normalizing**一般分为两步：

1. 均值归零：$\mu = \frac{1}{m}\sum_{i=1}^mx^{(i)},~ x := x - \mu$
2. 方差归一：$\sigma^2 = \frac{1}{m} \sum_{i=1}^mx^{(i)2}, ~ x /=\sigma^2 $

通过归一化可以将输入的不同特征控制在相似范围内，均值控制为0，从而加速梯度下降法的执行。

### 4 梯度爆炸与梯度消失

在层数很深的神经网络中，如果每层的参数$W$比单位矩阵$I$大一点，那么每层计算出的$z$都会指数级增长；相反，如果每层的参数$W$比单位矩阵小一点，那么计算出的$z$会指数级衰减。这种问题被称为**梯度爆炸**和**梯度消失**。

**随机初始化：**谨慎地随机初始化可以缓解上述问题。我们希望$W$既不太大，也不太小。对于一个输入数量$n$较多的神经元，将$W$中每个元素设置得应该更小。
$W^{[l]}$=np.random.randn(shape) * np.sqrt($\frac{2}{n^{[l-1]}}$)     # ReLU通常设为$\sqrt{\frac{2}{n^{[l-1]}}}$
tanh激活函数通常设置为$\sqrt{\frac{1}{n^{[l-1]}}}$，一些研究中也会用$\sqrt{\frac{2}{n^{[l-1]}+n^{[l]}}}$。
通过合理的初始化，一开始激活函数的$z$大小适中，从而加速了梯度下降的过程。

### 5 梯度检验

在复杂的计算模型中，**反向传播算法**或相似的梯度下降算法实现时很容易出现bug。有时整个过程表现得很正常，$J(\Theta)$会逐渐下降到一个最小值，但最终模型仍有很大的误差。**梯度检验** gradient checking可以大幅降低出错得可能性，在这类模型中可使用梯度检验来确保代码的正确性。

**梯度检验原理：**利用**双侧差分**计算出导数近似值，将近似值与反向传播的结果比较.
(双侧差分$\frac{\part}{\part\theta}J(\theta)\approx\frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}$的准确性高于单侧差分$\frac{J(\theta+\epsilon)-J(\theta)}{\epsilon}$，通常$\epsilon=10^{-7}$即可)

**梯度检验：**将神经网络中的所有参数$W^{[1]},b^{[1]}\cdots,W^{[L]},b^{[L]}$展开成一个长向量$\theta=[\theta_1,\theta_2,\cdots,\theta_n]$，接下来验证：
$\frac{\part}{\part\theta_1}J(\theta)\approx\frac{J(\theta_1+\epsilon,\theta_2,\cdots,\theta_n)-J(\theta_1-\epsilon,\theta_2,\cdots,\theta_n)}{2\epsilon}$
$\frac{\part}{\part\theta_2}J(\theta)\approx\frac{J(\theta_1,\theta_2+\epsilon,\cdots,\theta_n)-J(\theta_1,\theta_2-\epsilon,\cdots,\theta_n)}{2\epsilon}$
$\cdots\cdots$
$$\frac{\part}{\part\theta_n}J(\theta)\approx\frac{J(\theta_1,\theta_2,\cdots,\theta_n+\epsilon)-J(\theta_1,\theta_2,\cdots,\theta_n-\epsilon)}{2\epsilon}$$
式子左边为反向传播的计算结果，右边为双侧差分计算出的近似值.

**梯度检验实践：**

+ 由于计算开销很大，在训练时应关闭梯度检验
+ 如果梯度检验检测出问题，检查相关项寻找bug
+ 记得考虑$J$的正则项
+ 梯度检验与dropout不可同时使用
+ 随机初始后训练一段时间保持检查，因为有时$W,b$较大时才能暴露出问题

****

## L2 - 优化算法 Optimization Algorithms



