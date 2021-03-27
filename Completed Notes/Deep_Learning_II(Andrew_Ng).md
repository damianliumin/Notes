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

### 1 Mini-Batch梯度下降法

**Mini-Batch梯度下降法**每次迭代时不会遍历整个数据集，而是从中选取一个子集，这个子集的大小为mini-batch size。计算$J$也根据选取的mini-batch，因而使用该方法得到的$J$会有噪声，但是总体趋势不会改变。相比于应用整个数据集，Mini-Batch的优化速度快很多。

**Notation：**
$X^{\{t\}}$: 第$t$个mini-batch的输入
$Y^{\{t\}}:$ 第$t$个mini-batch的label
$epoch$: 遍历全部数据一次，称为一代

mini-batch size为$m$时，即为**Batch梯度下降法**，size为1时，即为**随机梯度下降法**.

+ Batch梯度下降法：遍历整个数据集，因而速度很慢，$m<2000$时可使用
+ 随机梯度下降法：噪声问题可通过减小学习率解决，但是每次只处理一个样本就失去了向量化加速，因而速度也较慢
+ Mini-Batch梯度下降法：每次遍历部分数据，计算可向量化，速度最快

**mini-batch size：**这也是很重要的超参数，一般在$2^6\sim2^9$之间，使用时也需要考虑CPU/GPU内存大小

### 2 动量梯度下降法

**指数加权平均 Expotentially weighted average**是一个统计学概念，给定一组数据$\theta_0,\theta_1\cdots$，可通过下式计算：
$$
v_t=\beta v_{t-1} + (1-\beta)\theta_t
$$
$v_t$大约是包括它的前$\frac{1}{1-\beta}$个数据的平均值。$\beta$的选取会对$v_t$有很大的影响：较大的$\beta$使得均值变化较为平缓，但是对数据变化的适应能力弱；较小的$\beta$使得均值对数据变化的适应力强，但是均值本身噪声较大，易受异常值影响。

尽管计算的是均值的近似，指数加权平均的计算效率很高且内存占用少。

**指数加权平均的偏差修正：**如果使用上面提到的式子，一开始设置$v_0=0$，初始阶段指数加权平均的值偏低。我们可以用$\frac{v_t}{1-\beta^t}$来进行修正，可以发现$t$越大分母越接近1，但在初期偏差修正会起到重要作用。通常机器学习中如果不关注初始阶段的预测结果，可以忽略偏差修正。

**动量梯度下降法：**
**动量梯度下降法 Gradient Descent with Momentum**基于指数加权平均对一般梯度下降法做了优化。一般的梯度下降法会在梯度图中来回摆动，最终到达近似的最优值。动量梯度下降法对近几次的偏导值做了平均，使得水平方向的摆动减缓，从而更加接近最优解方向。

On iteration t:
	Compute $dW, db$ on the current mini-batch
	$v_{dW}=\beta v_{dW}+(1-\beta)dW$
	$v_{db} = \beta v_{db} + (1-\beta)db$
	$W := W -\alpha v_{dW}, ~ b := b - \alpha v_{db}$

动量梯度下降法的超参数有$\alpha$和$\beta$，后者常取0.9。该算法中无需偏差修正，迭代多次后就几乎没有偏差了。

### 3 RMSprop算法

**RMSprop**全称是root mean square prop，它采取与动量梯度下降法相似的方法加速。假设有两个参数$dw,db$，优化目标函数的梯度图在$w$方向更加狭长，因而在此方向上更平滑，$S_{dw}$较小，$w$的变化会被加快；$b$方向与之相反，RMSprop抑制了$b$方向的抖动，从而使得整个优化过程更加平滑地接近最优解。

On iteration t:
	Compute $dw, db$ on the current mini-batch
	$S_{dw}=\beta_2 S_{dw}+(1-\beta_2)(dw)^2$
	$S_{db} = \beta_2 S_{db} + (1-\beta_2)(db)^2$
	$w = w -\alpha \frac{dw}{\sqrt{S_{dw}}+\epsilon} ~ b := b - \alpha \frac{db}{\sqrt{S_{db}}+\epsilon}$

这里超参数$\beta_2$要与动量梯度下降法的$\beta$区分开，$\epsilon$是为了防止$S_{dw}$近似于0，一般取$10^{-8}$即可。

### 4 Adam优化算法

**Adam算法**代表Adaptive Moment Esitimation，是动量梯度下降法和RMSprop的结合。这个算法和前两者一样能适应各种网络结构，在长期的实践中久经考验。

$v_{dw}=0, S_{dw}=0, v_{db}=0,S_{db}=0$
On iteration t:
	Compute $dw, db$ on the current mini-batch
    $v_{dw}=\beta_1 v_{dw}+(1-\beta_1)dw,~~v_{db} = \beta v_{db} + (1-\beta)db$
    $S_{dw}=\beta_2 S_{dw}+(1-\beta_2)(dw)^2,~~S_{db} = \beta_2 S_{db} + (1-\beta_2)(db)^2$
    $v_{dw}^{corrected}=\frac{v_{dw}}{1-\beta_1^t},~~v_{db}^{corrected}=\frac{v_{db}}{1-\beta_1^t}$             # typically we use bias correction here
	$S_{dw}^{corrected}=\frac{S_{dw}}{1-\beta^t_2},~~S_{db}^{corrected}=\frac{S_{db}}{1-\beta_2^t}$
	$w := w -\alpha \frac{v_{dw}^{corrected}}{\sqrt{S_{dw}^{corrected}}+\epsilon} ~ b := b - \alpha \frac{v_{db}^{corrected}}{\sqrt{S_{db}^{corrected}}+\epsilon}$

超参数选取如下：

+ $\alpha$: 需要实践挑选
+ $\beta_1:$ 推荐使用0.9
+ $\beta_2$: 推荐使用0.999
+ $\epsilon:$ 对结果影响不大，推荐$10^{-8}$

### 5 学习率衰减

Mini-Batch算法中，一开始用稍大的$\alpha$可以快速靠近最优解，但这会使得在最优解附近徘徊很久。如果$\alpha$能够逐渐衰减，训练过程会得到加快。通常采用如下方法实现学习率衰减：

+ $\alpha=\frac{1}{1+decayRate\times epochNum}\alpha_0$
+ $\alpha = 0.95^{epochNum} \alpha_0$ # 指数衰减
+ $\alpha = \frac{1}{\sqrt{epochNum}}\alpha_0$
+ $\alpha = \frac{1}{\sqrt{t}}\alpha_0$ # 迭代次数
+ 离散数值衰减

### 6 局部最优的问题

过去人们经常担心优化算法会被困在一个很差的局部最优解。不过随着深度学习理论的发展，人们发现在高维空间中，优化目标函数落入局部最优的概率极低，因为$J$对每个变量求偏导都为0的绝大多数情况会发生在马鞍面上。在这种平缓地带优化速度会很慢，因而上述加速的优化算法可以发挥出更大的优势。

****

## L3 - 超参数调试 Hyperparameter Tuning

### 1 调试处理

深度学习中有大量的超参数要调试，通常学习率$\alpha$是最重要的超参数，mini-batch size、隐藏单元数量、动量梯度下降法的$\beta$次之，还有一些参数不那么重要，或是选常用值即可。

由于很多时候很难比较超参数的重要性，深度学习中通常**随机选取**一系列超参数的组合进行训练，而非像传统机器学习那样按照grid的方式选取，然后从中挑选出效果最好的那个模型。

**随机选取方法：**通常选取超参数要在一定范围内，对于部分参数可以在普通数轴上随机选，对于$\alpha$等需要在对数数轴上随机选。例如$\alpha$在0.0001~1之间，可用alpha=10**(-4 * np.random.rand()实现)

### 2 Batch归一化

我们已经通过归一化输入加速了训练，实际上对于隐藏层，我们也可以用**Batch归一化 Batch Norm**使得$z$的均值和方差标准化。对于NN中某一层的$z$，计算出：
$\mu =\frac{1}{m}\sum_i z^{(i)} $
$\sigma^2 = \frac{1}{m}\sum_i (z^{(i)} - \mu)^2$
$z_{norm}^{(i)}=\frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}$
$\tilde{z}^{(i)}=\gamma z_{norm}^{(i)}+\beta$
$z_{norm}^{(i)}$是对第$i$个样本零均值化和方差标准化的结果，实践中有时不希望$z$集中在0附近，例如tanh函数在0附近近似线性，因而添加了可训练参数$\gamma,\beta$，计算出$\tilde{z}^{(i)}$。

**Batch Norm与神经网络：**
对与某层神经网络，添加Batch Norm后工作机制如下：
$a^{[l-1]}$ --($w^{[l]}, b^{[l]}$)--> $z^{[l]}$ --($\beta^{[l]},\gamma^{[l]}$, BN)--> $\tilde{z}^{[l]}$ ----> $a^{[l]}=g(\tilde{z}^{[l]})$
在实现上有一个细节需要注意：$z^{[l]}=w^{[l]}a^{[l-1]}+b^{[l]}$中，$b^{[l]}$会被Batch Norm消除掉，因而实践中可以省略这个参数。另外，$\beta^{[l]}$和$\gamma^{[l]}$的维度是$(n^{[l]},1)$。添加Batch Norm后梯度下降法如下：
for $t=1\cdots$ num of mini-batches
	Compute forward-prop on $X^{\{t\}}$ (in each layer replace $z^{[l]}$ with $\tilde{z}^{[l]}$)
	Use backprop to compute $dw^{[l]},d\beta^{[l]},d\gamma^{[l]}$
	Update parameters $w^{[l]}:=w^{[l]}-\alpha w^{[l]}\cdots$ 
	\# This also works with other optimization algorithms

**Batch Norm作用：**
1.Batch Norm和输入归一化一样，将$z$的大小控制在一定范围内从而加速算法。
2.Batch Norm最大的作用在于增强不同层的独立性。在深层神经网络的训练过程中，前几层参数$w,b$的变化可能会引起后面某层的输入$a^{[l-1]}$较大的变化，这就引发了covariate shift的问题，使得第$l$层不得不重新训练以得到准确的函数。通过Batch Norm的$\beta$和$\gamma$可以将$a^{[l-1]}$的均值和方差确定在一个较稳定的范围内，从而减弱浅层参数的变化对深层的影响。
3.Batch Norm也有一定的正则化作用。由于每次训练的$\sigma,\mu$都是在mini-batch上得到的，有一定的噪声，这会带来与dropout类似得效果。(测试和使用模型时对单一样本无法得到$\sigma,\mu$，通常在训练时对mini-batch用指数加权平均计算这两个值)

### 3 Softmax回归

多元分类问题中，输出$\hat{y}$应该归一化，即各个输出的和为1。通常用Softmax层作为输出层来实现这一点。Softmax回归在没有隐藏层的情况下，得到的决策边界是线性的，有了隐藏层之后可以得到复杂的决策边界。与Softmax对应的时Hardmax，它会将最大概率值映射到1，其余映射到0，相比之下Softmax的映射会温和很多。

**Softmax层：**首先计算出$z^{[L]}=w^{[L]}a^{[L-1]}+b^{[L]}$，然后用**Softmax激活函数**计算出$a^{[L]}=g(z^{[L]})$，其中$a^{[L]}_i = \frac{e^{z^{[L]}_i}}{\sum_{j=1}^C e^{z^{[L]}_j}}$.

对于单个样本，Softmax的损失函数如下：
$$
\mathcal{L}(y,\hat{y})=-\sum_{j=1}^C y_j\log\hat{y}_j
$$
当$C=2$时，Softmax实质上会退化为Logistic回归。