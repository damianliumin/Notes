# Machine Learning III

Author: Daniel Liu
Contact me:  191240030@smail.nju.edu.cn
Course: Machine Learning - AndrewNg

## L13 - 聚类 Clustering

### 1 无监督学习

监督学习中，训练集有标签：$\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\cdots,(x^{(m)},y^{(m)})\}$. 但在无监督学习中训练集变为$\{x^{(1)}, x^{(2)}, \cdots,x^{(m)} \}$，算法需要找到隐含在数据中的结构。例如，在数据中找到几组**簇 clusters**，这类算法被称为**聚类算法 clustering algorithm.**

### 2 K-means算法

**K-means算法**是一种应用广泛的聚类算法，将数据分为K个簇.

算法首先随机选取$K$个**聚类中心 cluster centroids**，加下来迭代。每次迭代分为两步：首先是**簇分配 cluster assignment**，将训练集中所有点划分到与之最近的聚类中心。接下来是**移动聚类中心 move centroids**，将所有分配到某聚类中心的点求平均，得到聚类中心的新位置。

**K-means Algorithm:**
Input:
-K (number of clusters)
-Training Set $\{x^{(1)}, x^{(2)}, \cdots,x^{(m)} \}$ (**drop** $x_0=1$ convention)

Randomly initialize $K$ cluster centroids $\mu_1,\mu_2,\cdots,\mu_K\in \mathbb{R^n}$
Repeat \{
	for $i=1$ to $m$ 
		$c^{(i)} :=$ index (1~$K$) of cluster centroid closest to $x^{(i)}$ 
	for $k=1$ to $K$ 
		$\mu_k:=$ mean of points assigned to cluster $k$
\}

有些应用场景下数据没有清晰地分成若干簇，此时K-means算法仍然可以应用。另外，若训练时某个聚类中心没有分配到任何点，通常的做法是将其删除，有时也可考虑重新随机分配聚类中心进行训练。

### 3 优化目标

**Notation:**
$c^{(i)}$：$x^{(i)}$被分配到的簇的下标
$\mu_k$：下标为$k$的簇 ($\mu_k\in\mathbb{R}^n$)
$\mu_{c^{(i)}}$：$x^{(i)}$被分配到的簇

和线性回归、SVM、神经网络等算法一样，K-means算法有优化目标：
$$
\min_{c^{(1)}\cdots c^{(m)},\\\mu_1\cdots \mu_K}J(c^{(1)},\cdots,c^{(m)},\mu_1,\cdots,\mu_K)=\frac{1}{m}\sum_{i=1}^m\left\| x^{(i)}-\mu_{c^{(i)}} \right\|^2
$$
K-means算法中簇分配实际上是假定$\mu_1,\cdots,\mu_K$固定，最小化代价函数$J(\cdots)$，移动聚类中心是假定$c^{(1)},\cdots,c^{(m)}$固定，最小化代价函数$J(\cdots)$.

### 4 随机初始化

K-means中一个很好的初始化方法是：随机挑选$K$**个训练样本**设为聚类中心.

**局部最优解：**K-means算法可能获得局部最优解。在K较小的情况下，可以通过多次(通常50~1000次)随机初始化的方式获得一个最好的局部最优解：
For $i=1$ to 100 {
	Randomly initialize K-means
	Run K-means, get $c^{(1)},\cdots,c^{(m)},\mu_1,\cdots,\mu_K$.
	Compute cost function $J(c^{(1)},\cdots,c^{(m)},\mu_1,\cdots,\mu_K)$
}
Pick clustering that gave lowest $J(c^{(1)},\cdots,c^{(m)},\mu_1,\cdots,\mu_K)$.
对于较大的K，多次随机初始化作用通常不太显著.

### 5 聚类数量

目前没有很好的自动化方法来决定聚类数量$K$，通常需要手动选择$K$。

**肘部法则 Elbow method：**增加$K$观察$J(\cdots)$，后者下降速度由快突然转慢的地方，即“肘部”，或许是很好的选择。但是有时绘制出的曲线无法清晰观察到一个“肘部”，这是肘部法则不再适用。

另外，很多时候需要根据后续目的(商业目的等)来确定聚类数量。

***

## L14 - 降维 Dimensionality Reduction

### 1 降维目标

**数据压缩：**将数据从3D降为2D，或从2D将为1D，这样可以减小数据的内存或磁盘空间，并且可以提高学习的速度.

**可视化：**高维数据降位低维以便可视化.

### 2 主成分分析 PCA

**主成分分析 Principle Component Analysis**是一种常用的降维算法，其核心思想是在高维空间中寻找低维平面，以最小化**投影误差 projection error**，即每个点与其投影点距离的平方和。

**数据预处理：**
给定训练集$\{x^{(1)}, x^{(2)}, \cdots,x^{(m)} \}$
首先**均值归一化**，计算$\mu_j=\frac{1}{m}\sum_{i=1}^mx_j^{(i)}$，用$x^{(i)}_j-\mu_j$替代$x_j^{(i)}$；
如果有必要，可以进行**特征缩放**，$x_j^{(i)}=\frac{x_j^{(i)}-\mu_j}{s_j}$.

**PCA算法：**PCA算法目标是将$n$维向量降到$k$维。首先计算出样本的**协方差矩阵 covariance matrix**，$\Sigma=\frac{1}{m}\sum_{i=1}^m (x^{(i)})(x^{(i)})^T$。接着用**奇异值分解SVD**(或其他方法)计算出$\Sigma$的特征向量，计算得到的特征向量按照列向量的形式存放在矩阵$U\in\mathbb{R}^{n\times n}$中。取$U$的前$k$列得到$U_{reduce}\in\mathbb{R}^{n\times k}$，则可将$x\in\mathbb{R}^n$降维：$z=U_{reduce}^Tx$，其中$z\in\mathbb{R}^k$。说明如此得到的低维平面具有最小的投影误差需要复杂的数学证明，在此不详细展开。

**PCA Algorithm：**
Compute covariance matrix: $\Sigma=\frac{1}{m}\sum_{i=1}^m (x^{(i)})(x^{(i)})^T$
Compute eigenvectors of matrix $\Sigma$: $U$, obtain $U_{reduce}$
$z = U_{reduce}^Tx$

**压缩重现：**$x_{approx}^{(i)}=U_{reduce}z^{(i)}$，通过这种方式可以将压缩后的数据重现为原高维数据的近似值。

### 3 主成分数量

PCA中需要选择主成分的数量$k$，这时需要考虑两个量.
**平均投影误差** average squared projection error: $\frac{1}{m}\sum_{i=1}^m\left\| x^{(i)}-x_{approx}^{(i)} \right\|^2$
**全变分** total variation: $\frac{1}{m}\sum_{i=1}^m\|x^{(i)} \|^2$

通常，选择满足下式的最小的$k$：
$$
\frac{\frac{1}{m}\sum_{i=1}^m\left\| x^{(i)}-x_{approx}^{(i)} \right\|^2}{\frac{1}{m}\sum_{i=1}^m\|x^{(i)} \|^2}\leq0.01
$$
这里0.01也可换成0.05之类的数据。将0.01理解为：保留99%的variance。

显然上式计算量很大。之前在用SVD方法求$U$时，还会得到奇异值矩阵$S$，$S$是一个$n\times n$矩阵，除对角线元素外都是0。可以得到下列关系：
$$
\frac{\frac{1}{m}\sum_{i=1}^m\left\| x^{(i)}-x_{approx}^{(i)} \right\|^2}{\frac{1}{m}\sum_{i=1}^m\|x^{(i)} \|^2}=1-\frac{\sum_{i=1}^kS_{ii}}{\sum_{i=1}^m S_{ii}}
$$
这样，我们就可以快速用$\frac{\sum_{i=1}^kS_{ii}}{\sum_{i=1}^m S_{ii}}\geq 0.99$找到$k$。

### 4 PCA应用

**加速学习：**通过将高维$x^{(i)}$降为低维$z^{(i)}$，可以获得特征更少的训练集$\{ (z^{(1)},y^{(1)}),(z^{(2)},y^{(2)}),\cdots,(z^{(m)},y^{(m)}) \}$。注意应该只在训练集上运行PCA，得到的$x\rightarrow z$的映射中考虑了均值归一化、特征缩放和$U_{reduced}$等。这个映射在$x_{cv}^{(i)}$和$x^{(i)}_{test}$上也能正常应用。这里PCA要注意对原信息的保留率。

**可视化：**将数据降为2D或3D可以可视化。

**不建议的应用场景：**不应用PCA减少特征数来降低过拟合的可能性。PCA训练时不考虑标签$y^{(i)}$，可能会丢失一些信息。另外，即使保留了95%或99%的variance，通常正则化的效果也至少和PCA一样。
另外，在设计项目时不建议直接使用PCA处理数据，除非有充分的证据表明用$x^{(i)}$无法运行。建议在使用PCA之前先用原数据进行训练。

***

## L15 - 异常检测 Anomaly Detection

### 1 异常检测

**模型描述：**给定数据集$\{x^{(1)},x^{(2)},\cdots,x^{(m)} \}$，判断$x_{test}$是否异常。建立一个概率模型$p(x)$，当$p(x)<\epsilon$时视为异常。

**异常检测**中要用到**高斯分布**：$x\sim \mathcal{N}(\mu,\sigma)$，可算出$p(x;\mu,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2})$. 对每个特征求高斯分布后，将所有的$p(x_j;\mu_j,\sigma_j)$求积即可。

**异常检测算法：**
Choose features $x_i$ that you think might be indicative of anomalous examples.
Fit parameters $\mu_1,\cdots,\mu_n,\sigma_1^2,\cdots,\sigma^2_n$.
	\- $\mu_j=\frac{1}{m}\sum_{i=1}^mx_j^{(i)}$
	\- $\sigma_j^2=\frac{1}{m}\sum_{i=1}^m(x_j^{(i)}-\mu_j)^2$
Given new examples $x$, compute $p(x)$:
	\- $p(x)=\prod_{j=1}^n p(x_j;\mu_j,\sigma_j^2)=\prod_{j=1}^n \frac{1}{\sqrt{2\pi}\sigma_j}\exp{(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})}$ 
Anomaly is $p(x)<\epsilon$

### 2 模型评估

为了评估异常检测模型的表现，需要将数据集进行划分。数据集中包含正常样本和异常样本，通常前者远多于后者。将正常样本按照6:2:2划分到训练集、验证集和测试集，异常样本分到验证集和测试集。训练集$\{x^{(1)},\cdots,x^{(m)} \}$没有标签，验证集$\{x_{cv}^{(1)},\cdots,x_{cv}^{(m_{cv})} \}$和测试集$\{x_{test}^{(1)},\cdots,x_{test}^{(m_{test})} \}$有标签，1表示异常，0表示正常。

数值评价标准可选取查准率、召回率以及$F_1$-Score。另外，异常检测的$\epsilon$也可通过验证集来选取。

### 3 异常检测与监督学习

**异常检测：**
\- 正类样本较少(0-20很常见)，负类样本很多
\- 异常种类较多，且样本不足以支持算法学习到正类的特征，未来的正类可能和现有正类差异很大，这些情况下考虑异常检测

**监督学习：**
\- 正类样本和父类样本都很多
\- 样本足以让算法学习到正类的特征，且未来的正类和现有正类较为相似，则可以考虑监督学习

### 4 特征选取

**非高斯分布特征：**选择不符合高斯分布的特征通常也能使算法运行，但是有一些方法可以将特征转化为近似高斯分布，例如$\log x$, $x^\alpha$等。

**误差分析：**当某个异常样未被识别出时，可专门分析它，找出其与众不同的特征，并将该特征加入算法内。

**特征选取：**通常再异常情况下变化幅度大的特征(特别小或特别大)非常合适，可以再原特征中选取，用$\frac{a}{b}$等方式组合出新特征用于训练。

### 5 多变量高斯分布

有时样本的特征$x_1$和$x_2$都在正常范围内，但是其组合是异常的，这是因为$x_1$和$x_2$有相关性。普通的异常检测可能无法捕捉到异常，这时可以考虑**多变量高斯分布**：$p(x;\mu,\Sigma)=\frac{1}{(2\pi)^\frac{n}{2}|\Sigma|^\frac{1}{2}}\exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$，其中$\Sigma$是协方差矩阵。

**异常检测算法：**
Choose features $x_i$ that you think might be indicative of anomalous examples.
Fit model $p(x)$ by setting:
	\- $\mu_j=\frac{1}{m}\sum_{i=1}^mx_j^{(i)}$
	\- $\Sigma=\frac{1}{m}\sum_{i=1}^m (x^{(i)}-\mu)^T(x^{(i)}-\mu)$
Given a new example x, compute:
	\- $p(x)=\frac{1}{(2\pi)^\frac{n}{2}|\Sigma|^\frac{1}{2}}\exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$
Flag 1 if $p(x)<\epsilon$.

原模型实际上是多变量高斯分布在$\Sigma$为对角矩阵时的特殊情形，其沿轴对称分布.

**原模型：**$\prod_{j=1}^n p(x_j;\mu_j,\sigma_j^2)$
\- 可以手动对原特征$x_1, x_2$进行组合得到新特征，用新特征进行训练
\- 计算代价小，对较大的特征数$n$适应能力强
\- 在训练样本数量$m$较小时也能运行

**多元高斯分布：**$p(x)=\frac{1}{(2\pi)^\frac{n}{2}|\Sigma|^\frac{1}{2}}\exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$
\- 自动捕捉特征间关联
\- 计算代价高
\- 要求$m>n$，否则$\Sigma$可能不可逆($\Sigma$不可逆的概率很小，如果发生了通常是因为$m<n$或存在冗余特征)。通常在$m\gg n$时才会考虑多元高斯分布的异常检测.

原模型在实践中应用更加广泛。

***

## L16 - 推荐系统 Recommender Systems

### 1 问题描述

**推荐系统**在现实中的应用场景非常广泛，很多科技公司都致力于提高推荐系统的性能。另外，我们已经看到特征的选取在机器学习中的重要性，现在一些系统能够自动选取合适的特征，而推荐系统就是其中之一。

推荐系统问题的一个主要形式是：根据已有的用户信息，填补用户未知的信息。以电影评分为例，大量用户会为许多电影评分，我们需要根据已知信息，推测用户对其未看过的电影的评分。

**Notation:**
$n_u$：用户数量
$n_m$：电影数量
$r(i,j)$：当用户$i$给电影$j$评过分时为1
$y^{(i,j)}：$用户$i$给电影$j$的评分，在$r(i,j)=1$时有定义
$m^{(j)}$：用户$j$评价的电影数量

### 2 基于内容的推荐算法

**基于内容推荐算法 content based recommendation**假设每部电影已知其特征向量$x^{(i)}$(该算法中加上$x_0=1$)，希望为每个用户找到参数$\theta^{(j)}$，通过$(\theta^{(j)})^Tx^{(i)}$可以计算出用户$i$对电影$j$的评价.
为了找到$\theta^{(j)}$，需要确定优化目标。对于单个用户$j$，优化目标为：$\min_{\theta^{(j)}}\frac{1}{2m^{(j)}}\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2m^{(j)}}\sum_{k=1}^n(\theta^{(j)}_k)^2$
整体形式和线性回归很像，第二项时正则项。式子可以乘以$2m^{(j)}$以简化常数。

推荐体统需要为每一个用户都进行优化，因而**优化目标**是：
$$
\min_{\theta^{(1)},\cdots,\theta^{(n_u)}}\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n(\theta^{(j)}_k)^2
$$
使用梯度下降法进行优化时，循环中的式子为：
$\theta_k^{(j)} := \theta^{(j)}_k -\alpha\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_k^{(i)} $ (for $k=0$)
$\theta_k^{(j)} := \theta^{(j)}_k -\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_k^{(i)} + \lambda\theta_k^{(j)} \right) $ (for $k\neq 0$)
从这里也可看出基于内容推荐算法与线性回归的相似之处.

### 3 协同过滤

基于内容的推荐算法假设每部电影的特征都是已知的，但现实中确定每部电影的各个特征代价是很大的。如果用户在提供评分之外，也提供了自己对每种电影特征的偏好程度，即参数$\theta^{(j)}$，我们也可以通过类似的办法确定电影的参数$x^{(i)}$。
这种情况下，对于每部电影优化目标是：$\min_{x^{(i)}}\frac{1}{2}\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{k=1}^n(x^{(i)}_k)^2$
考虑整个系统，优化目标是：
$$
\min_{x^{(1)},\cdots,x^{(n_m)}}\frac{1}{2}\sum_{i=1}^{n_m}\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n(x^{(i)}_k)^2
$$
可以发现，假设每个用户都给几部电影评了分，而每部电影都有几个用户评价过，这时可以在给定$\theta^{(1)},\cdots,\theta^{(n_u)}$的情况下学习$x^{(1)},\cdots,x^{(n_m)}$，也可以在给定后者的情况下学习前者。

**协同过滤 collaborative filtering**的思想是：先猜想$\theta$，通过$\theta$可以学习$x$，然后再学习$\theta$，如此循环往复，不断优化$\theta,x$。这个过程中每个用户提供评分信息都会提高推荐系统的准确性，因而谓之协同。在此基础上，**协同过滤算法**省去了$x$和$\theta$之间的来回迭代，用一个代价函数同时优化$\theta$和$x$：
$$
J(x^{(1)},\cdots,x^{(n_m)},\theta^{(1)},\cdots,\theta^{(n_u)})=\frac{1}{2}\sum_{(i,j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\\\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n(x^{(i)}_k)^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n(\theta^{(j)}_k)^2
$$
这个代价函数合并了之前两个过程的优化目标，可以实现同时优化$\theta$和$x$，从而将这个算法变得和其他机器学习算法类似。

要注意的是，由于特征$x$需要学习得到，这里放弃$x_0=1$的做法，规定$x\in\mathbb{R^n}$

**协同过滤算法：**
Initialize $x^{(1)},\cdots,x^{(n_m)},\theta^{(1)},\cdots,\theta^{(n_u)}$ to small random values.
Minimize $J(x^{(1)},\cdots,x^{(n_m)},\theta^{(1)},\cdots,\theta^{(n_u)})$ with gradient descendent for every $j=1,\cdots,n_u$, $i=1,\cdots, n_m$:
	$x_k^{(i)}:=x_k^{(i)}-\alpha\left(\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})\theta_k^{(j)} + \lambda x_k^{(i)} \right)$
	$\theta_k^{(j)} := \theta^{(j)}_k -\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_k^{(i)} + \lambda\theta_k^{(j)} \right) $
For a user with parameters $\theta$ and a movie with (learned) features $x$, predict a star rating of $\theta^Tx$.

### 4 协同过滤的应用与实现

**向量化实现**：用矩阵$Y$记录用户评分，其中可能存在未定义数据。计算用户预估评分时，可用矩阵$X=\left[ \begin{matrix} (x^{(1)})^T\\(x^{(2)})^T\\\cdots\\(x^{(n_m)})^T \end{matrix} \right]$记录电影特征，矩阵$\Theta=\left[ \begin{matrix} (\theta^{(1)})^T\\(\theta^{(2)})^T\\\cdots\\(\theta^{(n_u)})^T \end{matrix} \right]$记录用户参数，$X\Theta^T$即为预估分数。

**相似推荐：**算法通常可以学习到一些非常关键的特征，但是这些特征对人们来说往往难以理解。在推荐相似电影时，可以根据$\|x^{(i)}-x^{(j)}\|$来计算出与用户当前看过的电影中最相似的几个。

**均值归一化：**如果一个用户从未评价过任何电影，训练出的$\theta$会是全零的，这样其预估的电影评分都会是0。如果这不是我们所希望的，可以采用均值归一化。先对$Y$每行有效数据求平均得到向量$\mu$，然后$Y$每列减去$\mu$进行训练。然后计算预估值的时候，推测用户$j$给电影$i$的评分是$(\theta^{(j)})^Tx^{(i)}+\mu_i$，这样没有评价记录的用户算出的预估评分即为平均值。

***

## L17 - 大规模机器学习 Large Scale Machine Learning

### 1 大数据集

很多应用场景中，数据量会达到上亿的级别。在很多模型中，这样的$m$会使得计算代价相当大。例如，梯度下降法中的$\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$需要对$m$项求和。
可以考虑在$m$条数据中随机选择一部分进行训练，可通过绘制$J_{train}(\theta)$和$J_{cv}(\theta)$随着$m$变化的曲线来观察多大的$m$是可行的。
另外，也有一些方法可以用来解决过大的$m$带来的计算上的问题。

### 2 随机梯度下降法

许多机器学习算法中都有一个代价函数或优化目标，我们通常用梯度下降法进行优化。以线性回归为例，我们常用的算法被称作批量**梯度下降法 Batch Gradient Descendent**，因为每次循环都需要遍历数据集中全部$m$条数据：

**批量梯度下降法：**
$J_{train}(\theta)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2$
Repeat \{
	$\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$
	(for every $j=0,\cdots,n$)
\}

在$m$达到上亿条时，每次梯度下降法走一步遍历所有数据的开销都非常大。为了大规模数据集上批量梯度下降法的问题，我们可以采用**随机梯度下降法 Stochastic gradient descendent：**

**随机梯度下降法：**
$cost(\theta,(x^{(i)},y^{(i)}))=\frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2$ 
$J_{train}(\theta)=\frac{1}{m}\sum_{i=1}^mcost(\theta,(x^{(i)},y^{(i)}))$
Randomly shuffle (reorder) training examples.
Repeat \{
	for $i=1,\cdots,m$ \{
		$\theta_j:=\theta_j-\alpha(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$
		(for $j=0,\cdots,m$)
	\}
}

随机梯度下降法首先将数据随机打乱，然后在内循环中遍历每条数据。与批量梯度下降法不同的是，它在遍历每条数据时都会更新$\theta$。由于$m$很大，通常外循环进行1次就可以得到很好的假设函数了，一般外循环不超过10次。
从图像上看，批量梯度下降法每次循环都移动一步，移动形成的曲线通常平滑并直接指向最低点，最终在最低点收敛。随机梯度下降法由于每次用一条数据进行优化，因此移动有随机性，但是大体趋势还是向最低点靠近，最终在最低点附近的区域内活动。总体来看，随机梯度下降法的速度远高于批量梯度下降法，并且能得到一个很不错的假设函数。

**随机梯度下降法收敛：**批量梯度下降法中为了判断是否收敛，通常会绘制$J_{train}(\theta)$随着迭代次数的变化曲线，若最终趋于水平则已收敛。随机梯度下降法计算$J_{train}(\theta)$的代价过大，我们旨在内循环每次迭代时计算$cost(\theta,(x^{(i)},y^{(i)}))$，每进行一定次数迭代(如1000次)就绘制出这些$cost$的平均值，如果这条曲线趋于水平则收敛。另外，随机梯度下降法的曲线噪声可能比较大，可以增大迭代次数的间隔来求平均值、绘图。

**学习速率**$\alpha$：通常情况下$\alpha$用常数即可。随机梯度下降法最后会在最小值附近徘徊，但这影响不大。如果想要更精确的结果，也可以让$\alpha$逐渐变小，从而提高精度，例如使用$\alpha=\frac{constant_1}{iterationNum+constant_2}$.

### 3 Mini-Batch梯度下降法

批量梯度下降每次迭代处理$m$条数据，随机梯度下降每次处理1条数据。可以设想一种算法，每次处理$b$条数据，这就是**Mini-Batch梯度下降法**：

**Mini-Batch梯度下降法：**
Say $b=10, m=1000$
Repeat \{
	for $i=1,11,\cdots,991$ \{
		$\theta_j:=\theta_j-\alpha\frac{1}{10}\sum_{k=i}^{i+9}(h_\theta(x^{(k)})-y^{(k)})x^{(k)}_j$
		(for every $j=0,1,\cdots,n$)
	}
\}

Mini-Batch梯度下降法在合适的向量化计算下有时比随机梯度下降法更快.

### 4 在线学习

**在线学习 online learning**放弃了固定数据集的概念，在有连续数据流的情况下可以正常运行，在很多网站中会运用。另外，在线学习还有适应用户偏好变化的功能。如果不能保证不断的数据流，不建议使用在线学习。

假设某网站提供运输服务，特征$x$包含用户信息、出发地/目的地以及提供的价格，$y=1$表示用户购买了服务。为了训练参数$\theta$以预测用户的购买概率$p(y=1|x,\theta)$，可以用在线学习算法：

**在线学习：**
Repeat forever \{
	Get $(x,y)$ corresponding to user.
	Update $\theta$ : $\theta_j:=\theta_j-\alpha (h_\theta(x)-y)x_j$ (for $j=0,1,\cdots,n$)
\}

### 5 Map-reduce

一些规模庞大的机器学习问题用随机梯度下降法都难以解决，有时我们需要考虑用多台机器并行运算以提高计算速度，这就是**Map-reduce方法**。

假设$m=400$，需要计算$\theta_j:=\theta_j-\alpha\frac{1}{400}\sum_{i=1}^{400}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$，可以将求和的部分分到四台计算机上运行，例如第一台计算$temp_j^{(1)}=\sum_{i=1}^{100}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$。最后斯台计算机将结果传输给中心处理器，由它计算$\theta_j:=\theta_j-\alpha\frac{1}{400}(temp_j^{(1)}+temp_j^{(2)}+temp_j^{(3)}+temp_j^{(1=4)})$。如果不考虑网络延迟等因素，理论上计算速度会提高四倍。

需要大量求和的机器学习算法基本都可以用Map-reduce提速。另外，现在很多计算机的处理器都是多核的，也可以在多个核上应用Map-reduce方法。

***

**L18 - 机器学习应用: Photo OCR**