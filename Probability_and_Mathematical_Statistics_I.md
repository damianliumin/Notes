# Probability and Mathematical Statistics I

Author: Daniel Liu
Contact Me: 191240030@smail.nju.edu.cn
Course: Probability and Mathematical Statistics. NJU

## Ch1 随机事件与概率

### 1 随机事件及其运算

**概念：**随机试验，基本事件，样本空间，随机事件
**事件的关系与运算：**关系：事件的包含与等价，并、交、差，互不相容事件，对立事件，完备事件组；运算：交换律、结合律、分配律、德-摩根律(推广得对偶原理)
**概率的公理化定义**

### 2 古典概型

基本事件n个且等可能，事件A包含的基本事件数有m个，定义A的概率：
$$
P(A)=\frac{m}{n}=\frac{\text{A所包含的基本事件数}}{\text{基本事件总数}}
$$

### 3 几何概率

随机试验样本空间$\Omega$对应于测度有限的几何区域S，若样本点落在S内某一区域G的事件A的概率与区域G的测度成正比，而与G的形状无关，则：
$$
P(A)=\frac{G的测度}{S的测度}
$$

### 4 统计概率

在n次重复进行的随机试验中，当n很大时，事件A出现的频率$f_n(A)=\frac{n_A}{n}$稳定地在某一数值p附近摆动，称稳定值p为事件A发生的概率，记为：
$$
P(A)=p
$$

### 5 条件概率

已知事件B发生的条件下，事件A发生的条件概率：
$$
P(A|B)=\frac{P(AB)}{P(B)}
$$
**条件概率性质：**

1. $P(\Omega|B)=1,P(\emptyset|B)=0$
2. $P(A|B)=1-P(\overline{A}|B)$
3. $P(A_1\cup A_2|B)=P(A_1|B)+P(A_2|B)$-$P(A_1A_2|B)$
4. $A\subset B,P(A|B)=\frac{P(A)}{P(B)}$

**5.1 乘法公式**
$$
P(AB)=P(A)P(B|A)=P(B)P(A|B)
$$
对n个事件，若$P(A_1A_2\cdots A_{n-1})>0$，则有：
$$
P(A_1A_2\cdots A_n)=P(A_1)P(A_2|A_1)P(A_3|A_1A_2)\cdots P(A_n|A_1A_2\cdots A_{n-1})
$$
**5.2 全概率公式**
设$B_1,B_2,\cdots,B_n$是完备事件组，对任何事件$A$，有
$$
P(A)=\sum_{i=1}^nP(B_i)P(A|B_i)
$$
**5.3 贝叶斯公式**
设$B_1,B_2,\cdots,B_n$是完备事件组，对任何事件$A$，只要$P(A)>0$，则有
$$
P(B_j|A)=\frac{P(AB_j)}{P(A)}=\frac{P(B_j)P(A|B_j)}{\sum^n_{i=1}P(B_i)P(A|B_i)}
$$
这里$P(B_j)$为先验概率。贝叶斯公式可以在已知事件A发生的条件下计算出条件$B_j$发生的概率。

### 6 独立性

两个事件A, B满足$P(AB)=P(A)P(B)$，则其相互独立。对于多个事件$A_1,A_2,\cdots,A_n$，若满足下面$2^n-n-1$个等式：
$P(A_iA_j)=P(A_i)P(A_j)~~~(1\leq i < j \leq n)$
$P(A_iA_jA_k)=P(A_i)P(A_j)P(A_k)~~~(1\leq i < j < k \leq n)$
$\cdots \cdots~\cdots\cdots$
$P(A_1A_2\cdots A_n)=P(A_1)P(A_2)\cdots P(A_n)$
则称事件$A_1,A_2,\cdots,A_n$相互独立.
(注意事件两两独立$\neq$事件相互独立)

**独立性的性质：**

1. $A,B$相互独立且$P(A)>0, P(B)>0$，则$P(A|B)=P(A),P(B|A)=P(B)$
2. $A$与$B$，$\overline{A}$与$B$，$A$与$\overline{B}$，$\overline{A}$与$\overline{B}$事件中一对相互独立，则其他也相互独立
3. $P(A)=0$或$P(A)=1$，则$A$与任意事件独立
4. **分组独立性：**设事件$A_1,A_2,\cdots,A_n$相互独立，将其任意分为没有公共事件的$k$个组，每个组任意作事件运算得到$k$个新事件，则这$k$个新事件相互独立.

**独立事件至少发生一个概率计算：**
古典概率下，至少发生一个概率的计算通常很复杂，但是独立性可以简化计算：
$$
P(\bigcup_{i=1}^nA_i)=1-P(\overline{A_1\cup A_2\cup \cdots \cup A_n})\\=1-P(\overline{A_1}\overline{A_2}\cdots\overline{A_n})=1-P(\overline{A_1})P(\overline{A_2})\cdots P(\overline{A_n})
$$

若概率都为p，则$P(\bigcup_{i=1}^nA_i)=1-(1-p)^n$。$n\rightarrow \infty$时，概率趋于1，这就是**小概率事件原理**。

**相互独立事件与互不相容事件的区别**：
$P(A)>0,P(B)>0$，$A,B$相互独立则不可能互斥，$A,B$互斥则不可能相互独立

### 7 独立重复试验

独立重复试验也称**n重贝努利试验**，其概型为：
每次试验结果只有两个可能$A,\overline{A}$，且$P(A)=p,0<p<1$。试验重复n次，且n次试验相互独立。
$n$重贝努利试验事件A发生$k$次的概率为
$$
P_n(k)=C_n^kp^k(1-p)^{n-k},k=0,1,\cdots,n
$$

**泊松近似**
**泊松定理：**设$np_n\rightarrow\lambda$($\lambda$为正常数)，则对任一确定的正整数$k$，有
$$
\lim_{n\rightarrow \infty}C_n^kp^k_n(1-p_n)^{n-k}=\frac{\lambda^ke^{-\lambda}}{k!}.
$$
当$n$很大，$\lambda$很小时，我们有如下的**泊松近似**公式：
$$
C_n^kp^k(1-p)^{n-k}\approx \frac{(np)^ke^{-np}}{k!}
$$
对$k=0,1,\cdots$相加后，$\sum_{k=0}^\infty\frac{\lambda^ke^{-\lambda}}{k!}=1$。泊松近似可以用于计算独立重复试验概率的近似值。

***

## Ch2 随机变量及其分布

### 1 随机变量

**随机变量定义：**$\Omega$是随机试验的样本空间，对于每个试验的每一种可能结果$\omega\in \Omega$，都有唯一的实数$X(\omega)$与之对应，称这种**定义在样本空间上**的**实值函数**$X(\omega)$为**随机变量**。通常大写$X$表示随机变量，小写$x$表示函数变量。

**随机变量**在某个范围的**取值**表示**随机事件**。通常随机变量分为**离散型随机变量**(取值可列)与**连续型随机变量**(取值不可列)。

### 2 离散型随机变量与分布律

**定义：**若随机变量$X$的取值是有限个或可列无限个，称之为离散型随机变量。
设$X$可能取值为$x_1,\cdots, x_n,\cdots$，并设$P(X=x_n)=p_n~~(n=1,2\cdots)$，则称上式为离散型随机变量$X$的**分布律**，也可写作一个列表。

**分布律性质：**

+ $\forall n\in \mathbb{N}, p_n\geq 0$
+ $\sum_n p_n=1$

**2.1 0-1分布：**
若随机变量$X$的分布律为：
$$
P\{X=k\}=p^k(1-p)^{1-k}~~(k=0,1)
$$
则称$X$服从0-1分布，记为$X\sim (0-1)$。0-1分布实际上是二项分布的特殊情形。

**2.2 二项分布：**
若随机变量$X$的分布律为：
$$
P\{X=k\}=C_n^kp^k(1-p)^{n-k}~~(k=0,1,\cdots,n)
$$
则称$X$服从参数为$n,p$的二项分布，记作$X\sim B(n,p)$。

**2.3 泊松分布：**
若随机变量$X$的分布律为：
$$
P\{X=k\}=\frac{\lambda^k}{k!}e^{-\lambda}~~(k=0,1,2\cdots)
$$
则称$X$服从参数为$\lambda$的泊松分布，记作$X\sim P(\lambda)$。泊松分布是描述大量试验中稀有事件出现次数的概率模型，n很大p很小的情况下二项分布可以近似为泊松分布。

**2.4 超几何分布：**
若随机变量$X$的分布律为：
$$
P\{X=k\}=\frac{C_M^kC_{N-M}^{n-k}}{C_N^n}~~(k=0,1,\cdots,\min(M,n))
$$
则称$X$服从超几何分布，记作$X\sim H(n,N,M)$。$N\rightarrow \infty$时，超几何分布近似于二项分布，即N充分大时，可用不放回抽样代替有方回抽样。

**2.5 几何分布**
若随机变量$X$的分布律为：
$$
P(X=k)=(1-p)^{k-1}p~~~(k=1,2\cdots)
$$
则称$X$服从几何分布，记作$X\sim G(p)$。几何分布无记忆性，离散型随机变量若无及习性，则服从几何分布(即$P(X>s+t|X>t)=P(X>s)$)

### 3 分布函数

**分布函数定义：**对随机变量X和任意实数$x$，称$F(x)=P\{X\leq x\}$为随机变量X的分布函数。

**分布函数性质：**

+ $F(x)$是单调不减函数
+ $0\leq F(x)\leq 1,\lim_{x\rightarrow\infty}F(x)=1, \lim_{x\rightarrow-\infty}F(x)=0$
+ $F(x)$是右连续的，即$F(x+0)=F(x)$

### 4 连续型随机变量与概率密度函数

设随机变量X的分布函数为$F(x)$，若存在非负可积函数$p(x)$，使对于任意实数$x$，有
$$
F(x)=\int_{-\infty}^x p(t)\mathrm{d}t
$$
则$X$为**连续型随机变量**，称$p(x)$为$X$的**概率密度函数**。

**密度函数性质：**

+ $p(x)\geq 0$
+ $\int^{+\infty}_{-\infty} p(x)\mathrm{d}x=1$
+ $P\{a<X\leq b\}=F(b)-F(a)=\int_a^bp(x)\mathrm{d}x$
+ 在$p(x)$的连续点处，$p(x) = F'(x)$
+ $P(X=x_0)=0$

**4.1 均匀分布：**
若随机变量$X$的概率密度为：
$$
p(x)=\left\{ \begin{matrix} \frac{1}{b-a}, & a \leq x \leq b \\ 0, & otherwise \end{matrix} \right.
$$
则称$X$在区间$[a,b]$上服从均匀分布，记为$X\sim U[a,b]$。分布函数为：
$$
F(x)=\left\{ \begin{matrix} 0, &  x < b \\ \frac{x-a}{b-a}, & a\leq x < b \\ 1, & x\geq b \end{matrix} \right.
$$
**4.2 指数分布：**
若随机变量$X$的概率密度为：
$$
p(x)=\left\{ \begin{matrix} \lambda e^{-\lambda x}, &x\geq 0 \\ 0, & x < 0 \end{matrix} \right.
$$
则称$X$服从参数为$\lambda~(\lambda>0)$的指数分布，记为$X\sim E(\lambda)$。分布函数为：
$$
F(x)=\left\{ \begin{matrix} 1 - e^{-\lambda x}, &x\geq 0 \\ 0, & otherwise \end{matrix} \right.
$$
指数分布中$\frac{1}{\lambda}$是$X$取值的平均值。另外，指数分布具有**无记忆性**：$P(X>s+t|X>s)=P(X>t)$

**4.3 正态分布：**
若随机变量$X$的概率密度为：
$$
p(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}},~~-\infty < x < \infty
$$
则称$X$服从参数为的$\mu, \sigma^2$的正态分布，记为$X\sim N(\mu,\sigma^2)$。分布函数为：
$$
F(x)=\int_{-\infty}^x \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(t-\mu)^2}{2\sigma^2}}\mathrm{d}t
$$
正态分布沿$x=\mu$对称，在$x=\mu \pm \sigma$处有拐点。$\mu=0,\sigma=1$时，$X$服从标准正态分布，其密度函数和分布函数记为$\varphi(x),\Phi(x)$。标准正态分布有$\Phi(-x)=1-\Phi(x)$。
**Th: **若随机变量$X\sim N(\mu, \sigma^2)$，则$Z=\frac{X-\mu}{\sigma}\sim N(0,1)$
**3$\sigma$法则: **$X$的取值几乎肯定落入$[\mu-3\sigma,\mu+3\sigma]$中

### 5 随机变量函数的分布

**5.1 离散型随机变量的函数**
对于离散型随机变量$X$的函数$Y=g(X)$，可以求$Y$的分布律：
$$
P(Y=y_k)=\sum_{i:y_k=g(x_i)}P(X=x_i)
$$
**5.2 连续型随机变量的函数**
设$X$为连续型随机变量，$Y=g(X)$为连续实函数，则$Y=g(X)$也是连续型随机变量。已知$X$的密度函数$p_X(x)$可以求得$Y$的密度函数$p_Y(y)$：

**方法1 - 分布函数法：**
$$
F_Y(y)=P(Y\leq y) = P(g(X)\leq y) = \int_{x:g(x)\leq y}p_X(x)\mathrm{d}x
$$
然后计算$p_Y(y)=F_Y'(y)$.

**方法2 - Th: **设随机变量$X$的可能取值范围在$(a,b)$，$X$的概率密度为$p_X(x),~a<x<b~$($a,b$可为负、正无穷)。设函数$y=g(x)$处处可导，且恒有$g'(x)>0$ (或恒有$g'(x)<0$)，则$Y=g(X)$为连续型随机变量，其概率密度为
$$
p_Y(y)=\left\{ \begin{matrix} p_X[g^{-1}(y)]\cdot |[g^{-1}(y)]'| & \alpha < y < \beta \\ 0 & otherwise \end{matrix} \right.
$$
其中$\alpha=\min(g(a),g(b))$，$\beta=\max(g(a),g(b))$.

***

## Ch3 二维随机变量及其分布

### 1 二维随机变量及其分布函数

**1.1 联合分布函数**
设$(X,Y)$是一个二维随机变量，则对于任意实数$(x,y)$:
$$
F(x,y)=P\{X\leq x, Y\leq y\}
$$
是$(x,y)$的函数，称为二维随机变量$(X,Y)$的**联合分布函数**。

**几何意义：**平面上随机点$(X,Y)$落在以$(x,y)$为右上顶点的无穷矩形中的概率。

**性质：**

1. $F(x,y)$分别对每个变量单调不减
2. $0\leq F(x,y)\leq 1,\\ F(-\infty, y)=0,F(x,-\infty)=0,F(-\infty,-\infty)=0,F(\infty,\infty)=1$
3. $F(x,y)$关于每个变量右连续，$F(x,y)=F(x+0,y),F(x,y)=F(x,y+0)$
4. 对于$x_1<x_2,y_2<y_2$，有$F(x_2,y_2)-F(x_1,y_2)-F(x_2,y_1)+F(x_1,y_1)\geq 0$

任意二维随机变量的分布函数满足上述四条性质。反之，任意满足上述性质的$F(x,y)$是某个二维随机变量$(X,Y)$的分布函数。

**n维随机变量的分布函数​**：$F(x_1,x_2\cdots x_n)=P\{X_1\leq x_1,X_2\leq x_2,\cdots,X_n\leq x_n\}$

**1.2 边缘分布函数**
$F_X(x)=P(X\leq x,Y<\infty)=F(x,\infty)$为$X$的**边缘分布函数**，其中$F(x,\infty)=\lim_{y\rightarrow \infty}F(x,y)$。同理定义$Y$的边缘分布函数为$F_Y(y)=F(\infty,y)$.

**1.3 随机变量的独立性**
设$X,Y$为随机变量，若对任意实数$x,y$，随机事件$X\leq x$与$Y\leq y$相互独立，即$P(X\leq x,Y\leq y)=P(X\leq x)P(Y\leq y)$或$F(x,y)=F_X(x)F_Y(y)$对任意$(x,y)$成立，则称随机变量$X,Y$相互独立。

**Th：**设随机变量$X,Y$相互独立，$f(x)$与$g(y)$为连续或分段连续函数，则$f(X)$与$g(Y)$也相互独立.

### 2 二维离散型随机变量

**2.1 联合分布律**
若二维随机变量$(X,Y)$的取值是有限个或可列无穷多个，则称$(X,Y)$为**二维离散型随机变量**。设$(X,Y)$二维离散型随机变量，$X,Y$的取值为$(x_i,y_j),(i,j=1,2,\cdots,n,\cdots)$，则称$p_{ij}=P(X=x_i,Y=y_j),(i,j=1,2,\cdots)$为$(X,Y)$的**联合分布律**。

**性质：**

1. $p_{ij}=P(X=x_i,Y=y_j)\geq 0$
2. $\sum_i\sum_j p_{ij}=1$

**2.2 边缘分布律**
$P\{X=x_i\}=\sum_j P(X=x_i,Y=y_j)=\sum_{j=1}^{+\infty}p_{ij}=p_{i\cdot}$
$P\{Y=y_j\}=\sum_iP(X=x_i,Y=y_i)=\sum_{i=1}^{+\infty}p_{ij}=p_{\cdot j}$

**2.3 联合分布函数与边缘分布函数**
联合分布函数为：
$$
F(x,y)=\sum_{x_i\leq x}\sum_{y_j\leq y}p_{ij}
$$
边缘分布函数为：
$$
F_X(x)=P(X\leq x)=\sum_{x_i\leq x}p_{i\cdot}=\sum_{x_i\leq x}\sum_{j=1}^\infty p_{ij}\\
F_Y(y)=P(Y\leq y)=\sum_{y_j\leq y}p_{\cdot j}=\sum_{y_j\leq y}\sum_{i=1}^\infty p_{ij}
$$
**2.4 独立性条件**
若对一切$X,Y$的可能取值$(x_i,y_j)$有$P(X=x_i,Y=y_j)=P(X=x_i)P(Y=y_j),~(i,j=1,2,\cdots)$，即$p_{i,j}=p_{i\cdot}p_{\cdot j}$，则$X,Y$相互独立。

### 3 二维连续型随机变量

**3.1 二维连续型随机变量**
对于二维随机变量 $(X,Y)$的分布函数 $F (x , y )$，如果存在非负函数$p (x , y )$，使得对于任意的$ x,y$有：
$$
F(x,y)=\int_{-\infty}^x\int_{-\infty}^y p(u,v)dudv
$$
则称$(X,Y)$是**二维连续型随机变量**，函数$p(x,y)$称为$(X,Y)$的**联合密度函数**，简称**概率密度**。

**性质：**

1. $p(x,y)\geq 0$
2. $\int_{-\infty}^{+\infty} \int_{-\infty}^{+\infty}p(x,y)dxdy=F(+\infty,+\infty)=1$
3. 若$p(x,y)$在点$(x,y)$连续，则有$\frac{\part^2F(x,y)}{\part x\part y}=p(x,y)$
4. 设$G$是平面上一个区域，点$(X,Y)$落在$G$内的概率为$P\{(X,Y)\in G\}=\int\int_Gp(x,y)dxdy$

**3.2 边缘密度**
$F_X(x)=F(x,\infty)=\int_{-\infty}^x(\int_{-\infty}^{+\infty}p(u,y)dy)du$，$F_X(x)$为$X$的**边缘分布函数**，称$p_X(x)=F_X'(x)=\int_{-\infty}^{\infty}p(x,y)dy$为$X$的**边缘密度**。边缘分布函数与边缘密度满足：
$$
F_X(x)=\int_{-\infty}^xp_X(x)dx
$$
$Y$的边缘分布函数与边缘密度同理。

**3.3 独立性条件**
当$\forall x,y,F(x,y)=F_X(x)F_Y(y)$时，随机变量$X$与$Y$相互独立。用密度函数可表示为$\forall x,y,p(x,y)=p_X(x)p_Y(y)$

**3.4 二维均匀分布**
设$D$是平面上的有界区域，其面积为$A$，若二维随机变量$(X,Y)$的密度函数为
$$
p(x,y)=\left\{\begin{matrix} \frac{1}{A}&(x,y)\in D\\ 0&(x,y)\not\in D\end{matrix} \right.
$$
则称二维随机变量$(X,Y)$服从$D$上的**均匀分布**。

**几何意义：**$P((X,Y)\in G)=\frac{S_{G\cap D}}{S_D}$

**3.5 二维正态分布**
若二维随机变量$(X,Y)$具有概率密度
$$
p(x,y)=\frac{1}{2\pi \sigma_1\sigma_2\sqrt{1-\rho^2}}e^{-\frac{1}{2(1-\rho^2)}[(\frac{x-\mu_1}{\sigma_1})^2-2\rho (\frac{x-\mu_1}{\sigma_1})(\frac{y-\mu_2}{\sigma_2})+(\frac{y-\mu_2}{\sigma_2})^2]}
$$
其中$\mu_1,\mu_2,\sigma_1>0,\sigma_2>0,|\rho|<1$均为常数，则称$(X,Y)$服从**二维正态分布**，记作$(X,Y)\sim N(\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,\rho)$

**Th：**二维正态分布的边缘分布仍为正态分布：设$(X,Y)\sim N(\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,\rho)$，则$X\sim N(\mu_1,\sigma_1^2),Y\sim N(\mu_2,\sigma_2^2)$。即$p_X(x)=\frac{1}{\sqrt{2\pi}\sigma_1}e^{-\frac{(x-\mu_1)^2}{2\sigma_1^2}},p_Y(y)=\frac{1}{\sqrt{2\pi}\sigma_2}e^{-\frac{(y-\mu_2)^2}{2\sigma_2^2}}$

**Th：**设$(X,Y)\sim N(\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,\rho)$，则$X,Y$相互独立的充要条件为$\rho=0$

### 4 条件分布

**4.1 离散型随机向量的条件概率分布**
条件概率：
$$
P(Y=y_j|X=x_i)=\frac{P(X=x_i,Y=y_j)}{P(X=x_i)}=\frac{p_{ij}}{p_{i\cdot}}
$$
在$X=x_i$的条件下，$Y$的条件分布率为
$$
Y|_{X=x_i}\sim\begin{pmatrix} y_1&y_2&\cdots &y_j&\cdots\\p_{i1}/p_{i\cdot} & p_{i2}/p_{i\cdot}&\cdots &p_{ij}/p_{i\cdot}&\cdots \end{pmatrix}
$$
**全概率公式：**
$$
P(Y=y_j)=\sum_{i=1}^{+\infty}P(Y=y_j|X=x_i)P(X=x_i)
$$
**4.2 连续型随机向量的条件概率分布**
由于$P(X=x)=0$，不能用时间的条件概率来简单定义该分布，应从极限方法定义：
$$
P(Y\leq y | X= x)= \lim_{\Delta x \rightarrow 0+} P(Y\leq y|x< X \leq x + \Delta x)\\=\lim_{\Delta x\rightarrow 0+}\frac{\int_x^{x+\Delta x}\int _{-\infty}^y p(u,v) du dv }{\int_x^{x+\Delta x}p_X(u)du}
$$
若$p(x,y)$和$p_X(x)$均连续，且$p_X(x)>0$，有
$$
P(Y\leq y | X = x)=\int_{-\infty}^y\frac{p(x,v)}{p_X(x)}dv
$$
称$P(Y\leq y | X=x)$为在条件$X=x$的条件下$Y$的**条件分布函数**，记为$F_{Y|X=x}(y)$。可见条件分布函数可导，称导函数
$$
P_{Y|X=x}(y)=F'_{Y|X=x}(y)=\frac{p(x,y)}{p_X(x)}
$$
为在$X=x$条件下$Y$的**条件密度函数**。

**密度函数的乘法公式：**
$$
p(x,y)=p_{Y|X=x}(y)p_X(x)
$$
**全概率公式的密度函数形式：**
$$
p_Y(y)= \int_{-\infty}^{+\infty}p(x,y)dx = \int_{-\infty}^{+\infty}p_X(x)P_{Y|X=x}(y)dx
$$

### 5 二维随机向量函数的分布

**5.1 二维离散型随机向量函数的分布**
设$(X,Y)$是一个二维离散型随机向量，其联合分布律位$P(X=x_i, Y=y_i)=p_{ij},~(i,j=1,2,\cdots)$，若$Z=g(X,Y)$是一个随机变量，其所有可能取值集合为$\{z_k\}$，则其分布律可表示为：
$$
P(Z=z_k)=P(g(X,Y)=z_k)=\sum_{g(x_i,y_i)=z_k}p_{ij}
$$
以下为集中特殊的变换$Z=g(X,Y)$的分布律的计算公式：

**5.1.1 和的分布**
设$Z=X+Y$，则$Z$的所有可能取值集合为$\{z_k:\exists i,j,z_k=x_i+y_j \}$，其分布律为
$$
P(Z=z_k)=\sum_{x_i}P(X=x_i, Y= z_k - x_i)
$$
特别，当$X,Y$独立时，$P(Z=z_k)=\sum_{x_i}P(X=x_i)P(Y=z_k-x_i)$，这个公式为**卷积公式**。
随机变量差的分布可看成是$X,-Y$的和的分布律。

**5.1.2 商的分布**
设$P(Y=0)=0,~Z=X/Y$，则$Z$的取值集合为$\{z_k:\exists i,j,z_k=x_i/y_j\}$，分布律为
$$
P(Z=z_k)=\sum_{y_j}P(X=z_k\cdot y_j, Y=y_j)
$$
随机变量乘积的分布可以看成是$X,Y^{-1}$商的分布律。

**5.2 二维连续型随机向量函数的分布**
对于二维连续型随机向量$(X,Y)$，若已知其联合密度函数$f(x,y)$，且$X,Y$的函数$Z=g(X,Y)$也是一个连续型随机变量，要计算$Z$的密度函数时，一般可以先考虑分布函数
$$
F_Z(z)=P(Z\leq z) = P(g(X,Y)\leq z) = \iint_{g(x,y)\leq z}p(x,y)\mathrm{d}x\mathrm{d}y
$$
**5.2.1 和的分布**
设$(X,Y)$有联合密度函数$p(x,y)$，$Z=X+Y$，则$Z$是一个连续型的随机变量，其密度函数为
$$
p_Z(z)=\int_{-\infty}^{+\infty}p(x,z-x)\mathrm{d}x
$$
特别地，当$X,Y$独立时，$p_Z(z)=\int_{-\infty}^{+\infty}p_X(x)p_Y(z-x)\mathrm{d}x$，这个公式称为$p_X(x),p_Y(y)$的**卷积公式**。

**5.2.2 商的分布**
设$(X,Y)$有联合密度函数$p(x,y)$，$Y\neq 0$，定义$Y=X/Y$，则$Z$是一个连续型的随机变量，其密度函数为
$$
p_Z(z)=\int_{-\infty}^{+\infty}|y|p(zy,y)\mathrm{d}y
$$
**5.2.3 最大(小)值的分布**
设随机变量$X,Y$相互独立，分别有分布函数$F_X(x),F_Y(y)$，定义$M=\max(X,Y)$, $N=\min(X,Y)$，则$M,N$的分布函数为
$$
F_M(z)=F_X(z)F_Y(z)\\
F_N(z)=1 - (1-F_X(z))(1-F_Y(z))
$$
进而，如果$X,Y$都是连续型随机变量，则$M,N$密度函数分别为
$$
p_M(z) = p_X(z)F_Y(z)+F_Z(z)p_Y(z)\\
p_N(z) = p_X(z)(1-F_Y(z)) + p_Y(z)(1-F_X(z))
$$
