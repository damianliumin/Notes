# Probability and Mathematical Statistics II

Author: Daniel Liu
Contact Me: 191240030@smail.nju.edu.cn
Course: Probability and Mathematical Statistics. NJU

## Ch5 极限理论

极限理论主要包含**大数定律**与**中心极限定理**，它们反映了随机变量序列的**频率稳 定性**与**分布稳定性**问题。

### 1 大数定律

**1.1 大数定律的定义**
**依概率收敛定义**：设$X_1,\cdots,X_n,\cdots$为一列随机变量，若存在随机变量$X$，使得任意给定的$\epsilon>0$，
$$
\lim_{n\rightarrow \infty}P(|X_n-X|\geq \epsilon)=0, ~or\\
\lim_{n\rightarrow \infty}P(|X_n-X|<\epsilon)=1
$$
则称**随机变量序列**$\{X_n\}$**依概率收敛于**随机变量$X$，记为$X_n\xrightarrow{P} X$.

**大数定律定义：**对随机变量序列$\{X_n\}$，若任意$\epsilon > 0$，有
$$
\lim_{n\rightarrow \infty}P\left(\left| \frac{1}{n} \sum_{k=1}^nX_k - \frac{1}{n}\sum_{k=1}^n EX_k \right| \geq \epsilon \right)=0\\
\lim_{n\rightarrow \infty} P\left(\left| \frac{1}{n} \sum_{k=1}^nX_k - \frac{1}{n}\sum_{k=1}^n EX_k \right| < \epsilon \right)=1
$$
即$\frac{1}{n}\sum_{k=1}^n X_k \xrightarrow{P} \frac{1}{n}\sum_{k=1}^n EX_k $，则称$\{X_n\}$服从**大数定律**.
(随机变量的平均值依概率收敛于它们数学期望的平均值)

**1.2 马尔可夫大数定律**
设随机变量$\{X_n\}$满足$\frac{1}{n^2}D(\sum_{k=1}^nX_k)\rightarrow 0~(n\rightarrow \infty)$，则$\{X_n\}$服从大数定律.

**1.3 切比雪夫大数定律**
设随机变量$X_1, X_2, \cdots, X_N, \cdots$**两两互不相关**，则存在常数$C>0$，使对每个$X_k$，有$D(X_k)<C$，则$\{X_n\}$服从大数定律.

**1.4 独立同分布大数定律**
设随机变量$X_1, X_2, \cdots, X_N, \cdots$**独立同分布**，$EX_k=\mu, DX_k=\sigma^2<\infty$，则$\{X_n\}$服从大数定律，即$\frac{1}{n}\sum_{k=1}^n X_k\xrightarrow{P}\mu$.

**1.5 贝努里大数定律**
设$\mu_n$为$n$重贝努里试验中事件$A$发生的次数，$p=P(A)$，则对任意$\varepsilon > 0$有$\lim_{n\rightarrow \infty}P\{|\frac{\mu_n}{n}-p|\geq \varepsilon\}=0$或$\lim_{n\rightarrow \infty}P\{|\frac{\mu_n}{n}-p|< \varepsilon\}=1$.

该定理给出了频率的稳定性的严格的数学意义。

### 2 中心极限定理

**2.1 标准化的随机变量**
若随机变量$X$满足$EX=0,DX=1$，则称$X$为标准化的随机变量。对任意随机变量$X$，$Y=\frac{X-EX}{\sqrt{DX}}$是标准化的随机变量.

**2.2 中心极限定理**
**定理1：独立同分布的中心极限定理**
设$X_1,\cdots, X_n,\cdots$独立同分布，且$EX_k=\mu, DX_k=\sigma^2,(k=1,2,\cdots)$。令$Z_n=\frac{\sum_{k=1}^nX_k-n\mu}{\sqrt{n}\sigma}$，则对任意$x$有
$$
\lim_{n\rightarrow\infty} P(Z_n\leq x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^x e^{-\frac{t^2}{2}}dt=\Phi(x)
$$
表示$Z_n$的极限分布为标准正态分布。

（中心极限定理的本质是：$\sum_{k=1}^nX_k$的极限分布为正态分布，渐近分布为$N\{E(\sum_{k=1}^nX_k), D(\sum_{k=1}^n X_k)\}$）

**定理2：贝努里情形的中心极限定理（拉普拉斯中心极限定理）**
设$\mu_n$是$n$重贝努里试验中$A$发生的次数，$p=P(A)$，则对任意$x\in\R$有：
$$
\lim_{n\rightarrow \infty} P\{\frac{\mu_n - np}{\sqrt{npq}}\leq x \} =\frac{1}{\sqrt{2\pi}}\int_{-\infty}^x e^{-\frac{t^2}{2}}dt=\Phi(x)
$$
（本质是$\mu_n\sim B(n, p)$，则$\mu_n$的极限分布是正态分布）

***

## Ch6 样本及抽样分布

### 1 基本概念

**1.1 样本**
**总体：**研究对象的某项数量指标的值的全体
**个体：**总体中的每个元素为个体

**样本定义：**设随机变量$X$的分布函数是$F(x)$，若$X_1,\cdots,X_n$是**具有同一分布函数**$F$的**相互独立**的随机变量， 则称$X_1,\cdots,X_n$为从总体$X$中得到的容量为$n$的**简单随机样本**，简称为样本，其观察值$x_1,\cdots, x_n$称为样本值。
由定义可知，若$X_1,\cdots,X_n$为$X$的一个样本，则$X_1,\cdots,X_n$的联合分布函数为：
$$
F^*(x_1,\cdots,x_n)=\prod_{i=1}^nF(x_i)
$$
设$X$的密度为$p(x)$，则$X_1,\cdots, X_n$的联合概率密度为：
$$
p^*(x_1,\cdots,x_n)=\prod_{i=1}^np(x_i)
$$
**1.2 统计量**
**统计量定义：**设$X_1,\cdots,X_n$为来自总体$X$的一个样本，$g(X_1,\cdots,X_n)$是$X_1,\cdots,X_n$的函数，若$g$是连续函数，且$g$中不含任何**未知**参数，则称$g(X_1,\cdots,X_n)$为一个**统计量**。设$x_1,\cdots,x_n$是$(X_1,\cdots,X_n)$的样本值，则称$g(x_1,\cdots,x_n)$是$g(X_1,\cdots,X_n)$的**观察值**。
（注意：**统计量是随机变量**）

**常用的统计量：**
样本均值：$\overline{X}=\frac{1}{n}\sum_{i=1}^nX_i$
样本方差：$S_n^2=\frac{1}{n}\sum_{i=1}^n(X_i-\overline{X})^2=\frac{1}{n}\sum_{i=1}^nX_i^2-\bar{X}^2$
样本方差(修正)：$S_{n-1}^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\overline{X})^2$
样本标准差：$S_n=\sqrt{S_n^2}=\sqrt{\frac{1}{n}\sum_{i=1}^n(X_i-\overline{X})^2}$
样本$k$阶(原点)矩：$A_k=\frac{1}{n}\sum_{i=1}^nX_i^k~~k=1,2,\cdots$
样本$k$阶中心矩：$B_k=\frac{1}{n}\sum_{i=1}^n(X_i-\overline{X})^k~~k=1,2,\cdots$

根据上述统计量可计算其观察值。统计量是样本的函数，它是一个随机变量，统计量的分布称为**抽样分布**。

**结论1：**设$X_1,\cdots,X_n$为来自总体$X$的一个样本，$EX=\mu, DX=\sigma^2$，$\overline{X}$为样本均值，则$E\overline{X}=\mu, D\overline{X}=\frac{\sigma^2}{n}$.
**结论2：**设$X_1,\cdots,X_n$为来自总体$X$的一个样本，$EX=\mu, DX=\sigma^2$，$S_n^2$为样本方差，则$ES_n^2=\frac{n-1}{n}\sigma^2$.

### 2 抽样分布

**2.1 正态总体样本的线性函数的分布**
设$X_1,\cdots,X_n$是来自正态总体$X\sim N(\mu, \sigma^2)$的样本，则随机变量$U=a_1X_1+a_2X_2+\cdots+a_nX_n$也服从正态分布：
$$
U\sim N(\mu\sum_{i=1}^na_i, \sigma^2\sum_{i=1}^n a_i^2)
$$
特别地，若取$a_i=\frac{1}{n},(i=1,2,\cdots,n)$，则有$U=\overline{X}$，$\frac{\overline{X}-\mu}{\sigma / \sqrt{n}}\sim N(0,1)$.

**标准正态分布的上$\alpha$分位点**：设$X\sim N(0,1)$满足$P(X>u_\alpha)=\alpha$的$u_\alpha$值称为$N(0,1)$的上$\alpha$分位点。
$\Phi(u_\alpha)=P(X\leq u_\alpha)=1-\alpha$

**2.2 $\chi^2$分布**
**定义：**设$X_1,X_2,\cdots,X_n$相互独立，都服从正态分布$N(0,1)$，则称随机变量
$$
\chi^2=X_1^2+X_2^2+\cdots+X_n^2
$$
所服从的分布为自由度为$n$的$\chi^2$分布，记为$\chi^2\sim \chi^2(n)$.

$\chi^2(n)$**概率密度**为
$$
p(x)=\left\{ \begin{matrix}\frac{1}{2^{\frac{n}{2}}\Gamma(\frac{n}{2})}e^{-\frac{x}{2}}x^{\frac{n}{2}-1} & x>0 \\
0 & x \leq 0 \end{matrix} \right.
$$
**性质：**

1. 设$X_1\sim \chi^2(n_1), X_2\sim \chi^2(n_2)$，且$X_1, X_2$相互独立，则$X_1+X_2\sim \chi^2(n_1+n_2)$. (可加性)
2. 若$X\sim \chi^2(n)$，则$E(X)=n, D(X)=2n$.

**分位点：**对于给定的$\alpha~(0<\alpha < 1)$，称满足条件$P\{\chi^2>\chi^2_\alpha(n)\}=\alpha$的点$\chi_\alpha^2(n)$为$\chi^2(n)$分布的**上$\alpha$分位点.** 对于不同的$\alpha, n$可通过查表求得上$\alpha$分位点的值.
（一般1~45可查表，$n>45$有近似公式$\chi_\alpha^2(n)=\frac{1}{2}(u_\alpha+\sqrt{2n-1})^2$）

**2.3 t分布**(student 分布)
**定义：**设$X\sim N(0,1), Y\sim\chi^2(n)$，且$X,Y$独立，则称随机变量$T=\frac{X}{\sqrt{Y/n}}$服从自由度为$n$的$t$分布，记为$T\sim t(n)$.

**概率密度函数**：
$$
p(t)=\frac{\Gamma(\frac{n+1}{2})}{\sqrt{n\pi}\Gamma(\frac{n}{2})}(1+\frac{t^2}{n})^{-\frac{n+1}{2}},~~-\infty<t<+\infty
$$
**性质：**$t$分布的密度函数关于$t=0$对称，当$n$充分大时，密度函数$p(t)$近似于$N(0,1)$的密度$\Phi(t)$. 即
$$
\lim_{n\rightarrow \infty}p(t) = \frac{1}{\sqrt{2\pi}}e^{-t^2/2}
$$
**分位点：**$T\sim t(n)$，对于给定的$\alpha~(0<\alpha < 1)$，称满足条件$P\{T>t_\alpha(n)\}=\alpha$的点$t_\alpha(n)$为$t(n)$分布的**上$\alpha$分位点**。可通过查表获取$t_\alpha(n)$。
$t$分布的上$\alpha$分位点有性质：$t_{1-\alpha}(n) = -t_\alpha(n)$.

**2.4 F分布**
定义：设$U\sim \chi^2(n_1), V\sim \chi^2(n_2)$，且$U,V$独立，则称随机变量$F=\frac{U/n_1}{V/n_2}$服从自由度为$(n_1,n_2)$的$F$分布，记为$F\sim F(n_1,n_2)$.

**概率密度：**
$$
p(x)=\left\{\begin{matrix} \frac{\Gamma(\frac{n_1+n_2}{2})(\frac{n_1}{n_2})^{\frac{n_1}{2}}x^{\frac{n_1}{2}-1}}{\Gamma(\frac{n_1}{2})\Gamma(\frac{n_2}{2}[1+(\frac{n_1x}{n_2})]^{\frac{n_1+n_2}{2}})} & x> 0\\
0 & otherwise

\end{matrix}\right.
$$
**分位点：**对于给定的$\alpha~(0< \alpha < 1)$，称满足条件$p\{F>F_\alpha(n_1,n_2)\}=\alpha$的点$F_\alpha(n_1, n_2)$为$F(n_1, n_2)$分布的上$\alpha$分位点。

**性质：**

1. 若$F\sim F(n_1, n_2)$，则$\frac{1}{F}\sim F(n_2, n_1)$.
2. $F_{1-\alpha}(n_1, n_2) = \frac{1}{F_\alpha(n_2, n_1)}$.

($X\sim t(n)$，则$X^2\sim F(1, n)$)

 **2.5 正态总体的样本均值与样本方差的分布**
**定理1：样本均值的分布：**设$X_1,\cdots,X_n$是来自正态总体$X\sim N(\mu, \sigma^2)$的样本，$\overline{X}=\frac{1}{n}\sum_{i=1}^n X_i$:
$$
\frac{\overline{X}-\mu}{\sigma / \sqrt{n}}\sim N(0,1)
$$
**定理2：样本方差的分布：**设$X_1,\cdots,X_n$是来自正态总体$N(\mu, \sigma^2)$的样本，$\overline{X}$和$S_n^2$分别为样本均值和样本方差，则有

1. $\frac{nS_n^2}{\sigma^2}\sim \chi^2(n-1)$ （或$\frac{(n-1)S_{n-1}^2}{\sigma^2}\sim \chi^2(n-1)$）
2. $\overline{X}$与$S_n^2$独立

**定理3：**设$X_1,\cdots,X_n$是来自正态总体$N(\mu, \sigma^2)$的样本，$\overline{X}$和$S_n^2$分别为样本均值和样本方差，则有
$$
\frac{\overline{X}-\mu}{S_n / \sqrt{n-1}}\sim t(n-1)~(\text{or }\frac{\overline{X}-\mu}{S_{n-1} / \sqrt{n}}\sim t(n-1))
$$
**定理4: (双正态总体样本均值差、样本方差比的分布)**设$X\sim N(\mu_1, \sigma_1^2), Y\sim N(\mu_2, \sigma_2^2)$，且$X$与$Y$独立，$X_1,\cdots,X_{n_1}$是来自$X$的样本，$Y_1,\cdots, Y_{n_2}$是来自$Y$的样本，$\overline{X}$和$\overline{Y}$分别是这两个样本的样本均值，$S_1^2$和$S_2^2$分别是$X,Y$的修正样本方差：$S_1^2=\frac{1}{n_1-1}\sum_{i=1}^{n_1}(X_i-\overline{X})^2~~S_2^2=\frac{1}{n_2-1}\sum_{i=1}^{n_2}(Y_i-\overline{Y})^2$,则有

1. $\frac{S_1^2/S_2^2}{\sigma_1^2/\sigma_2^2}\sim F(n_1-1, n_2 - 1)$
2. $U = \frac{(\overline{X}-\overline{Y})-(\mu_1 - \mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}}\sim N(0,1)$
3. 若$\sigma_1^2 = \sigma_2^2 = \sigma^2$, 则$\frac{(\overline{X}-\overline{Y})-(\mu_1 - \mu_2)}{S_w\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}\sim t(n_1+n_2-2)$,其中$S_w^2=\frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1 + n_2 - 2}, S_w = \sqrt{S_w^2}$.

***

## Ch7 参数估计

方法分类：点估计(矩估计、极大似然估计)、区间估计

**点估计：**对总体分布中的未知参数$\theta$, 以样本$X_1, X_2,\cdots,X_n$构造统计量$\hat{θ}(X_1,X_2,\cdots,Xn)$作为参数$\theta$的估计, 称$\hat{θ}(X_1,X_2,\cdots,X_n)$为参数$\theta$**估计量**。当测得样本值$(x_1 , x_2 ,\cdots, x_n)$时, 代入$\hat{\theta}$即可得到参数$\theta$**估计值**：$\hat{θ}(X_1,X_2,\cdots,Xn)$.

**区间估计：**对总体分布中的未知参数$\theta$, 以样本$X_1, X_2,\cdots,X_n$构造 2个统计量$\hat{θ}_1(X_1,X_2,\cdots,X_n)$和$\hat{θ}_2(X_1,X_2,\cdots,X_n)$以区间$(\hat{\theta}_1, \hat{\theta}_2)$作为参数$\theta$的估计。对 给定的概率$1-\alpha$，满足：
$$
P(\hat{θ}_1(X_1,X_2,\cdots,X_n) < \theta < \hat{θ}_2(X_1,X_2,\cdots,X_n))=1-\alpha
$$

### 1 矩估计

**原则：**以样本矩作为总体矩的估从而得到参数的估计量.

**矩估计法：**设总体$X$的**分布类型已知**，$X$的分布函数为$F(x;\theta_1,\theta_2,\cdots,\theta_k)$，其中$\theta_1,\theta_2,\cdots,\theta_k$为未知参数。设$X_1,X_2,\cdots,X_n$为来自总体$X$的样本。若$m$阶总体矩$EX^m=\mu_m(\theta_1,\theta_2,\cdots,\theta_k)$存在，$(m=1,2,\cdots,k)$，则有$m$阶样本矩$A_m = \frac{1}{n}\sum_{i=1}^nX_i^m~(m=1,2,\cdots, k)$。令
$$
\left\{\begin{matrix}\mu_1(\theta_1,\theta_2,\cdots,\theta_k)=A_1\\\cdots\\ \mu_k(\theta_1,\theta_2,\cdots, \theta_k)=A_k  \end{matrix}\right.
$$
这时包含$k$个未知参数$\theta_1,\cdots,\theta_k$的方程组，从中解出方程组的解$\hat{\theta_1},\cdots,\hat{\theta_k}$。
用$\hat{\theta_1},\cdots,\hat{\theta_k}$作为$\theta_1,\cdots,\theta_k$的估计量，就是**矩估计法**，这种估计量称为**矩估计量**，观察值称为**矩估计值**。
（无论总体$X$服从何种分布，总体均值$EX=\mu$, 总体方差$DX=\sigma^2$作为未知参数， 其矩估计量一定是样本均值和样本方差）

相关系数的矩估计：
$$
\rho_{XY}=\frac{cov(X,Y)}{\sqrt{D(X)D(Y)}}=\frac{E([X-E(X)][Y-E(Y)])}{\sqrt{(E([X-E(X)]^2)E([Y-E(Y)]^2))}}\\
r=\frac{\frac{1}{n}\sum_{i=1}^n(x_i - \overline{x})(y_i - \overline{y})}{\sqrt{\frac{1}{n}\sum_{i=1}^n(x_i-\overline{x})^2 \cdot \frac{1}{n}\sum_{i=1}^n(y_i - \overline{y})^2}}
$$
定义**样本相关系数**：
$$
\hat{\rho}=\frac{\frac{1}{n}\sum_{i=1}^n X_iY_i-\overline{X}\cdot \overline{Y}}{\sqrt{ [\frac{1}{n}\sum_{i=1}^n X_i^2 - \overline{X}^2][\frac{1}{n}\sum_{i=1}^n Y_i^2 - \overline{Y}^2] }}=\frac{S_{XY}}{S_{nX}S_{nY}}
$$

### 2 极大似然估计

**原则：**以样本$X_1,X_2,\cdots,X_n$的观测值$x_1,\cdots, x_n$来估计参数$\theta_1,\theta_2,\cdots, \theta_k$，若选取$\hat\theta_1,\hat\theta_2,\cdots, \hat\theta_k$使观测值出现的概率最大，把$\hat\theta_1,\hat\theta_2,\cdots, \hat\theta_k$作为参数$\theta_1,\theta_2,\cdots, \theta_k$的估计量.

**离散型极大似然估计：**总体$X$为离散型，其分布律$P\{X=x\}=p(x,\theta)$的形式为已知，$\theta$为待估参数。又设$x_1,\cdots,x_n$为$X_1,\cdots, X_n$的一个样本值；样本$X_1,\cdots, X_n$取$x_1,\cdots,x_n$的概率，即事件$\{X_1=x_1,\cdots,X_n=x_n\}$的概率为：$P\{X_1=x_1,\cdots,X_n=x_n\}=\prod_{i=1}^np(x_i,\theta)$，记为$L(\theta)=L(x_1,\cdots,x_n;\theta)=\prod_{i=1}^np(x_i,\theta)$，$L(\theta)$称为样本的**似然函数**。
由极大似然估计法，选择使$L(x_1,\cdots,x_n;\theta)$最大的参数$\hat\theta$作为$\theta$的估计值，即取$\hat\theta$使得：
$$
L(x_1,\cdots,x_n;\hat\theta)=\max_\theta L(x_1,\cdots,x_n;\theta)
$$
$\hat\theta$与$x_1,\cdots,x_n$有关，记为$\hat\theta(x_1,\cdots,x_n)$; 称其为参数$\theta$的**极大似然估计值**，$\hat\theta(X_1,\cdots,X_n)$称为$\theta$的**极大似然估计量**。

**连续型极大似然估计：**设概率密度$p(x,\theta)$，$x_1,\cdots,x_n$是$X_1,\cdots,X_n$的样本观测值，$X_1,\cdots,X_n$落在$x_1,\cdots,x_n$邻域(边长为$dx_1,\cdots,dx_n$的n维立方体)内的概率近似为$\prod_{i=1}^np(x_i,\theta)dx_i$。取$\theta=\hat\theta$，使上述概率达到最大值。由于$dx_i$不随$\theta$改变，故只考虑$L(\theta)=L(x_1,\cdots,x_n;\theta)=\prod_{i=1}^np(x_i,\theta)$的最大值，这个函数称为样本的**似然函数**。若：
$$
L(x_1,\cdots,x_n;\hat\theta)=\max_\theta L(x_1,\cdots,x_n;\theta)
$$
则称$\hat\theta(x_1,\cdots,x_n)$为$\theta$的极大似然估计值，称$\hat\theta(X_1,\cdots,X_n)$为$\theta$的极大似然估计量。

**求解方法：**令$\frac{dL(\theta)}{d\theta}=0$，求得$\theta$。由于$L(\theta)$是乘积形式，求导后复杂，且$L(\theta)$与$\ln L(\theta)$极值点相同，因此$\theta$的极大似然估计可从下述方程解得：
$$
\frac{d}{d\theta}\ln L(\theta)=0
$$
若总体分布中包含多个参数$\theta_1,\cdots,\theta_k$，则分别求到得到似然方程组或对数似然方程组。

**极大似然估计的不变性：**设$\hat\theta$是$\theta$的极大似然估计, $u=u(\theta)$是$\theta$的函数, 且有单值反函数$\theta=\theta(u)$，则$\hat{u} = u(\hat{\theta})$是$u(\theta)$的极大似然估计。

### 3 估计量的评选标准

标准：无偏性，有效性，一致性

**3.1 无偏性**
**定义：**$X_1,X_2,\cdots,X_n$为总体$X$的一个样本，设$\theta$的估计量$\hat\theta=\theta(X_1,X_2,\cdots,X_n)$. 若$E(\hat\theta)=\theta$，则称$\hat\theta$是$\theta$的无偏估计量.

$k$阶样本矩$A_k$是$k$阶总体矩$\mu_k$的无偏估计。特别地，样本均值$\overline{X}$是总体均值$\mu_1 =E( X )$的无偏估计量，样本二阶矩$A_2=\frac{1}{n}\sum_{i=1}^nX_i^2$是总体二阶矩$\mu_2=E(X^2)$的无偏估计量（样本方差$S_{n}^2$**不是**总体方差$\sigma^2$的无偏估计，但修正的样本方差$S_{n-1}^2$是）.

**3.2 有效性**
**定义：**设$\hat \theta_1=\hat\theta_1(X_1,\cdots,X_n)$和$\hat\theta_2 = \hat\theta_2(X_1,\cdots,X_n)$都是参数的无偏估计量，若对任意$n$，$D(\hat\theta_1)\leq D(\hat\theta_2)$，则称$\hat\theta_1$较$\hat\theta_2$有效。

**3.3 一致性**
**定义：**设$\hat\theta_n=\hat\theta_n(X_1,X_2,\cdots,X_n)$是总体参数$\theta$的估计量，若当$n\rightarrow \infty$时，$\hat\theta_n$依概率收敛于$\theta$，即$\forall \varepsilon > 0$，有$\lim_{n\rightarrow \infty}P(|\hat\theta_n-\theta|<\varepsilon)=1$，则称$\hat\theta_n$是总体参数$\theta$的**一致估计量**.

常用结论：

1. 样本$k$阶矩是总体$k$阶矩的一致估计量.
2. 设$\hat\theta_n$是$\theta$的无偏估计量，且$\lim_{n\rightarrow \infty}D(\hat\theta_n)=0$，则$\hat\theta_n$是$\theta$的一致估计量.

### 4 区间估计

**区间估计：**根据样本给出未知参数的 一个范围，并保证真参数以指定的较大概率属于这个范围.

**4.1 单正态总体情形**
设$X_1,\cdots,X_n$为总体$X\sim N(\mu,\sigma^2)$的一个样本。求$\mu$或$\sigma^2$的区间估计$[\hat\theta_1,\hat\theta_2]$.

**置信区间与置信度：**设总体$X$含未知参数$\theta$，对于样本$X_1,\cdots,X_n$，找出统计量$\hat\theta_i=\theta_i(X_1,\cdots,X_n)~(i=1,2),\hat\theta_1<\hat\theta_2$，使得$P\{\hat\theta_1\leq \theta\leq \hat\theta\}=1-\alpha, (0<\alpha<1)$。称区间$[\hat\theta_1,\hat\theta_2]$为$\theta$的**置信区间**，$1-\alpha$为该区间的**置信度**。

1. 置信区间长度$L$反映了估计精度，$L$越小, 估计精度越高.
2. $\alpha$反映了估计的可靠度，$\alpha$越小，$1-\alpha$越大，估计的可靠度越高，但这时，$L$ 往往增大，因而估计精度降低.
3. $\alpha$确定后，置信区间的选取方法不唯一，常选长度最小的一个

**正态总体，求均值$\mu$区间估计**
设$X_1,\cdots,X_n$为总体 的一个样本。$X\sim N(\mu,\sigma^2)$在置信度$1-\alpha$下，来确定$\mu$的置信区间$[\theta_1,\theta_2]$ 。
**(1) 已知方差，估计均值**
已知方差$\sigma^2=\sigma^2_0$，则$U=\frac{\overline{X}-\mu}{\sigma_0/\sqrt{n}}\sim N(0,1)$。对于给定的置信度，找出临界值$\lambda_1,\lambda_2$使$P\{\lambda_1\leq U\leq \lambda_2\}=1-\alpha$。取对称区间$L$最小，因而$\lambda=u_{\alpha/2}$，$\Phi(u_{\alpha/2})=1-\alpha/2$。可解出$\mu$置信区间:
$$
[\overline{X}-u_{\alpha/2}\cdot \frac{\sigma_0}{\sqrt{n}}, \overline{X} + u_{\alpha/2}\cdot \frac{\sigma_0}{\sqrt{n}}]
$$
**(2) 未知方差，估计均值**

用$S_n^2=\frac{1}{n}\sum_{i=1}^n(X_i - \overline{X})^2$代替$\sigma^1$，或$T=\frac{\overline{X}-\mu}{S_{n-1}/\sqrt{n}}\sim t(n-1)$. 对于给定的$1-\alpha$，选取对称区间使$P\{|T|\leq \lambda\}=1-\alpha$。解得$\mu$区间：
$$
[\overline{X}-t_{\alpha/2}(n-1)\cdot\frac{S_n}{\sqrt{n-1}}, \overline{X}+t_{\alpha/2}(n-1)\cdot \frac{S_n}{\sqrt{n-1}}]
$$
**正态总体，求方差$\sigma^2$的区间估计**
取样本函数$\chi^2 = \frac{nS_{n}^2}{\sigma^2}\sim \chi^2(n-1)$. $P\{\lambda_1\leq \chi^2 \leq \lambda_2\}=1-\alpha$。由于$\chi^2$分布不对称，可采用使概率对称的区间$P\{\chi^2<\lambda_1\} = P\{\chi^2 > \lambda_2\}=\alpha/2$. 解得$\sigma^2$的置信区间：
$$
[\frac{nS_n^2}{\chi^2_{\alpha/2}(n-1)}, \frac{nS_n^2}{\chi^2_{1-\alpha/2}(n-1)}]
$$
**4.2 双正态总体情形**
$(X_1,\cdots,X_{n-1})$为总体$X\sim N(\mu_1,\sigma_1^2)$的样本，$(Y_1,\cdots,Y_{n_2})$为总体$Y\sim N(\mu_2,\sigma^2_2)$的样本，$\overline{X},S_1^2,\overline{Y},S_2^2$分别表示$X,Y$的样本均值与修正样本方差。设$X,Y$独立，置信度为$1-\alpha$. 求$\mu_1-\mu_2,\sigma_1^2/\sigma_2^2$的区间估计.

**(1) $\sigma_1^2,\sigma_2^2$已知，求$\mu_1-\mu_2$的置信区间**
利用$\frac{(\overline{X}-\overline{Y})-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}}\sim N(0,1)$，解得置信区间
$$
\left( (\overline{X}-\overline{Y}) - u_{\alpha/2}\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}, (\overline{X}-\overline{Y}) + u_{\alpha/2}\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}} \right)
$$
**(2) 方差比$\sigma_1^2/\sigma_2^2$的置信区间**
利用$F=\frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2}\sim F(n_1-1,n_2-1)$，解得置信区间
$$
\left( \frac{S_1^2}{S_2^2}\frac{1}{F_{\alpha/2}(n_1-1,n_2-1)}, \frac{S_1^2}{S_2^2}\frac{1}{F_{1-\alpha/2}(n_1-1,n_2-1)} \right)
$$
**4.3 单侧置信区间**
对于$0<\alpha<1$，样本$X_1,\cdots,X_n$，确定统计量$\hat\theta_1(X_1,\cdots,X_n)$，使$P(\theta>\hat\theta_1)=1-\alpha$，则称$(\hat\theta_1,+\infty)$是$\theta$的置信度$1-\alpha$的单侧置信区间。$\hat\theta_1$称为**单侧置信下限**。同理根据$P(\theta<\hat\theta_2)=1-\alpha$可定义**单侧置信上限**。

**4.4 非正态总体均值的区间估计**
当总体分布非正态时，一般很难求出统计量的具体分布。此时采用大样本发。在样本量较大时，利用极限定理求出枢轴变量的近似分布，然后求得参数的区间估计。例如，对样本$X_1,\cdots,X_n$求$\mu$置信度为$1-\alpha$的区间估计，则$n$充分大时，根据中心极限定理有
$$
\frac{\overline{X}-\mu}{\sigma/\sqrt{n}}\rightarrow N(0,1)
$$
若$\sigma$未知，可用样本标准差$S_{n-1}$无偏代替，得
$$
U=\frac{\overline{X}-\mu}{S_{n-1}/\sqrt{n}}\sim N(0,1)
$$
可解得$\mu$的置信区间为
$$
(\overline{X}-u_{\alpha/2}\frac{S_{n-1}}{\sqrt{n}}, \overline{X}+u_{\alpha/2}\frac{S_{n-1}}{\sqrt{n}})
$$

