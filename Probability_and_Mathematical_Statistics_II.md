# Probability and Mathematical Statistics II

Author: Daniel Liu
Contact Me: 191240030@smail.nju.edu.cn
Course: Probability and Mathematical Statistics. NJU

## Ch4 随机变量的数字特征

### 1 数学期望

**1.1 定义**
(1) 设**离散型随机变量X**的分布律为$P(X=x_i)=p_i,i=1,2,\cdots$，若级数$\sum_{i=1}^{+\infty}|x_i|p_i$收敛(绝对收敛)，则称$\sum_{i=1}^{+\infty}x_ip_i$为$X$的数学期望，即
$$
EX=\sum_{i=1}^{+\infty}x_ip_i
$$
(2) 设连续型随机变量X的密度为$p(x)$，若积分$\int_{-\infty}^{+\infty}|x|p(x)dx < \infty$，则定义$X$的数学期望EX为
$$
EX=\int_{-\infty}^{+\infty}xp(x)dx
$$
**1.2 常见分布的数学期望**
(1) 0-1分布 $B(p)$
$$
EX = p
$$
(2) 二项分布 $B(n,p),0<p<1$，$P(X=k)=C_n^kp^k(1-p)^{n-k},k=0,1,\cdots,n$
$$
EX=np
$$
(3) 泊松分布 $P(\lambda),\lambda>0$，$P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda},k=0,1,\cdots$
$$
EX=\lambda
$$
(4) 几何分布 $G(p)$，$P(X=k)=(1-p)^{k-1}p,k=1,2\cdots$
$$
EX=\frac{1}{p}
$$
(5) 均匀分布 $U[a,b],-\infty < a < b < +\infty$，$p(x)=\frac{1}{b-a}I_{[a,b]}(x)$
$$
EX=\frac{a+b}{2}
$$
(6) 指数分布 $e(\lambda),\lambda>0$，$p(x)=\lambda e^{-\lambda x},x>0$
$$
EX=\frac{1}{\lambda}
$$
(7) 正态分布 $N(\mu, \sigma^2)$，$p(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-(x-\mu)^2}{2\sigma^2}},x\in\R$
$$
EX=\mu
$$
**PS. 示性函数**
$I_A=\left\{\begin{matrix}1,&\textbf{A发生}\\0,&\textbf{A不发生}  \end{matrix}\right. $，$EI_A=P(A)$.

**1.3 随机变量函数的数学期望**
**定理1：**若X的函数$Y=g(X)$也是一个随机变量，设$g(X)$的数学期望存在：
(1) 若$X$为离散型随机变量，其分布律为$P(X=x_k)=p_k,k=1,2,\cdots$，则Y的数学期望为
$$
EY=E[g(X)]=\sum_{k=1}^{+\infty}g(x_k)P(X=x_k)=\sum_{k=1}^{+\infty}g(x_k)p_k
$$
(2) 若$X$为连续型随机变量，其密度为$p(x)$，则
$$
EY=E[g(X)]=\int_{-\infty}^{+\infty}g(x)p(x)dx
$$
**定理2：**设随机向量$(X,Y)$的函数$Z=g(X,Y)$是一个随机变量，且期望存在：
(1) 若$(X,Y)$为离散型随机向量，其联合分布律为$P(X=x_i,Y=y_j)=p_{ij},i,j=1,2,\cdots$，则
$$
EZ=E[g(X,Y)]=\sum_{i=1}^{+\infty}\sum_{j=1}^{+\infty}g(x_i,y_j)p_{ij}
$$
(2) 若$(X,Y)$为连续型随机向量，其联合密度为$p(x,y)$，则
$$
EZ=E[g(X,Y)]=\int_{-\infty}^{+\infty}\int_{-\infty}^{\infty}g(x,y)p(x,y)dxdy
$$
**1.4 数学期望的性质**

1. 对常数$a,b$，若$a\leq X\leq b$，则$a\leq EX\leq b$，特别地，$E(a)=a$
2. **任意**随机变量$X,Y$，若它们的数学期望存在，则对任意有限常数$a,b,E(aX+bY)=aEX+bEY$.
3. 若$X,Y$**相互独立**，则$E(XY)=E(X)E(Y)$

### 2 方差

**2.1 方差定义**
若$EX^2<+\infty$，则称$E(X-EX)^2$为随机变量$X$的方差，记为$D(X)$或$Var(X)$. 称$\sigma(X)=\sqrt{D(X)}$为$X$的**均方差**或**标准差**。
**计算方法**：$D(X)=EX^2-(EX)^2$

**2.2 方差的性质**

1. 设C为常数，$D(C)=0$。另外，$D(X)=0\iff X$以概率1取常数$C$
2. $D(CX)=C^2D(X)$，特别地$D(-X)=D(X)$
3. $D(X\pm Y)=D(X)+D(Y)\pm 2E\{(X-E(X))(Y-E(Y))\}$
   $X,Y$独立时，$D(X\pm Y)=D(X)+D(Y)$
4. (**切比雪夫不等式**)设随机变量$X$的期望$EX$和方差$DX$均存在，则$\forall \epsilon > 0$，$P(|X-EX|\geq \epsilon)\leq \frac{DX}{\epsilon^2}$

**2.3 常用分布的方差**
(1) 0-1分布：$D(X)=pq$
(2) 二项分布：$D(X)=npq$
(3) 泊松分布：$D(X)=\lambda$
(4) 几何分布：$D(X)=\frac{1-p}{p^2}$
(5) 均匀分布：$D(X)=\frac{(b-a)^2}{12}$
(6) 指数分布：$D(X)=\frac{1}{\lambda^2}$
(7) 正态分布：$D(X)=\sigma^2$

### 3 矩、协方差与相关系数

**3.1 矩**
**定义：**对随机变量$X$和非负整数$k$，若$E(|X|^k)<\infty$，则称$EX^k$为$X$的$k$**阶原点矩**，简称$k$**阶矩**。若$E(|X-EX|^k)<\infty$，则称$E(X-EX)^k$为$X$的$k$**阶中心矩**.

一阶矩为$X$的期望，二阶中心矩为方差，二阶矩$EX^2=DX+(EX)^2$

对于标准正态分布$X\sim N(0,1)$，$k$阶矩为：
$$
EX^k = \left\{ \begin{matrix} 0,&\text{k为奇数}\\(k-1)!!,&\text{k为偶数} \end{matrix} \right .
$$
**3.2 协方差**
**定义：**对随机变量$X,Y$，若$E|X|,E|Y|$和$E|(X-EX)(Y-EY)|$都有限，则定义$E[(X-EX)(Y-EY)]$为$X$和$Y$的**协方差**，记为$cov(X,Y)$.

**随机变量方差公式：**
$$
D(X\pm Y)=D(X)+D(Y)\pm 2cov(X,Y)\\
D(X_1+\cdots+X_n) = \sum_{k=1}^nD(X_k)+2\sum_{1\leq i < j \leq n}cov(X_i,X_j)
$$
**协方差计算公式：**
$$
cov(X,Y)=E(XY)-EX\cdot EY
$$
**性质：**

1. $cov(X,k) = 0$
2. 若$X,Y$独立，则$cov(X,Y)=0$，反之不成立
3. 方差是特殊的协方差：$cov(X,X)=D(X)$
4. $cov(X,Y)=cov(Y,X)$
5. $cov(aX+c_1, bY+c_2)=ab\,cov(X,Y)$
6. 若$X_1,X_2,Y$的二阶矩有限，则$cov(X_1+X_2,Y)=cov(X_1,Y)+cov(X_2,Y)$
7. **Cauchy-Schwarz不等式:** $[cov(X,Y)]^2\leq D(X)D(Y)$

**3.3 相关系数**
**定义：**设随机变量$X$和$Y$的二阶矩有限，则$D(X)>0,D(Y)>0$，则称
$$
\frac{cov(X,Y)}{\sqrt{D(X)D(Y)}}
$$
为$X$和$Y$的**相关系数**，记为$\rho_{XY}$.

**性质：**

1. $|\rho_{XY}|\leq 1$
2. $|\rho_{XY}|=1\iff\exists a,b(a\neq 0),P\{Y=aX+b\}=1$。
   且$a>0,\rho=1$，正相关，$a<0,\rho=-1$，负相关。
   $\rho_{XY}$表示$X,Y$存在**线性关系的强弱程度**，$|\rho_{XY}|=0$表示$X,Y$不存在线性关系，称为**不相关**. 
   $X,Y$独立$\rightarrow X,Y$不相关(反之不成立)$\leftrightarrow \rho_{XY}=0\leftrightarrow cov(x,y)=0\leftrightarrow E(XY)=EX\cdot EY \leftrightarrow\\ D(X\pm Y)=D(X)\pm D(Y)$
   **二维正态分布：**$X,Y$独立$\leftrightarrow X,Y$不相关

***

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





