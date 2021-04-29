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
4. (切比雪夫不等式)设随机变量$X$的期望$EX$和方差$DX$均存在，则$\forall \epsilon > 0$，$P(|X-EX|\geq \epsilon)\leq \frac{DX}{\epsilon^2}$

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
**定义**

**3.2 协方差**
**定义：**
**性质：**

1. $cov(X,k) = 0$

**3.3 相关系数**
**定义：**
**性质：**

1. $|\rho_{XY}|\leq 1$
2. $|\rho_{XY}|=1\iff\exists a,b(a\neq 0),P\{Y=aX+B\}=1$。
   且$a>0,\rho=1$，正相关，$a<0,\rho=-1$，负相关。
   $\rho_{XY}$表示$X,Y$存在**线性关系的强弱程度**，$|\rho_{XY}|=0$表示$X,Y$不存在线性关系，称为**不相关**. 
   ($X,Y$独立$\rightarrow X,Y$不相关$\leftrightarrow \rho_{XY}=0\leftrightarrow cov(x,y)=0\leftrightarrow$)
   **二维正态分布：**$X,Y$独立$\rightarrow X,Y$不相关
3. 



***

## Ch5 极限理论

### 1 大数定律

**1.1 大数定律的定义**

**1.2 马尔可夫大数定律**

Homework: Ch4 - 15~18 20 22  





