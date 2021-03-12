# Probability and Mathematical Statistics

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
P(A_1A_2\cdots A_n)=P(A_1)P(A_2|A_1)P(A_3|A_1A_1)\cdots P(A_n|A_1A_2\cdots A_{n-1})
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



