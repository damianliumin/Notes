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