# Problem Solving IV (PART II)

Author: Daniel Liu
Contact me:  191240030@smail.nju.edu.cn
Course: Problem Solving IV. NJU

## 4-8 问题的形式化描述 

### 1 语言

**字符表的定义：**任意非空有限集称为**字符表alphabet**，字符表$\Sigma$的任意元素称为$\Sigma$的**符号symbol**.

**词的定义：**令$\Sigma$为字符表，$\Sigma$上有限长的符号序列为**词word**，空词$\lambda$是唯一由0个元素组成的词。$\Sigma$上所有词组成的集合记为$\Sigma^*$.
**词的长度：**定义$\Sigma$上一个词$w$的长度为$w$中符号的个数，记为$|w|$。$\#_a(w)$为$w$中符号$a$出现的次数。定义$\Sigma^n=\{x\in \Sigma^*||x|=n\}$，特别地，$\Sigma^+=\Sigma^*-\{\lambda\}$.
**词的连接的定义：**定义$\Sigma$上两个词$v,w$的**连接concatenation**为$v$的符号按原序接上原序的$u$的符号，记为$vw$。$\forall w\in \Sigma^*$，定义$(i)w^0=\lambda,(ii)\forall n \in \N^*, w^{n+1}=w\cdot w^n$.
**标准序定义：**$\Sigma=\{s_1,s_2,\cdots,s_m\},m\geq 1$是一个字符表，令$s_1<s_2<\cdots < s_m$为$\Sigma$上一个线性序，定义$\Sigma^*$上的标准序canonical ordering如下：
$\forall u,v\in \Sigma^*,u<v~if~|v|<|u| ~ or |v|=|u|,u=xs_iu'\land v=xs_jv',i<j$.

**语言的定义：**$\Sigma$为一个字符表，任意$L\subseteq \Sigma^*$都是$\Sigma$上的**语言language**，$L$的**补complement**为$L^C=\Sigma^*-L$。语言$L_1\subseteq \Sigma_1^*,L_2\subseteq\Sigma_2^*$的**连接concatenation**定义为$L_1L_2=L_1\circ L_2=\{uv\in(\Sigma_1\cup \Sigma_2)^*|u\in L_1\land v\in L_2\}$.

### 2 算法问题

**2.1 决策问题**
**决策问题的定义：** A decision problem is a triple $(L,U,\Sigma)$ where $\Sigma$ is an alphabet and $L\subseteq U \subseteq \Sigma^*$. An algorithm A solves (decides) the decision problem $(L,U,\Sigma)$ if, for every $x\in U$,
(i) $A(x) = 1 ~if ~x\in L$
(ii) $A(x)=0~if~ X\in U-L~(x\not\in L).$

**Problem** ($L,U,\Sigma$)
	Input: An $x\in U$
	Output: "yes" if $x\in L$
					"no" otherwise

**EX1 素性检验**
$(PRIM, \Sigma_{bool})$, 其中$PRIM=\{w\in\{0,1\}^*|Number(w)~is ~ a ~ prime\}$.
**Primality testing**
	Input: An $x\in \Sigma^*_{bool}$
	Output: "yes" if Number(x) is a prime
					"no" otherwise

**EX2 Equivalence Problem for Polynomials**
**EQ-POL**
	Input: A prime $p$, 2 polynomials $p_1$ and $p_2$ over variables from $X=\{x_1,x_2,\cdots\}$
	Output: "yes" if $p_1\equiv p_2$ in the field $\Z_p$
					"no" otherwise

**EX3 Equivalence Problem for One-time-only Branching Programs**
**EQ-1BP**
	Input: One-time-only branching program $B_1$ and $B_2$ over a set of Boolean 				variables $X=\{x_1,x_2,x_3\cdots\}$
	Output: "yes" if $B_1,B_2$ are equivalent (represent the same Boolean function)
					"no" otherwise

**EX4 Satisfiability Problem**
The satisfiability problem is to decide, for a given formula in the CNF$ (\vee\wedge-nf)$, whether it is satisfiable or not.
(SAT, $\Sigma^+_{logic}$), $\text{SAT} = \{ w \in \Sigma_{logic}^+ | w \text{ is a code of a satisfiable formula in CNF} \}$

**EX5 Clique Problem**
**Clique Problem**
	Input: A positive integer $k$ and a graph $G$
	Output: "yes" if G contains a clique of size $k$,
					"no" otherwise.

**2.2 优化问题**
**优化问题的定义：** An optimization problem is a 7-tuple $U=(\Sigma_I,\Sigma_O,L,L_I,\mathcal{M},cost, goal)$, where
(i) $\Sigma_I$ is the **input alphabet** of $U$
(ii) $\Sigma_O$ is the **output alphabet** of $U$
(iii) $L\subseteq \Sigma_I^*$ is the **language of feasible problem instances**
(iv) $L_I\subseteq L$ is the **language of the (actual) problem instances of U**
(v) $\mathcal{M}$ is a function from $L$ to $Pot(\Sigma_O^*)$, and, for every $x\in L$, $\mathcal{M}(x)$ is called the **set of feasible solutions** for x.
(vi) cost is the **cost function**​ that, for every pair $(u, x)$, where $u\in \mathcal{M}(x)$ for some $x\in L$, assigns a positive real number $cost(u,x)$.
(vii) $goal\in\{minimum, maximum\}$

For every $x \in L_I$, a feasible solution $y\in \mathcal{M}(x)$ is called **optimal for x and U** if 
$$
cost(y, x) = goal\{cost(z, x) | z \in \mathcal{M}(x) \}.
$$
For an optinal solution $y\in \mathcal{M}(x)$, we denote $cost(y, x)$ by $Opt_U(x)$. *U* is called a **maximization problem** if $goal = \text{maximum}$ and *U* is a **minimization problem** if $goal = \text{minimum}$. In what follows $Output_U(x) \subseteq \mathcal{M}(x)$ denotes the set of all optimal solutions for the instance $x$ of *U*.

An algorithm *A* is **consistent** for *U* if, for every $x \in L_I$, the output $A(x) \in \mathcal{M}(x)$. We say that an algorithm *B* **solves** the optimization problem *U* if
(i) *B* is consistent for *U*
(ii) for every $x \in L_I$, $B(x)$ is an optimal solution for $x$ and $U$.

**EX1 Traveling Salesperson Problem (TSP)**
**EX2 Makespan Scheduling Problem (MS)**
**EX3 Cover Problems (MIN-VCP, SCP, WEIGHT-VCP)**
**EX4 Maximum Clique Problem (MAX-CL)**
**EX5 Cut Problems (MAX-CUT)**
**EX6 Knapsack Problem (KP)**
**EX7 Bin-Packing Problem (BIN-P)**
**EX8 Maximum Satisfiability Problem (MAX-SAT)**
**EX9 Linear Programming (LP)**

### 3 复杂性理论

**3.1 复杂度**
**时间复杂度与空间复杂度的定义：**Let $\Sigma_I$ and $\Sigma_O$ be alphabets. Let *A* be an algorithm that realizes a mapping from $\Sigma_I^*$ to $\Sigma_O^*$. For every $x \in \Sigma_I^*$,  $Time_A(x)$ denotes the time complexity (according to the logarithmic cost) of the computation of *A* on the input $x$, and $Space_A(x)$ denotes the space complexity (according to the logarithmic cost measurement) of the computation of *A* on $x$.
**(worst case) time complexity:** $Time_A(n) = \max\{Time_A(x)|x\in \Sigma_I^n\}$.

**定理：**存在决策问题$(L, \Sigma_{bool})$，任意决定$L$的算法$A$都有算法$B$满足$Time_B(n)=\log_2Time_A(n)$对无穷多个$n$成立。
因此，算法问题无法定义复杂度，只能定义复杂度的上界和下界。

**复杂度上界和下界的定义：** Let *U* be an algorithmic problem, and let *f*, *g* be functions from $\mathbb{N}$ to $\mathbb{R}^+$:
We say that $O(g(n))$ is an **upper bound on the time complexity of U** if there exists an algorithm *A* solving *U* with $Time_A(n) \in O(g(n))$.
We say that $\Omega(f(n))$ is a **lower bound on the time complexity of U** if every algorithm *B* solving *U* has $Time_B(n) \in \Omega(f(n))$.
算法$C$最优：$Time_C(n)\in O(g(n))$且$\Omega(g(n))$是*U*的下界

**3.2 P与NP**
**Church-Turing Thesis: **a problem *U* can be solved by an algorithm (computer program in any programming language formalism) if and only if there exists a Turing machine solving *U*.
Using the formalism of TMs it was proved that for every increasing function $f: \mathbb{N} \rightarrow \mathbb{R}^+$:

1. there exists a decision problem such that every TM solving it has the time complexity in $\Omega(f(n))$
2. but there is a TM solving it in $O(f(n) \cdot \log f(n))$ time.

One can say that the main objective of the complexity theory is

1. to find a formal specification of the class of pratically solvable problems, and
2. to develop methods enabling the classification of algorithmic problems according to their membership in this class.

Let, for every TM (algorithm) *M*, *L(M)* denote the language decided by *M*.

**Definition 2.3.3.5** We define the complexity class **P** of languages decidable in polynomial-time by 
$$
\textbf{P} = \{ L = L(M) | M \text{ is a TM (an algorithm) with } Time_M(n) \in O(n^c) \}
$$
A language (decision problem) *L* is called **tractable (practically solvable)** if $L \in \text{P}$. A language *L* is called **intractable** if $L \notin \text{P}$.

To prove the membership of a decision problem *L* to **P**, it is sufficient to design a polynomial-time algorithm for *L*.

**Definition 2.3.3.6** Let *M* be a nondeterministic TM (algorithm). We say that M accepts a languege $L$, $L = L(M)$, if

1. $\forall x \in L$, there exists at least one computation of M that accepts $x$
2. for every $y \notin L$, all computations of M reject $y$.

For every input $w \in L$, the time complexity $Time_M(w)$ of $M$ on $w$ is the time complexity of the shortest accepting computation of *M* on *w*. The **time complexity of M** is the function $Time_M$ from $\mathbb{N}$ to $\mathbb{N}$ defined by
$$
Time_M(n) = \max \{ Time_M(x) | x \in L(M) \cap \Sigma^n \}.
$$
We define the class 
$$
\textbf{NP} = \{L(M) | M \text{ if a polynomial-time nondeterministic TM}\}
$$
as the class of decision problems decided nondeterministically in polynomial time.

**Definition 2.3.3.7** Let $L \subseteq \Sigma^*$ be a language. An algorithm *A* working on inputs from $\Sigma^* \times \{0, 1\}^*$ is called a **verifier for L**, denoted L = V(A), if
$$
L = \{w \in \Sigma^* | A \text{ accepts } (w, c) \text{ for some } c \in \{0, 1\}^* \}.
$$
If *A* accepts (*w*,*c*), we say that *c* is a **proof (certifiate)** of the fact $w \in L$.

A verifier *A* for *L* is called a **polynomial-time verifier** if there exists a positive integer *d* such that, for every $w \in L$, $Time_A(w, c) \in O(\vert w \vert ^d)$ for a proof *c* of $w \in L$.

We define the **class of polynomially verifiable languages** as 
$$
\textbf{VP} = \{ V(A) | A \text{ is a polynomial-time verifier.} \}
$$
**Definition 2.3.3.10** Let $L_1 \subseteq \Sigma_1^*$ and $L_2 \subseteq \Sigma_2^*$ be two languages. We say that *L*1 is **polynomial-time reducible** to *L*2, $L_1 \leqslant_p L_2$, if there exists a polynomial-time algorithm *A* that computes a mapping from $\Sigma_1^*$ to $\Sigma_2^*$ such that, for every $x \in \Sigma_1^*$, $x \in L_1 \Longleftrightarrow A(x) \in L_2$. $A$ is called the **polynomial-time reduction** from *L*1 to *L*2.

A language *L* is called **NP-hard** if, for every $U \in \text{NP}$, $U \leqslant_p L$.

A language L*L* is called **NP-complete** if (1) $L \in \text{NP}$ and (2) *L* is NP-hard.

**Lemma 2.3.3.11** If *L* is NP-hard and $L \in P$, then $P = NP$.

**Theorem 2.3.3.12 (Cook’s Theorem)** SAT is NP-complete.

**Observation 2.3.3.13** Let *L*1 and *L*2 be two languages. If $L_1 \leqslant_p L_2$ and *L*1 is NP-hard, then *L*2 is NP-hard.

**Lemma 2.3.3.15** SAT $\leqslant_p$ Clique.

**Lemma 2.3.3.16** Clique $\leqslant_p$ VC.

**Lemma 2.3.3.19** 3SAT $\leqslant_p$ Sol-0/1-LP.

**Definition 2.3.3.21** **NPO** is the class of optimization problems, where $U = (\Sigma_I, \Sigma_O, L, L_I, \mathcal{M}, cost, goal) \in \text{NPO}$ if the following conditions hold:

1. $L_1 \in \text{P}$
2. there exists a polynomial $p_U$ such that
   1. for every $x \in L_I$, and every $y \in \mathcal{M}(x)$, $\vert y \vert \leqslant p_U(\vert x \vert)$
   2. there exists a polynomial-time algorithm that, for every $y \in \Sigma_O^*$ and every $x \in L_I$ such that $\vert y \vert \leqslant p_U(\vert x \vert)$, devides whether $y \in \mathcal{M}(x)$
3. the function cost is computable in polynomial time.

**Definition 2.3.3.23** **PO** is the class of optimization problems $U = (\Sigma_I, \Sigma_O, L, L_I, \mathcal{M}, cost, goal)$ such that

1. $U \in \text{NPO}$
2. there is a polynomial-time algorithm that, for every $x \in L_I$, computes an optimal solution for $x$.

**Definition 2.3.3.24** Let $U = (\Sigma_I, \Sigma_O, L, L_I, \mathcal{M}, cost, goal)$ be an optimization problem from NPO. We define the **threshold language of U** as
$$
Lang_U = \{(x, a) \in L_1 \times \Sigma_{bool}^* | Opt_U(x) \leqslant Number(a)\}
$$
if $goal = \texttt{minimum}$, and as
$$
Lang_U = \{(x, a) \in L_1 \times \Sigma_{bool}^* | Opt_U(x) \geqslant Number(a)\}
$$
if $goal = \texttt{maximum}$.

We say that **U is NP-hard** if $Lang_U$ is NP-hard.

**Lemma 2.3.3.25** If an optimization problem $U \in \text{PO}$, then $Lang_U \in \text{P}$.

**Theorem 2.3.3.26** Let *U* be an optimization problem. If $Lang_U$ is NP-hard and $P \neq NP$, then $ U \notin PO$.

**Lemma 2.3.3.27** Max-SAT is NP-hard.

**Lemma 2.3.3.28** Max-CL is NP-hard.

***

## 4-9 NP完全理论初步 NP-Completeness

### 1 多项式时间

**1.1 抽象问题**
抽象问题$Q$是问题实例集合$I$和问题解的集合$S$上的一个二元关系。**判定问题 decision problem**可以看作从$I$映射到$\{0,1\}$的函数，**优化问题 optimization problem**需要最大化或最小化某值。优化问题的难度通常可以转化为一个与之对应的判定问题的难度。

**1.2 问题编码**
定义问题$Q$的一种编码方式$e(Q): I\rightarrow \{0,1\}^*$，多项式时间内可解的实例问题的集合定义为$P$。
这里需要注意的是编码方式，为了防止复杂度受编码方式的影响，我们定义**多项式相关 polynomially related**（例如二进制编码和三进制多项式相关，不和一进制多项式相关）。因此，我们只需要对某种对象规定一个标准编码，与之多项式相关的其他编码对问题复杂性的分析没有影响。

**1.3 语言**
定义语言$L$为字符表$\Sigma$上任意的字符串集合，空串记为$\epsilon$，空语言记为$\emptyset$，所有字符串组成的语言记为$\Sigma^*$。定义$L=L_1L_2=\{x_1x_2:x_1\in L_1\text{ and }x_2\in L_2\}$，$\overline{L}=\Sigma^* - L$, $L^*=\{\epsilon\}\cup L\cup L^2\cup L^3\cup \cdots$

算法$A$**接受 accept**字符串$x\in\{0,1\}^*$：$A(x)=1$，**拒绝 reject** $x$：$A(x)=0$
算法$A$**接受 accept**语言$L$：$L=\{x\in\{0,1\}^*: A(x)=1\}$
算法$A$**判定 decide**语言$L：$$A$接受$L$且$\forall x\notin L, A(x)=0$

在此基础上可以定义：
$$
P=\{ L\subseteq \{0, 1\}*: \text{there exists an algorithm $A$ such that decides $L$}\\\text{in polynomial time}\}
$$
有如下定理：
$$
P=\{L:L\text{ is accepted by a polynomial-time algorithm}\}
$$

### 2 多项式时间验证

**2.1 验证算法**
验证算法$A$接受两个参数：$x$和$y$，后者称为**certificate**。若存在$y$使得$A(x,y)=1$，称$A$验证了输入$x$。由算法$A$验证过的语言为：
$$
L=\{x\in\{0,1\}^*: \text{there exists $y\in \{0,1\}^*$ such that $A(x,y)=1$}\}
$$

**2.2 NP**
定义可以在多项式时间内被验证的语言类为$NP$，也即下列$L$属于$NP$:
$$
L=\{x\in\{0,1\}^*: \text{there exists a certificate $y$ with $|y|=O(|x|^c)$}\\\text{such that $A(x,y)=1$}\}
$$
尽管尚无确切证明，学术界倾向于认为$P\neq NP$。除了这个问题，目前也没法证明NP在补运算下是否封闭，也即**NP=co-NP**，其中co-NP=$\{L:\overline{L}\in NP\}$。可以证明**P** $\subseteq$ **NP** $\cap$ **co-NP**.

### 3 NPC和归约

**3.1 归约**
$L_1,L_2$为语言，若存在多项式时间内可计算的函数$f:\{0,1\}^*\rightarrow\{0,1\}^*$使得对于任意$x\in \{0,1\}^*$，$x\in L_1\leftrightarrow f(x)\in L_2$，则称$L_1$可在**多项式时间内归约 polynomial-time reducible**为$L_2$，记为$L_1\leq_P L_2$。其中$f$为**归约函数 reduction function**，计算出$f$的多项式时间算法$F$为**归约算法 reduction algorithm**。

**引理：**$L_1,L_2\subseteq \{0,1\}^*$是满足$L_1\leq_P L_2$的语言，则$L_2\in P\rightarrow L_1\in P$

**3.2 NP完全性**
定义$L\subseteq \{0,1\}^*$是**NP-完全 NP-complete**的：
(1) $L\in NP$
(2) $\forall L'\in NP,~ L' \leq_P L$
若$L$满足(2)但不确保满足(1)，则是NP-hard的。我们定义**NPC**为NP-complete的语言的类。

**定理：**如果任意NP-complete的问题在多项式时间内可解，则$P=NP$。逆否命题为，如果任意NP中的问题不能在多项式时间内解决，则NP-complete问题不能在多项式时间内解决。

**3.3 CIRCUIT-SAT**
只需要证明一个问题是NPC的，则可以通过归约推出很多其他问题是NPC的。可以将CIRCUIT-SAT作为第一个证明。

$\text{CIRCUIT-SAT}=\{\left<C\right>: C\text{ is a satisfiable boolean combinational circuit}\}$
 证明思路：

1. 证明C-SAT $\in$ NP
2. 证明C-SAT是NP-hard的：通过计算机状态内存的模型构造出一个F

### 4 NP-complete证明

**4.1 证明方法**
**引理：**$L$是一个语言，若存在$L'\in NPC$满足$L'\leq_P L$，则$L$是NP-hard的。如果另有$L\in NP$，则$L\in NPC$.

已证$C-SAT\in NPC$，结合上述引理，证明$L$是NP-complete步骤如下：

1. 证明$L$
2. 选择一个已知为NP-complete的语言$L'$
3. 设计算法，计算函数$f$将$L'$的每个实例$x\in \{0,1\}^*$映射到$L$的实例$f(x)$
4. 证明$f$满足：$\forall x\in \{0,1\}^*,x\in L' \leftrightarrow  f(x)\in L$
5. 证明计算$f$的算法是在多项式时间内的

### 5 NP-complete问题

**5.1 C-SAT**
**5.2 SAT** (from C-SAT)
**5.3 3-CNF-SAT** (from SAT)
**5.4 CLIQUE** (from 3-CNF-SAT)
$\text{CLIQUE}=\{\left< G, k\right>:\text{$G$ is a graph containing a clique of size $k$}\}$
**5.5 VERTEX-COVER** (from CLIQUE)
$\text{VERTEX-COVER} = \{\left< G,k \right>: \text{$G$ has a vertex cover of size $k$} \}$
**5.6 HAM-CYCLE** (from VERTEX-COVER)
**5.7 TSP** (from HAM-CYCLE)
$\text{TSP}=\{\left<G,c,k\right>: G=(V, E)\text{ is a complete graph,}\\~~~~\text{c is a function from $V\times V\rightarrow \Z$,} \\~~~~\text{$k\in \Z$,}\\~~~~\text{G has a traveling-salesman tour with cost at most $k$} \} $
**5.8 SUBSET-SUM** (from 3-CNF-SAT)
$\text{SUBSET-SUM}=\{\left<S,t\right>:\text{$\exists$ subset $S'\subseteq S$ such that $t=\sum_{s\in S'}s$}\}$


***

## 4-10 近似算法 Approximation Algorithms

### 1 近似算法概念

**1.1 概念1**
考虑优化问题$U = (\Sigma_I, \Sigma_O, L, L_I, \mathcal{M}, cost, goal)$及它的一个算法$A$，$\forall x\in L_I$，$A$在$x$上的**相对误差 relative error** $\varepsilon_A(x)$定义为：
$$
\varepsilon_A(x)=\frac{|cost(A(x)) - Opt_U(x)|}{Opt_U(x)}
$$
$\forall n\in \N$, 定义$A$的相对误差为：
$$
\varepsilon_A(n)=\max\{ \varepsilon_A(x)|x\in L_1\cap (\Sigma_I)^n \}
$$
$\forall x\in L_I$，$A$在$x$上的**近似比 approximation ratio** $R_A(x)$定义为：
$$
R_A(x) = \max \left\{ \frac{cost(A(x))}{Opt_U(x)}, \frac{Opt_U(x)}{cost(A(x))} \right\}
$$
$\forall n\in \N$, 定义$A$的近似比为：
$$
R_A(n)=\max\{ R_A(x)|x\in L_1\cap (\Sigma_I)^n \}
$$
**1.2 概念2**
$\forall \delta>1$，若$\forall x\in L_I, R_A(x)\leq \delta$，称$A$为$\delta$-近似算法；
$\forall f:\N\rightarrow \R^+$，若$\forall n\in \N, R_A(n)\leq f(n)$，称$A$为$f(n)$-近似算法.

**1.3 概念3**
对最小化问题$U$，有
$$
R_A(x) = \frac{cost(A(x))}{Opt_U(x)}=1+\varepsilon_A(x)
$$
对最大化问题$U$，有
$$
R_A(x)=\frac{Opt_U(x)}{cost(A(x))}
$$

### 2 优化问题分类

**PTAS与FPTAS定义：**对于优化问题$U = (\Sigma_I, \Sigma_O, L, L_I, \mathcal{M}, cost, goal)$，若对于任意$(x,\varepsilon)\in L_I\times \R^+$，算法$A$能以最多为$\varepsilon$的误差计算出可行解$A(x)$，且计算时间$Time_A(x,\varepsilon^{-1})$在$|x|$的多项式内，则称$A$为PTAS (polynomial-time approximation scheme)。若$Time_A(x,\varepsilon^{-1})$也是在$\varepsilon^{-1}$内的，则称$A$为FPTAS (fully polynomial-time approximation scheme)。PTAS和FPTAS算法可在运行时间和误差大小间做取舍。

对$U\in NPO$分类：
**NPO(I):** 存在FPTAS
**NPO(II):** 存在PTAS
**NPO(III): ** $\exists\delta$，存在$\delta$-近似算法($\delta>1$)且$\forall d<\delta$，不存在$d$-近似算法（也即近似率有精确数值下界的近似算法）
**NPO(IV):** 不存在绝对的$\delta$-近似算法，但存在$f(n)$-近似算法，其中$f(n)$为界于$n$的多项式的函数
**NPO(V): **存在$f(n)$-近似函数，但是$f(n)$不界于任何$n$的多项式函数

### 3 近似稳定性

**定义：**令$U = (\Sigma_I, \Sigma_O, L, L_I, \mathcal{M}, cost, goal)$和$\overline{U} = (\Sigma_I, \Sigma_O, L, L, \mathcal{M}, cost, goal)$为两个优化问题，其中$L_I\sub L$。任意满足下列条件的函数$h_L:L\rightarrow \R^{\geq0}$称为$\overline{U}$根据$L_I$的**距离函数 distance function**:

+ $\forall x\in L_I, h_L(x)=0$
+ $h$是多项式时间可计算的

可以定义:
$$
Ball_{r,h}(L_I)=\{ w\in L| h(w)\leq r\}
$$
令$p$为正实数，若$\forall 0<r\leq p,~\exists \delta_{r,\varepsilon}\in \R^{>1}$使得$A$为问题$U_r = (\Sigma_I, \Sigma_O, L, Ball_{r, h}(L_I), \mathcal{M}, cost, goal)$的$\delta_{r,\varepsilon}$-近似算法，则称$A$为$p$-stable according to $h$. 若$\forall p\in \R^+$，$A$为p-stable according to h的，则称$A$为stable according to $h$，否则为unstable.

对任意任意正整数$r$，每个函数$f_r:\N\rightarrow \R^{>1}$，若$A$为问题$U_r = (\Sigma_I, \Sigma_O, L, Ball_{r, h}(L_I), \mathcal{M}, cost, goal)$的$f_r(n)$-近似函数，我们称$A$是$(f,f_r(n))$-quasistable according to $h$的。

**定义：**令$U = (\Sigma_I, \Sigma_O, L, L, \mathcal{M}, cost, goal)$和$\overline{U} = (\Sigma_I, \Sigma_O, L, L, \mathcal{M}, cost, goal)$为两个优化问题，且$L_I\sub L$，令$h$为$\overline{U}$根据$L_I$的距离函数，令$\forall r \in \R^+,U_r = (\Sigma_I, \Sigma_O, L, Ball_{r, h}(L_I), \mathcal{M}, cost, goal)$，$A=\{A_\varepsilon\}_{\varepsilon>0}$为$U$的一个PTAS。若$\forall r>0,\forall \varepsilon > 0$，$A_\varepsilon$为$U_r$的一个$\delta_{r,\varepsilon}$-近似算法，则PTAS算法$A$为stable according to $h$的。
若$\delta_{r,\varepsilon}\leq f(\epsilon)\cdot g(r)$，其中

+ $f$和$g$为从$\R^{\geq 0}$映射到$\R^+$的函数
+ $\lim_{\varepsilon\rightarrow 0}f(\varepsilon)=0$

则我们称PTAS算法$A$为superstable according to $h$的。

### 4 dual approximation algorithms

**定义1：**令$U = (\Sigma_I, \Sigma_O, L, L, \mathcal{M}, cost, goal)$为一个优化问题，$U$的一个**受限距离函数 constraint distance function**为满足下列条件的任意函数$h:L_I\times \Sigma_O^*\rightarrow \R^{\geq 0}$:

+ $\forall S\in \mathcal{M(x)},h(x,S)=0$
+ $\forall S\notin \mathcal{M(x)},h(x,S)>0$
+ $h$是多项式时间内可计算的

$\forall \varepsilon \in \R^+, \forall x\in L_I,\mathcal{M}_\varepsilon^h(x)=\{S\in \Sigma_O^*|h(x,S)\leq \varepsilon\}$为$\mathcal{M(x)}$的$\varepsilon$-ball according to $h$.

**定义2：**令$U = (\Sigma_I, \Sigma_O, L, L, \mathcal{M}, cost, goal)$为一个优化问题，$h$为U的一个受限距离函数。若$\forall x\in L_I$，满足以下条件时$U$的一个优化算法$A$称为$h$-dual $\varepsilon$-approximation algorithm for $U$：

+ $A(x)\in \mathcal{M}_\varepsilon^h(x)$
+ $cost(A(x))\geq Opt_U(x)$ if $goal = maximum$ and $cost(A(x))\leq Opt_U(x)$ if $goal = minimum$

**定义3：**令$U = (\Sigma_I, \Sigma_O, L, L, \mathcal{M}, cost, goal)$为一个优化问题，令$h$为$U$的一个受限距离函数。若满足以下条件，则称$A$为$h$-dual polynomial-time approximation scheme ($h$-dual PTAS for U):

+ $\forall$ input $(x,\varepsilon)\in L_I\times \R^+, A(x,\varepsilon)\in \mathcal{M}_\varepsilon^h(x)$
+ $cost(A(x))\geq Opt_U(x)$ if $goal = maximum$ and $cost(A(x))\leq Opt_U(x)$ if $goal = minimum$
+ $Time_A(x,\epsilon^{-1})$是界于$|x|$的多项式的函数

若$Time_A(x,\varepsilon^{-1})$同时也界于$\varepsilon^{-1}$，则称$A$为$h$-dual fully polynomial-time approximation scheme ($h$-dual FPTAS) for $U$.

***

## 4-11 随机算法 Randomized Algorithms

### 1 随机算法基本概念

**1.1 概率空间**
**定义：**随机算法$A$在输入$x$上的一次运行$C$，记概率为$Prob_{A,x}(C)$。$A$在$x$上输出$y$的概率$Prob(A(x)=y)$为$C$输出$y$的概率$Prob_{A,x}(C)$的和。随机算法的目标是：若$y$为$x$的正确输出，尽量提高$Prob(A(x)=y)$.

记$Random_A(x)$为算法$A$在$x$上所有随机运行过程中随机位数的最大值。$\forall n \in \N$，有
$$
Random_A(n) = \max\{Random_A(x) | x \text{ is an input of size }n \}
$$
**1.2 时间复杂度**
随机算法的时间复杂度用**期望时间复杂度**表示，有两种表示方法：
$$
Exp-Time_A(x)=E[Time]=\sum_C Prob_{A,x}(C)\times Times(C)\\
Exp-Time_A(n)=\max\{Exp-Time_A(x)|x\text{ is an input of size }n \}
$$
或：
$$
Time_A(x) = \max\{Time(C)|C\text{ is a run of $A$ on $x$}\}\\
Time_A(n) = \max\{Time_A(x)|x \text{ is an input of size }n \}
$$
前者给出了更精确的复杂度，但是很难计算。后者是期望的最坏情况复杂度。

### 2 随机算法分类

**2.1 Las Vegas 算法**
**第一类定义：**随机算法$A$是计算问题$F$的Las Vegas算法 if $F$的任意输入实例$x$满足：
$$
Prob(A(x)=F(x))=1
$$
其中$F(x)$是$F$在输入实例下的解。这一类Las Vegas算法保证结果正确，用$Exp-Time_A(n)$分析复杂度。

**第二类定义：**随机算法$A$是计算问题$F$的Las Vegas算法 if $F$的任意输入实例$x$满足：
$$
Prob(A(x)=F(x))\geq \frac{1}{2}\\
Prob(A(x)=?)=1-Prob(A(x)=F(x))\leq \frac{1}{2}
$$

其中$F(x)$是$F$在输入实例下的解。这一类Las Vegas算法多数情况下正确，并且可以保持沉默但用不犯错（找到正确解的概率随着计算时间的增加而显著提高，正确概率可以接受）。其复杂度用$Time_A(n)$表示。

通常用第一类处理函数计算问题，相对而言计算复杂度不是很大，利用随机特性进行优化或避免什么；用第二类处理判定问题，相对计算复杂度很大，用随机特性进行降阶。

**2.2 One-Sided Error Monte Carlo算法**
**定义：**考虑决策问题。令$L$为一个语言，$A$为随机算法。$A$为one-sided-error Monte Carlo Algorithm recognizing $L$ if

1. $\forall x\in L, Prob(A(x)=1)\geq \frac{1}{2}$
2. $\forall x\notin L, Prob(A(x)=0)=1$

**2.3 Two-Sided Error Monte Carlo算法**
**定义：**令$F$为问题，称随机算法$A$为two-sided-error Monte Carlo Algorithm computing $F$ if $\exists \epsilon, 0<\epsilon <\frac{1}{2}$，使得对于$F$的每一个输入$x$
$$
Prob(A(x)=F(x))\geq \frac{1}{2}+\epsilon
$$
**2.4 Unbounded-Error Monte Carlo算法**
**定义：**这些随机算法统称为蒙特卡洛算法 Monte Carlo algorithms。令$F$为问题，称随机算法$A$为Unbounded-error Monte Carlo Algorithm computing $F$ if $F$的每个输入$x$
$$
Prob(A(x)=F(x))>\frac{1}{2}
$$
**2.5 Randomized Optimization Algorithms**

### 3 随机算法设计范式

1. Foling an adversary.
2. Abundance of witnesses.
3. Fingerprinting.
4. Random sampling.
5. Relaxation and random rounding.


***

## 4-12 启发式算法 Heuristics

### 1 模拟退火算法

+ the set of system states - the set of feasible solutions
+ the energy of a state - the cost of a feasible solution
+ perturbation mechanism - a random choice from the neighborhood
+ an optimal state - an optimal feasible solution
+ tenperature - a control parameter

**Simulated Annealing for $U$ with respect to Neigh**
**SA(Neigh)**

+ Input: An input instance $x\in L_I$
+ Step 1: Compute or select (randomly) an initial feasible solution $\alpha \in \mathcal{M}(x)$.
  Select an initial temperature (control parameter) T.
  Select a temperature reduction function $f$ as a function of two parameters $T$ and time.
+ Step 2: $I:=0$;
  **While** $T>0$ (or $T$ is not too close to 0) **do**
          **begin** randomly select a $\beta \in Neigh_x(\alpha)$;
                  **if** $cost(\beta) \leq cost(\alpha)$ then $\alpha:=\beta$
                  **else begin** 
                          generate a random number $r$ uniformly in the range $(0,1)$;
                          **if** $r < e^{-\frac{cost(\beta)-cost(\alpha)}{T}}$
                          **then** $\alpha := \beta$
                   **end**
                   $I:=I+1$
                   $T:=f(T,I)$
          **end**
+ Step 3: **output**($\alpha$)

**1.1 初始温度选取**
方法一：选择一个足够大的$T$(例如两个相邻解代价的最大差值)
方法二：随机选择$T$，增大$T$直到几乎总是选取某个相邻解$\beta$

**1.2 温度下降函数的选取**
$T:=r\cdot T~~(0.8\leq r \leq 0.99)$和$T_k:=\frac{T}{\log_2(k+2)}$很常用

**1.3 终止条件**
方法一：$\alpha$一段时间内不再改变
方法二：$term\leq \frac{\epsilon}{(\ln[\mathcal{M(x)}]-1)/p}$

**1.4 实践经验：**

+ 模拟退火算法可通过巨大的计算代价换取高质量的可行解
+ 输出的质量未必与初始点相关
+ 模拟退火最重要的参数是温度下降函数的下降率和对相邻解的选择
+ 平均复杂度接近最坏情况复杂度
+ 在相邻解的选取相同的情况下，模拟退火的效果远好于local search和multi-start local search

### 2 遗传算法

+ an individual - a feasible solution
+ a gene - an item of the solution representation
+ fitness value - cost function
+ population - a subset of the set of feasible solutions
+ mmutation - a random local transformation

**Genetic Algorithm Scheme (GAS)**

+ Input: An instance $x$ of an optimization problem $U=(\Sigma_I, \Sigma_O, L, L_I, \mathcal{M}, cost, goal)$
+ Step 1: Create (possibly randomly) an initial population $P=\{\alpha_1,\cdots,\alpha_k\}$  of size $k$
  $t:=0$ (the number of created populations)
+ Step 2: Compute $fitness(\alpha_i)$ for $i=1,\cdots,k$ (may be  $cost(\alpha_i)$)
  Use $fitness(\alpha_i)$ to estimate a probability distribution $Prob_P$ on $P$ in such a way that feasible solutions with high fitnesses get assigned higer probabiliries.
+ Step 3: Use $Prob_P$ to randomly choose $k/2$ pairs of feasible solutions $(\beta_1^1, \beta_1^2),\cdots,(\beta_{k/2}^1, \beta_{k/2}^2)$. Use the crossover operation on every pair of parents $(\beta_i^1, \beta_i^2)$ for $i=1,\cdots,k/2$ to create new individuals, and put them into $P$.
+ Step 4: Apply randomly the mutation operation to each individual of $P$
+ Step 5: Compute the fitness $fitness(\gamma)$ of all individuals $\gamma$ in $P$ and use it to choose $P'\subseteq P$ of cardinality $k$.
  Possibly improve every individual of $P'$ by local search with respect to a neighborhood.
+ Step 6: $t:=t+1$
  $P:=P'$
  **if** the stop criterion is not fulfilled **goto** Step 2
  **else** give the best individuals of $P$ as the output

**Adjustment of Free Parameters**

+ population size
+ selection of the initial population
+ fitness estimation and selection mechanism for parents
+ representation of individuals and the crossover operation
+ probability of mutation
+ selection mechanism for a new population
+ stop criterion

***

## 4-14 串匹配 String Matching

串匹配String Matching是在$T[1\cdots n]$中寻找模式$P(1\cdots m),m\leq n$，$P,T$中的字符从一个有限字符表$\Sigma$中选取。若$P[1..m]=T[s+1..s+m]$，称$s$是一个合法的shift. 串匹配问题的目标是寻找到所有的合法shift。

**Notation:**
$\Sigma^*:$ 从$\Sigma$中选取字符组成的所有有限长字符串
$\epsilon:$空串
$|x|:$ x的长度
$xy:$ 连接字符串$x,y$
$w\sqsubset x$: w是x的前缀，即$\exists y\in \Sigma^*,x=wy$
$w\sqsupset x$: w是x的后缀，即$\exists y\in \Sigma^*, x = yw$
$P_m:$ $P[1..m]$

### 1 朴素的串匹配算法

```pseudocode
NAIVE-STRING-MATCHER(T, P)
    n = T.length
    m = P.length
    for s = 0 to n-m
        if P[1..m] == T[s+1..s+m]
            print "Pattern occurs with shift s"
```

预处理复杂度: 0
匹配复杂度: $O((n-m+1)m)$

### 2 Rabin-Karp算法

**算法思想：**朴素串匹配算法效率低是因为每次重新扫描T中$s+1$开始的$m$个字符时，它就忘记了上一次从$s$开始扫描得到的信息，实际上这两个字符串相似度相当高。提高了信息的利用率后，Rabin-Karp算法实际效率高很多。
对于每个串匹配问题，首先根据$\Sigma$选择一个基数$d$，这样就可以就算出：
$p=P[m]+d(P[m-1]+d(P[m-2]+cdots+d(P[2]+dP[1])\cdots))$
同理，可以计算出$t_0$。计算$t_{s+1}$时，可以用以下递推式：
$t_{s+1} = d(t_s-d^{m-1}T[s+1])+T[s+m+1]$
可以发现，$P[1..m]=T[s+1..s+m]$即意味着$p=t_s$。这种做法的问题在于，当$d$或$m$很大时，$p$和$t_s$会是相当大的数字。为了解决这一问题，可以对$p$和$t_s$模$q$。这样上述递推式变为：
$t_{s+1} = (d(t_s-T[s+1]h)+T[s+m+1])\mod{q},~~h\equiv d^{m-1}(\mod q)$
引入模运算导致$t_{s+1}\equiv p(\mod q)$时串实际上不匹配。这时检查是否匹配即可。

```pseudocode
RABIN-KARP-MATCHER(T, P, d, q)
	n = T.length
	m = P.length
	h = d**(m-1) / q
	p = 0
	t[0] = 0
	for i = 1 to m		               // preprocessing
		p = (dp + P[i]) mod q
		t[0] = (dt[0] + T[i]) mod q
	for s = 0 to n-m                   // matching
		if p == t[s]
			if P[1..m] == T[s+1..s+m]
				print "Pattern occurs with shift s"
		if s < n - m
			t[s+1] = (d * (t[s] - T[s+1]*h) + T[s+m-1]) mod q
```

预处理复杂度: $O(m)$
匹配复杂度: $O((n-m+1)m)$

注意匹配复杂度是最坏情况下的。实际上，通常核查$t_s=p$的时间开销是$O(m(v+n/q))$，其中$v$为合法匹配数。如果$v=O(1)$且$n\leq q$，则核查匹配开销为$O(m)$，又因为$m\leq n$，匹配的复杂度为$O(n)$.

### 3 Finite automaton算法

Finite automation算法将**有限状态机**的思想引入串匹配：

+ $Q$: 状态机集合
+ $q_0$: 初始状态
+ $A\subseteq Q$: 接受状态
+ $\Sigma:$ 输入字符表
+ $\delta:Q\times \Sigma\rightarrow Q$ 状态转移函数

对于串匹配状态机可作如下定义：

+ $Q=\{0,1,\cdots,m\}$，初始状态是$0$，状态$m$是唯一的接受状态
+ 定义$\sigma(x)$为$x$与$P$共有的最长后缀$P_k$的长度$k$，则对任意状态$q$和字符$a$有$\delta(q,a)=\sigma(P_qa)$

```pseudocode
FINITE-AUTOMATON-MATCHER(T, delta, m)
	n = T.length
	q = 0
	for i = 1 to n
		q = delta(q, T[i])
		if q == m
			print "Pattern occurs with shift i-m"

COMPUTE-TRANSITION-FUNCTION(P, Sigma)
	m = P.length
	for q = 0 to m
		for each character a in Sigma
			k = min(m + 1, q + 2)
			repeat
				k = k - 1
			until (P_k) is prefix of (P_q)a
			delta(q, a) = k
	return delta
```

COMPUTE-TRANSITION-FUNCTION进行预处理，其复杂度为$O(m^3|\Sigma|)$，实际上有$O(m|\Sigma|)$的算法能找到$\delta$状态转移函数.
FINITE-AUTOMATON-MATCHER进行匹配，复杂度为$O(n)$.

### 4 Knuth-Morris-Pratt算法

KMP算法计算出数组$\pi$从而无需在预处理阶段完整计算出$\delta$函数。匹配过程中可以通过$\pi[q]$计算出$\delta(q,a)$。$\pi$的定义如下：
$$
\pi[q]=\max\{k:k<q\land P_k \sqsupset P_q \}
$$
KMP的整体思路是在预处理阶段计算出$\pi$，然后进行匹配。这两个算法很相似，预处理是将P与自身比较，匹配是将P与T比较。外层循环内的while和if语句和$\delta$的作用类似。

```pseudocode
KMP-MATCHER(T, P)
	n = T.length
	m = P.length
	Pi = COMPUTE-PREFIX-FUNCTION(P)
	q = 0
	for i = 1 to n
		while q > 0 and P[q + 1] != T[i]
			q = Pi[q]
		if P[q + 1] == T[i]
			q = q + 1
		if q == m
			print "Pattern occurs without shift i-m"
			q = Pi[q]

COMPUTE-PREFIX-FUNCTION(P)
	m = P.length
	let Pi[1..m] be a new array
	Pi[1] = 0
	k = 0
	for q = 2 to m
		while k > 0 and P[k + 1] != P[q]
			k = Pi[k]
		if P[k + 1] == P[q]
			k = k + 1
		Pi[q] = k
	return Pi[1..m]
```

通过摊还分析可以证明预处理的复杂度为$\Theta(m)$，匹配的复杂度为$\Theta(n)$.

***



