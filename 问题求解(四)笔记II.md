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
算法$A$**决定 decide**语言$L：$$A$接受$L$且$\forall x\notin L, A(x)=0$

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








***

## 4-10 串匹配 String Matching

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





