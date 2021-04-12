# Problem Solving IV (PART II)

Author: Daniel Liu
Contact me:  191240030@smail.nju.edu.cn
Course: Problem Solving IV. NJU

## 4-8 串匹配 String Matching

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





