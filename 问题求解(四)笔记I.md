# Problem Solving IV (PART I)

Author: Daniel Liu
Contact me:  191240030@smail.nju.edu.cn
Course: Problem Solving IV. NJU

## 4-1 群论初步 Groups

### 1 群的定义与基本性质

在**集合**上定义**二元运算**，若运算具有**封闭性**，则形成**代数系统**。运算的内容及其性质决定了代数系统的结构。**群**是一种公理化定义的代数系统。

**群的定义：**在集合G上定义二元运算，任意$(a,b)\in G\times G$对应唯一的$a\circ b\in G$(封闭性)。满足下述公理的集合G和运算$(a,b)\mapsto a\circ b$形成一个群$(G,\circ)$:

+ 结合律：$(a\circ b)\circ c=a\circ (b\circ c)$ for $a,b,c\in G$.
+ 单位元 identity element：$\exist e\in G, \forall a \in G, e\circ a=a\circ e = a$.
+ 逆元 inverse element：$\forall a \in G,\exist a^{-1}\in G,a\circ a^{-1}=a^{-1}\circ a=e$.

若群G满足交换律$\forall a, b\in G,a\circ b = b\circ a$，称G为**交换群**或**阿贝尔群**(abelian)。若群中包含有限个元素，称其为有限的，反之称为无限的。

**群的基本性质：**

1. 群G的单位元$e$唯一

2. $g\in G$，则$g^{-1}$唯一(逆元唯一)

3. $a,b\in G\rightarrow (ab)^{-1}=b^{-1}a^{-1}$

4. $\forall a\in G\rightarrow (a^{-1})^{-1}=a$

5. $\forall a, b\in G$，方程$ax=b$和$xa=b$有唯一解
   (**群的第二定义：**代数系统$(G,\circ)$满足结合律且形如$ax=b$和$xa=b$的方程均具有唯一解)

6. $a,b,c\in G, ba=ca\rightarrow b=c \land ab = ac\rightarrow b=c$(**消去律**)

7. 群中**幂运算**法则成立，$\forall g,h\in G$，

   1. $\forall m,n\in \mathbb{Z}, g^mg^n=g^{m+n}$
   2. $\forall m, n\in \mathbb{Z}, (g^m)^m=g^{mn}$
   3. $\forall n\in\mathbb{Z},(gh)^n=(h^{-1}g^{-1})^{-n}$. 若G是阿贝尔群，则$(gh)^n=g^nh^n$

   (若群是$\mathbb{Z}$或$\mathbb{Z}_n$，我们通常用乘法$ng$表示幂运算，加法$g+h$表示群运算)

$U(n)$包含$[1,n-1]$中所有与n互质的整数，称为**group of units** of $\mathbb{Z}_n$.

**整数等价类(模运算)**

整数的模运算可以形成一个代数系统。设$\mathbb{Z}_n$为整数模n形成的等价类集合，$a, b, c\in \mathbb{Z}_n$:

+ 加法、乘法满足交换律
+ 加法、乘法满足结合律
+ 加法和乘法都有单位元(1, 0)
+ 满足乘法分配律: $a(b+c)=ab+ac$   (mod n)
+ 每个整数$a$有一个加法逆元$-a$
+ $a$是非零整数，则gcd($a,n$)=1$\leftrightarrow a$有一个乘法逆元$b$，$ab\equiv 1$  (mod n)

### 2 子群

**子群的定义：**H是群G的子集，若H在群运算下也是一个群，则H是G的子群.

任意群G有至少两个子群，即$\{e\}$和G本身。

**子群的定理：**

1. 群G的子集H是子群 $\leftrightarrow$ H满足如下条件：
   1. G的单位元$e$在H中
   2. $h_1, h_2\in H \rightarrow h_1h_2\in H$ (运算**封闭性**)
   3. $h\in H\rightarrow h^{-1}\in H$
2. 群G的子集H是子群 $\leftrightarrow$ H$\neq\emptyset$，且$\forall g,h\in H\rightarrow gh^{-1}\in H$
3. 群G的**有限**子集H是子群 $\leftrightarrow$ $\forall g,h\in H, gh\in H$

### 3 循环群

**基本定理：**$a$是群G的任意元素，则$\langle a\rangle=\{a^k:k\in\mathbb{Z}\}$是G包含$a$的最小子群
(使用"+"运算则为$\langle a\rangle=\{na:n\in\mathbb{Z}\}$)

**循环群的定义：**对于$a\in G$，称$\langle a \rangle$为$a$生成的**循环子群**。若G包含元素$a$且G$=\langle a \rangle$，则G是**循环群**，$a$为G的生成元。定义$a$的**度数**为使得$a^n=e$的最小正整数$n$，写作$|a|=n$，若不存在这样的整数$n$，则称$a$的度数无穷大，即$|a|=\infty$.

**循环群的定理：**

1. 任意循环群都是阿贝尔群

2. 循环群的任意子群都是循环群

   **推论：**$\mathbb{Z}$的子群是$n\mathbb{Z}$，n=0,1,2...

3. G是度数为n的循环群且$a$是G的生成元，则$a^k=e\leftrightarrow n$能整除$k$

4. G是度数为n的循环群且$a$是G的生成元，则$b=a^k$的度数是$n/$gcd($k,n$)

   **推论：**$\mathbb{Z}_n$的生成元是满足$1\leq r<n$且gcd($r,n$)=1的整数$r$
   
5. n阶循环群G的生成元的个数恰好等于不大于n的与n互质的正整数的个数

**循环群应用1：乘法运算下的复数群**

**基本概念：**$z=a+bi$为直角坐标表示，$z=r(cos\theta + i sin\theta)=rcis\theta$为极坐标表示。$\mathbb{C}^*$是定义了乘法运算的非零复数的群，不同于$\mathbb{Q}^*$和$\mathbb{R}^*$，它有很多具有特殊性质的子群，例如**circle group**：$\mathbb{T}=\{z\in \mathbb{C}:|z|=1\}$.

**极坐标表示下的定理：**

1. $z=r\,cis\theta$和$w=s\,cis\phi$是非零复数，则$zw=rs\,cis(\theta+\phi)$
2. **DeMoivre定理：**$z=r\,cis\theta$是非零复数，则$[r\,cis\theta]^n=r^ncis(n\theta)$
3. circle group$\mathbb{T}$是$\mathbb{C}^*$的子群

**定义：**称满足方程$z^n=1$的复数为the n***th roots of unity***，称其生成元为**primitive**

**nth roots of unity的定理：**

1. 若$z^n=1$，则the n*th* roots of unity是$z=cis(\frac{2k\pi}{n}),k=0,1,...,n-1$。
   此外，the n*th* roots of unity是$\mathbb{T}$的子群.

**循环群应用2：模系统下计算高次幂**

**方法：**通过平方可以快速计算$a^{2^0},a^{2^1},\cdots,a^{2^k}(mod\,n)$

***

## 4-2 置换群和拉格朗日定理 Permutation Groups and Lagrange's Theorem

### 1 置换群

**定义：**通常集合$X$的**置换 permutation**可以形成一个群$S_X$。如果$X$是有限集合，可以假设$X=\{1,2,\cdots,n\}$，将群记作$S_n$。$S_n$被称为**对称群 symmetric group**，它有$n!$个元素，二元操作是**映射的复合**。$S_n$的子群为**置换群 permutation group**.

注：由于映射的运算顺序从右到左，$(\sigma\circ\tau)(x)$理解为$\sigma(\tau(x))$。另外，不是所有置换复合都满足交换律。

**置换的逆：**$((a_1,a_2,\cdots,a_n))^{-1}=(a_n,a_{n-1},\cdots,a_1)$

**1.1 轮换表示 Cycle Notation：**
如果存在$a_1,a_2,\cdots,a_k\in X$，$\sigma(a_1)=a_2,\sigma(a_2)=a_3,\cdots,\sigma(a_k)=a_1$，对于其余的$x\in X$有$\sigma(x)=x$，称$\sigma\in S_X$是长度为$k$的轮换。可以用$(a_1,a_2,\cdots,a_k)$ 表示$\sigma$。很容易计算出轮换的**乘积**，例如$(1352)(256)=(1356)$。若两个轮换$\sigma=\{a_1,\cdots,a_k\}$和$\tau=\{b_1,\cdots,b_l\}$有$\forall i,j, a_i\neq b_j$，我们称$\sigma$和$\tau$**不相交 disjoint**。

**轮换的定理：**

1. $\sigma$和$\tau$是$S_X$中两个**不相交**的轮换，则$\sigma\tau=\tau\sigma$
2. $S_n$中任意**置换**都可表示为不相交的**轮换的乘积**

**1.2 对换 Transposition：**
将长度为2的轮换称为**对换 transposition**。对换乘积通常有很多种等价的表示形式。可以发现$(a_1,a_2,\cdots,a_n)=(a_1,a_n)(a_1,a_{n-1})\cdots(a_1,a_2)$.

**对换的定理：**

1. 在元素数至少为2的有限集合中，任意置换可以表示为**对换的乘积**

   **引理：**如果单位元表示为$r$个对换的乘积，$(1)=\tau_1\cdots\tau_r$，则$r$为偶数

2. 如果置换$\sigma$可表示为偶数个对换的乘积，则$\sigma$其他的对换表示形式也一定是偶数个对换的乘积。对奇数个也是同理。

   PS: 根据置换可表示为奇数个或偶数个对换，可定义**置换的奇偶**。

**1.3 交错群 Alternating Groups：**
置换群$S_n$中全部**偶置换**组成的子群称为**交错群 Alternating Groups**，记作$A_n$。

**交错群的定理：**

1. 置换群$S_n$中奇置换和偶置换数量相同，因此$A_n$度数为$\frac{n!}{2}$.

**1.4 二面体群 Dihedral Groups：**
**二面体群**是置换群的一个子群，是正$n$边形上刚性变换组成的群，记作$D_n$。

**二面体群的定理：**

1. 二面体群$D_n$是置换群$S_n$的度数为$2n$的一个子群
2. $n\geq 3$时，二面体群$D_n$由所有满足下列条件的两元素$r,s$的乘积组成：
   + $r^n=1$
   + $s^2=1$
   + $srs=r^{-1}$

**PS：正方体**刚性变换构成的群就是$S_4$，包含24个元素

### 2 拉格朗日定理

**2.1 陪集**
**陪集的定义：**$H$是群$G$的子群，定义H的代表为$g\in G$的**左陪集 left coset**为集合$gH=\{gh:h\in H\}$，**右陪集**可以定义为$Hg=\{hg:h\in H\}$。

左陪集和右陪集不一定相等。在整数集上陪集可用'+'的形式书写。

**引理：**H是G的子群，假设$g_1,g_2\in G$，下述条件**等价**：

1. $g_1H=g_2H$
2. $Hg_1^{-1}=Hg_2^{-1}$
3. $g_1H\subset g_2H$
4. $g_2\in g_1 H$
5. $g_1^{-1}g_2\in H$

**陪集的定理：**

1. H是G的子群，则H的左陪集(右陪集)是对$G$的一个划分partition
2. H是G的子群，则H在G中左陪集和右陪集的数量相等
   **定义：**H左陪集的数量称为H在G中的**index**，记作$[G:H]$.

**2.2 拉格朗日定理**
需要首先明确的是，H中**元素的数量**和其左陪集、右陪集相等。另外，

**拉格朗日定理：**G是有限群，H是G的子群。$|G|/|H|=[G:H]$是H在G中不同左陪集的个数。特别地，子群H中元素的个数一定能整除G中元素的个数。

**拉格朗日定理的引理：**

1. G是有限群且$g\in G$，则$g$的度数一定能整除G中元素个数
2. |G|=p，p是素数，则G是循环的，且G中除单位元外都是生成元
3. H和K都是有限群G的子群，且$G\supset H \supset K$，则$[G:K]=[G:H][H:K]$

要注意的是，拉格朗日定理的逆命题是错误的。根据拉格朗日定理可以推断$A_4$可能有度数为1, 2, 3, 4, 6的子群，但是实际上$A_4$不存在度数为6的子群。

**PS: 定理：**$S_n$中两个轮换$\tau$和$\mu$有相同长度$\leftrightarrow$$\exists \sigma\in S_n,\mu=\sigma \tau \sigma^{-1}$

**2.3 费马定理与欧拉定理**
**欧拉函数** Euler $\phi$-function:  $\phi:\mathbb{N}\rightarrow \mathbb{N}$。$n=1$时$\phi(n)=1$，$n>1$时$\phi(n)$为满足$1\leq m < n$和gcd($m,n$)=1的正整数$m$的个数。

**定理：**$U(n)$是$\mathbb{Z}_n$中的group of units，则$|U(n)|=\phi(n)$

**欧拉定理：**$a$和$n$是满足$n>0$且gcd($a,n$)=1的整数，则$a^{\phi(n)}\equiv 1$ (mod $n$)

**费马小定理：**$p$是质数且$p$不能整除$a$，则$a^{p-1}\equiv 1$ (mod $p$)。另外，任意整数$b$有$b^p\equiv b$ (mod $p$).

***

## 4-3 群同态基本定理与正规子群 Homomorphisms

### 1 同构

**同构定义：**如果两个群$(G,\cdot),(H,\circ)$间存在满射$\phi:G\rightarrow H$使得群运算得到维持，即$\forall a,b\in G,\phi(a\cdot b)=\phi(a)\circ \phi(b)$，称这两个群**同构 isomorphic**，记作$G\cong H$。映射$\phi$是**同构函数 isomorphism**.

同构可在任意代数系统上讨论。在某个结构上成立的命题，在与之同构的结构上也会成立，因而深入研究某系统即可掌握与之同构的所有系统。

**同构的定理：**

1. $\phi: G\rightarrow H$是两个群之间的同构函数，则以下陈述均正确：

   1. $\phi^{-1}:H\rightarrow G$是同构函数
   2. $|G|=|H|$
   3. $G$是阿贝尔群$\rightarrow$$H$是阿贝尔群
   4. $G$是循环群$\rightarrow$$H$是循环群
   5. $G$有度数为$n$的子群$\rightarrow$$H$有度数为$n$的子群

2. 所有**无穷**的**循环群**同构于$\mathbb{Z}$

3. 若G是度数为$n$的**循环群**，则G同构于$\mathbb{Z}_n$

   **引理：**若G的度数$p$是**素数**，则G同构于$\mathbb{Z}_p$

4. 群的同构决定了全部群上的等价关系

5. **(Cayley.)**任意群都同构于某个置换构成的群

   (证明要点：定义$\lambda_g:G\rightarrow G,\lambda_g(a)=ga$，证明G和$\overline{G}=\{\lambda_g:g\in G\}$间存在双射$\phi: g\rightarrow \lambda_g$)

### 2 直积

**2.1 外直积**
**外直积的定义：**$(G,\cdot)$和$(H,\circ)$的笛卡尔积是$G\times H$，令$(g,h)\in G\times H$，在$G\times H$上定义二元运算：$(g_1,h_1)(g_2,h_2)=(g_1\cdot g_2, h_1\circ h_2)$。$G\times H$加上二元运算得到的是一个群，$G\times H$是G和H的**外直积 external direct product.** 同理，可以定义$\prod_{i=1}^n G_i=G_1\times G_2\times \cdots \times G_n$为$G_1，G_2， \cdots，G_n$的外直积，或后者全部相同可以将外直积写作$G^n$.

**外直积的定理：**

1. $(g,h)\in G\times H$，若$g,h$度数有限且分别为$r,s$，则$(g,h)$在$G\times H$中的度数为$r,s$的最小公倍数$\text{lcm}(r,s)$.
   **引理：**$(g_1,\cdots,g_n)\in \prod G_i$，若$g_i$度数有限且在$G_i$中为$r_i$，则$(g_1,\cdots,g_n)$在$\prod G_i$中度数为$\text{lcm}(r_1,\cdots,r_n)$
2. $\mathbb{Z}_m\times \mathbb{Z}_n$同构于$\mathbb{Z}_{mn}\leftrightarrow$ gcd($m,n$)$=1$
   **引理1：**$\prod_{i=1}^k\mathbb{Z}_{n_i}\cong \mathbb{Z}_{n_1\cdots n_k}\leftrightarrow\text{gcd}(n_i,n_j)=1\text{ for }i\neq j$
   **引理2：**$m=p_1^{e_1}\cdots p_k^{e_k}$，其中$p_i$是不同的质数$\rightarrow \mathbb{Z}_m\cong \mathbb{Z}_{p_1^{e_1}}\times\cdots\times \mathbb{Z}_{p_k^{e_k}}$
   (所有有限阿贝尔群同构于引理2中的直积)

**2.2 内直积**
**内直积的定义：**若群G的子群H, K满足下列条件：

+ $G=HK=\{hk:h\in H, k\in K\}$
+ $H\cap K=\{e\}$
+ $hk=kh$ for all $k\in K$ and $h\in H$

则G是H和K的**内直积 internal direct product.**
类似地，可将内直积的概念拓展到多个子群$H_1,\cdots,H_n$上.

**内直积的定理：**

1. G是子群H和K的内直积，则$G\cong H\times K$
2. G是子群$H_i$的内直积，$i=1,2,\cdots,n$，则$G\cong \prod_i H_i$

### 3 正规子群与商群

**3.1 正规子群 Normal Subgroups：**
**定义：**H是G的子群，若$\forall g\in G, gH=Hg$，则H是G的**正规子群**.

**正规子群的定理：**若N是G的子群，则下列陈述等价：

1. N是G的正规子群
2. $\forall g\in G, gNg^{-1}\subset N$
3. $\forall g\in G, gNg^{-1}=N$

**3.2 商群 Factor(/Quotient) Groups：**
**定义：**N是G的**正规子群**，则N在G中的**陪集**会形成一个群G/N，满足运算$(aN)(bN)=abN$。这个由陪集组成的群G/N即为**商群**，其阶数为$[G:N]$。

子群的陪集可以构成群的划分，但是不能保证陪集形成一个群。如果是正规子群，就可以定义陪集上的操作，进而形成陪集上的一个群：单位元就是子群N本身，$aN$的逆元是$a^{-1}N$.

**3.3 simple groups**
**定义：**如果一个群除了$\{e\}$没有**真正规子群**，则它是**simple group**。
**定理：**$n\geq5$时，交错群$A_n$是simple group.

### 4 同态

**4.1 群同态 Group Homomorphisms：**
**同态定义：**群$(G,\cdot)$和$(H,\circ)$间存在映射$\phi:G\rightarrow H$使得$\phi(g_1\cdot g_2)=\phi(g_1)\circ \phi(g_2)$，$g_1,g_2\in G$。定义这种映射为**同态**，$\phi$在H中的范围称为$\phi$的**同态像 homomorphic image**.

**Kernel定义：**$\phi^{-1}(\{e\})$是G的**正规子群**，称之为$\phi$的kernel，记作$\text{ker }\phi$.

**同态的定理：**

1. 令$\phi:G_1\rightarrow G_2$为群的同态，则：
   1. 如果$e$是$G_1$的单位元，则$\phi(e)$是$G_2$的单位元
   2. $\forall g\in G_1,\phi(g^{-1})=[\phi(g)]^{-1}$
   3. $H_1$是$G_1$的子群$\rightarrow \phi(H_1)$是$G_2$的子群
   4. $H_2$是$G_2$的子群$\rightarrow \phi^{-1}(H_2)=\{g\in G_1:\phi(g)\in H_2 \}$是$G_1$的子群.
      此外，如果$H_2$是$G_2$的正规子群，则$\phi^{-1}(H_2)$是$G_1$的正规子群.
2. $\phi: G\rightarrow H$是群同态，则$\phi$的kernel是G的正规子群.

**4.2 同构定理：**
**定义：**H是G的正规子群，定义**标准同态 canonical homomorphism**为$\phi: G\rightarrow G/H$，且$\phi(g)=gH$.
(标准同态也称自然同态)

**第一同构定理：**如果$\psi: G\rightarrow H$是群同态，且$K=\text{ker }\psi$，则$K$是G的正规子群。令$\phi:G\rightarrow G/K$为标准同态，则存在唯一的同构$\eta:G/K\rightarrow \psi(G)$，其中$\psi=\eta \phi$.

**第二同构定理：**H是G的子群且N是G的正规子群，则HN是G的子群，H$\cap$N是H的正规子群，$H/H\cap N\cong HN/N$.

**Correspondence Theorem：**N是G的正规子群，则$H\mapsto H/N$是双射(其中H是G包含N的子群，H/N是G/N的子群)。另外，G包含N的正规子群与G/N一一对应.

**第三同构定理：**H和N是G的正规子群，且$N\subset H$，则$G/H\cong \frac{G/N}{H/N}$.

***

## 4-4 数论基础 Number Theory

### 1 数学归纳法

**第一数学归纳法：**$S(n)$是关于$n\in \N$的陈述，且假设对于某个$n_0$，$S(n_0)$是正确的。如果对于所有整数$k\geq n_0$，$S(k)$正确则$S(k+1)$正确，那么对于所有$n\geq n_0$有$S(n)$正确。

**第二数学归纳法：**$S(n)$是关于$n\in \N$的陈述，且假设对于某个$n_0$，$S(n_0)$是正确的。如果对于所有整数$k\geq n_0$，$S(n_0),S(n_0+1)\cdots S(k)$正确则$S(k+1)$正确，那么对于所有$n\geq n_0$有$S(n)$正确。

**良序原理：**自然数的任意非空子集都是良序的。

**引理：**数学归纳法可推导出1是最小的正自然数。
**定理：**数学归纳法可推导出良序原理，即$\N$的每个非空子集包含一个最小元素。

### 2 Division Algorithm

**2.1 Division Algorithm**
**Division Algorithm：**$a$和$b$是整数，且$b>0$，存在唯一的整数$q$和$r$使得$a=bq+r$，其中$0\leq r < b$。
注：这个定理是数论中最常用的定理之一

**定理：**若$a$和$b$是非零整数，则存在$r$和$s$使得$\gcd(a,b)=ar+bs$。此外，$a$和$b$的最大公因数唯一。(gcd(a,b)是a,b线性组合中最小的正整数)
**引理：**若$a$和$b$互质，则存在整数$r$和$s$使得$ar+bs=1$ (逆命题也正确)

**2.2 欧几里得算法**
**欧几里得算法 Euclidean Algorithm**可以求解$a$和$b$的最大公因数

```C++
int Euclid(int a, int b){
    if(b == 0)
        return a;
    else
        return Euclid(b, a % b);
}
```

**2.3 质数**
**定理：**

1. **引理:** $a$和$b$都是整数，且$p$是质数。如果$p|ab$，则$p|a$或$p|b$
2. 质数有无限个
3. **算数基本定理：**$n$为任意大于1的整数，则$n=p_1p_2\cdots p_k$，其中$p_1,\cdots,p_k$是素数(不一定互异).

### 3 逆元和最大公因数

**3.1 模n系统下的方程解和逆元**
**逆元定义：**若$\exists a',a'\cdot_n a = 1$，则称$a$在$\Z_n$中存在**乘法逆元**。

**引理：**若$a$在$\Z_n$中存在乘法逆元$a'$，则$\forall b \in \Z_n$，方程$a\cdot_nx=b$有唯一解：$x=a'\cdot_nb$.
**推论：**若$\exists b\in \Z_n$使得方程$a\cdot_nx=b$无解，则$a$在$\Z_n$中没有乘法逆元.
**定理：**若一个元素在$\Z_n$中有乘法逆元，则该逆元是唯一的.

**3.2 模运算下的方程转化为一般方程**
**引理：**方程$a\cdot_n x = 1$在$Z_n$中有解$\leftrightarrow$存在整数$x,y$使$ax+ny=1$.

**3.3 最大公因数**
**定理：**

1. $a$在$\Z_n$中存在乘法逆元$\leftrightarrow$存在整数$x,y$使$ax+ny=1$.
   **推论：**如果$a\in\Z_n$和整数$x,y$满足$ax+ny=1$，则$a$在$\Z_n$中的乘法逆元是$x$ mod $n$.
2. **引理：**给定$a$和$n$，如果存在整数$x,y$使$ax+ny=1$，则$\gcd(a,n)=1$，也即$a$和$n$互质.
3. **引理：**若$j,k,q,r$是正整数，$k=jq+r$，则$\gcd(j,k)=\gcd(r,j)$.

**拓展欧几里得算法：**给定两个整数$j,k$，该算法同时计算出$\gcd(j,k)$和满足$\gcd(j,k)=jx+ky$的两个整数$x,y$.

```c++
int Extended_Euclid(int a, int b, int &x, int &y){
    if(b == 0){
        x = 1;
        y = 0;
        return a;
    }
    int ret = Extended_Euclid(b, a % b, x, y);
    int tmp = x;
    x = y;
    y = tmp - a / b * y;
    return ret;
}
```

**3.4 计算乘法逆元**
**定理：**

1. **推论：**对任意整数$n$：$\Z_n$中的一个元素$a$有逆元$\leftrightarrow\gcd(a,n)=1$.
2. **推论：**对任意质数$p$，$\Z_p$的每个非零元素$a$有一个逆元.
3. 如果$\Z_n$中的元素$a$有逆元，则可通过拓展欧几里得算法计算出满足$ax+ny=1$的整数$x,y$，$a$在$\Z_n$中的逆元为$x$ mod $n$.

***

## 4-5 数论算法 Number-Theoretic Algorithms

### 1 数论基本概念

**整除相关定理：**

1. **Division Theorem：**$a$和$b$是整数，且$b>0$，存在唯一的整数$q$和$r$使得$a=bq+r$，其中$0\leq r < b$
2. 若$a,b$是两个非零整数，则$\gcd(a,b)$是$a,b$线性组合$\{ax+by|x,y\in\Z\}$中最小的元素
   **推论：**$\forall a,b\in \Z, d|a\land d|b \rightarrow d|\gcd(a,b)$
   **推论：**$\forall a, b \in \Z, n\in \N,\gcd(an,bn)=n\gcd(a,b)$
   **推论：**$\forall n,a,b\in \N, n|ab\land \gcd(a,n)=1\rightarrow n|b$

**质数相关定理：**

1. $\forall a,b,p\in \Z, \gcd(a,p)=1\land \gcd(b,p)=1\rightarrow \gcd(ab,p)=1$
2. $\forall p\in Primes, \forall a, b\in \Z, p|ab\rightarrow p|a \lor p|b$
3. 对任意合数$a$，仅存在一种写法$a=p_1^{e_1}p_2^{e_2}\cdots p_r^{e_r}$，其中$p_i$是素数，$e_i$是正整数，$p_1<p_2<\cdots <p_r$.

### 2 最大公因数

**引理：**$\forall a\in \N, b\in \N^*,\gcd(a,b)=\gcd(b,a\mod b)$

**欧几里得算法：**

```pseudocode
EUCLID(a, b)
	if b == 0
		return a
	else return EUCLID(b, a % b)
	
EXTENDED-EUCLID(a, b)
	if b == 0
		return (a, 1, 0)
	else (d1, x1, y1) = EXTENDED-EUCLID(b, a % b)
		(d, x, y) = (d1, y1, x1 - floor(a / b) * y1)
		return (d, x, y)
```

**Euclid复杂度相关定理：**

1. $a>b\geq 1$且EUCLID($a,b$)执行$k\geq 1$次递归调用，则$a\geq F_{k+2}$且$b\geq F_{k+1}$
2. $\forall k \geq 1,a > b \geq 1 \land b < F_{k+1}\rightarrow$ EUCLID($a,b$)递归调用次数小于$k$

### 3 模运算

**模运算规则：** 
$[a]_n + _n [b]_n = [a+b]_n$
$[a]_n \cdot_n [b]_n = [ab]_n$
$[a]_n -_n [b]_n = [a-b]_n$

**定义:** 模n加法群为$(\Z_n, +_n)$，$|\Z_n|=n$。模n乘法群为$(\Z_n^*,\cdot_n)$，其中$\Z_n^* = \{[a]_n\in\Z:\gcd(a,n)=1\}$，$|\Z_n|=\phi(n)$。这两个群都是有限阿贝尔群。

**欧拉函数：**
$$
\phi(n)=n\prod_{p:~p ~is~prime~\land~p|n}\left(1-\frac{1}{p} \right)
$$
欧拉函数$\phi(n)$表示小于等于$n$的整数中与$n$互质的正整数个数，其范围满足$\frac{n}{e^\gamma \ln\ln n+\frac{3}{\ln\ln n}}< \phi(n) \leq n-1$，其中$\gamma$是欧拉常数.

### 4 模运算下的线性方程组

对于线性方程组$ax \equiv b ~ ($mod $n)$，其中$a>0,n>0$，我们定义$\left<a\right> = \{a^{(x)} : x>0 \} = \{ ax \mod n : x > 0\}$

**定理：**

1. $\forall a,n\in \N^*,d=\gcd(a,n)\rightarrow \left< a \right> = \left< d \right> =\{ 0,d,2d,\cdots,((n/d)-1)d \}$ in $\Z_n$，因此，$|\left< a \right>| = n/d$.

   **推论：**方程$ax\equiv b$ (mod $n$)有解 $\leftrightarrow$ $d|b$，其中$d=\gcd(a,n)$
   **推论：**方程$ax\equiv b$ (mod $n$)要么在$\Z_n$有$d=\gcd(a,n)$个不同解，要么无解

2. 令$d=\gcd(a,n)$，$d=ax'+ny'$。若$d|b$，则方程$ax\equiv b$(mod $n$)其中一个解为$x_0 = x'(b/d)$ mod $n$

3. 若方程$ax\equiv b$ (mod $n$)有解，且$x_0$是其中任意一解，则方程有$d$个不同解：$x_i=x_0+i(n/d),~i=0,1,\cdots,d-1$

   **推论：**$\forall n>1,\gcd(a,n)=1\rightarrow ax\equiv b~(mod ~n)$在$\Z_n$下有唯一解

   **推论：**$\forall n>1, \gcd(a,n)=1\rightarrow$ $ax\equiv 1~(mod~n)$在$\Z_n$下​有唯一解; $\gcd(a,n)\neq 1\rightarrow ax\equiv 1~(mod~n)$无解.

**模运算方程算法：**

```pseudocode
MODULAR-LINEAR-EQUATION-SOLVER(a, b, n)
	(d, x, y) = EXTENDED-EUCLID(a, n)
	if d | b
		x0 = x(b/d) mod n
		for i = 0 to d - 1
			print(x0 + i(n/d)) mod n
	else print "no solutions"
```

### 5 中国剩余定理

**中国剩余定理：**$n=n_1n_2\cdots n_k$，其中$n_i$两两互质。考虑如下对应关系：$a\leftrightarrow (a_1,a_2,\cdots,a_k)$，其中$a\in\Z_n, a_i\in \Z_{n_i}$，且$a_i= a\mod n_i,~i=1,2\cdots k$，这个对应关系是$\Z_n$与$\Z_{n_1}\times \Z_{n_2}\times \cdots \times \Z_{n_k}$间的一一映射。在$\Z_n$上进行的运算等价于在$k-$元组上对每个元素分别运算，也即：
若$a\leftrightarrow (a_1,a_2,\cdots,a_k)$，$b\leftrightarrow (b_1,b_2,\cdots,b_k)$，
则 $(a+b)~mod~n~\leftrightarrow~ ((a_1+b_1)~mod ~ n_1,\cdots,(a_k+b_k)~mod~n_k),$
$(a-b)~mod~n~\leftrightarrow~ ((a_1-b_1)~mod ~ n_1,\cdots,(a_k-b_k)~mod~n_k),$
$(ab)~mod~n~\leftrightarrow~ ((a_1b_1)~mod ~ n_1,\cdots,(a_kb_k)~mod~n_k)$

**推论：**

1. 若$n_1,n_2,\cdots,n_k$两两互质且$n=n_1n_2\cdots n_k$，则$\forall a_1,a_2,\cdots, a_k\in \Z$，方程组$x\equiv a_i~(mod ~ n_i),~i=1,2,\cdots,k$在模$n$下有唯一解
2. 若$n_1,n_2,\cdots, n_k$两两互质且$n=n_1n_2\cdots n_k$，则$\forall x,a\in\Z$，$x\equiv a~(mod~n_i),i=1,2,\cdots,k\leftrightarrow x\equiv a~(mod ~ n)$

**计算方法：**$a\rightarrow (a_1,a_2,\cdots,a_k)$很简单，$(a_1,a_2,\cdots,a_k)\rightarrow a$计算方法如下：

1. 定义$m_i=n/n_i, ~i=1,2,\cdots,k$
2. 定义$c_i=m_i(m_i^{-1}~mod~n_i)$
3. $a\equiv (a_1c_1+a_2c_2+\cdots+a_kc_k)~(mod~n)$

### 6 幂运算取模

**幂运算定理：**

1. **欧拉定理：**$\forall n>1,\forall a\in\Z^*_n ,a^{\phi(n)}\equiv 1~(mod~n)$

2. **费马小定理：**$p$是质数，则$\forall a\in\Z^*_p, a^{p-1}\equiv 1~(mod~p)$

3. 使得$\Z_n^*$为循环群的大于1的整数$n$包括$2,4,p^e$和$2p^e$，其中$p$是素数，$e\in \N^*$

4. $g$是$\Z_n^*$的生成元，则方程$g^x\equiv g^y~(mod~n)$成立 $\leftrightarrow$ $x\equiv y~(mod~\phi(n))$

5. $p$是奇质数且$e\geq 1$，则方程$x^2\equiv 1~(mod ~ p^e)$仅有两个解: $x=1,-1$

   **推论：**若$x^2\equiv 1~(mod~n)$有$x=1,-1$以外的解，则$n$是合数

**快速幂算法：**

```pseudocode
MODULAR-EXPONENTIATION(a, b, n)
	d = 1
	let <bk, b(k-1), ..., b0> be the binary rep of b
	for i = k downto 0
		d = (d * d) mod n
		if bi == 1
			d = (d * a) mod n
	return d
```

若$a,b,n$长度为$\beta$，则复杂度为$O(\beta^3)$.

***

## 4-6 密码算法 Cryptography

### 1 私钥系统

私钥系统中仅有一个key，设为函数$f$及其逆函数$f^{-1}$，需要由发送者和接收者秘密保管。

+ monoalphabetic cryptosystem:
  1. shift code: $f(p)=p+a\mod 26,~f^{-1}(p)=p-a\mod 26$
  2. affine cryptosystem: $f(p)=ap+b\mod 26$, $f^{-1}(p)=a^{-1}p-a^{-1}b\mod 26$, 其中$\gcd(a,26)=1$
+ polyalphabetic cryptosystem: $f(p)=Ap+b,$ $f^{-1}(p)=A^{-1}p-A^{-1}b\mod 26$，其中$p,b$是列向量

### 2 公钥系统

**1.1 公钥系统**
**公钥系统 public-key cryptosystems (PKC)**中，每个人有**公钥**和**私钥**，满足$M=P(S(M))=S(P(M))$。Alice, Bob两人，其公钥、私钥记为$P_A, S_A, P_B, S_B$，用PKC有如下两种通讯方式：

**B->A 加密：**

1. B得到A的公钥$P_A$
2. B根据信息$M$计算出密文$C=P_A(M)$并发给A
3. A接收到密文$C$，通过私钥得到原信息：$S_A(C)=S_A(P_A(M))=m$

**A->B 数字签名 digital signature：**

1. A用私钥$S_A$为信息$M'$计算出数字签名$\sigma=S_A(M')$
2. A发送信息/数字签名对$(M',\sigma)$给B
3. B接收到$(M',\sigma)$，用A的公钥计算$P_A(\sigma)$，若$P_A(\sigma)=M$，则验证出$M'$确实发送自$A$，否则说明信息在传送过程中受损或是伪造信息

加密可以防止信息传输过程被窃听，数字签名验证了发送者的身份和信息的准确性。如果使用数字签名时，通过B的公钥$P_B$计算出$P(M')$，则可以同时确保信息的私密性和安全性。

**1.2 RSA公钥系统**
通过如下方式计算公钥和私钥：

1. 选取两个大素数$p,q~(p\neq q)$，每个数或许是1024 bit
2. 计算$n=pq$
3. 选取一个小的奇数$e$，满足$e$与$\phi(n)=(p-1)(q-1)$互质
4. 计算$e$在模$\phi(n)$系统下的乘法逆元$d$
5. RSA公钥为$(e,n)$
6. RSA私钥为$(d,n)$

加密方式为$P(M)=M^e\mod n,~S(C)=C^d\mod n$。如果$\lg e = O(1)$，公钥需要$O(1)$次模乘法和$O(\beta^2)$次位运算，私钥需要$O(\beta)$次模乘法和$O(\beta^3)$次位运算。
由于大数很难分解为两个质因子，破解RSA公钥系统的计算量是很不现实的。目前尚未证明大数分解为两个质因子很困难就会导致破解RSA很困难，但是多年来人们也没有发现一个很好的破解方法。

### 3 整数分解

Pollard's rho启发式算法：

```pseudocode
POLLARD-RHO(n)
	i = 1
	x1 = RAMDOM(0, n-1)
	y = x[1]
	k = 2
	while TRUE
		i = i + 1
		x[i] = (x[i-1] * x[i-1] - 1) mod n
		d = gcd(y - x[i], n)
		if d != 1 and d != n
			return d
		if i == k
			y = x[i]
			k = 2k
```

该算法的复杂度和正确性都无法完全保证，在实践中通常认为分解出一个素因子$p$的期望复杂度为$\Theta(\sqrt{p})$，并且内存开销很小。算法中$x_i = (x_{i-1}^2 - 1)\mod n$会生成一个$\rho$形的混沌环，通过生日悖论可以发现，该环长度的期望是$\sqrt{p}$。$x_i$和$y$像快慢指针在环上移动，直至找到一个素因子$d=\gcd(y-x_i, n)$。

***



