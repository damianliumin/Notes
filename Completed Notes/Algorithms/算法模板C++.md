# 算法模板(C++)

Author: Daniel Liu
Contact me: 191240030@smail.nju.edu.cn

## Ch1 图论算法

### 1 最小生成树

#### 1-1 Kruskal算法

```C++
// 适合稀疏图（点多边少）
struct edge{
    int from, to, w;
}
edge edges[MAXM] = {0};

int find_set{
    ...
}

void union_set{
    ...
}

void kruskal(){
    sort(edges， edges + m, cmp);
    int ans = 0, t = 0;
    for(int i = 1 ;i <= n-1; ++i){
        while(find_set(edges[t].from) == find_set(edges[t].to))
            ++t;
        ans += edges[t].weight;
        union_set(edges[t].from, edges[t].to);
    }
}
```

#### 1-2 Prim算法

```c++
// 适合稠密图（点少边多）
bool vis[MAXN] = {0};
int low[MAXN]; // distance to the closest vertice in tree
vector<pair<int, int>> edge[MAXN]; // save edge and weight
int ans = 0;
void prim(){
    // initialize
    vis[1] = true;
    for(int i = 1 ;i <= n ;++i)
        low[i] = INF;
    low[1] = 0;
    for(auto &x: edge[1])
        low[x.first] = x.second;
    // solve
    for(int i = 1 ;i <= n-1 ;++i){
        int low_dis = INF, low_vet = 0;
        for(int j = 1 ;j <= n ;++j)
            if(!vis[j] && low[j] < low_dis){ // travese unvis
                low_dis = low[j], low_vet = j;
            }
        vis[los_vet] = true;
        ans += low_dis;
        for(auto x: edge[low_vet]) // maintain low[]
            low[x.first] = min(low[x.first], x.second);  
	}
}
```

### 2 图的遍历

#### 2-1 广度优先搜索 BFS

```c++
//record father and dis
bool vis[MAXN] = {0};
int father[MAXN] = {0};
int dis[MAXN] = {0};  // distance from s
vector<int> connect[MAXN];
void bfs(int s){
    queue<int> q;  // should be replaced by 			  	
    q.push(s);		// priority_queue in some cases
    father[s] = s;
    dis[s] = 0;
    while(!q.empty()){
     	int u = q.front();
        q.pop();
        for(auto &v: connect[u])
            if(!vis[v]){
                vis[v] = 1;
                dis[v] = dis[u] + 1;
            	father[v] = u;
                q.push(v);
            }
    }
}
```

#### 2-2 深度优先搜索 DFS

```c++
// record time stamp
int time = 0;
bool vis[MAXN];
int father[MAXN];
vector<int> connect[MAXN];
int time_d[MAXN], time_f[MAXN];
void dfs(){
    for(int i = 0 ; i < MAXN ;++i)
        if(!vis[i])
            dfs_vis(i);
}
void dfs_vis(int u){
    time_d[u] = time++;
    vis[u] = 1;
    for(auto &v: connect[u])
        if(!vis[v]){
            father[v] = u;
            dfs_visit(v);
        }
    time_f[u] = time++；
}
```

### 3 拓扑排序

```pseudocode
TOPOLOGICAL-SORT(G){
	DFS(G) // compute v.f (finish time) for each vertex
	// add a vertex to the front of list L when it finishes
	return L
}
```

### 4 有向图强连通分支 

#### 4-1 Tarjan算法

实际运行效果Tarjan算法优于Kosaraju算法.
一些无向图的题目也会需要用Tarjan来寻找特殊的连通分量（例如任意两点间存在经过不同edge的path），在无向图上用Tarjan才需要pre参数

```c++
int ts = 0， cnt = 0;
int dfn[MAXN] = {0};
int low[MAXN] = {0};
int belong[MAXN] = {0};
bool instack[MAXN] = {0};
vector<int> edges[MAXN];
stack<int> s;

void dfs(int pos [,int pre]) {   // 仅在无向图需要pre
    instack[pos] = true;
    s.emplace(pos);
    low[pos] = dfn[pos] = ++ts;
    for (auto &nex : edges[pos]) {
        if (dfn[nex] == 0) {  // not visited yet
            dfs(nex [, pos]);
            low[pos] = min(low[pos], low[nex]);
        } else if (instack[nex] [&& nex != pre])
            low[pos] = min(low[pos], dfn[nex]);
    }
    if (low[pos] == dfn[pos] && instack[pos]) {
        ++cnt;    // total strong connected components
        while (!s.empty() && s.top() != pos) {
            belong[s.top()] = cnt;
            instack[s.top()] = false;
            s.pop();
        }
        if (!s.empty()) s.pop();
        belong[pos] = cnt;
        instack[pos] = false;
    }
}
```

#### 4-2 Kosaraju算法

```pseudocode
// O(V+E)
STRONGLY-CONNECTED-COMPONENTS{
	DFS(G) // compute v.f
	compute G’ // invert every edge of G
	DFS(G’)  // main loop of DFS in decreasing order of v.f
	// each tree in DFS(G') is a strongly-connected-component
}
```

### 5 单源最短路径算法

#### 5-1 BELLMAN-FORD算法

Bellman-Ford算法可以检验出负权回路。

```c++
// O(VE)
int dis[MAXN] = {0};
struct edge{
	int from, to, w;
} edges[MAXN];
int n, m;

bool bellman_ford(int s){
    for(int i = 1 ;i <= n ;++i)
        dis[i] = INF;
    dis[s] = 0;
    for(int i = 1 ;i <= n ;++i)
        for(int j = 1 ;j <= m ;++j) {
            int u = edges[j].from, v = edges[j].to;
            dis[v] = min(dis[v], dis[u] + edges[j].w);
        }
    for(int j = 1 ;j <= m ;++j){
        int u = edges[j].from, v = edges[j].to;
        if(dis[u] + edges[j].w < dis[v])
            return false;
    }
    return true;
}
```

#### 5-2 DAG单源最短路径

```pseudocode
// O(V+E)
DAG-SHORTEST-PATH(G, s){
	topologically sort the vertices of G
	INITIALIZE-SINGLE-SOURCE(G, s)
	for each vertex u, taken in topologically sorted order
		for each vertex v in Adj[u]
			if distance[u] + weight(u,v) < distance[v]{
				distance[v] = distance[u] + weight(u,v)
				predecessor[v] = u
			}	
}
```

#### 5-3 Dijkstra算法

```C++
// O((V+E)logV) with priority queue
// 图必须是非负边权的
int n;
bool vis[MAXN] = {0};
int dis[MAXN] = {0};
vector<pair<int, int>> edges[MAXN];
void dijkstra(G, s){ // can add predecessor if necessary
	for(int i = 1 ;i <= n ;++i)		// init
		dis[i] = INF;
	dis[s] = 0;
	priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
	pq.emplace(0, s);
	while(!pq.empty()){
		int u = pq.top().second;
		pq.pop();
		if(vis[u]) continue;
		vis[u] = true;
		for(auto &x: edges[u]){
			int v = x.second, weight = x.first;
			if(dis[u] + weight < dis[v]){
				dis[v] = dis[u] + weight;
				pq.emplace(dis[v], v);
			}
		}
	}
}
```

### 6 多源最短路径算法

#### 6-1 Floyd算法

```c++
int dis[MAXN][MAXN]; // store edges before floyd
    
void floyd(){
  for(int k = 1 ;k <= n ;++k)
    for(int i = 1 ;i <= n ;++i)
      for(int j = 1 ;j <= n ;++j)
        dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j]);
}
```

可以用Floyd算法传递闭包：

```C++
int T[MAXN][MAXN];
// (i, j)exists or i == j   =>   T[i][j] = 1
// otherwise  =>  T[i][j] = 0

void floyd(){
    for(int k = 1 ;k <= n ;++k)
        for(int i = 1 ;i <= n ;++i)
            for(int j = 1 ;j <= n ;++j)
                T[i][j] = T[i][j] || (T[i][k] && T[k][j]);
}
```

### 7 匹配算法

#### 7-1 匈牙利算法-二部图最大匹配

```c++
//O(VE)
bool used[MAXN];
int belong[MAXN];

bool find(int x){
    for(auto& y : edges[x]){
        if(!used[y]){
            used[y] = true;
            if(belong[y]==0 || find(belong[y])){ 
                belong[y] = x;
                return true;
            }
        }
    }
    return false;
}

int hungarian(){
    int total = 0;
    for(int i = 1 ;i <= n ;++i){
        memset(used, 0, sizeof(used)); // clear used every time
        if(find(i))
            ++total;
    }
    return total;
}
```

### 8 网络流

#### 8-1 Dinic算法(邻接矩阵)

```c++
// O(V^2E)
int edge[MAXN][MAXN];
int depth[MAXN];
bool vis[MAXN];
int src, dst;

int dfs(int u, int lim){
    if(u == dst)
        return lim;
    vis[u] = true;
    for(int v = 1 ;v <= MAXN ;++v){ // careful with range
        int ret;
        if(!vis[v] && edge[u][v]>0 && depth[v] == depth[u]+1 
           && (ret = dfs(v, min(edge[u][v], lim)))){
            edge[u][v] -= ret;
            edge[v][u] += ret;
            return ret;
        }
    }
    return 0;
}

bool bfs(){
    memset(depth, -1, sizeof(depth));
    queue<int> q;
    q.push(src);
    depth[src] = 0;
    while(!q.empty()){
        int cur = q.front();
        q.pop();
        for(int i = 1 ;i <= MAXN ;++i)
            if(edge[cur][i] > 0 && depth[i] == -1){
                depth[i] = depth[cur] + 1;
                q.push(i);
            }
    }
    return depth[dst] != -1;
}

int dinic(){
    int flow = 0;
    while(bfs()){
        int ret;
        memset(vis, 0, sizeof(vis));
        while((ret = dfs(src, INF)) != 0) // caution INF here
        	flow += ret;
    }
    return flow;
}

// add edges:
// directed single edge: edge[u][v] = w;
// directed multiple edge: edge[u][v] += w;
// undirected single edge: edge[u][v] = edge[v][u] = w;
// undirected multiple edge: edge[u][v] += w;edge[v][u] += w;

```

#### 8-2 Dinic算法(vector)

```c++
int src, dst;
struct node{
    int to, cap, rev;
    node(int x, int y, int z):to(x), cap(y), rev(z){};
};
vector<node> edge[MAXN];
int depth[MAXN];
bool vis[MAXN];

int dfs(int u, int lim){
    if(u == dst)
        return lim;
    vis[u] = true;
    for(auto &v: edge[u]){
        int ret;
        if(!vis[v.to] && v.cap && depth[v.to] == depth[u] + 1          && (ret = dfs(v.to, min(v.cap, lim)))){
            v.cap -= ret;
            edge[v.to][v.rev].cap += ret;
            return ret;
        }
    }
    return 0;
}

bool bfs(){
    memset(depth, -1, sizeof(depth));
    queue<int> q;
    q.push(src);
    depth[src] = 0;
    while(!q.empty()){
    	int cur = q.front();
        q.pop();
        for(auto &v: edge[cur])
        	if(depth[v.to] == -1 && v.cap > 0){
                q.push(v.to);
                depth[v.to] = depth[cur] + 1;
            }    
    }
    return depth[dst] != -1;
}

int dinic(){
    int flow = 0;
    while(bfs()){
        int ret;
        memset(vis, 0, sizeof(vis));
        while((ret = dfs(src, INF)))
            flow += ret;
    }
    return flow;
}

// add edges (assume single directed edge)
// cin >> u >> v >> w;
// edge[u].emplace_back(v, w, edge[v].size());
// edge[v].emplace_back(u, w, edge[u].size() - 1);
```

***

## Ch2 数论算法

### 1 大数运算

#### 1-1 快速幂

```c++
// compute a ^ b (mod q)
// if q >= 2^32, use (__int128)
#define ll long long

ll qpow(ll a, ll b, ll q){
    ll ret = 1, rec = a;
    while(b != 0){
        if(b & 1)
            ret = ret * rec % q;
        rec = rec * rec % q;
        b /= 2;
    }
    return ret;
}
```

#### 1-2 快速乘

```c++
ll qmult(ll a, ll b, ll q){
    ll ret = 0, rec = a;
    while(b != 0){
        if(b & 1)
            ret = (ret + rec) % q;
        rec = (rec << 1) % q;
        b /= 2;
    }
    return ret;
}

// the following method depends on platform
inline ll qmult(ll a, ll b, ll q){
    return (__int128)a * (__int128)b % q;
}
```

### 2 欧几里得算法

#### 2-1 欧几里得算法

```c++
// 尾递归实现
int gcd(int a, int b){
    if(b == 0)
        return a;
    else
        return euclid(b, a % b);
}
// 循环实现
int gcd(int a, int b){
    while(b != 0){
        int tmp = a;
        a = b;
        b = tmp % b;
    }
    return a;
}
```

#### 2-2 拓展欧几里得算法

返回最大公因数的同时，得到$ax+by=d$的两个整数$x,y$.

```c++
ll exgcd(ll a, ll b, ll &x, ll &y){  // ax + by = d
    if(b == 0){
        x = 1, y = 0;
        return a;
    }
    ll d = exgcd(b, a % b, x, y);
    ll tmp = x;
    x = y;
    y = tmp - a / b * y;
    return d;
}
```

### 3 欧拉函数

欧拉函数$\varphi(n)$是$[1,n]$中与$n$互质的元素个数。本算法借助了质数打表的思想，并利用公式$\varphi(n)=n\prod_i(1-\frac{1}{p_i})$，其中$p_i$是$n$所有的质因子。通过本算法可快速计算出欧拉函数的数值表。

```c++
double phi[MAXN];

void Euler() {
    phi[1] = 1;
    for(int i = 2; i < MAXN; i++)
        if(!phi[i])       // i is a prime
            for(int j = i; j < MAXN; j += i){
                if(!phi[j])
                    phi[j] = j;
                phi[j] = (phi[j] / i) * (i-1);
            }
}
```

### 4 大质数判定

#### 4-1 Miller-Rabin算法

复杂度为$O(k\lg n)$，其中$k$为数组A中元素个数。费马素性检测通过费马定理，在$2^{32}$内有很高的正确率。米勒-拉宾素性检验在费马定理的基础上加以拓展，能在更大的范围内保证高正确率，模板中的A可以确保$x$在$2^{64}$内算法的正确性。如果$x$是奇素数，则$a^{d},a^{2d},\cdots,a^{2^rd}$要么全是1，要么中间某数为-1(后面肯定全1)。

```c++
# define ll long long

bool miller_rabin(ll x){
    if(x < 3) return x == 2;
    if(x % 2 == 0) return false;
    // miller rabin
    ll A[] = {2, 325, 9375, 28178, 
              450775, 9780504, 1795265022};
    ll d = x - 1, r = 0;
    while(d % 2 == 0) // a^(x-1) = a^(2^r*d) = 1 (mod x)
        d /= 2, ++r;
    for(auto a : A) {
        ll v = qpow(a, d, x);  // compute a^d
        if (v <= 1 || v == x - 1) // x|a or a^d=-1 continue
            continue;
        for(int i = 0; i < r; ++i) {
            v = (__int128)v * v % x;
            if(v == x - 1 && i != r - 1){
                v = 1;
                break;
            }
            if(v == 1)
                return false;
        }
        if(v != 1)
            return false;
    }
    return true;
}
```

### 5 模运算系统

#### 5-1 线性模方程算法

求解$ax\equiv b (\mod n)$.

```c++
int solve(int a, int b, int n){
    int x, y, d;
	d = exgcd(a, n, x, y);
    x = (x + b) % b;
	if(b % d == 0){
		int x0 = (x * (b / d)) % n;  // find first solution
		for(int i = 0 ;i < d ;++i)
            cout << (x0 + i * (n / d)) % n;
        return 0;
    } else 
        return -1;  // no solution
}
```

#### 5-2 乘法逆元

```c++
ll inv(ll a, ll b){ // use ax + by = 1
    ll x, y;
    ll d = exgcd(a, b, x, y);
    assert(d == 1);
	return (x + b) % b;
}

ll inv(ll a, ll q){  // Fermat Th, q must be a prime!
    return qpow(a, q-2, q);
}
```

#### 5-3 中国剩余定理

输入变量n为$m=m_1\cdots m_n$中$m_i$个数，a[]为$a\mod m_i$，m[]存放$m_i$，$p=m$。
输出变量$a\leftarrow(a_1,a_2\cdots a_n)$

```c++
ll crt(int n, ll a[], ll m[]){
    ll w, x, r = 0;
    ll p = 1;
    for(int i = 0 ;i < n ;++i)
    	p *= m[i];
    for(int i = 0 ;i < n ;++i){
        w = p / m[i];
        x = inv(w, m[i]);
        r = (r + qmult(qmult(a[i], w, p), x, p)) % p;
    }
    return (r + p) % p;
}
```

#### 5-4 扩展中国剩余定理

```c++
// m1 ... mn not pairwise coprime

ll excrt(int n, ll a[], ll m[]){
    ll a1 = a[0], m1 = m[0];
    for(int i = 1 ;i < n ;++i){
        ll a2 = a[i], m2 = m[i];
        ll x, y;
        ll d = exgcd(m1, m2, x, y);
        x = x * ((a2 - a1 % m2 + m2) % m2) / d % (m2 / d);
        a1 += m1 * x;
        m1 *= m2 / d;
        a1 = (a1 % m1 + m1) % m1;
    }
	return a1;    
}
```

#### 5-5 BSGS算法

BSGS(baby step & giant step): 求解$a^x\equiv b\mod p$时，取一个整数$t$将方程转化为$a^{At-B}\equiv b\mod p$，则原方程变为$a^{At}\equiv a^Bb\mod p$。暴力枚举列出全部$a^x$，而BSGS仅列出$a^{At}$和$a^Bp$。取$t=\lceil \sqrt{p} \rceil$，$a^Bb$可能取值为$b,ab,a^2b,\cdots,a^{t-1}b (\mod p)$, $a^{At}$可能取值为$a^{t},a^{2t},a^{3t},\cdots,a^{t^2}$。A从小取，B从大取，可得到最小的$x=At-B$

```c++
#define ll long long
ll t, a, b, p, at; // at is a^t
map<ll, ll> saveB;  // save (a^Bb, B)

void listB(){
    ll tmp = b;
    at = 1;
    for(int i = 0 ;i < t ;++i){
        saveb[tmp] = i;
        tmp = tmp * a % p;
        at = at * a % p;
    }
}

ll bsgs(){
    t = ceil(sqrt((double)p));
    listB();
    ll tmp = at;
    for(int i = 1 ;i <= t ;++i){
        auto it = saveB.find(tmp);
        if(it != saveB.end())
            return i * t - it->second; // minimal x
        tmp = tmp * at % p;
    }
    return -1;  // no solution
}
```

### 6 组合数

#### 6-1 组合数计算

组合数的计算可以用杨辉三角/动态规划打表：$C_n^m = C_{n-1}^m+C_{n-1}^{m-1}$

```c++
ll C[MAXN][MAXN];

void cal_comb(){
    for(int i = 1 ;i < MAXN ;++i)
        C[i][0] = C[i][i] = 1;
    for(int i = 2 ;i < MAXN ;++i)
        for(int j = 1 ;j < i ;++j)
            C[i][j] = C[i-1][j-1] + C[i-1][j];
}
```

如果组合数的结果对一个数$q$取模，且模数比$n,m$大，可用乘法逆元计算：$\binom{n}{m}=\frac{n\cdot (n-1)\cdots(n-m+1)}{1\cdot 2\cdots m}$

```c++
void choose(ll n, ll m, ll q){
    ll ret = 1;
    m = min(m, n - m);  // accelerate
    for(int i = 1 ;i <= m ;++i) // divided by (1x2x...x m)
        ret = qmult(ret, i, q);
    ret = inv(ret, q);
    for(int i = 0 ;i <= m - 1 ;++i)
        ret = qmult(n - i, ret, q);
    return ret;
}
```

#### 6-2 卢卡斯定理

卢卡斯定理：$\binom{n}{m}\mod p = \binom{n\mod p}{m\mod p}\binom{\lfloor n/p \rfloor}{\lfloor m/p \rfloor}\mod p$. 数据很大时可用Lucas。

```c++
ll lucas(ll n, ll m, ll q){
    if(m == 0)
        return 1;
    return qmult(choose(n%q, m%q, q), lucas(n/q, m/q, q), q);
}
```

***

## Ch3 其他算法

### 1 二分法

#### 1-1 整数二分

```c++
int ans = 0, l, r;
// some input, find minimum
while(l <= r){
    int m = (l + r) / 2;
    if(check(m)){
        ans = m;
        l = m + 1;
    } else {
        r = m - 1;
    }
}
```

### 2 矩阵计算

#### 2-1 Gauss消元法

```c++
// 求解线性方程组，求解成功后A的左边方阵为单位阵，最右边一列即为解
// O(n^3)
double A[MAXN][MAXN];   // A is the nX(n+1) augmented matrix

void gauss(){
    for(int j = 1 ;j <= N ;++j){  // vis col
        int max_idx = j;
        for(int i = j + 1 ;i <= N ;++i)  // careful!
            if(abs(A[i][j]) > abs(A[max_idx][j]))
                max_idx = i;
        if(max_idx != j)
            for(int k = 1 ;k <= N + 1 ;++k)
                swap(A[max_idx][k], A[j][k]);
        else
            if(A[j][j] == 0){
                error("unsolvable");
                return;
            }
        //
        double rec_ajj = A[j][j];  // careful here!
        for(int k = 1 ;k <= N + 1 ;++k)
            A[j][k] /= rec_ajj;
        //
        for(int i = 1 ;i <= N ;++i)
            if(i != j) {
                double rec_aij = A[i][j]; // careful here!
                for (int k = 1; k <= N + 1; ++k) {
                    A[i][k] -= rec_aij * A[j][k];
                }
            }
    }
}
```

### 3 线性规划

#### 3-1 单纯形算法

目标函数：$f=\sum_{j=1}^n A[0][j]$

第$i$个约束条件：$\sum_{j=1}^nA[i][j]\leq A[i][0]$

```c++
const double EPS = 1e-8;

int id[MAXN+MAXM] = {};
double v[MAXN] = {};
double a[MAXM][MAXN] = {};

inline int sgn(double x) {   // get sign of double
  if (x < -EPS) return -1;
  return x > EPS ? 1 : 0;
}

void pivot(int r, int c) {
  swap(id[n + r], id[c]);
  double x = -a[r][c];
  a[r][c] = -1;
  for (int i = 0; i <= n; ++i) a[r][i] /= x;
  for (int i = 0; i <= m; ++i)
    if (sgn(a[i][c]) && i != r) {
      x = a[i][c];
      a[i][c] = 0;
      for (int j = 0; j <= n; ++j) a[i][j] += x * a[r][j];
    }
}

int simplex() {
  /* important: revert symbols of conditions */
  for (int i = 1; i <= m; ++i) 
    for (int j = 1; j <= n; ++j) 
       a[i][j] *= -1;
  for (int i = 1; i <= n; ++i) id[i] = i;
  /* initial-simplex */
  while (true) {
    int x = 0, y = 0;
    for (int i = 1; i <= m; ++i)
      if (sgn(a[i][0]) < 0) { x = i; break; }
    if (!x) break;
    for (int i = 1; i <= n; ++i)
      if (sgn(a[x][i]) > 0) { y = i; break; }
    if (!y) return -1; // infeasible
    pivot(x, y);
  }
  /* solve-simplex */
  while (true) {
    int x = 0, y = 0;
    for (int i = 1; i <= n; ++i)
      if (sgn(a[0][i]) > 0) { x = i; break; }
    if (!x) break; // finished
    double w = 0, t = 0; bool f = true;
    for (int i = 1; i <= m; ++i)
      if (sgn(a[i][x]) < 0) {
        t = -a[i][0] / a[i][x];
        if (f || t < w) {
          w = t, y = i, f = false;
        }
      }
    if (!y) return 1; // unbounded
    pivot(y, x);
  }
  for (int i = 1; i <= n; ++i) v[i] = 0;
  for (int i = n + 1; i <= n + m; ++i) v[id[i]] = a[i - n][0];
  return 0;
}
```

### 4 并查集

树实现的并查集算法：

```c++
//O(m\alpha(n))
struct node{
    int val;
    int rank;
    node* p;
    node(int v): val(v), rank(1), p(it){}
}
node* make_set(int val){
    return new node(val);
}
void link(node* x, node* y){
    if(x == y) return;
    if(x->rank > y->rank)
        y->p = x;
    else{
        x->p = y;
        if(x->rank == y->rank)
            y->rank ++;
    }
}
node* find_set(node* x){
    if(x->p != x)
        x->p = find_set(x->p);  // path compression
    return x->p;
}
void union_set(node* x, node* y){
    link(find_set(x), find_set(y));
}
```

### 5 串匹配

#### 5-1 KMP算法

```c++
int s[MAXN]; // string length n
int t[MAXN]; // string length m
int p[MAXN]; // prefix

void prefix(){
    p[1] = 0;
    int k = 0;
    for(int i = 2 ;i <= m ;++i){
        while(k > 0 && t[k+1] != t[i])
            k = p[k];
        if(t[k+1] == t[i])
            k ++;
        p[i] = k;
    }
}

void kmp(){
   	prefix();
    int k = 0;
    for(int i = 1 ;i <= n ;++i){
        while(k > 0 && t[k+1] != s[i])
            k = p[k];
        if(t[k+1] == s[i])
            k ++;
        if(k == m){
     		cout << "find one" << endl;       
            k = p[k];
        }
    }
}

```

### 6 排序算法

#### 6-1 Quick-Sort

```c++
void qsort(int obj[], int l, int r){
    if(l >= r) return;
    swap(obj[r], obj[rand() % (r-l+1) + l]);
    int i = l - 1;
    for(int j = l ;j < r ;++j)
        if(obj[l] <= obj[r])
            swap(obj[++i], obj[j]);
    swap(obj[i+1], obj[r]);
    qsort(obj, l, i);
    qsort(obj, i+2, r);
}
```

