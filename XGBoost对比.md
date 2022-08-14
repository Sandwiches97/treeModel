@[TOC]

# BDT，GBDT，XGBoost 对比


流程：类似 BDT、GBDT 算法，$\color{yellow}\text{\colorbox{black}{（BDT是GBDT 选择 square loss 损失时的特例,因此看懂GDBT就行）}}$。

输入：训练数据集 $T=\{(x_1, y_1), \cdots, (x_N,y_N) \}$, $x_i\in \mathcal{X}\subseteq \mathbb{R}^n$, $y_i\in \mathcal{Y}\subseteq \mathbb{R}$， Loss function：$L(y, f(x))$

输出：a XGBoost Tree $f_M(x)$

- (1) 初始化

  - $\color{yellow}\text{\colorbox{black}{BDT}}$：$f_0(x)=0$;
  - $\color{magenta}\text{\colorbox{black}{GBDT （CART）}}$：$f_0(x) = \argmin_c \sum^N_{i=1}L(y_i, c)$; 只有一个根节点的树，c为树的参数
  - $\color{red}\text{\colorbox{black}{XGBoost}}$：$f_0(x) = \argmin_c \sum^N_{i=1}L(y_i, c)$;
- (2) 对 $m=1，2，...，M$;

  - $\color{yellow}\text{\colorbox{black}{BDT}}$:
    - $\color{yellow}\text{\colorbox{black}{计算残差}}$：$r_{mi} = y_i - f_{m-1}(x_i), \text{ i=1,2,...,N}$;
    - $\color{yellow}\text{\colorbox{black}{拟合残差}}$ $r_{mi}$ 学习一个回归树，得到 $T(x; \Theta_m)$;
    - $\color{yellow}\text{\colorbox{black}{更新 树}}$ $f_m(x) = f_{m-1}(x) + T(x; \Theta_m)$
  - $\color{magenta}\text{\colorbox{black}{GBDT}}$：
    - $\color{magenta}\text{\colorbox{black}{loss的一阶泰勒展开，计算梯度函数}}$：$r_{mi} = -\left[\frac{\partial L(y_i,f_{m-1}(x_i))}{\partial f(x_i)}\right]_{f(x)=f_{m-1}(x)}, \text{ i=1,2,...,N}$;
    - $\color{magenta}\text{\colorbox{black}{拟合负梯度}}$ $r_{mi}$ 学习一个回归树（此时得到的树的参数），将第 $m$ 棵树的叶结点$\color{magenta}\text{\colorbox{black}{分区}}$：$R_{mj}, \text{ j=1,2,...,J}$;
    - 对 $\text{ j=1,2,...,J}$，$\color{magenta}\text{\colorbox{black}{获得最优分割点 j 以及树的参数c}}$：
      - $$
        c_{mj} = \argmin_c \sum_{x_i\in R_{mj}} L(y_i, f_{m-1}(x_i) +c)

        $$
    - $\color{magenta}\text{\colorbox{black}{更新}}$ $f_m(x) = f_{m-1}(x) + \sum^J_{j=1}c_{mj}I(x\in R_{mj})$.
  - $\color{red}\text{\colorbox{black}{XGBoost}}$：
    - $\color{red}\text{\colorbox{black}{loss的二阶泰勒展开，计算一阶、二阶导数}}$：
      - $r_{mi} = -\left[\frac{\partial L(y_i,f_{m-1}(x_i))}{\partial f(x_i)}，\frac{\partial^2  L(y_i,f_{m-1}(x_i))}{(\partial f(x_i))^2}\right]_{f(x)=f_{m-1}(x)} = {\color{yellow}\sum_{i=1}^N\left[ g_i {\color{red}f_m(x_i)} + \frac{1}{2}h_i {\color{red}f^2_m(x_i)}\right]} + {\color{orange}\Omega(f_m)}   , \text{ i=1,2,...,N}$;
    - $\color{red}\text{\colorbox{black}{拟合}} r_{mi} 与正则项$ 学习一个回归树(此时得到的树的参数)，将第 $m$ 棵树的叶结点$\color{red}\text{\colorbox{black}{分区}}$：$R_{mj}, \text{ j=1,2,...,J}$;
    - 对 $\text{ j=1,2,...,J}$，$\color{red}\text{\colorbox{black}{获得最优分割点 j 以及树的参数c}}$：
      - $$
        {\color{green}f_{mj}} = \argmin_c \sum_{x_i\in R_{mj}} L(y_i, f_{m-1}(x_i) + {\color{red}f_m(x_i)}) + {\color{orange}\Omega(f_m)}

        $$
      - 对于暴力迭代求解的优化：贪心法、近似法
    - $\color{red}\text{\colorbox{black}{更新}}$ $f_m(x) = f_{m-1}(x) + \color{green}T(x; \Theta_m)$
- (3) 得到回归问题 提升树：

  $$
  f_M(x) = \sum_{m=1}^M T(x; \Theta_m)

  $$

①提升树的思想是基于加法模型，不断拟合残差。
②GBDT和Xgboost都是基于提升树的思想。
③GBDT的全称是Gradient Boosting Decision Tree，之所以有Gradient是因为GBDT中引入了梯度作为提升树中“残差”的近似值（提升树的每次迭代都是为了使当前模型拟合残差，就是使求得的增量模型尽可能等于残差）。
④Xgboost可以说是GBDT的一种，因为其也是基于Gradient和Boosting思想，但是和原始GBDT的不同是：Xgboost中引入了二阶导数和正则化，除此之外，Xgboost的作者陈天奇博士在工程实现方面做了大量的优化策略。

## 随机森林、GBDT和Xgboost区别

### Random Forest和GBDT区别如下：

- ①RF的基分类器可以是分类树也可以是回归树，GBDT只能是回归树。
  ②RF不同基分类器可以并行，GBDT只能串行。
  ③RF最终结果采用的策略是多数投票、一票否则、加权投票等，而GBDT是将所有结果（加权）累加起来。
  ④RF对异常值不敏感，GBDT对异常值敏感
  ⑤RF对训练集一视同仁，GBDT基于Boosting思想，基于权值，分类器越弱，权值越小
  ⑥RF主要减少模型方差，所以在噪声较大的数据上容易过拟合，而GBDT主要较少模型偏差。
  ⑦RF随机选择样本，GBDT使用所有样本。

Xgboost就是GBDT的一种，所以Xgboost和RF的区别和GBDT一样。

### GBDT和Xgboost的区别如下：

- ①基分类器的选择： 传统GBDT以 CART 作为基分类器，XGBoost还支持线性分类器，这个时候XGBoost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。
  ②梯度信息： 传统GBDT只引入了一阶导数信息，Xgboost引入了一阶导数和二阶导数信息，其对目标函数引入了二阶近似，求得解析解, 用解析解作为 Gain 来建立决策树, 使得目标函数最优（Gain求到的是解析解）。另外，XGBoost工具支持自定义损失函数，只要函数可一阶和二阶求导。
  ③正则项： Xgboost引入了正则项部分，这是传统GBDT中没有的。加入正则项可以控制模型的复杂度，防止过拟合。
  ④特征采样： Xgboost引入了特征子采样，像随机森林那样，既可以降低过拟合，也可以减少计算。
  ⑤节点分裂方式：GBDT是用的基尼系数，XGBoost是经过优化推导后的。
  ⑥并行化： 传统GBDT由于树之间的强依赖关系是无法实现并行处理的，但是Xgboost支持并行处理，XGBoost的并行不是在模型上的并行，而是在特征上的并行，将特征列排序后以block的形式存储在内存中，在后面的迭代中重复使用这个结构。这个block也使得并行化成为了可能，其次在进行节点分裂时，计算每个特征的增益，最终选择增益最大的那个特征去做分割，那么各个特征的增益计算就可以开多线程进行。
  ⑦除此之外，Xgboost实现了分裂点寻找近似算法、缺失值处理、列抽样（降低过拟合，还能减少计算）等包括一些工程上的优化，LightGBM是Xgboost的更高效实现。
