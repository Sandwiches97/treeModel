@[TOC]

# XGBoost 简介

XGBoost 全称为：Extreme Gradient Boosting Decision Tree，是Gradient Boosting的Cpp优化实现

## XGBoost 原理

## 1. 从目标函数开始，生成一棵树

XGBoost, GBDT 都是 Boosting Tree 方法

### 1.1 学习第 t 颗树

XGBoost 是由 k 个基模型组成的一个 **加法模型**，使用 **前向分步算法** 求解。

假设第 t 次迭代要训练的树模型是 $\color{red}f_t(x)$, 则有：

$$
\hat{y}_i^{(t)} = \sum_{k=1}^tf_k(x_i) = \hat{y}^{(t-1)}_i + \color{red}f_t(x_i), \tag{1}

$$

其中，

- $\hat{y}_i^{(t)}$ 表示第 $t$ 次迭代后样本 $i$ 的预测结果，
- $\hat{y}^{(t-1)}_i$ 表示前 $t-1$ 颗树的预测结果，
- $\color{red}f_t(x_i)$ 表示第 $t$ 棵树的模型。

### 1.2 XGBoost 的目标函数

$\colorbox{black}{\color{yellow}损失函数}$可由预测值 $\hat{y}_i$ 与真实值 $y_i$ 表示：

$$
L = \sum^n_{i=1}l(y_i, \hat{y}_i)

$$

其中，$n$ 为样本数量。

**模型的预测精度**由模型的 $\color{yellow}\text{Bais}$ 和 $\color{orange}\text{Variance}$ 共同决定，

- $\colorbox{black}{\color{yellow}\text{Loss Function}}$ 代表模型的 $\colorbox{black}{\color{yellow}\text{Bais}}$，
- 想要 $\color{orange}方差$ 小，则需要在 $\color{white}\textbf{目标函数}$ 中添加 $\color{orange}\text{Regularization}$，用于防止过拟合。

因此，$\textbf{\colorbox{black}{Objective Function}}^{(t)}$ 由模型的 $\colorbox{black}{\color{yellow}\text{Loss Function：L}}$ 与 抑制模型复杂度的 $\color{orange}\text{Regularization}：\Omega$ 组成，其定义如下：

$$
\textbf{\colorbox{black}{Objective Function}}^{(t)} = {\color{yellow}\sum^n_{i=1} l(y_i, \hat{y}_i^{(t)})} + \color{orange}\sum^t_{i=1}\Omega(f_i)

$$

其中，$\color{orange}\sum^t_{i=1}\Omega(f_i)$ 是将全部 $t$ 棵树的复杂度进行求和，添加到目标函数中作为 Regularization，用于防止过拟合。

由 XGBoost 是 Boostiong 族的算法，所以遵循 $\textbf{前向分布算法}$ ，以第 $t$ 步模型为例，模型对第 $i$ 个样本的预测值为：

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \color{red}f_t(x_i)

$$

其中，$\hat{y}^{(t-1)}_i$ 是由第 t-1 步的模型给出的预测值，是已知常数，$\color{red}f_t(x_i)$ 是这次需要加入的$\color{red}新模型的预测值$。此时，目标函数可以写成：

$$
\begin{aligned}
\text{\colorbox{black}{Objective Function}}^{(t)} &= {\color{yellow}\sum_{i=1}^nl(y_i, \hat{y}_t^{(t)})} + {\color{orange}\sum^t_{i=1}\Omega(f_i)}\\
&= {\color{yellow}\sum_{i=1}^nl(y_i, \hat{y}_t^{(t-1)} + {\color{red}f_t(x_i)})} + {\color{orange}\sum^t_{i=1}\Omega(f_i)}\\
&={\color{yellow}\sum_{i=1}^nl(y_i, \hat{y}_t^{(t-1)} + {\color{red}f_t(x_i)})} + {\color{orange}\Omega(f_t)+\text{constant}} 
\end{aligned}

$$

注意上式中，$\textbf{只有第 t 棵树 是变量}$，$\textbf{其余都是已知量}$（或可被计算得知）

- 第 $t$ 棵树 ${\color{red}f_t(x_i)}$，
- 第 $t$ 棵树的 $\color{orange}正则项$ ：将 正则项 进行拆分，由于前 $t-1$ 课树的结构已经确定，${\color{orange}因此前\ t-1\ 棵树的复杂度之和可以用一个常量表示}$，如下：

$$
\begin{aligned} 
\sum^t_{i=1}\Omega(f_i) &= {\color{orange}\Omega(f_t)} + \sum^{t-1}_{i=1} \Omega(f_i)\\
&= {\color{orange}\Omega(f_t)}+\text{constant}
\end{aligned}

$$

### 1.3 泰勒公式展开

泰勒公式的介绍，略。

若函数 $f(x)$ 包含 $x_0$ 的某个闭区间 $[a,b]$ 上具有 $n$ 阶导数，且在开区间 $(a,b)$ 上具有 $n+1$ 阶导数，则对闭空间 $[a,b]$ 上任意一点 $x$ 有：

$$
f(x) = \sum^n_{i=0}\frac{f^{(i)}(x)}{i!} + R_n(x)

$$

其中的多项式称为函数在 $x_0$ 处的泰勒展开，$R_n(x)$ 是泰勒公式的余项，且是 $(x-x_0)^n$ 的高阶无穷小。

把函数 $f(x+ \triangle)$ 在点 $x$ 处进行泰勒的二阶展开，可得：

$$
f(x+\triangle x) \approx f(x) + f'(x)\triangle x + \frac{1}{2}f''(x)\triangle x^2

$$

回到 XGBoost 的 $\color{white}\textbf{Objective Function}$ 上，

- $\colorbox{black}{\color{yellow}f(x)}$ 对应损失函数 $\color{yellow}l(y_i, \hat{y}_i^{(t-1)}+ {\color{red}f_t(x_i)})$,
- $x$ 对应前 $t-1$ 棵树的预测值 $\hat{y}^{(t-1)}_t$,
- $\color{red}\triangle x$ 对应于我们正在训练的第 $t$ 棵树 $\color{red}f_t(x_i)$ ，

则可以将损失函数写为

$$
{\color{yellow}l(y_i, \hat{y}_i^{(t-1)}+ {\color{red}f_t(x_i)}) = l(y_i, \hat{y}_i^{(t-1)})+g_i {\color{red}f_t(x_i)} + \frac{1}{2}h_i {\color{red}f^2_t(x_i)}}

$$

其中，$g_i$ 为损失函数的一阶导，$h_i$ 为损失函数的二阶导，注意这里的求导是对 $\hat{y}^{(t-1)}_i$ 求导。

**举例**：我们以平方损失函数为例：

$$
l(y_i, \hat{y}_i^{(t-1)}) = (y_i - \hat{y}^{(t-1)}_i)^2

$$

则：

$$
\begin{aligned}
& g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial\hat{y}_i^{(t-1)}} = -2 (y_i-\hat{y}_i^{(t-1)})\\
& h_i = \frac{\partial l^2(y_i, \hat{y}_i^{(t-1)})}{\partial(\hat{y}_i^{(t-1)})^2} = 2
\end{aligned}

$$

将上述的二阶展开式，带入到 XGBoost 的目标函数中，可以得到目标函数的近似值：

$$
\text{\colorbox{black}{Objective Function}}^{(t)} \simeq {\color{yellow}\sum_{i=1}^n\left[ l(y_i, \hat{y}_i^{(t-1)})+g_i {\color{red}f_t(x_i)} + \frac{1}{2}h_i {\color{red}f^2_t(x_i)}\right]} + {\color{orange}\Omega(f_t)+\text{constant}}

$$

由于在第 $t$ 步时，$\hat{y}_i^{(t-1)}$ 其实是一个已知的值，所以 $\color{yellow}l(y_i, \hat{y}_i^{(t-1)})$ 是一个常数，其对函数的优化不会产生影响。因此，$\text{\colorbox{black}{去掉全部的常数项}}$，得到的 Objective Function 为：

$$
\text{\colorbox{black}{Objective Function}}^{(t)} \simeq {\color{yellow}\sum_{i=1}^n\left[ g_i {\color{red}f_t(x_i)} + \frac{1}{2}h_i {\color{red}f^2_t(x_i)}\right]} + {\color{orange}\Omega(f_t)}

$$

所以，我们只需要求出每一步 $\color{yellow}\text{\colorbox{black}{Loss Function}}$ 的 一阶导、二阶导的值（由$\hat{y}_i^{(t-1)}$求得），然后最优化目标函数，就可以得到每一步的 $\color{red}f(x)$, 最后根据假发模型得到一个整体模型。

### 1.4 定义一棵树 $\color{red}f_t(x)$

XGBoost 的基模型不仅支持决策树，还支持 Linear Model，下面介绍基于决策树的 $\text{\colorbox{black}{Objective Function}}$。

定义一棵树，包括两部分：

- 叶子节点的权重向量 $w$;
- 实例 (sample) 到 叶子结点 的映射关系 $q$ （本质是树的分支结构）

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://img2022.cnblogs.com/blog/1235684/202204/1235684-20220425163340184-1352207651.jpg"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 1 
  	</div>
</center>

### 1.5 树的复杂度 $\Omega$ ~ 模型泛化能力 ~ $\color{orange}{正则项}$

决策树的复杂度 $\Omega$ 可由叶子数 $T$ 组成，叶子结点越少模型越简单，此外叶子结点不应该含有过高的权重 $w$（类别 $LR$ 的每个变量的权重），所以 $\text{\colorbox{black}{Objective Function}}$ 的 $\color{orange}{正则项}$ 由生成的 所有决策树 的:

1. 叶子结点数量和
2. 所有结点权重所组成的向量的 L2 范式（权重的平方和）共同决定。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://img2022.cnblogs.com/blog/1235684/202204/1235684-20220425163440272-2117719556.png"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 1 
  	</div>
</center>

### 1.6 叶子节点归组

$I_j = \{ i|q(x_i) = j\}$ 表示将第 $j$ 个${\color{magenta}叶子节点}$的所有样本 $x_i$ 划入到一个集合 $I_j$ 中，那么 XGBoost 的 $\text{\colorbox{black}{Objective Function}}$ 可以写成：

$$
\begin{aligned}
\text{\colorbox{black}{Objective Function}}^{(t)} &\simeq {\color{yellow}\sum_{i=1}^n\left[ g_i {\color{red}f_t(x_i)} + \frac{1}{2}h_i {\color{red}f^2_t(x_i)}\right]} + {\color{orange}\Omega(f_t)}\\
&= {\color{yellow}\sum_{i=1}^n\left[ g_i {\color{red}w_{q(x_i)}} + \frac{1}{2}h_i {\color{red}w^2_{q(x_i)}}\right]} + {\color{orange}\gamma T + \frac{1}{2}\lambda\sum^T_{j=1}w_j^2 }\\
&= {\color{yellow}\sum_{\color{magenta}j=1}^{\color{magenta}T}\left[ \left(\sum_{i\in I_j }g_i\right) {\color{red}w_j} + \frac{1}{2}\left(\sum_{i\in I_j }h_i + \lambda\right) {\color{red}w^2_j}\right]} + {\color{orange}\gamma {\color{magenta}T}}\\
\end{aligned}

$$

公式的 line 2 -> line 3 解释：

- line 2：遍历${\color{magenta}所有的样本}$后，求每个样本的损失函数。但样本最终会落在叶子结点上，因此：
- line 3：
  - 最外层遍历${\color{magenta}叶子结点}$，
  - 里层遍历：获取叶子结点上的样本集合，
  - 最后求 $\color{yellow}\text{\colorbox{black}{Loss Function}}$。

参数解释：

- $w_j$: 第 $j$ 个叶子结点取值
- $G_j=\sum_{i\in I_j}g_i$：叶子结点 $j$ 所包含 samples 的一阶偏导数累加之和，是一个常量；
- $H_j = \sum_{i\in I_j}h_i$：叶子结点 $j$ 所包含 samples 的二阶偏导数累加之和，是一个常量。

将 $G_j,\ H_j$ 带入 XGBoost $\text{\colorbox{black}{Objective Function}}$：

$$
\begin{aligned}
\text{\colorbox{black}{Objective Function}}^{(t)} &\simeq {\color{yellow}\sum_{\color{magenta}j=1}^{\color{magenta}T}\left[ G_j {\color{red}w_j} + \frac{1}{2}\left(H_j + \lambda\right) {\color{red}w^2_j}\right]} + {\color{orange}\gamma {\color{magenta}T}}\\
\end{aligned}

$$

### 1.7 树结构打分

假设一元二次函数：

$$
Gx + \frac{1}{2}H x^2, \ \ H>0

$$

根据最值公式，求出最值点：

$$
\begin{aligned}
x^* &= -\frac{b}{2a} = -\frac{G}{H} \\
y^* &= \frac{4ac-b^2}{4a} = -\frac{G^2}{2H}
\end{aligned}

$$

回到 XGBoost 的最终目标函数 $\text{\colorbox{black}{Objective Function}}^{(t)}  $, how to get 他的最值？

$$
\begin{aligned}
\text{\colorbox{black}{Objective Function}}^{(t)} &= {\color{yellow}\sum_{\color{magenta}j=1}^{\color{magenta}T}\left[ G_j {\color{red}w_j} + \frac{1}{2}\left(H_j + \lambda\right) {\color{red}w^2_j}\right]} + {\color{orange}\gamma {\color{magenta}T}}\\
\end{aligned}

$$

1. 对于每个叶子节点 $j$ ， 可以将它从 $\text{\colorbox{black}{Objective Function}}^{(t)}  $ 中拆解出来: $G_j {\color{red}w_j} + \frac{1}{2}\left(H_j + \lambda\right) {\color{red}w^2_j}$,
   在 [section 1.5]() 中，$G_j,\ H_j$ 相对于第 $t$ 棵树来说，可以计算出来。则，这个式子就是个一元二次函数 ；
2. 再次分析 $\text{\colorbox{black}{Objective Function}}^{(t)}  $，可以发现，$\color{magenta}{各个叶子节点的目标子式是相互独立的}$,
3. 因此，$\color{magenta}假设目前树的结构已经固定$，套用一元二次函数最值公式，可得最优 ${\color{green}w_j^*}$ 值为

$$
\color{green}w^* = -\frac{b}{2a} = -\frac{G_j}{H_j+\lambda}

$$

因此，$\color{magenta}当前树结构的$ $\text{\colorbox{black}{\color{green}objective Function}}^{(t) \color{green}}  $ 为：

$$
\begin{aligned}
\text{\colorbox{black}{\color{green}objective Function}}^{(t)\color{green}} &= \color{green}-\frac{1}{2}{\sum_{\color{magenta}j=1}^{\color{magenta}T}\left[ \frac{(G_j)^2}{H_j+\lambda}\right]} + {\color{orange}\gamma {\color{magenta}T}}\\
\end{aligned}

$$

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://img2022.cnblogs.com/blog/1235684/202204/1235684-20220425163908171-140807500.png"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 3
  	</div>
</center>

上图给出 $\color{magenta}固定树结构$ 的  $\text{\colorbox{black}{Objective Function}}^{(t)}  $ 最优值  $\text{\colorbox{black}{\color{green}objective Function}}^{(t) \color{green}}  $ 求解的例子，

1. 先求每个结点每个样本的一阶导数 $g_i$ 和二阶导数 $h_i$,
2. 然后针对每个结点对所含样本求和得到 $G_j$, $H_j$，
3. 最后遍历决策树的结点，即可得到 $\text{\colorbox{black}{\color{green}objective Function}}^{(t) \color{green}}  $;

注意，我们$\color{red}后续还需要找到树结构的最优结构$，就需要这个函数：$\text{\colorbox{black}{\color{green}objective Function}}^{(t) \color{green}}$;

##
