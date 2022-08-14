2.
3. 一棵树生成的细节 (类似CART)

### 2.1 最优切分点划分算法 (不同于树桩，完全遍历多层决策树太复杂了)

在实际训练过程中，当建立第 $t$ 棵树时，一个非常关键的问题就是：$\color{red}如何找到树的最优结构$（$\color{red}叶子结点的最优切分点$）

XGBoost 支持两种分裂结点的方法 —— $贪心算法、近似算法$：

#### 2.1.1 贪心算法 （只考虑当前结点最优）

1. 从树深度为 0 开始，对每个结点枚举所有可用特征；
2. 针对每个特征，把 $\color{yellow}\text{\colorbox{black}{属于该结点的}}$ 训练样本 $\color{red}根据该特征值$ 进行 $\color{yellow}\text{\colorbox{black}{升序排列}}$；
3. （根据特征分裂收益）选择$\color{yellow}最佳分裂特征$，$\color{yellow}最佳分裂点$，将该结点上分裂处左右两个新的叶节点，并为每个新结点关联对应的样本集；
4. 回到第 1 步，递归执行 直到满足停止条件为止。

##### 计算每个特征的分裂收益

假设在某结点完成特征分裂，则分裂前的目标函数为：

$$
\begin{aligned}
\text{\colorbox{black}{\color{green}objective Function}}_1 &= {\color{green}-\frac{1}{2}{\left[ \frac{(G_L + G_R)^2}{H_L + H_R +\lambda}\right]}} + {\color{orange}\gamma \cdot {\color{magenta}1}}\\
\end{aligned}

$$

分裂后的目标函数为：

$$
\begin{aligned}
\text{\colorbox{black}{\color{green}objective Function}}_2 &= {\color{green}-\frac{1}{2}{\left[ \frac{(G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R +\lambda}\right]}} + {\color{orange}\gamma \cdot {\color{magenta}2}}\\
\end{aligned}

$$

对目标函数来说，分裂后的收益为：

$$
{\color{green}Gain = -\frac{1}{2}{\left[ \frac{(G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R +\lambda} - \frac{(G_L + G_R)^2}{H_L + H_R +\lambda}\right]}} - {\color{orange}\gamma \cdot {\color{magenta}1}}

$$

该特征收益，可作为 特征重要性 输出的依据。

##### 对于每次分裂，都需要枚举所有特征可能的分割方案，如何高效地枚举所有的分割？

假设我们要枚举某个特征所有 $\color{magenta}x<a\text{ （分割条件）}$ 这样条件的样本，对于 $\color{magenta}\text{分割点 }a$ 计算左边和右边的导数和。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://img2022.cnblogs.com/blog/1235684/202204/1235684-20220425164001517-1283658461.png"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Fig. 4
  	</div>
</center>

对于一个特征，对特征取值 $\color{yellow}\text{\colorbox{block}{排完序}}$ 后，$\color{yellow}\text{\colorbox{block}{枚举}}$ 所有的 $\color{magenta}\text{分割点 }a$，只要从左到右扫描就可以枚举出所有分割的梯度  $G_L$、$G_R$，计算增益。假设树的高度 $H$，特征数 $d$，则复杂度为 $O(H\cdot d\cdot n\log n)$。其中，排序为 $O(n\log n)$，每个特征都要排序乘以 $d$，每层都要排序乘以 $H$。

观察分裂后的收益，我们会发现 $\color{red}\text{\colorbox{white}{结点划分不一定会使得结果变好}}$，因为我们有一个引入$\textbf{新叶子的惩罚项}$，也就是说引入的分割带来的增益如果小于一个阈值的时候，我们可以 $剪掉这个分割$。

#### 2.1.2 近似算法 Approximate Algorithm

贪心算法可以得到最优解，但 $\color{red}当数据量太大$ 则无法读入内存进行计算，$\color{yellow}\text{\colorbox{black}{近似算法}}$ 主要针对贪心算法这一缺点给出了近似最优解。

对于每个特征，$\color{yellow}\text{\colorbox{black}{只考察分割点}}$ 可以减少计算复杂度。

该算法首先根据特征分布的分位数提出 $\color{magenta}\text{候选划分点}$，然后将连续型特征映射到由这些候选点划分的桶中，然后聚合统计信息找到所有区间的 $\color{green}最佳分裂点$。

在提出候选切分点时的两种策略：

1. $Global$：学习每棵树前就提出候选切分点，并在每次分裂时都采用这种分割；
2. $Local$：每次分裂前重新提出候选切分点。

直观上看，$Local$ 策略需要更多的计算步骤，而 $Global$ 策略因为结点已有划分所以需要更多的候选点。

下图给出不同种分裂策略的 $AUC$ 变化曲线，横坐标为迭代次数，纵坐标为测试集 $AUC$，`eps`为近似算法的精度，其倒数为桶的数量。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://img2022.cnblogs.com/blog/1235684/202204/1235684-20220425164207756-1602386895.png"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">

</center>

从上图看出，$Global$ 策略在候选点多时（`eps`小）可以和 $Local$ 策略在候选点少时（`eps` 大）具有相似的精度。此外还发现，`eps` 取值合理的情况下，分位数策略可以获得与贪心算法相同的精度。

近似算法：根据特征 $k$ 的分布来确定 $l$ 个候选切分点 $S_k=\{s_{k1}, s_{k2},...,s_{kl}\}$，然后根据候选切分点把相应的样本放入对应的桶中，对每个的 $G,H$ 进行累加，在候选切分点集合上进行精确贪心查找。算法描述：

$\text{Algorithm 2: Approximate Algorithm for Split Finding}$：

- for k=1 to m do:（分成m个区间）
  - propose $S_k = \{s_{k1}, s_{k2}, ..., s_{kl}\}$ by percentiles on feature $k$.
  - proposal can be done per tree (global), or per split (local)
- end
- for k=1 to m do:
  - $G_{kv}  ←= \sum_{j\in\{j|s(k,v) \ge x_{jk} > s(k, v-1)\} } g_i$;
  - $H_{kv}  ←= \sum_{j\in\{j|s(k,v) \ge x_{jk} > s(k, v-1)\} } h_i$;
- end
- Follow same step as in previous section to find max score only among proposed splits

算法讲解：

- 第一个 $for$ 循环：对特征 $k$ 根据该特征分布的分位数找到切割点的候选集合 $S_k = \{s_{k1}, s_{k2}, ..., s_{kl}\}$。这样做的目的是：$\color{red}提取出部分的切分点，而不用遍历所有的切分点$。其中获取某个特征 $k$ 的候选切割点的方式叫 $proposal (策略)$。XGBoost 支持 Global 策略 和 Local 策略。
- 第二个 $for$ 循环：$\color{red}将每个特征的取值映射到由该特征对应的候选点集划分的分桶区间$，即 $s(k,v) \ge x_{jk} > s(k, v-1)$。对每个桶间内的样本统计值 $G, H$ 并进行累加，最后在这些累计的统计量上寻找最佳分裂点

实例：近似算法举例，以三分位为例：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://img2022.cnblogs.com/blog/1235684/202204/1235684-20220425164445030-635621431.png

根据样本特征进行排序，然后基于分位数进行划分，并统计三个桶内的 $G,H$ 值，最终求结点划分的增益

### 2.2 加权分位数缩略图

XGBoost 不是简单地按照样本个数进行分位，而是$\color{red}\text{以二阶导数值 } h_i$ 作为样本的 $\color{red}权重$ 进行划分。为了处理带权重的候选切分点的选取，提出了加权分位数缩略图算法。

加权分位数缩略图算法提出了一种数据结构，这种数据结构支持 $merge$ 和 $prune$ 操作。

加权分位数缩略图候选点的选取方式：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://img2022.cnblogs.com/blog/1235684/202204/1235684-20220425164550511-447887820.png

**为什么用二阶梯度  $h_i$ 进行样本加权？**

模型的目标函数：

$$
\begin{aligned}
\text{\colorbox{black}{Objective Function}}^{(t)} &\simeq {\color{yellow}\sum_{i=1}^n\left[ g_i {\color{red}f_t(x_i)} + \frac{1}{2}h_i {\color{red}f^2_t(x_i)}\right]} + {\color{orange}\Omega(f_t)}
\end{aligned}

$$

把目标函数整理成一下形式，可以看出 $\color{red} h_i \text{ 对 loss 加权的作用}  $。

$$
\begin{aligned}
\text{\colorbox{black}{Objective Function}}^{(t)} &\simeq {\color{yellow}\sum_{i=1}^n\left[ g_i {\color{red}f_t(x_i)} + \frac{1}{2}h_i {\color{red}f^2_t(x_i)}\right]} + {\color{orange}\Omega(f_t)}\\
&= {\color{yellow}\sum_{i=1}^n \frac{1}{2}h_i \left[ \frac{2 g_i {\color{red}f_t(x_i)}}{h_i} + {\color{red}f^2_t(x_i)} \right]} + {\color{orange}\Omega(f_t)}\\   
&= {\color{yellow}\sum_{i=1}^n \frac{1}{2}h_i \left[ \frac{2 g_i {\color{red}f_t(x_i)}}{h_i} + {\color{red}f^2_t(x_i)} + (\frac{g_i}{h_i})^2 -(\frac{g_i}{h_i})^2  \right]} + {\color{orange}\Omega(f_t)}\\   =
&=  {\color{yellow}\sum_{i=1}^n \frac{1}{2}h_i \left[ {\color{red}f_t(x_i)}  -(-\frac{g_i}{h_i}) \right]^2 + {\color{orange}\Omega(f_t)} - \sum^n_{i=1}\frac{1}{2}\frac{g^2_i}{h_i}}\\  
&={\color{yellow}\sum_{i=1}^n \frac{1}{2} {\color{magenta} h_i} \left[ {\color{red}f_t(x_i)}  -(-\frac{g_i}{h_i}) \right]^2 + {\color{orange}\Omega(f_t)} - constant}  
\end{aligned}

$$

其中，加入 $\frac{1}{2}\frac{g^2_i}{h_i}$ 是因为 $g_i, h_i$ 是上一轮的损失函数求导与  constant 皆为常数。我们可以看到 $\color{magenta}h_i$ 就是平方损失函数中样本的权重。

## 2.3 稀疏感知算法

实际工程中，比如数据的缺失、one−hot 编码都会造成输入数据稀疏。XGBoost 在构建树的结点过程中只考虑非缺失值的数据遍历。

为每个结点增加了一个缺省方向，当样本相应的特征值缺失时，可以被归类到缺省方向上，最优的缺省方向可以从数据中学到。至于如何学到缺省值的分支，其实很简单，分别枚举特征缺省值的样本归为左右分支后的增益，选择增益最大的枚举项为最优缺省方向。

在构建树的过程中需要枚举特征缺失的样本，乍一看这个算法会多出相当于一倍的计算量，但其实不是的。因为在算法的迭代中只考虑了非缺失值数据的遍历，缺失值数据直接被分配到左右结点，所需要遍历的样本量大大减小。

通过在 Allstate-10K 数据集上进行实验，从结果看到稀疏算法比普通算法在数据处理上快了 50 倍。

dd
