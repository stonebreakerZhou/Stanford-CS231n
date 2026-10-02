//全局配置 (放在最上面，只写一次)
#set page(
  paper: "us-letter",
  columns: 2,
  margin: (x: 1in, y: 1in), //缩减边距，让双栏更美观
  //设置页码的计数
  footer: context {
    let page_number = counter(page).at(here()).first()
    grid(
      columns: (1fr, 1fr, 1fr),
      align(left)[#text(size: 8pt, font: "New Computer Modern", fill: gray)[© Xiaoyang Zhou]],
      align(center)[#text(size: 9pt, font: "New Computer Modern")[#page_number]],
      align(right)[],
    )
  },
)


//首页标题 (跨栏显示)
#place(top, scope: "parent", float: true)[
  #align(center)[
    #v(0.5in)
    #text(size: 25pt, weight: "bold")[Notes in CS231n]
    #v(1em)
    #text(size: 14pt)[Sean] \
    #text(size: 10pt)[#link("stonebreaker365@163.com")]
    #v(1em)
    #block(width: 90%, stroke: (y: 0.5pt), inset: 1em)[
      #set align(left)
      *Abstract* --- This note records Sean's notes of Stanford course CS231n taught by Fei-fei Li. (recordings from bilibili)
    ]
    #v(2em)
  ]
]









#pagebreak()








#place(top, scope: "parent", float: true)[
  #align(center + horizon)[  // horizon 让它垂直居中页顶区域，更美观
    #text(font: "Georgia", weight: "bold", size: 24pt)[§ Lec I]  //
    #v(0em)
    #line(length: 100%, stroke: 1pt)  // 可选：加一条装饰线
  ]
]

== Lead-in

\

CS231n overview :

- Deep Learning Basics
- Perceving and Understanding the Visual World
- Generative and Interactive Visual Intelligence
- Human-Centered Applicatins and Implications

\

#figure(
  image("images/Lec1_cs231n_covering.png", width: 100%),
  caption: [CS231n covering],
)

\

#figure(
  image("images/Lec1_AI_timeline.png", width: 100%),
  caption: [AI timeline],
)

\

#figure(
  image("images/Lec1_syllabus.png", width: 100%),
  caption: [CS231n syllabus],
)







#pagebreak()








#place(top, scope: "parent", float: true)[
  #align(center + horizon)[  // horizon 让它垂直居中页顶区域，更美观
    #text(font: "Georgia", weight: "bold", size: 24pt)[§ Lec II]  //
    #v(0em)
    #line(length: 100%, stroke: 1pt)  // 可选：加一条装饰线
  ]
]

== Image Classificationwith Linear Classifiers
\
outline :
- The image classification task
- Two basic data-driven approaches to image classification : *_K-nearest neighbor + linear classifier_*

\
\

Image Classification Task
\

Challenges :
① Viewpoint variation\
② Illumination\
③ Background Clutter\
④ Occlusion\
⑤ Deformation\
⑥ Intraclass variation\
⑦ Context

\
\

Machine Learning : *Data-Driven Approach*\
① Collect a dataset of images and labels\
② Use Machine Learning algorithms to train a classifier\
③ Evaluate the classifier on new images

\
\
\
\
\
\
\
\
\
\

=== 1. First classifier : Nearest Neighbor

~~~~It computes the distance betwee the query data and  training data(with labels).


- - *Distance Metric*

$L 1$ distance :
$
  d_1(I_1, I_2) = sum_p |I^p_1 - I^p_2|
$
~~~~That's the sum over all absolute values of all the pixel differences.

~~~~If we have $N$ examples for prediction, then the predictgion time is $O(N)$.\
~~~~This is not good because we want prediction to be fast.

~~~~One way we can use is *nearest neighbor* :

#figure(
  image("images/Lec2_1_nearest_neighbor.png", width: 60%),
  caption: [nearest neighbor],
)

~~~~However, in the figure above, the small yellow region in the middle may mean an outlier.

~~~~To make it more robust, we can use *$k$-nearest neighbors*. Instead of copying label from nearest neighbor, we take majority vote from K closest points

#figure(
  image("images/Lec2_k_nearest_neighbors.png", width: 100%),
  caption: [$k$-nearest neighbor],
)


~~~~Note that the white regions here are areas we cannot make a decision.



~~~~Meanwhile, the distance metric we take is :

- - *$L 1$ (Manhattan) distance*

~~~~定义为各维差值的绝对值之和：

$
  d_1(I_1, I_2) = sum_p ||I^p_1 - I^p_2||
$
#figure(
  image("images/Lec2_L1_Manhattan_distance.png", width: 45%),
  caption: [$L 1$ Manhattan distance],
)

\
- - *$L 2$ (Euclidean) distance*

~~~~定义为各维差值平方和的平方根：

$
  d_2(I_1, I_2) = sqrt(sum_p (I^p_1 - I^p_2)^2)
$
#figure(
  image("images/Lec2_L2_Euclidean_distance.png", width: 45%),
  caption: [$L 2$ Euclidean distance],
)

\

~~~~如果我们想保留特征的话，通常 $ell_1$ distance 依赖于数据向量的轴，一旦轴旋转，距离就会变化，所以需要找到“特定有意义”的轴作为向量的不同维度；而由于 $ell_2$ 无论轴是否旋转距离值始终不变，所以适用于特征可以任意采取的情况。

\
\

~~~~可以看出 $ell_2$ 的 k-nearest neighbors 得到的边界更加光滑，直观上因为 $ell_2$ 对应等距线是圆，拼接出来的边界显然比 $ell_1$ 使用方形线拼接出来的分类边界更加圆滑。




#figure(
  image("images/Lec2_k_nearest_neighbors_with_L1-metric_comparison.png", width: 100%),
  caption: [$k$-nearest neighbor distance metric comparison],
)

\
\
\
\
- *Setting Hyperparameters*

*Distance metric and $k$ are hyperparameters.*


*Idea 1 :* Choose hyperparameters that work best on the training data. (bad : $k = 1$ always works perfectly on training data)

*Idea 2 :* choose hyperparameters that work best on test data. (bad : cna't generalize)

*Idea 3 :* Split data into train, *_val_*; choose hyperparameters on val and evaluate on test. (only run on the test set once at the very end)

*Idea 4 :* *($k$-fold) Cross-validation* : Split data into folds, _try each fold as validation and average the results_.

~~~~It's useful for small datasets, but not used too frequently in deep learning




~~~~In practice, $k$-Nearest Neighbor with pixel distance is never used because distance metrics on pixels are not informative.





#pagebreak()





=== 2. Linear Classifier

\
Parametric Approach :

$
  f(x, W) = W x + b -> 10 "numbers giving class scores"
$

#figure(
  image("images/Lec2_linear_classifier_illustration.png", width: 80%),
  caption: [Linear Classifier algebraic viewpoint],
)

#figure(
  image("images/Lec2_Linear_classifier_visual_viewpoint.png", width: 60%),
  caption: [Linear Classifier visual viewpoint],
)


~~~~In a geometric viewpoint, linear classifier would find the hyperplanes that classifies different classes in the input space.

#figure(
  image("images/Lec2_Linear_classifier_geometric_viewpoint.png", width: 80%),
  caption: [Linear Classifier geometric viewpoint],
)

\
\
\
\
\
\

- *A simple example : Softmax Classifier*

~~~~我们希望把分类器的原始输出分数 $bold(s)$ 解释为概率。

~~~~给定输入 $bold(x) in RR^D$，线性分类器产生原始分数：

$ bold(s) = f(bold(x); bold(W), bold(b)) = bold(W) bold(x) + bold(b) #h(1em) in RR^K $

~~~~对原始分数向量 $bold(s)$ 应用 softmax 函数，得到概率向量 $bold(p)$：

$ p(Y = k | x) = frac(e^(s_k), sum_(j=1)^K e^(s_j)), quad k = 1, dots, K $

现在我们思考怎样定义目标函数：

\

法一：\

~~~~输出的概率向量可以看作是我们得到的预测概率分布，我们可以计算真实概率分布（对应向量应该是独热的）与这个预测概率分布之间的 KL 散度 $D_(K L)(P||Q)$ 值 ：


$
  D_(K L) (P || Q) & = sum_k P_k log frac(P_k, Q_k) \
                   & = sum_k P_k log P_k - sum_k P_k log Q_k
$

~~~~其中：

$ H(P) = - sum_k P_k log P_k $

是真实分布 $P$ 的熵。

$ H(P, Q) = - sum_k P_k log Q_k $

是交叉熵。

~~~~所以：

$ D_(K L) (P parallel Q) = H(P, Q) - H(P) $

~~~~由于此时 P 是真实标签分布，它不依赖于模型参数。所以 $H(P)$ 是一个常数 :
$ arg min_Q D_(K L)(P parallel Q) = arg min_Q H(P, Q) $

~~~~因此我们实际上就是使用的交叉熵损失函数公式。


\
\
\
\



法二：\

我们用最大似然估计（*MLE*）：

~~~~整个数据集出现的似然是每个样本正确类别概率的连乘：

$ L(W, b) = product_(i=1)^N p(y_i | x_i; W, b) $
~~~~MLE 的目标就是：

$ max_(W,b) L(W, b) = max_(W,b) product_(i=1)^N p(y_i | x_i; W, b) $

~~~~取对数后取负为：
$ -ell(W, b) = -log L(W, b) = -sum_(i=1)^N log p(y_i | x_i; W, b) $

~~~~则最终的目标就是：
$ min_(W,b) -ell(W, b) = min_(W,b) -sum_(i=1)^N log p(y_i | x_i; W,b) $

~~~~之前对单个样本，真实分布 $P_i$​ 是 one-hot，模型预测分布是 $Q_i$，其交叉熵为：
$
  H(P_i, Q_i) & = - sum_(k=1)^K P_(i, k) log Q_(i, k) \
              & = −log p(y_i​ | x_i ​; W ,b)
$

~~~~所以对于这个分类任务，最大似然估计与交叉熵会推出相同的目标函数进行优化。


\
\
\
\
\
\
\

*补充 : hinge loss 与 regularization*
\

~~~~前面讲了 Softmax 分类器：\
~~~~它把线性分数 $s = W x + b$ 通过 Softmax 变成概率，再用交叉熵损失衡量预测分布和真实 one-hot 分布的差距。

~~~~现在换一个思路：\
~~~~我们不一定非要输出概率，也可以只要求正确类别的分数比错误类别的分数高出至少一个间隔。

这就是 SVM（Support Vector Machine）所使用的 合页损失（hinge loss）。

#figure(
  image("images/Lec3_SVM_vs_Softmax.png", width: 100%),
  caption: [Softmax vs. SVM],
)
\

- *Hinge Loss*

~~~~对于第 $i$ 个样本，正确类别是 $y_i$，线性分类器给出的分数是 $s$，SVM 损失定义为：

$ L_i = sum_(j != y_i) max(0, s_j - s_(y_i) + Delta) $

~~~~通常取间隔 $Delta = 1$，所以：

$ L_i = sum_(j != y_i) max(0, s_j - s_(y_i) + 1) $

含义：\
① 如果正确类别的分数 $s_(y_i)$ 比某个错误类别 $s_j$ 高出至少 1，那么这一项为 0，没有损失；\
② 否则，损失等于“还差多少才满足间隔”。

~~~~整个训练集的损失是：

$ L = frac(1, N) sum_(i=1)^N L_i $

~~~~但是我们选用 hinge loss 会出现一个问题，即满足损失值相同的权重值 $W$ 并非唯一！有些时候 $W$ 与 $2 W$ 都会得到相同的损失值！ \

e.g : 一个损失为 0 的例子

假设某个样本的正确类别是 frog，分数如下：

#table(
  columns: 2,
  align: (left, right),
  stroke: none,
  [*类别*], [*分数*],
  [cat], [1.3],
  [frog], [4.9],
  [car], [2.0],
)

~~~~正确类别 frog 的分数是 4.9\
~~~~对 cat 和 car 分别计算：

$ max(0, 1.3 - 4.9 + 1) = max(0, -2.6) = 0 $

$ max(0, 2.0 - 4.9 + 1) = max(0, -1.9) = 0 $

~~~~所以这个样本的 SVM 损失为 0。\
~~~~正确类别已经比错误类别高出超过 1，满足间隔要求。

\

~~~现在把权重 $W$ 变成 $2W$，所有分数也翻倍：

#table(
  columns: 3,
  align: (left, right, right),
  stroke: none,
  [*类别*], [*原分数*], [*$2W$ 下的分数*],
  [cat], [1.3], [2.6],
  [frog], [4.9], [9.8],
  [car], [2.0], [4.0],
)

再算 SVM 损失：

$ max(0, 2.6 - 9.8 + 1) = max(0, -6.2) = 0 $

$ max(0, 4.0 - 9.8 + 1) = max(0, -4.8) = 0 $

\
~~~~我们会发现 $W$ 与 $2 W$ 下损失值均为0！\

~~~~但 $W$, $2 W$ 对应两种截然不同的模型：2W 的权重更大，对输入变化更敏感；可能更容易过拟合；数值上也可能更不稳定。\

~~~~所以最简单的方式就是加入 L2 正则化！！！这样模型会选取 W 而非 2W ！





#pagebreak()








#place(top, scope: "parent", float: true)[
  #align(center + horizon)[  // horizon 让它垂直居中页顶区域，更美观
    #text(font: "Georgia", weight: "bold", size: 24pt)[§ Lec III]  //
    #v(0em)
    #line(length: 100%, stroke: 1pt)  // 可选：加一条装饰线
  ]
]

== Regularization + Optimization
\


=== 1. Regularization


$
  "总损失" #h(0.5em) L(W) & = 1 / N sum_(i=1)^N L_i (f(x_i, W), y_i) + lambda R(W) \
                          & = "Data loss" + "regularization"
$

*$lambda$* : regularization strength (hyperparameter)

*Occam’s Razor* : Among multiple competing hypotheses, _the simplest is the best_.

\

- *Simple examples*

- - *L2 regularization*

$ R_"L2" (W) = sum_k sum_l W_(k,l)^2 = norm(W)_2^2 $


- - *L1 regularization*

$ R_"L1" (W) = sum_k sum_l abs(W_(k,l)) = norm(W)_1 $

- - *Elastic net (L1 + L2)*

$ R_"ElasticNet" (W) = sum_k sum_l (beta (W_(k,l))^2 + abs(W_"k, l")) $


~~~~In practice, we'll get more zero values in the weight matrix $W$ when we use L1 regularization. And L2 regularization allows spread out values that are close to zero but non-zero in $W$.

- - *More complex*

dropout, batch normalization, stochastic depth, fractional pooling, etc.

\

#rect[
  Q : Why regularize?\
  A : ① Express preferences over weights ② Make the model simple so it works on test data ③ Improve optimization by adding curvature

]


#figure(
  image("images/Lec3_regularization_recap_figure.png", width: 100%),
  caption: [loss recap],
)

\
\
\

=== 2. Optimization
\

~~~~*Numerical gradient*: approximate, slow, easy to write\
~~~~*Analytic gradient*: exact, fast, error-prone\
=>\
~~~~In practice: Always use analytic gradient, but check implementation with numerical gradient. This is called a _gradient check_.

\
\
\
- *Stochastic Gradient Descent (SGD)*

- - *Problem #1 with SGD*

~~~~场景：损失函数在一个方向变化快，另一个方向变化慢

~~~~假设参数只有两个：$w_1$ 和 $w_2$。

~~~~损失函数 $L(w_1, w_2)$ 的等高线画出来，是一个狭长的椭圆（如下图）：

#figure(
  image("images/Lec3_SGD_problem1_loss_contour_figure.png", width: 100%),
  caption: [loss contour],
)

~~~~沿着 $w_2$ 方向：等高线很密，这个方向曲率大；沿着 $w_1$ 方向：等高线很稀疏，这个方向*曲率小*。

~~~~当我们使用梯度下降，在“陡峭方向上”（即 $w_2$ 方向上）会来回震荡！

#figure(
  image("images/Lec3_SGD_problem1_contour_route.png", width: 100%),
  caption: [SGD route],
)


~~~~一个全局学习率无法同时适应所有方向：\
~~~~如果将学习率 $alpha$ 调大，让平坦方向走得快一点，那么陡峭方向就会震荡得更厉害，甚至发散；如果将学习率 $alpha$ 调小，让陡峭方向不震荡，那么平坦方向会进展非常缓慢。




~~~~数学上，我们说：Loss function has high condition number: ratio of largest to smallest singular value of the Hessian matrix is large.

#rect[

  补充说明：引入 *Hessian 矩阵 $H$*。

  - $H$ 是损失函数的二阶导数矩阵：
    $ H_(i j) = frac(partial^2 L, partial w_i partial w_j) $
  - 它描述了损失曲面在各个方向上的*曲率*。
  - Hessian 的特征值表示不同方向上的曲率大小：
    - 大特征值 $arrow$ 陡峭方向；
    - 小特征值 $arrow$ 平坦方向。

  *条件数*定义为：

  $ "condition number" = frac(lambda_max, lambda_min) $

  也就是最大曲率除以最小曲率。

  - 如果条件数接近 1，说明各个方向曲率差不多，曲面像个圆碗，梯度下降很好走。
  - 如果条件数很大，说明有的方向极陡，有的方向极平，曲面像个又长又窄的山谷。

  所以条件数很大时优化很难。
]


- - *Problem 2 with SGD*

~~~~场景：梯度下降遇到损失函数局部最小值 (local minima) 或鞍点 (saddle point) 时，梯度为零，参数不再更新，优化停滞。

#figure(
  image("images/Lec3_SGD_problem2_figure.png", width: 60%),
  caption: [SGD problem2 figure],
)

~~~~Saddle points are much more common in higher dimension.

\


- - *Problem 3 with SGD*

~~~~Our gradients come from minibatches so they can be noisy!

\
\
\
\
\
- - *Solution 1 : SGD + Momentum*

*Update rule :* （经典 Momentum 做法）
*$ v_(t+1) & = rho v_t + nabla f(x_t) \
w_(t+1) & = x_t - alpha v_(t+1) $*

① Build up “velocity” as a running mean of gradients\
② $rho$ gives “friction”; typically $rho = 0.9$ or 0.99

#figure(
  image("images/Lec3_classic_momentum_update_figure.png", width: 60%),
  caption: [classic momentum update],
)

~~~~Combine gradient at current point with velocity to get step used to update weights. （是在当前点算梯度，再用速度累积）


~~~~应用 Momentum 后的优化图示 ：
#figure(
  image("images/Lec3_sol1_to_problem1.png", width: 100%),
  caption: [SGD+momentum $->$ poor conditioning],
)

#figure(
  image("images/Lec3_sol1_to_problem2.png", width: 85%),
  caption: [SGD+momentum $->$ local minima+saddle points],
)

#figure(
  image("images/Lec3_sol1_to_problem3.png", width: 60%),
  caption: [SGD+momentum $->$ gradient noise],
)

\
\
\
\
\
\
\
\
\
\
\
\
\
\

*补充 1 ：Nestrov Momentum*

~~~~先用当前速度往前看一步，到那个预测位置算梯度，再把这个梯度和速度混合，得到实际更新方向。

~~~~原始 Nesterov 更新公式：
*$ v_(t+1) & = mu v_t - alpha nabla f(x_t + mu v_t) \
x_(t+1) & = x_t + v_(t+1) $*

~~~~与 classic momentum 的关键区别：\
① classic Momentum：梯度在 $x_t$ 算；\
② Nesterov：梯度在 $x_t + mu v_t$ 算。

#figure(
  image("images/Lec3_Nesterov_momentum_update.png", width: 65%),
  caption: [Nesterov momentum update],
)

~~~~直接使用这个公式不直观：梯度不在当前点 $x_t$​ 算，而在 $x_t + mu v_t$​ 算；这导致更新公式里同时有 $x_t$ 和 $v_t$​
\

~~~~我们不妨考虑变量代换：
$
  tilde(x_t) = x_t + mu v_t
$

~~~~于是写成 ：
$
         v_(t+1) & = mu v_t - alpha nabla f(tilde(x_t)) \
  tilde(x)_(t+1) & = x_(t+1) + mu v_(t+1) = x_t + (1 + mu) v_(t+1) \
                 & = (tilde(x_t) - mu v_t) + (1 + mu) v_(t+1) \
                 & = tilde(x_t) + v_(t+1) + mu(v_(t+1) - v_t)
$

~~~~这个形式和原始 Nesterov 完全等价。

~~~~实现时跟踪 $tilde(x)_t$，只需保存旧速度 $v_t$；\
~~~~梯度仍在 $tilde(x)_t$ 点计算；\
~~~~更新表达式更接近“梯度 + 速度 → 实际步”的统一框架；\
~~~~若要回到原始参数，可用 $x_t = tilde(x)_t - mu v_t$ 换算。



#pagebreak()

*补充 2 ：AdaGrad*
\

~~~~前面的两种 Momentum 方法是在参数更新方向上进行优化，它们解决的是梯度方向来回震荡、收敛慢的问题。
\

~~~~但它们有一个共同点：所有参数维度共享同一个学习率 $alpha$.
\

~~~~于是现在我们考虑参数不同维度各自具有不同的学习率：即自适应学习率方法。

\
\

~~~~AdaGrad 的做法：\
① 对每个参数维度，单独累积历史梯度平方；\
② 用这个累积量去逐元素缩放梯度。

~~~~设梯度：

$ g_t = nabla f(x_t) $

~~~~AdaGrad 维护一个和参数同维度的累积变量 $s_t$，初始为 0：

$ s_t = s_(t-1) + g_t dot.o g_t $

~~~~注意这里是累加，不是平均，也不是指数衰减。\
~~~~所以 $s_t$ 的每个分量，就是对应维度从训练开始到现在的梯度平方和。

~~~~参数更新公式：

$ x_(t+1) = x_t - alpha frac(g_t, sqrt(s_t) + epsilon) $

~~~~逐维度写：

$ x_(t+1,i) = x_(t,i) - alpha frac(g_(t,i), sqrt(s_(t,i)) + epsilon) $

~~~~因此每个维度的有效学习率是：

$ alpha_i = frac(alpha, sqrt(s_(t,i)) + epsilon) $

~~~~这就是 per-parameter learning rates 或 adaptive learning rates。
\

~~~~最终使得陡峭方向（梯度大）的进展被抑制；平坦方向（梯度小）的进展被加速。



~~~~但由于 $s_t$ 是单调累积变量，所以最终更新步长会衰减到0。这导致训练后期参数几乎不再更新，模型可能过早停止优化！

\
\
\

~~~~于是这便引出了 RMSProp 对于 AdaGrad 改进的核心思想：\

① AdaGrad 的问题根源在于：它把所有历史梯度平方都累加起来，从不遗忘。

一个自然的改进思路是：

② 不要累加所有历史，而是只保留最近一段时间的梯度平方，让旧信息慢慢“漏掉”。（也就是引入一个“记忆衰减系数”！）

~~~~因此 RMSProp 也被称为 "Leaky AdaGrad".

#figure(
  image("images/Lec3_AdaGrad_to_RMSProp.png", width: 100%),
  caption: [AdaGrad $->$ RMSProp],
)





#pagebreak()





- - *More complex optimizers : RMSProp*

~~~~RMSProp 不看梯度的“方向”，而是看每个参数维度上梯度的“历史大小”：\
~~~~梯度一直大的维度，这个方向上容易震荡，于是减小其学习率；梯度一直小的维度，这个方向上较为平坦，于是增大其学习率。\
~~~~这样每个参数都有自己独立的学习率。

~~~~RMSProp 的具体更新公式 ：

~~~~设：\
#let hadamard = math.op("⊙")
~~~~$beta$：衰减率，通常取 $0.9$ 或 $0.99$（$beta$ 越大越关注较远的历史）；\
~~~~$s_t$：缓存变量，和参数同维度，初始为 $arrow(0)$；
~~~~$hadamard$：逐元素乘法。


*$ g_t = nabla_w L(w_t) $*

*$ s_t = beta s_(t-1) + (1-beta) g_t hadamard g_t $*

*$ w_(t+1) = w_t - eta frac(g_t, sqrt(s_t) + epsilon) $*

其中根号、除法都是逐元素操作。



```python
cache = 0
for t in range(steps):
    g = compute_gradient(w)
    cache = beta * cache + (1 - beta) * g * g
    w = w - learning_rate * g / (np.sqrt(cache) + eps)
```


\


~~~~普通 SGD 更新：

$ w_(t+1) = w_t - eta g_t $

~~~~所有参数共用一个学习率 $eta$。

~~~~RMSProp 不同，它对每个维度单独缩放：

$ w_(t+1,i) = w_(t,i) - frac(eta, sqrt(s_(t,i)) + epsilon) g_(t,i) $

~~~~每个参数 $w_i$ 的有效学习率是：

$ eta_i = frac(eta, sqrt(s_(t,i)) + epsilon) $

\
\

- - *Adam $approx$ Momentum + RMSProp*
\
*1 ) Adam (almost form) :*
#figure(
  image("images/Lec3_Adam(almost).png", width: 100%),
  caption: [Adam (almost)],
)

*Step 1 :*
$ m_t = beta_1 m_(t-1) + (1 - beta_1) g_t $

~~~~$g_t$ 是当前的梯度值，所以这一步是在更新动量 $m_t$ 的值（梯度的一阶矩），记录梯度的平均方向。


*Step 2 :*
$ v_t = beta_2 v_(t-1) + (1 - beta_2) g_t^2 $

~~~~$v_t$ 记录的是每个参数维度上梯度过去有多大（梯度平方的二阶矩）。

*Step 3 (update) :*
$ x arrow.l x - alpha frac(m_t, sqrt(v_t) + epsilon) $

~~~~最后一步就是用 Momentum 的方向，除以 RMSProp 的缩放因子（逐维度除法来缩放），最后得到每个参数的实际更新量。

\
\

*2 ) Adam (full form) :*

#figure(
  image("images/Lec3_Adam(full_form).png", width: 100%),
  caption: [Adam (full form)],
)

~~~~完整版 Adam 在 Momentum 与 RMSProp 之间新增关键一步 ：偏差校正

$ hat(m)_t = frac(m_t, 1 - beta_1^t) $

$ hat(v)_t = frac(v_t, 1 - beta_2^t) $

~~~然后最后在 RMSProp 处用校正后的值更新参数：

$ x_(t+1) = x_t - alpha frac(hat(m)_t, sqrt(hat(v)_t) + epsilon) $


~~~~The full form of Adam adds *bias correction* for the fact that first and second moment estimates start at zero.\

\
\
\

Adam (full form) 新增的偏差校正背后的原理 ：
\

1. 指数加权平均本身带了偏差

~~~~由于：

$ m_t = beta m_(t-1) + (1 - beta) g_t $

~~~~且初始 $m_0 = 0$，展开得 ：

$
  m_t & = (1 - beta) g_t + beta (1 - beta) g_(t-1) + beta^2 (1 - beta) g_(t-2) + dots \
      & = (1 - beta) sum_(i=1)^t beta^(t-i) g_i
$

~~~~而“真正的”加权平均应该是权重归一化的：

$ "真实加权平均" = frac(sum_(i=1)^t beta^(t-i) g_i, sum_(i=1)^t beta^(t-i)) $

~~~~这里的分母 ：

$ sum_(i=1)^t beta^(t-i) = frac(1 - beta^t, 1 - beta) $

~~~~所以 “真实的加权平均” 为：

$
  hat(m)_t = frac(m_t, 1 - beta_1^t)
$

~~~~这就是偏差的来源。\
当 $t$ 很大时，$(1 - beta^t) -> 1$，偏差消失。\
当 $t$ 很小时，比如 $t = 1$，$(1 - beta^1) = 1 - beta = 0.1$，偏差非常严重。

~~~~$v_t$ 同理





2. 偏差校正就是除以这个系数

$
  hat(m)_t & = frac(m_t, 1 - beta_1^t) = "真实一阶矩" \
  hat(v)_t & = frac(v_t, 1 - beta_2^t) = "真实二阶矩"
$

~~~~偏差校正是为了在数学上恢复到无偏估计。

\
\
\
\
\
3. 如果不做校正，会出什么问题？

~~~~之前我们的参数更新量是：

$ Delta x = alpha frac(m_t, sqrt(v_t) + epsilon) $

~~~~因为 $m_t$ 和 $v_t$ 相比真实的一阶、二阶矩都被缩小了，所以这个比例式的值会受到因子：

$ frac(1 - beta_1^t, sqrt(1 - beta_2^t)) $

的影响。

~~~~这个因子在训练初期可能很大（比如 $t=1, beta_1 = 0.9, beta_2 = 0.999$ 时约 3.16），也可能很小。
它完全取决于所选的 $beta_1, beta_2$。
\
\

~~~~这就带来几个问题：

① 早期步长不可控 : 实际第一步步长是：$3.16 alpha$

② 对超参数过于敏感 ：不同 $beta_1, beta_2$ 会导致不同的初始偏差。没有校正，Adam 的表现会非常依赖这些超参数，调参困难。

③ 早期训练不稳定 ：如果某个维度的梯度很小，$v_1$ 可能比 $epsilon$ 还小，分母被 $epsilon$ 主导，步长又会变得非常小。
所以早期更新量可能一会儿很大、一会儿很小，不稳定。

\
\
\
\

4. 校正后的好处

~~~~校正后：

$ Delta x_i = alpha frac(g_(1,i), abs(g_(1,i)) + epsilon) approx alpha dot op("sign")(g_(1,i)) $

~~~~每个维度的第一步步长都接近 $alpha$，与梯度大小无关，与 $beta_1, beta_2$ 无关。这就让：\
~~~~初期步长可控；训练稳定；对超参数不敏感。
\
\
\


#text(fill: red)[
  ~~~~*_Adam with $beta_1 = 0.9, beta_2 = 0.999$ and $alpha("learning rate") = 1e-3 "or" 5e-4$ is a great starting point for many models!_*
]


#figure(
  image("images/Lec3_Adam_optimization_route_figure.png", width: 100%),
  caption: [Adam optimization route],
)

\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
- - *AdamW (Adam variant with weight decay)*
\
*1. *什么是 *weight decay* ? Why weight decay ?\

~~~~Weight decay 实际上就是我们希望训练得到的权重 $W$ 不要过大，过大的权重实际上是对某些数值特别敏感，也会导致数值不稳定（激活值、梯度爆炸），也会导致模型过拟合。所以 weight decay 要做的就是一直把模型权重向着原点方向拉动。
\
\
\
\

*2. SGD 中正则项与 weight decay 之间的关系？*

~~~~在 SGD 中如果我们加入了 L2 正则项 ：
$
  L = L_"data" ​+ λ / 2 ||w||^2
$
~~~~此时训练梯度变为 ：
$
  g & = nabla_w L \
    & = nabla_w L_"data" + lambda w \
    & = g_"data" + lambda w
$

~~~~SGD 更新中 ：
$ w arrow.l w - alpha g_"data" - alpha lambda w $

~~~~最后一项就是把 $w$ 向着0方向拉动，并且对每个参数$w_t$都是$- alpha lambda w_t$​，和梯度大小无关，即#underline[每个参数按同样比例衰减！！！]

~~~~所以说 SGD 中的 L2 正则项就具有 weight decay 的作用！

\
\
\
\
\
*3. Standard Adam 中使用 L2 正则项 ？*

~~~~计算梯度为 ：

$ g_t = nabla_w L_"data" + lambda w_t $

~~~~这个 $g_t$ 被送进 Adam 然后做参数更新 ：

$ m_t = beta_1 m_(t-1) + (1 - beta_1) g_t $
$ v_t = beta_2 v_(t-1) + (1 - beta_2) g_t^2 $
$ hat(m)_t = frac(m_t, 1 - beta_1^t) $
$ hat(v)_t = frac(v_t, 1 - beta_2^t) $
$ w_(t+1) = w_t - eta frac(hat(m)_t, sqrt(hat(v)_t) + epsilon) $

~~~~假设训练已经稳定，$m_t approx g_t$​，$v_t approx g_t^2$\

~~~~那么更新量近似为：

$ Delta w approx - eta frac(g_"data" + lambda w, sqrt((g_"data" + lambda w)^2 + epsilon)) $

如果 $abs(g_"data")$ 远大于 $lambda w$，那么：

$
  Delta w & approx - eta frac(g_"data" + lambda w, abs(g_"data")) \
          & approx - eta frac(g_"data", abs(g_"data")) - eta frac(lambda w, abs(g_"data"))
$


~~~~第二项就是在做 weight decay，但此时：\

~~~~梯度大的参数，$|g_"data"|$ 大，分母大，weight decay 作用被压小；\
~~~~梯度小的参数，$g_"data"$ 小，分母小，weight decay 作用被放大。

~~~~本来是要对参数均匀正则化，但现在“梯度小的参数被过度正则化，梯度大的参数几乎不正则化”，这完全违背了 weight decay 的初衷！

\
\
\
\
\
\
\
\
\
\

*4. AdamW 的做法*

~~~~AdamW 把 weight decay 从梯度里拿出来：

$ g_t = g_"data" $

~~~~先做纯 Adam 更新：

$ w_(t+1) = w_t - eta frac(hat(m)_t, sqrt(hat(v)_t) + epsilon) $

~~~~然后单独做 weight decay：

$ w_(t+1) arrow.l w_(t+1) - eta lambda w_t $

~~~~这样 weight decay 不进入 $m_t$、$v_t$，不受 $sqrt(hat(v)_t)$ 缩放。
每个参数都按同样的比例衰减，恢复到 SGD 里 weight decay 的行为。

\
\

#figure(
  image("images/Lec3_Adam_AdamW_comparison.png", width: 80%),
  caption: [Standard Adam with L2 vs. AdamW],
)




#pagebreak()





- *Hyperparameter : Learning rate*
\
#figure(
  image("images/Lec3_loss_figure_with_different_learning_rates.png", width: 70%),
  caption: [different learning rates],
)

~~~~事实上，这些都可以成为好的学习率，因为在现代深度学习中我们在训练时可以切换改变好几种学习率。

\
\
\
\
\
- - *Learning rate decays over time*

~~~~As training continues, current learning rate is too high to converge any further. So we need learning rate decay over time.

\
*Choice 1. Reduce learning rate at a few fixed points.*\
e.g. For ResNets, multiply LR by 0.1 after epochs 30, 60, and 90.

#figure(
  image("images/Lec3_learning_rate_decay_1.png", width: 70%),
  caption: [learning rate decay choice 1],
)

\

*Choice 2. Cosine function :*
*$ alpha_t = 1/2 alpha_0 (1 + cos((t pi) / T)) $*

#figure(
  image("images/Lec3_learning_rate_decay_2.png", width: 70%),
  caption: [learning rate decay choice 2],
)

~~~~The corresponding loss figure will look like this :

#figure(
  image("images/Lec3_training_loss_figure_under_learning_rate_decay_2.png", width: 70%),
  caption: [traning loss figure under Cosine learning rate],
)

\
\
*Choice 3. Linear decay*

$
  alpha_t = alpha_0 (1 - t/T)
$
#figure(
  image("images/Lec3_learning_rate_decay_3.png", width: 70%),
  caption: [learning rate decay choice 3],
)


\
\

*Choice 4. Inverse sqrt decay*

$
  alpha_t = alpha_0 / (sqrt(t))
$
#figure(
  image("images/Lec3_learning_rate_decay_4.png", width: 70%),
  caption: [learning rate decay choice 4],
)


\

- - *Linear warmup*

#figure(
  image("images/Lec3_learning_rate_decay_linear_warmup.png", width: 70%),
  caption: [linear warmup],
)

~~~~High initial learning rates can make loss explode; linearly increasing learning rate from 0 over the first ~5,000 iterations can prevent this.

\

#rect[
  *Empirical rule of thumb* :\
  ~~~~*_If you increase the batch size by $N$, then also scale the initial learning rate by $N$_*
]

\
\
\
\
\
\
\
\
\


- *Second-order Optimization*
\
- - *Use gradient form linear approximation*

~~~~在点 $w_t$ 处，我们用一阶泰勒展开对损失做局部线性近似：

$ L(w) approx L(w_t) + nabla L(w_t)^top (w - w_t) $

~~~~这个线性近似没有全局最小值，由于负梯度方向是当前点处使线性近似下降最快的方向，所以我们选择负梯度方向作为下降方向，并取一个步长 $alpha$ （线性近似只在局部有效，所以步长不能过大）：

$ w_(t+1) = w_t - alpha nabla L(w_t) $

~~~~每一步只使用一阶导数信息，因此这类方法称为一阶优化 (first-order optimization)

#figure(
  image("images/Lec3_first-order_optimization.png", width: 80%),
  caption: [first-order optimization],
)

\
\
\
\
- - *Use gradient and Hessian to form quadratic approximation*

~~~~Second-order Taylor expansion at $theta_0$:

$ J(theta) approx J(theta_0) + (theta - theta_0) nabla_theta J(theta_0) + 1/2 (theta - theta_0)^T H (theta - theta_0) $

~~~~其中 $H = nabla_theta^2 J(theta_0)$ 是 Hessian 矩阵，满足 ：
$ H_(i j) = frac(partial^2 J, partial theta_i partial theta_j) $

~~~~对近似函数关于 $theta$ 求梯度：

$ nabla_theta [ J(theta_0) + (theta - theta_0)^top nabla J(theta_0) + 1/2 (theta - theta_0)^top H (theta - theta_0) ] $

~~~~逐项求导：

$J(theta_0)$ 是常数，导数为 $0$；\
$(theta - theta_0)^top nabla J(theta_0)$ 对 $theta$ 求导得 $nabla J(theta_0)$；\
$1/2 (theta - theta_0)^top H (theta - theta_0)$ 对 $theta$ 求导得 $H (theta - theta_0)$。

~~~~所以：

$ nabla J(theta_0) + H (theta - theta_0) = 0 $

~~~~解出：

$ H (theta - theta_0) = - nabla J(theta_0) $

*$ theta^* = theta_0 - H^(-1) nabla J(theta_0) $*

~~~~这就是牛顿法更新公式。

#figure(
  image("images/Lec3_second-order_optimization.png", width: 85%),
  caption: [second-order optimization],
)

~~~~二阶优化实际上就是按照二阶泰勒展开用一个二次曲面来近似当前点处的原损失函数曲面，然后优化结果就是直接使参数到达二次曲面的最低点。\
~~~~这样看来，二阶优化仿佛与一阶优化不同，仿佛没有步长的限制！（一阶优化线性近似没有最低点，所以才有步长的限制）当然我们也可以人为加上一个步长限制，使用“阻尼牛顿法”。

\
~~~~相比之下，Adam 实际就是一种“廉价”的二阶近似。因为 Adam 优化中使用$frac(1, sqrt(v_t) + epsilon)$ 逐维度缩放梯度 $hat(m)_t$，而牛顿法是在使用 $H^(-1)$ 缩放梯度，自适应步长。



\
#rect[
  ~~~~However, this is *_bad_* for deep learning !\
  ~~~~Because Hessian $H$ has $O(N^2)$ elements, inverting takes $O(N^3)$, and $N$ is a very large number in deep learning ! （参数过多）
]




*BFGS , L-BFGS* :\

~~~~由于严格牛顿法的 $H^(-1)$ 取逆参数爆炸问题，我们换而维护一个近似矩阵 $B approx H^(-1)$ ，然后用 $B$ 代替 $H^(-1)$ 做参数更新：
*$ theta_(t+1) = theta_t - B_t nabla J(theta_t) $*

我们令以下符号：\
~~~~$s_t = theta_(t+1) - theta_t$：参数变化；\
~~~~$y_t = nabla J(theta_(t+1)) - nabla J(theta_t)$：梯度变化；\
~~~~$y_t^top s_t > 0$（曲率条件成立）。

~~~~我们的近似逆矩阵的计算（更新）公式为：

$
  B_(t+1) = lr(I - frac(s_t y_t^top, y_t^top s_t)) B_t lr(I - frac(y_t s_t^top, y_t^top s_t)) + frac(s_t s_t^top, y_t^top s_t)
$

~~~~满足割线方程：

$ B_(t+1) y_t = s_t $

~~~~这样每次矩阵取逆操作数从 $O(N^3)$ 降为 $O(N^2)$ 就可以得到一个近似的逆矩阵。
\

~~~~这个方法：BFGS 仍然要存一个 $N times N$ 的近似逆 Hessian，参数量仍有 $O(N^2)$，当 $N$ 很大时，参数量也撑不住。
\

~~~~于是还有进一步的方法：L-BFGS（Limited memory BFGS），其思路是：不存完整的近似逆 Hessian，只存最近 $m$ 步的梯度差和参数差。用这些历史信息隐式地计算 $H^(-1) nabla J$，而不显式构造矩阵。这样参数量降为 $O(m N)$. 但注意 L-BFGS 主要适用于全批量模式（full-batch mode），对于 mini-batch 而言其数据噪声大，近似出的梯度不准确。




\


#rect[
  *In practice* :
  \

  ① *Adam(W)* is a good default choice in many cases; it
  often works ok even with constant learning rate\

  ② *SGD + Momentum* can outperform Adam but may require more tuning of LR and schedule\

  ③ If you can afford to do full batch updates then try out *L-BFGS* (and don't forget to disable all sources of noise).
]












#pagebreak()
















