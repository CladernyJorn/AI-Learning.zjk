# Deep Learning 笔记 

## 一.数学基础

（只含补充内容和符号定义）

#### 1.线性代数部分

**张量**：超过二维的数组，一般用A表示
$$
张量\textbf{A}中坐标为(i,j,k)的元素为\textbf{A}_{i,j,k}
$$
**广播**：将一个向量**b**加到矩阵**A**的每一**行**
$$
记为\textbf{C}=\textbf{A}+\textbf{b}\quad (C_{i,j}=A_{i,j}+b_j)
$$
**元素对应乘积/Hadamard乘积**：
$$
\textbf{A}\odot\textbf{B}
$$
**范数**：
$$
p范数L^p:||x||_p=(\Sigma|x_i|^p)^{\frac{1}{p}},p\geq1\\
欧几里得范数L^2:||x||_2=xx^T\\
最大范数L^{\infty}:||x||_\infty=max|x_i|\\
Frobenius矩阵范数：||\textbf{A}||_F=\sqrt{\Sigma A_{i,j}^2}
$$

- 一般考虑每个元素的倒数和整个向量大小时采用2-范数（欧几里得范数），但是它在原点增长缓慢
- 若要考虑向量中零和非零元素时，可以使用1-范数（即元素绝对值之和）

**奇异值分解SVD**：对非方阵的矩阵进行相似对角化操作，相似变换矩阵是正交可逆方阵。

**Moore-Penrose伪逆**：对一个非方阵矩阵取“逆”（“逆”是在求解线性方程组下定义的），他与原矩阵的向量解不等价，并且伪逆矩阵的解在矩阵行列数不同关系下具有不同的特点（原方程有无穷解时，逆矩阵所求的解是2-范数最小的解；原方程无解时，所求的是使原方程差值向量的2-范数最小的解）。

**迹运算**：即对角元素之和，具有循环转换不变性，并且定义标量的迹是它自己。
$$
Tr(\textbf{A})=\Sigma A_{i,i}
$$
可用迹运算表示F-范数：
$$
||\textbf{A}||_F=\sqrt{Tr(\textbf{AA}^T)}
$$
**主成分分析**：对m个点的n个特征降维，变成m个点的l维数据。（l<n)
$$
\begin{align}
&PCA推导\\
&步骤1:对一个点向量x求解最优解码码矩阵D(x^*=Dc)\\
&\quad\quad\quad即求解c=argmin_c||x-Dc||\\
&\quad\Rightarrow c=D^Tx,即用一个l*n的矩阵对n*1的x进行降维至l*1的c\\
&步骤2:对总体m个点的矩阵X(m*n)，求解（降维到1维）最优编码矩阵d(n*1),其中||d||=1且d^Td=I_l\\
&\quad\quad\quad即求解d=argmin_d(||X-Xdd^T||_F)\\
&\quad\Rightarrow d=argmax_dTr(d^TX^TXd) ,d为X^TX最大特征值对应的单位特征向量\\
&步骤3:对降维到l的情况，使归纳可以证明D为由X^TX前l个最大特征值的特征向量组成的规范正交矩阵
\end{align}
$$

##### **矩阵求导**

梯度：
$$
对m*n矩阵A，f:\R^{m*n}\rightarrow R\\
\frac{\partial f(A)}{\partial A}=\nabla_Af(A)=
\begin{bmatrix}
\frac{\partial f(A)}{\partial A_{11}}&\frac{\partial f(A)}{\partial A_{12}}&\cdots&\frac{\partial f(A)}{\partial A_{1n}}\\
\frac{\partial f(A)}{\partial A_{21}}&\frac{\partial f(A)}{\partial A_{22}}&\cdots&\frac{\partial f(A)}{\partial A_{2n}}\\
\vdots&\vdots&\ddots&\vdots\\
\frac{\partial f(A)}{\partial A_{m1}}&\frac{\partial f(A)}{\partial A_{m2}}&\cdots&\frac{\partial f(A)}{\partial A_{mn}}
\end{bmatrix}
$$
向量梯度即矩阵梯度的特例：
$$
对向量x：\\
\nabla_xf(x)=
\begin{bmatrix}
\frac{\partial f(x)}{\partial x_{1}} \\
\frac{\partial f(x)}{\partial x_{2}} \\
\vdots \\
\frac{\partial f(x)}{\partial x_{n}} 
\end{bmatrix}\\
$$
一些结论：
$$
\frac{\partial b^Tx}{\partial x}=b\\
\frac{\partial a^TXb}{\partial X}=ab^T\\
\frac{\partial a^TX^Tb}{\partial X}=ba^T\\
\frac{\partial x^TBx}{\partial x}=(B+B^T)x\\



$$
Hessian矩阵：
$$
对n维向量x，对应Hessian矩阵为n*n矩阵，记为：\\
H(x)=\nabla _x^2f(x)=
\begin{bmatrix}
\frac{\partial^2 f(x)}{\partial x_{1}^2}&\frac{\partial^2 f(x)}{\partial x_{1}\partial x_{2}}&\cdots&\frac{\partial^2 f(x)}{\partial x_{1}\partial x_{n}}\\
\frac{\partial^2 f(x)}{\partial x_{2}\partial x_{1}}&\frac{\partial^2 f(x)}{\partial x_{2}^2 }&\cdots&\frac{\partial^2 f(x)}{\partial x_{2}\partial x_{n}}\\
\vdots&\vdots&\ddots&\vdots\\
\frac{\partial^2 f(x)}{\partial x_{n}\partial x_{1}}&\frac{\partial^2 f(x)}{\partial x_{n}\partial x_{2}}&\cdots&\frac{\partial^2 f(x)}{\partial x_{n}^2}
\end{bmatrix}\\
重要结论：对f(x)=x^TAx,	其中x为n维向量，A为n阶方阵，有\\
\nabla_x^2f(x)=2A
$$

#### 

#### 2.概率论与常用生成函数

**独立**：
$$
x\bot y:x与y独立\\
x\bot y|z:x	与y条件独立（在z条件下，x与y独立）\\
$$
**Laplace分布**：指数分布平移后对称到两侧的分布（即对称的指数分布）：
$$
Laplace(x;\mu ,\gamma)=\frac{1}{2\gamma}e^{-\frac{|x-\mu|}{\gamma}}
$$
**Dirac分布**：在某一点越变的分布（即离散变量分布写为概率密度函数）：
$$
记为：p(x)=\delta (x-\mu)
$$
**Logistic sigmoid函数**：即生物模型里的函数，用于生成伯努利分布的p值
$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$
**softplus函数**：max(0,x)的平滑函数，可用于生成正态分布的方差
$$
\zeta(x)=log(1+e^x)
$$

## 二.（广义）线性回归Regression

**线性回归定义**：向量x，y为x的真实值，表示输入，$\hat y$表示x的预测标量，定义$\hat y=h_\theta(x)=\theta^Tx+b，其中w记为参数向量$

若有m个输入样本 $\{x_1,x_2\dots\},\{y_1,y_2\dots\}$ ,使用均方误差表示线性回归模型的效果，定义:（基本原理来源于中心极限定理+最大似然估计）
$$
cost\quad function :\quad J(\theta)=\frac{1}{2}\sum_{i=0}^n(h_\theta(x_i)-y_i)^2
$$

**Logistic回归**：对于离散的二分类目标，可以使用sigmoid函数作为预测函数：
$$
\hat y=h_\theta(x)=g(\theta^T x)=\frac{1}{1+e^{-\theta^Tx}}
$$
类似的，使用伯努利概率模型定义它的损失函数：
$$
L(\theta)=\sum_{i=1}^n(y_ilog(h_\theta(x_i))+(1-y_i)log(1-h_\theta(x_i)))
$$
最后将线性模型的分类分布用于二项分布后，能够得到**softmax**函数，这一部分的公式和推导在CVLearning中有详细阐述。（使用较为广泛）

#### 优化算法

- **最小均方规则-LMS算法**

最优化求$J(\theta)$最小值，使用梯度下降方法$\theta\leftarrow \theta -\alpha\frac{\partial}{\partial \theta}J(\theta)$
$$
\begin{align}
\frac{\partial}{\partial \theta}J(\theta)=&\frac{\partial}{\partial \theta}\frac{1}{2}\sum_{i=0}^n(\theta^T x_i-y_i)^2\rightarrow\frac{\partial}{\partial \theta}\frac{1}{2}\sum_{i=0}^n(\theta^T x_i-y_i)^T(\theta^T x_i-y_i)\\
=&(h_\theta(x)-y)\cdot x\\
&注意：这个结论对于linear-regression和logistic均成立，事实上使用概率模型处理的损失函数都有这样的形式
\end{align}
$$

- **牛顿法**，提高梯度下降学习步长

$$
\theta\leftarrow \theta -H^{-1}\nabla_\theta J(\theta),	其中H^{-1}表示J(\theta)的Hession矩阵的逆
$$

#### **总结**

——使用的回归类型决定于最终分类呈现的概率分布

分类为**高斯**分布的使用**线性回归**；**伯努利**分布的使用**logistic回归（sigmoid）**；**二项**分布的使用**softmax**。









