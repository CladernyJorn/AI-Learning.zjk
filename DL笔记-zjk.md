# 《DeepLearning》笔记 

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
