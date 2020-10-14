# Kalman filter library

## Introduction 介绍 
The kalman filter framework described here is an incredibly powerful tool for any optimization problem,
but particularly for visual odometry, sensor fusion localization or SLAM. It is designed to provide very
accurate results, work online or offline, be fairly computationally efficient, be easy to design filters with in
python.

此处描述的卡尔曼滤波器框架是解决任何优化问题的强大工具，特别是对于视觉里程表，传感器融合定位或SLAM。
它旨在提供非常准确的结果，可以在线或离线工作，计算效率相当高，易于在python中设计过滤器。

![](examples/kinematic_kf.png)


## Feature walkthrough

### Extended Kalman Filter with symbolic Jacobian computation 具有符号雅可比计算的扩展卡尔曼滤波器 
Most dynamic systems can be described as a Hidden Markov Process. To estimate the state of such a system with noisy
measurements one can use a Recursive Bayesian estimator. For a linear Markov Process a regular linear Kalman filter is optimal.
Unfortunately, a lot of systems are non-linear. Extended Kalman Filters can model systems by linearizing the non-linear
system at every step, this provides a close to optimal estimator when the linearization is good enough. If the linearization
introduces too much noise, one can use an Iterated Extended Kalman Filter, Unscented Kalman Filter or a Particle Filter. For
most applications those estimators are overkill. They add a lot of complexity and require a lot of additional compute.

Conventionally Extended Kalman Filters are implemented by writing the system's dynamic equations and then manually symbolically
calculating the Jacobians for the linearization. For complex systems this is time consuming and very prone to calculation errors.
This library symbolically computes the Jacobians using sympy to simplify the system's definition and remove the possibility of introducing calculation errors.

大多数动态系统可以描述为隐马尔可夫过程。为了用噪声测量来估计这种系统的状态，可以使用递归贝叶斯估计器。对于线性马尔可夫
过程，常规线性卡尔曼滤波器是最佳的。不幸的是，许多系统都是非线性的。扩展卡尔曼滤波器可以通过线性化非线性系统的每一步来
对系统建模，当线性化足够好时，这可以提供接近最佳的估计量。如果线性化引入了过多的噪声，则可以使用迭代扩展卡尔曼滤波器，
无味卡尔曼滤波器或粒子滤波器。在大多数应用中，这些估计量是过大的。它们增加了很多复杂性，并需要大量额外的计算。 传统的
扩展卡尔曼滤波器是通过编写系统的动态方程，然后手动符号化地计算雅可比矩阵以实现线性化来实现的。对于复杂的系统，这很耗时
并且很容易产生计算错误。该库使用sympy象征性地计算Jacobian值，以简化系统的定义并消除引入计算错误的可能性。

### Error State Kalman Filter 状态误差卡尔曼滤波器
3D localization algorithms usually also require estimating orientation of an object in 3D. Orientation is generally represented
with euler angles or quaternions.

Euler angles have several problems, there are multiple ways to represent the same orientation,
gimbal lock can cause the loss of a degree of freedom and lastly their behaviour is very non-linear when errors are large.
Quaternions with one strictly positive dimension don't suffer from these issues, but have another set of problems.
Quaternions need to be normalized otherwise they will grow unbounded, this is cannot be cleanly enforced in a kalman filter.
Most importantly though a quaternion has 4 dimensions, but only represents 3 degrees of freedom, so there is one redundant dimension.

Kalman filters are designed to minimize the error of the system's state. It is possible to have a kalman filter where state and the error of the state are represented in a different space. As long as there is an error function that can compute the error based on the true state and estimated state. It is problematic to have redundant dimensions in the error of the kalman filter, but not in the state. A good compromise then, is to use the quaternion to represent the system's attitude state and use euler angles to describe the error in attitude. This library supports and defining an arbitrary error that is in  a different space than the state. [Joan Solà](https://arxiv.org/abs/1711.02508) has written a comprehensive description of using ESKFs for robust 3D orientation estimation.

3D定位算法通常还需要估计3D对象的方向。方向通常用欧拉角或四元数表示。 欧拉角有几个问题，有多种方法可以表示相同的方向，
万向节锁定会导致自由度的损失，最后，当误差较大时，它们的行为非常非线性。具有严格正维的四元数不会遇到这些问题，但会遇
到另一组问题。需要对四元数进行规范化，否则它们将无限制地增长，这不能在卡尔曼过滤器中明确实施。最重要的是，虽然四元数
具有4个维度，但仅表示3个自由度，因此存在一个冗余维度。 卡尔曼滤波器旨在最大程度地减少系统状态的误差。可能有一个卡尔曼
滤波器，其中状态和状态错误在不同的空间中表示。只要有一个误差函数可以根据真实状态和估计状态计算误差。在卡尔曼滤波器的
误差中具有冗余尺寸而在状态中没有冗余尺寸是有问题的。因此，一个好的折衷方法是使用四元数表示系统的姿态状态，并使用欧拉角
描述姿态误差。该库支持并定义与状态不同的任意错误。 JoanSolà撰写了有关使用ESKF进行可靠的3D方向估计的全面说明。

### Multi-State Constraint Kalman Filter
How do you integrate feature-based visual odometry with a Kalman filter? The problem is that one cannot write an observation equation for 2D feature observations in image space for a localization kalman filter. One needs to give the feature observation a depth so it has a 3D position, then one can write an obvervation equation in the kalman filter. This is possible by tracking the feature across frames and then estimating the depth. However, the solution is not that simple, the depth estimated by tracking the feature across frames depends on the location of the camera at those frames, and thus the state of the kalman filter. This creates a positive feedback loop where the kalman filter wrongly gains confidence in it's position because the feature position updates reinforce it.

The solution is to use an [MSCKF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.437.1085&rep=rep1&type=pdf), which this library fully supports.

多状态约束卡尔曼滤波器 您如何将基于特征的视觉里程表与卡尔曼滤波器集成在一起？问题在于，无法为本地化卡尔曼滤波器编写图像
空间中2D特征观测的观测方程。一个人需要给特征观察一个深度，使其具有3D位置，然后可以在卡尔曼滤波器中编写一个观察方程。通过
跨帧跟踪特征，然后估算深度，这是可能的。但是，解决方案不是那么简单，通过跨帧跟踪特征估计的深度取决于摄像机在那些帧处的位
置，并因此取决于卡尔曼滤波器的状态。这会创建一个正反馈回路，在该回路中，卡尔曼滤波器会错误地获得对其位置的信心，因为特征
位置更新会对其增强。 解决方案是使用此库完全支持的MSCKF。

### Rauch–Tung–Striebel smoothing
When doing offline estimation with a kalman filter there can be an initialization period where states are badly estimated.
Global estimators don't suffer from this, to make our kalman filter competitive with global optimizers we can run the filter
backwards using an RTS smoother. Those combined with potentially multiple forward and backwards passes of the data should make
performance very close to global optimization.

Rauch-Tung-Striebel平滑 使用卡尔曼滤波器进行离线估算时，可能会有一个初始化阶段，其中状态估计不正确。全局估计器不受此影响，
为了使我们的卡尔曼滤波器与全局优化器具有竞争力，我们可以使用更平滑的RTS向后运行滤波器。那些与潜在的多次向前和向后传递数据
相结合的性能应该使性能非常接近全局优化。

### Mahalanobis distance outlier rejector
A lot of measurements do not come from a Gaussian distribution and as such have outliers that do not fit the statistical model
of the Kalman filter. This can cause a lot of performance issues if not dealt with. This library allows the use of a mahalanobis
distance statistical test on the incoming measurements to deal with this. Note that good initialization is critical to prevent
good measurements from being rejected.

马氏距离离群值剔除器 许多测量值并非来自高斯分布，因此存在离群值与卡尔曼滤波器的统计模型不符的异常值。如果不解决，这可能会
导致很多性能问题。该库允许对传入的测量值使用马哈拉诺比斯距离统计测试来解决此问题。注意，良好的初始化对于防止拒绝良好的测
量至关重要。
