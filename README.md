# Advanced Kalman Filtering and Sensor Fusion

## What is Sensor Fusion?
- Sensor fusion is the process of combining sensor data or data derived from disparate sources such that the resulting information has less uncertainty than would be possible when these sources were used individually.
- Data Fusion is the process of integrating **multiple** data sources to form useful information that is more **consistent** and more **accurate** than the original data sources.
<p align="center"><img src="images/dynamics_of_aircraft.png" /></p>
<p align="center"><img src="images/sensor_fusion_speed.png" /></p>

## How does sensor fusion work?
Data fusion is a wide subject area, so we will concentrate on the most powerful method in this, which is Bayesian or probabilistic data fusion for the application of sensor fusion or state estimation. This is where the state of the dynamic system to be estimated is encoded as a probability distribution. This probability distribution is then updated with the probability distributions of sensor measurements or other sources of information to form the most optimum estimatge of the state of the system.
数据融合是一个广泛的学科领域，因此我们将专注于其中最强大的方法，即贝叶斯或概率数据融合，用于传感器融合或状态估计的应用。这是将要估计的动态系统的状态编码为概率分布的地方。然后用传感器测量或其他信息源的概率分布更新该概率分布，以形成系统状态的最佳估计
Probability provides a very powerful framework to build data fusion algorithms upon, it helps encapsulate the inherent uncertainty about the model and measurement processes and helps us to know how accurate are the state estimates.
概率提供了一个非常强大的框架来构建数据融合算法，它有助于封装模型和测量过程的固有不确定性，还有助于我们了解状态估计的准确性。
Sensor data fusion is usually split into two phases and most estimation processes follow a similar process of breaking the problem down into two recursive steps:
传感器数据融合通常分为两个阶段，大多数估计过程遵循将问题分解为两个递归步骤的类似过程：
**Prediction Step**: to calculate the best estimate of the state for the current time using all the available information we have at that time. This is where the uncertainty in the estimates grows.
预测步骤：使用我们当时拥有的所有可用信息计算当前时间状态的最佳估计。这是估计的不确定性增加的地方。
**Update Step**: where the current best state estimate is updated measurements or information when they become available, to produced an even better estimate. This is where the uncertainty in the estimate shrinks.
更新步骤：当前最佳状态估计是在可用时更新测量或信息，以产生更好的估计。这是估计中的不确定性缩小的地方。
Now these two steps don't have to be run in sequence, you might have different sensors or measurement being made at different rates, so you might make multiple prediction steps before the next measurement update is fused in. You can also fuse in multiple sensors at the same time.
现在这两个步骤不必按顺序运行，您可能有不同的传感器或以不同的速率进行测量，因此您可以在融合下一个测量更新之前进行多个预测步骤。您也可以融合多个传感器同时。
