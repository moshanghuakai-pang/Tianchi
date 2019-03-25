## 这是什么
2018寒假参加的天池数据挖掘大赛：天体光谱分类。

## 简单介绍
- 时间：2018.2.10-2018.3.5
- 综合排名40/700

## 思路简介：
- 对数量较多的天体进行 kmeans 聚类后进行下采样，对样本数量较少的类别进行重采样；
- 基于 Pytorch 搭建 CNN,针对每一类天体建模，将多分类问题转化为多个二分类问题；
- 依据验证集上的性能优劣将模型融合，最后四类天体的 F1 平均为 76%，综合排名 40/700。
