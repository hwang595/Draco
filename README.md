# DRACO: Byzantine-resilient Distributed Training via Redundant Gradients
This repository contains source code for Draco, a scalable framework for robust distributed training that uses ideas from coding theory. Please check [https://arxiv.org/abs/1803.09877](https://arxiv.org/abs/1803.09877) for detailed information about this project.

# Overview:
Draco is a scalable framework for robust distributed training that uses ideasfrom coding theory. In Draco, compute nodes evaluate redundant gradients that are then used by the parameter server (PS) to eliminate the effects of adversarial updates.

<div align="center"><img src="https://github.com/hwang595/Draco/blob/master/images/Draco.jpg" height="400" width="450" ></div>
