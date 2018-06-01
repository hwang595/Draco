# DRACO: Byzantine-resilient Distributed Training via Redundant Gradients
This repository contains source code for Draco, a scalable framework for robust distributed training that uses ideas from coding theory. Please check [https://arxiv.org/abs/1803.09877](https://arxiv.org/abs/1803.09877) for detailed information about this project.

# Overview:
Draco is a scalable framework for robust distributed training that uses ideasfrom coding theory. In Draco, compute nodes evaluate redundant gradients that are then used by the parameter server (PS) to eliminate the effects of adversarial updates.

<div align="center"><img src="https://github.com/hwang595/Draco/blob/master/images/Draco.jpg" height="400" width="450" ></div>

In Draco, each compute node processes *rB/P* gradients and sends a linear combination of those to the PS. This means that Draco incurs a computational redundancy ratio of *r*. Upon receiving the *P* gradient sums, the PS uses a “decoding” function to remove the effect of the adversarial nodes and reconstruct the original desired sum of the B gradients. With redundancy ratio *r*, we show that Draco can tolerate up to *r − 1)/2* adversaries, which is information theoretically tight. 
