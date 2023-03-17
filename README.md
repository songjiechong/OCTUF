# Optimization-Inspired Cross-Attention Transformer for Compressive Sensing (CVPR 2023)
This repository is for OCTUF introduced in the following paper：

[Jiechong Song](https://scholar.google.com/citations?hl=en&user=EBOtupAAAAAJ), Chong Mou, Shiqi Wang, Siwei Ma, [Jian Zhang](http://jianzhang.tech/), "Optimization-Inspired Cross-Attention Transformer for Compressive Sensing", in the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. [PDF](https://arxiv.org/abs/2110.09766)

## Introduction

By integrating certain optimization solvers with deep neural networks, deep unfolding network (DUN) with good interpretability and high performance has attracted growing attention in compressive sensing (CS). However, existing DUNs often improve the visual quality at the price of a large number of parameters and have the problem of feature information loss during iteration. In this paper, we propose an Optimization-inspired Cross-attention Transformer (OCT) module as an iterative process, leading to a lightweight \textbf{OCT}-based \textbf{U}nfolding \textbf{F}ramework (\textbf{OCTUF}) for image CS. Specifically, we design a novel Dual Cross Attention (Dual-CA) sub-module, which consists of an Inertia-Supplied Cross Attention (ISCA) block and a Projection-Guided Cross Attention (PGCA) block. ISCA block introduces multi-channel inertia forces and increases the memory effect by a cross attention mechanism between adjacent iterations. And, PGCA block achieves an enhanced information interaction, which introduces the inertia force into the gradient descent step through a cross attention block. Extensive CS experiments manifest that our OCTUF achieves superior performance compared to state-of-the-art methods while training lower complexity.

<img width="1001" alt="PMM_MAPMM" src="https://user-images.githubusercontent.com/62560218/161186801-95d503f6-f2fa-4dcc-8c60-fc80aab65079.png">


## Dataset

### Train data

[train400](https://drive.google.com/file/d/15FatS3wYupcoJq44jxwkm6Kdr0rATPd0/view?usp=sharing)

### Test data

Set11

[CBSD68](https://drive.google.com/file/d/1Q_tcV0d8bPU5g0lNhVSZXLFw0whFl8Nt/view?usp=sharing)

[Urban100](https://drive.google.com/file/d/1cmYjEJlR2S6cqrPq8oQm3tF9lO2sU0gV/view?usp=sharing)

## Command

### Train

`python Train_CS_MADUN.py --cs_ratio 10/25/30/40/50                  ` 

### Test

`python TEST_CS_MADUN.py --cs_ratio 10/25/30/40/50 --test_name Set11/CBSD68/Urban100`

## Citation

If you find our work helpful in your resarch or work, please cite the following paper.

```
@inproceedings{song2021memory,
  title={Memory-Augmented Deep Unfolding Network for Compressive Sensing},
  author={Song, Jiechong and Chen, Bin and Zhang, Jian},
  booktitle={Proceedings of the ACM International Conference on Multimedia (ACM MM)},
  year={2021}
}
```
