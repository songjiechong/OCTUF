# Optimization-Inspired Cross-Attention Transformer for Compressive Sensing (CVPR 2023)
This repository is for OCTUF introduced in the following paper：

[Jiechong Song](https://scholar.google.com/citations?hl=en&user=EBOtupAAAAAJ), [Chong Mou](https://scholar.google.com/citations?user=SYQoDk0AAAAJ&hl=en&oi=ao), Shiqi Wang, Siwei Ma, [Jian Zhang](http://jianzhang.tech/), "Optimization-Inspired Cross-Attention Transformer for Compressive Sensing", in the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. [PDF]

## :art: Abstract

By integrating certain optimization solvers with deep neural networks, deep unfolding network (DUN) with good interpretability and high performance has attracted growing attention in compressive sensing (CS). However, existing DUNs often improve the visual quality at the price of a large number of parameters and have the problem of feature information loss during iteration. In this paper, we propose an Optimization-inspired Cross-attention Transformer (OCT) module as an iterative process, leading to a lightweight OCT-based Unfolding Framework (OCTUF) for image CS. Specifically, we design a novel Dual Cross Attention (Dual-CA) sub-module, which consists of an Inertia-Supplied Cross Attention (ISCA) block and a Projection-Guided Cross Attention (PGCA) block. ISCA block introduces multi-channel inertia forces and increases the memory effect by a cross attention mechanism between adjacent iterations. And, PGCA block achieves an enhanced information interaction, which introduces the inertia force into the gradient descent step through a cross attention block. Extensive CS experiments manifest that our OCTUF achieves superior performance compared to state-of-the-art methods while training lower complexity. 

## :fire: Network Architecture
![Network](/Figs/network.png)


![Network](/Figs/OCT.png)

## 🚩 Results
![Network](/Figs/psnr_para.png)

## 👀 Datasets
- Train data: [train400](https://drive.google.com/file/d/15FatS3wYupcoJq44jxwkm6Kdr0rATPd0/view?usp=sharing)
- Test data: Set11, [CBSD68](https://drive.google.com/file/d/1Q_tcV0d8bPU5g0lNhVSZXLFw0whFl8Nt/view?usp=sharing), [Urban100](https://drive.google.com/file/d/1cmYjEJlR2S6cqrPq8oQm3tF9lO2sU0gV/view?usp=sharing), [DIV2K](https://drive.google.com/file/d/1olYhGPuX8QJlewu9riPbiHQ7XiFx98ac/view?usp=sharing)

## :e-mail: Contact
If you have any question, please email `songjiechong@pku.edu.cn`.

## :hugs: Acknowledgements
This code is built on [FSOINet](https://github.com/cwjjun/fsoinet). We thank the authors for sharing their codes. 
