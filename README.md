# Optimization-Inspired Cross-Attention Transformer for Compressive Sensing (CVPR 2023)
This repository is for OCTUF introduced in the following paper：

[Jiechong Song](https://scholar.google.com/citations?hl=en&user=EBOtupAAAAAJ), Chong Mou, Shiqi Wang, Siwei Ma, [Jian Zhang](http://jianzhang.tech/), "Optimization-Inspired Cross-Attention Transformer for Compressive Sensing", in the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. [PDF](https://arxiv.org/abs/2110.09766)

## Introduction

Mapping a truncated optimization method into a deep neural network, deep unfolding network (DUN) has attracted growing attention in compressive sensing (CS) due to its good interpretability and high performance. Each stage in DUNs corresponds to one iteration in optimization. By understanding DUNs from the perspective of the human brain’s memory processing, we find there exists two issues in existing DUNs. One is the information between every two adjacent stages, which can be regarded as short-term memory, is usually lost seriously. The other is no explicit mechanism to ensure that the previous stages affect the current stage, which means memory is easily forgotten. To solve these issues, in this paper, a novel DUN with persistent memory for CS is proposed, dubbed Memory-Augmented Deep Unfolding Network (MADUN). We design a memory-augmented proximal mapping module (MAPMM) by combining two types of memory augmentation mechanisms, namely High-throughput Short-term Memory (HSM) and Cross-stage Long-term Memory (CLM). HSM is exploited to allow DUNs to transmit multi-channel short-term memory, which greatly reduces information loss between adjacent stages. CLM is utilized to develop the dependency of deep information across cascading stages, which greatly enhances network representation capability. Extensive CS experiments on natural and MR images show that with the strong ability to maintain and balance information our MADUN outperforms existing state-of-the-art methods by a large margin. 

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
