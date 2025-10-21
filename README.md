## AdaM: Adaptive Multi-View Clustering


This repository contains the implementation of AdaM (Adaptive Multi-View Clustering), a novel approach for multi-view clustering that adaptively learns to fuse information from multiple views.

### Training a new model

Run the main training script:

```bash
python AdaM.py
```

You can modify the dataset by changing the `Dataname` variable in the script, or pass it as an argument.

### Testing a trained model

```bash
python test.py
```

### Citation

If you find this work useful in your research, please consider citing:

@article{XUE2026112409,
title = {Adaptive multi-view consistency clustering via structure-enhanced contrastive learning},
journal = {Pattern Recognition},
volume = {172},
pages = {112409},
year = {2026},
author = {Xuqian Xue and Qi Cai and Zhanwei Zhang and Yiming Lei and Hongming Shan and Junping Zhang}
}

For any questions or issues, please contact: xqxue22@m.fudan.edu.cn

### Acknowledgments

This work builds upon and extends the GCFAgg framework proposed by Yan et al. (CVPR 2023). We acknowledge their foundational contributions to multi-view clustering and refer readers to their original implementation: [GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering](https://github.com/Galaxy922/GCFAggMVC).

