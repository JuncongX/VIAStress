[README.MD](https://github.com/user-attachments/files/21857591/README.MD)# Code for VIAStress

The code for our proposed method will be released once we receive the review feedback.
> Reference Paper: Lin X, Zhou C, Wu J, et al. Towards flexible and adaptive neural process for cold-start recommendation[J]. IEEE Transactions on Knowledge and Data Engineering, 2023.

> Reference Paper: Dong H, Nejjar I, Sun H, et al. SimMMDG: A simple and effective framework for multi-modal domain generalization[J]. Advances in Neural Information Processing Systems, 2023, 36: 78674-78695.

> Reference Code (official): https://github.com/donghao51/SimMMDG

## encoder structure reference
> Sánchez-Reolid R, de la Rosa F L, López M T, et al. One-dimensional convolutional neural networks for low/high arousal classification from electrodermal activity[J]. Biomedical Signal Processing and Control, 2022, 71: 103203.

> Biswas D, Everson L, Liu M, et al. CorNET: Deep learning framework for PPG-based heart rate estimation and biometric identification in ambulant environment[J]. IEEE transactions on biomedical circuits and systems, 2019, 13(2): 282-291.


## The methods used to compared

#### TCNTransformer
> Reference Paper: Wu Y, Daoudi M, Amad A. Transformer-based self-supervised multimodal representation learning for wearable emotion recognition[J]. IEEE Transactions on Affective Computing, 2023, 15(1): 157-172.

#### BCSA
> Reference Paper: Zhang X, Wei X, Zhou Z, et al. Dynamic alignment and fusion of multimodal physiological patterns for stress recognition[J]. IEEE Transactions on Affective Computing, 2023, 15(2): 685-696.

> Reference Code (unofficial): https://github.com/WillPowellUk/Modular-Multimodal-Stress-Detector/tree/master/src/ml_pipeline/models/attention_models

#### ITTA
> Reference Paper: Chen L, Zhang Y, Song Y, et al. Improved test-time adaptation for domain generalization[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 24172-24182.

> Reference Code (official): https://github.com/liangchen527/ITTA

#### Mixup
> Reference Paper: Zhang H, Cisse M, Dauphin Y N, et al. mixup: Beyond Empirical Risk Minimization[C]//International Conference on Learning Representations. 2018.

> Reference Code (official): https://github.com/facebookresearch/mixup-cifar10

#### SelfReg
> Kim D, Yoo Y, Park S, et al. Selfreg: Self-supervised contrastive regularization for domain generalization[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 9619-9628.

> Reference Code (official): https://github.com/dnap512/SelfReg

#### MLDG
> Reference Paper: Learning to generalize: Meta-learning for domain generalization

> Reference Code (official): https://github.com/HAHA-DL/MLDG

> Reference Code (unofficial): https://github.com/VerdantE1/Meta-Learning-for-Domain-Generalization-MLdg-Implementation-with-PyTorch/blob/master/MLDG.py

> Reference Code (unofficial): https://github.com/alexrame/fishr/blob/main/domainbed/algorithms.py#L473

#### Fish
> Reference Paper: Shi Y, Seely J, Torr P H S, et al. Gradient matching for domain generalization[J]. arXiv preprint arXiv:2104.09937, 2021.

> Reference Code (official): https://github.com/YugeTen/fish

> Reference Code (unofficial): https://github.com/alexrame/fishr/blob/main/domainbed/algorithms.py

#### SAGM 
> Reference Paper: Wang P, Zhang Z, Lei Z, et al. Sharpness-aware gradient matching for domain generalization[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 3769-3778.

> Reference Code (official): https://github.com/Wang-pengfei/SAGM


Since some networks or methods do not have publicly available code, we reproduced them based on unofficial implementations from GitHub and relevant descriptions in the papers. 
If there are any issues with the network implementation of the above methods, feel free to discuss them with me in the Issues or directly submit code modifications.
The original authors are especially welcome to review and verify the code.

## Ablation Study
### Discussion on Adaptive Classifier Structures
#### (R) MetaAge
> Reference Paper: Li W, Lu J, Wuerkaixi A, et al. MetaAge: Meta-learning personalized age estimators[J]. IEEE Transactions on Image Processing, 2022, 31: 4761-4775.

> Reference Code (official): https://github.com/Li-Wanhua/MetaAge/blob/main/MetaAge_model.py

#### (R) WCN
> Reference Paper: Akbari A, Martinez J, Jafari R. A meta-learning approach for fast personalization of modality translation models in wearable physiological sensing[J]. IEEE journal of biomedical and health informatics, 2021, 26(4): 1516-1527.

___
> Training Trick:  Pretraining PPG VPD and EDA VAE with an Annealing Strategy
