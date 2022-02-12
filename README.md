## Learn to forget: Realizing data removal in Machine Learning Models
Undergraduate Research Program
## Abstract
The project proposes and analyzes several ways of removing information from a machine learning system. Algorithms delete requested data from a model making it behave like it never interacted with this data before; whereas their complexity overtakes the straightforward retraining from scratch. The two proposed methods encode the target dataset in a way that once fed into the network, it destroys the filters containing local information about itself while preserving the global features. The other algorithm builds on converging into the final loss curve by performing iterative updates. Experiments proved that developed methods work with deep neural networks on regression and classification tasks and can be applied to other problems. Furthermore, the careful evaluation shows their superiority over trivial forgetting algorithms and equality with recently published successful cases.
![alt text](https://github.com/zero-or-one/URP-Summer-Fall-2021/blob/main/abstact-img.PNG)
## Reference
* ### Papers:
1. https://arxiv.org/abs/1911.04933
2. https://arxiv.org/abs/1804.00792

* ### Code: 
1. https://github.com/AdityaGolatkar/SelectiveForgetting
2. https://github.com/oval-group/dfw
3. https://github.com/zleizzo/datadeletion
4. https://github.com/ashafahi/inceptionv3-transferLearn-poison
5. https://github.com/wannabeOG/MAS-PyTorch
6. https://github.com/King-Of-Knights/overcoming-catastrophic
