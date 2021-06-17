# Multi-Stage Differential Privacy

**Multi-Stage Differential Privacy** (MSDP) is a theoretical framework that improves the privacy-utility tradeoff incurred when 
training differentially private deep learning  models. It allows training both centralised and decentralised models using 
Federated Learning, referring to the second scenario as **Multi-Stage Differentially Private Federated Learning** (MSDPFL). 
MSDP works by achieving differential privacy at multiple stages across the training pipeline, by perturbing the training 
data, the gradients and the parameters of the model. 

* **Stage I**: Based on Kang et al. 2020; achieves differential privacy by input perturbation; applied to the training set
* **Stage II**: Implemented using Opacus; achieves differential privacy by gradient perturbation; applied to the optimiser 
  level, during training
* **Stage III** and **Stage IV**: Based on Wei et al. 2019; achieves differential privacy by parameter perturbation; for 
  centralised training, only stage III is applied after the training has finished; for federated learning, stage III is 
  applied by each client after the local training is finished and stage IV is applied to the aggreagated model.



## Setup

To use this library we first recommend using Python 3.8 or above. For installing the required libraries, install the packages 
from the ``requirements.txt`` file.

## Basic usage

For examples of usage, please check the `run.py` file that also can be used to replicate our experiments, but bear in mind that 
some parameters from `config.py` may need to be changed.

The framework allows training centralised and decentralised (using Federated Learning) models. For centralised training, create 
an instance of the class `MSDPTrainer`  and attach the stages with the desired hyperparameters. For Federated Learning 
simulation, create an instance of `FLEnvironment`; this will create the clients, the aggregator and will start the simulation. 
Please check the documentation of each class for a complete list of arguments and their usage.

## Future work

* Add more differential privacy verification techniques
  
* Test different aggregation techniques to compensate for the effects of differential privacy when the clients have non-i.i.d data
 partitions. For this, simply extend the `Aggregator` class and override the model averaging methods.
  
* Abstract the code and provide wrappers for TensorFlow

* Add secure aggregation to reduce the noise injected by stages III and IV during MSDPFL training.

* Any other contributions are welcome

## References
* Yilin Kang, Yong Liu, Ben Niu, Xinyi Tong, Likun Zhang, and Weiping Wang. Input  Perturbation: A New Paradigm between Central 
and Local Differential Privacy, 2020. URL https://arxiv.org/abs/2002.08570
  
* Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal
Talwar, and Li Zhang. Deep learning with differential privacy. CCS ’16, page 308–318, New
York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450341394. doi:
10.1145/2976749.2978318. URL https://doi.org/10.1145/2976749.2978318.
  
* Kang Wei, Jun Li, Ming Ding, Chuan Ma, Howard H. Yang, Farhad Farokhi, Shi Jin, Tony
Q. S. Quek, and H. Vincent Poor. Federated learning with differential privacy: Algorithms
and performance analysis. IEEE Transactions on Information Forensics and Security, 15:
3454–3469, 2020. doi: 10.1109/TIFS.2020.2988575. URL https://doi.org/10.1109/TIFS.
2020.2988575.
  
* Opacus: a library that enables training PyTorch models with differential privacy. URL
https://github.com/pytorch/opacus.
