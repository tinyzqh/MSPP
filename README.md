# Erlang planning network & MSPP

## Introduction

This is the Github repository for the Erlang Planning Network project and MSPP project, which is an implementation of the paper:

-  ["Erlang planning network: An iterative model-based reinforcement learning with multi-perspective." Pattern Recognition 128 (2022): 108668.](https://www.sciencedirect.com/science/article/abs/pii/S0031320322001492).


In this paper, we propose a bi-level Erlang Planning Network (EPN) architecture, which is composed of an upper-level agent and several multi-scale parallel sub-agents, trained in an iterative way. The proposed method focuses upon the expansion of representation by environment: a multi-perspective over the world model, which presents a varied way to represent an agent’s knowledge about the world that alleviates the problem of falling into local optimal points and enhances robustness during the progress of model planning. 

- [Understanding world models through multi-step pruning policy via reinforcement learning](https://www.sciencedirect.com/science/article/abs/pii/S0020025524012751)

This article introduces a novel approach that explores a variety of policies instead of focusing on either world model bias or singular policy bias. Specifically, we introduce the Multi-Step Pruning Policy (MSPP), which aims to reduce redundant actions and compress the action and state spaces. This approach encourages a different perspective within the same world model. To achieve this, we use multiple pruning policies in parallel and integrate their outputs using the cross-entropy method. Additionally, we provide a convergence analysis of the pruning policy theory in tabular form and an updated parameter theoretical framework. In the experimental section, the newly proposed MSPP method demonstrates a comprehensive understanding of the world model and outperforms existing state-of-the-art model-based reinforcement learning baseline techniques.



## Installation

To use the Erlang Planning Network algorithm and MSPP, you need to have Python 3.6 or higher installed on your system. 

## Result
### Erlang planning network
- Result in different views.

<div align=center>
    <span class='gp-n'>
        <img src='https://github.com/tinyzqh/Kagebunsin-no-jyutu/blob/master/figures/acrobots-swingupFigure_1.png' width="250" alt="多尺度acrobots-swingup"/>
        <img src='https://github.com/tinyzqh/Kagebunsin-no-jyutu/blob/master/figures/cartpole-balanceFigure_1.png' width="250" alt="多尺度cartpole-balance"/>
        <img src='https://github.com/tinyzqh/Kagebunsin-no-jyutu/blob/master/figures/hopper-hopFigure_1.png' width="250" alt="多尺度hopper-hop"/>
    </span>
</div>

- The result diagram of the first algorithm fusing multiple scale strategies, compared with Dreamer, a model-based sota algorithm.

<div align=center>
    <span class='gp-n'>
        <img src='https://github.com/tinyzqh/Kagebunsin-no-jyutu/blob/master/figures/acrobots-swingupall_algorithms_Figure_1.png' width="250" alt="多尺度acrobots-swingup"/>
        <img src='https://github.com/tinyzqh/Kagebunsin-no-jyutu/blob/master/figures/cartpole-balanceall_algorithms_Figure_1.png' width="250" alt="多尺度cartpole-balance"/>
        <img src='https://github.com/tinyzqh/Kagebunsin-no-jyutu/blob/master/figures/hopper-hopall_algorithms_Figure_1.png' width="250" alt="多尺度hopper-hop"/>
    </span>
</div>


### MSPP

See the paper. (I am lazy...)


## Cite

```bash
@article{he2024understanding,
  title={Understanding World Models through Multi-Step Pruning Policy via Reinforcement Learning},
  author={He, Zhiqiang and Qiu, Wen and Zhao, Wei and Shao, Xun and Liu, Zhi},
  journal={Information Sciences},
  pages={121361},
  year={2024},
  publisher={Elsevier}
}
```


```bash
@article{wang2022erlang,
  title={Erlang planning network: An iterative model-based reinforcement learning with multi-perspective},
  author={Wang, Jiao and Zhang, Lemin and He, Zhiqiang and Zhu, Can and Zhao, Zihui},
  journal={Pattern Recognition},
  volume={128},
  pages={108668},
  year={2022},
  publisher={Elsevier}
}
```





