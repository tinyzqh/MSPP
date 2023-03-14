# Erlang planning network

## Introduction

This is the Github repository for the Erlang Planning Network project, which is an implementation of the paper ["Erlang planning network: An iterative model-based reinforcement learning with multi-perspective." Pattern Recognition 128 (2022): 108668.](https://www.sciencedirect.com/science/article/abs/pii/S0031320322001492).

In this paper, we propose a bi-level Erlang Planning Network (EPN) architecture, which is composed of an upper-level agent and several multi-scale parallel sub-agents, trained in an iterative way. The proposed method focuses upon the expansion of representation by environment: a multi-perspective over the world model, which presents a varied way to represent an agent’s knowledge about the world that alleviates the problem of falling into local optimal points and enhances robustness during the progress of model planning. 

## Installation

To use the Erlang Planning Network algorithm, you need to have Python 3.6 or higher installed on your system. 

## Result

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

- The result diagram of the second algorithm fusing multiple scale strategies, compared with Dreamer, a model-based sota algorithm.

<div align=center>
    <span class='gp-n'>
        <img src='https://github.com/tinyzqh/Kagebunsin-no-jyutu/blob/master/figures/cartpole-balance_aap_all_algorithms_Figure_1.png' width="250" alt="多尺度acrobots-swingup"/>
    </span>
</div>
