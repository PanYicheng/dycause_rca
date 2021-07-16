# DyCause
This repository is the source code of our paper "Faster, Deeper, Easier: Crowdsourcing Diagnosis of Microservice Kernel Failure from User Space" at ISSTA'2021.

ACM Reference Format:
> Yicheng Pan, Meng Ma, Xinrui Jiang, and Ping Wang. 2021. Faster, Deeper, Easier: Crowdsourcing Diagnosis of Microservice Kernel Failure from User
Space. In Proceedings of the 30th ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA ’21), July 11–17, 2021, Virtual, Denmark. ACM, NewYork,NY, USA, 12 pages. https://doi.org/10.1145/3460319.3464805
## Getting Started
### Requirements
DyCause is written in Python 3.7.5. The dependencies are listed in `environment.yaml`. We recommand using `anaconda` to create an environment for testing DyCause. In current directory, run `conda env create --file environment.yaml -n dycauseenv`
### Demo run DyCause
The entry code of DyCause is `main_dycause_mp.py` for multiprocess version or `main_dycause.py` for single process version. The multiprocess version is faster to test.
The required parameters are documented in the code and can be showed by running with `-h`.

* For a demo execution of DyCause on PyMicro dataset, run:

  `python main_dycause_mp.py pymicro 16 1 --start 1200 --step 30 --bef 100 --aft 0 --lag 9  --num_sel 1 --edge_thres 0.8  --verbose 2 --mean arithmetic`
* For a demo execution of DyCause on our real-world microservice dataset, run:
  * (Multithread version)
    `python main_dycause.py real_micro_service 14 6 28 30 31 --verbose 2`
  * (Multiprocess version)
    `python main_dycause_mp.py real_micro_service 14 6 28 30 31 --verbose 2`
  
### Demo run baselines
The baselines (TBAC, Netmedic, CloudRanger, MonitorRank, MicroCause) we implemented are also included here. Run `python test_all.py` to get a demo result for each baselines and DyCause.

For MicroCause, our implementation is in `microcause.ipynb`.

## Detailed Description
We have tested each methods with different parameters to get reliable results.

### PR@k and RankScore
For the methods TBAC, Netmedic and MonitorRank, we test them in `parameter_tune.ipynb`. For the methods CloudRanger and DyCause, we first generate the experiment results using `cloudranger_params_tune.py` and `dycause_params_tune.py`,  and then analyze the results. 

We provide the code to analyze the data and reproduce the figures (Figure 5, 7, 8, 9, 10, 11) in our paper in `dycause_exp.ipynb`. Also, we provide our experiment record files in the folder `exp_records`. Using these files, one can get the same preformance results as we do.

### Efficiency tests
In order to test the efficiency of different methods, we also record the running time in some of our experiments. And we use the code in `dycause_exp.ipynb` to analyze them and generate the running time figures in our paper.