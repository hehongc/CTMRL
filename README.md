# Improving Generalization in Context-based Offline Meta-Reinforcement Learning via Cross-task Contexts, ICML 2025

## Additional Experimental Results in the Meta-World ML1 Environment

### Table 1. Additional experimental results in the Meta-World ML1 environment.
| Task set                  | Dataset return | CTMRL (ours)  | GENTLE        | IDAQ          | CSRO          | ANOLE         | CORRO     | FOCAL++       | FOCAL     | MACAW         | BOReL     |
| ------------------------- | -------------- | ------------- | ------------- | ------------- | ------------- | ------------- | --------- | ------------- | --------- | ------------- | --------- |
| Bin-Picking               | 3807.73        | 0.10±0.04     | 0.09±0.04     | 0.12±06       | 0.15±0.08     | 0.06±0.05     | 0.11±0.06 | 0.08±0.04     | 0.07±0.03 | **0.18±0.10** | 0.00±0.00 |
| Button-Press              | 553.36         | **1.26±0.04** | **1.26±0.06** | 1.18±0.10     | 1.10±0.08     | 1.25±0.03     | 1.15±0.06 | 1.24±0.07     | 1.09±0.18 | 0.04±0.01     | 0.01±0.01 |
| Button-Press-Topdown-Wall | 3774.26        | **0.47±0.05** | 0.46±0.02     | 0.43±0.03     | 0.37±0.02     | 0.45±0.03     | 0.35±0.05 | 0.42±0.02     | 0.40±0.07 | 0.05±0.02     | 0.05±0.01 |
| Button-Press-Wall         | 2886.57        | **1.06±0.01** | **1.06±0.01** | 1.04±0.04     | 1.04±0.01     | 1.03±0.02     | 1.02±0.04 | 0.98±0.07     | 0.99±0.06 | 0.02±0.00     | 0.01±0.00 |
| Coffee-Button             | 507.65         | **1.29±0.16** | 1.10±0.05     | 1.02±0.18     | 0.90±0.14     | 0.87±0.09     | 1.06±0.04 | 1.04±0.17     | 0.66±0.16 | 0.22±0.15     | 0.02±0.00 |
| Coffee-Pull               | 4205.87        | 0.45±0.05     | 0.33±0.08     | 0.40±0.05     | **0.48±0.01** | 0.46±0.04     | 0.43±0.04 | 0.32±0.04     | 0.23±0.04 | 0.19±0.12     | 0.00±0.00 |
| Coffee-Push               | 1531.27        | **1.28±0.08** | 1.26±0.09     | 1.22±0.13     | 1.22±0.01     | 1.18±0.01     | 1.17±0.16 | 1.00±0.05     | 0.64±0.07 | 0.01±0.01     | 0.00±0.00 |
| Door-Lock                 | 3352.69        | **0.97±0.01** | **0.97±0.01** | **0.97±0.01** | 0.94±0.02     | 0.92±0.03     | 0.89±0.05 | 0.96±0.00     | 0.90±0.02 | 0.25±0.11     | 0.14±0.00 |
| Faucet-Close              | 4033.40        | 1.11±0.01     | **1.12±0.01** | **1.12±0.01** | 1.10±0.01     | 1.08±0.00     | 1.11±0.01 | 1.11±0.00     | 1.06±0.02 | 0.07±0.01     | 0.13±0.03 |
| Hammer                    | 1274.02        | **0.85±0.10** | 0.75±0.08     | 0.74±0.06     | 0.68±0.05     | 0.70±0.05     | 0.70±0.04 | 0.73±0.04     | 0.52±0.06 | 0.09±0.01     | 0.09±0.01 |
| Handle-Press              | 4794.29        | 0.45±0.04     | 0.32±0.05     | 0.53±0.05     | 0.34±0.08     | 0.48±0.07     | 0.25±0.04 | **0.54±0.02** | 0.51±0.02 | 0.19±0.09     | 0.01±0.00 |
| Handle-Press-Side         | 4969.13        | **0.53±0.05** | 0.50±0.10     | 0.51±0.06     | 0.33±0.09     | **0.53±0.03** | 0.24±0.11 | 0.48±0.04     | 0.46±0.03 | 0.29±0.21     | 0.02±0.02 |
| Handle-Pull               | 3907.89        | **0.48±0.14** | 0.26±0.09     | 0.32±0.16     | 0.16±0.12     | 0.22±0.08     | 0.23±0.10 | 0.24±0.04     | 0.21±0.04 | 0.00±0.00     | 0.00±0.00 |
| Handle-Pull-Side          | 2838.50        | 0.24±0.06     | 0.21±0.02     | 0.12±0.04     | 0.14±0.02     | 0.23±0.09     | 0.10±0.06 | **0.26±0.07** | 0.11±0.08 | 0.00±0.00     | 0.00±0.00 |
| Peg-Unplug-Side           | 1128.56        | **0.78±0.04** | **0.78±0.05** | 0.76±0.06     | 0.30±0.08     | 0.73±0.06     | 0.29±0.09 | 0.68±0.07     | 0.26±0.13 | 0.00±0.00     | 0.00±0.00 |
| Pick-Place-Wall           | 4137.32        | **0.37±0.06** | 0.21±0.10     | 0.32±0.15     | 0.36±0.13     | 0.36±0.12     | 0.33±0.17 | 0.21±0.06     | 0.14±0.04 | 0.37±0.22     | 0.00±0.00 |
| Plate-Slide               | 4390.86        | 1.00±0.02     | 0.96±0.02     | 1.01±0.03     | 1.00±0.00     | **1.02±0.01** | 0.91±0.02 | 0.92±0.01     | 0.83±0.09 | 0.01±0.00     | 0.01±0.00 |
| Plate-Slide-Back          | 1132.48        | **1.48±0.23** | 1.24±0.11     | 1.23±0.14     | 0.38±0.17     | 1.13±0.16     | 1.00±0.24 | 1.15±0.04     | 0.71±0.06 | 0.26±0.15     | 0.01±0.00 |
| Plate-Slide-Back-Side     | 2121.69        | 0.77±0.06     | 0.66±0.16     | **0.79±0.07** | 0.58±0.06     | 0.68±0.12     | 0.52±0.20 | 0.78±0.05     | 0.63±0.13 | 0.02±0.01     | 0.01±0.00 |
| Plate-Slide-Side          | 3517.98        | **1.10±0.03** | 1.05±0.02     | 1.07±0.08     | 0.98±0.03     | 1.01±0.04     | 0.99±0.01 | 0.99±0.07     | 0.70±0.14 | 0.00±0.00     | 0.00±0.00 |
| Push                      | 3016.54        | 0.80±0.04     | **0.82±0.06** | 0.55±0.10     | 0.60±0.07     | 0.58±0.08     | 0.57±0.08 | 0.62±0.09     | 0.34±0.14 | 0.28±0.19     | 0.00±0.00 |
| Push-Back                 | 309.87         | **0.83±0.06** | 0.71±0.02     | 0.82±0.05     | 0.67±0.14     | 0.76±0.06     | 0.56±0.19 | 0.59±0.07     | 0.36±0.10 | 0.00±0.00     | 0.00±0.00 |
| Push-Wall                 | 3721.30        | **0.89±0.05** | 0.83±0.08     | 0.71±0.15     | 0.71±0.02     | 0.62±0.08     | 0.69±0.07 | 0.74±0.07     | 0.43±0.06 | 0.23±0.18     | 0.00±0.00 |
| Reach-Wall                | 4804.65        | **0.94±0.02** | 0.93±0.02     | 0.93±0.05     | 0.91±0.01     | 0.93±0.02     | 0.84±0.07 | 0.92±0.03     | 0.53±0.18 | 0.82±0.02     | 0.06±0.00 |
| Shelf-Place               | 2802.44        | **0.76±0.11** | 0.65±0.20     | 0.70±0.18     | 0.54±0.05     | 0.51±0.09     | 0.59±0.15 | 0.53±0.04     | 0.32±0.11 | 0.01±0.01     | 0.00±0.00 |
| Stick-Pull                | 1543.67        | **0.94±0.15** | 0.87±0.04     | 0.52±0.13     | 0.34±0.05     | 0.37±0.07     | 0.54±0.04 | 0.54±0.05     | 0.39±0.10 | 0.00±0.00     | 0.00±0.00 |
| Stick-Push                | 1123.76        | 0.67±0.08     | 0.65±0.06     | 0.67±0.10     | 0.73±0.06     | **0.74±0.05** | 0.71±0.05 | 0.73±0.07     | 0.56±0.05 | 0.17±0.16     | 0.00±0.00 |
| Sweep                     | 4356.22        | 0.74±0.05     | **0.78±0.04** | 0.77±0.04     | 0.75±0.06     | 0.74±0.07     | 0.53±0.18 | 0.37±0.11     | 0.32±0.08 | 0.20±0.20     | 0.00±0.00 |
| Sweep-Into                | 2269.74        | **1.02±0.03** | 0.93±0.05     | 0.93±0.04     | 0.89±0.02     | 0.90±0.03     | 0.78±0.03 | 0.69±0.05     | 0.53±0.07 | 0.00±0.00     | 0.01±0.00 |
| Window-Open               | 2120.12        | **1.02±0.05** | 0.97±0.02     | 0.96±0.02     | 0.98±0.03     | 0.95±0.02     | 0.93±0.02 | 0.86±0.04     | 0.84±0.05 | 0.17±0.10     | 0.03±0.00 |


## Analysis of the Number of Codes During Meta-training

### Table 2. Comparison of the performance of using different numbers of codes during meta-training.

|                                 | Training with a single code | Training with all codes |
| ------------------------------- | --------------------------- | ----------------------- |
| Performance                     | -85.44±7.47                 | -120.05±16.22           |
| Cosine similarity within codes  | 0.5011721253395081          | 0.7657914757728577      |
| Euclidean distance within codes | 2.9890005588531494          | 1.0864074230194092      |

## Analysis of the Use of Cosine Similarity in Contrastive Loss

### Table 3. Comparison with the contrastive loss based on cosine similarity.

| Environment / Task set | CTMRL        | CTMRL with cosine similarity in contrastive loss |
| ---------------------- | ------------ | ------------------------------------------------ |
| Half-Cheetanh-Vel      | -85.44±7.47  | -85.89±7.09                                      |
| Hopper-Rand-Params     | 360.91±4.89  | 360.04±6.13                                      |
| Walker-Rand-Params     | 391.92±28.26 | 389.33±23.44                                     |
| Reach                  | 0.92±0.02    | 0.92±0.03                                        |




## Additional Analysis of w^{spe} and w^{cro}

### Table 4. Additional analysis of w^{spe} and w^{cro}.

| Environment / Task set | CTMRL        | w^{spe}=w^{cro}=0.5 | w^{spe}=1, w^{cro}=0 |
| ---------------------- | ------------ | ------------------- | -------------------- |
| Half-Cheetanh-Vel      | -85.44±7.47  | -86.26±7.91         | -124.03±15.44        |
| Hopper-Rand-Params     | 360.91±4.89  | 350.63±11.90        | 309.39±19.50         |
| Walker-Rand-Params     | 391.92±28.26 | 372.66±12.05        | 322.97±28.93         |
| Reach                  | 0.92±0.02    | 0.90±0.02           | 0.77±0.10            |


## Installation
To install locally, you will need to first install [MuJoCo](https://www.roboti.us/index.html). 
To handle task distributions with varying reward functions, such as those found in the Cheetah and Ant environments, it is recommended to install MuJoCo150 or a more recent version.
Set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) as well as the gpu drivers (something like `/usr/lib/nvidia-390`, you can find your version by running `nvidia-smi`).

To set up the remaining dependencies, create a conda environment using the following steps:
```
conda env create -f environment.yaml
```
Install the wandb:
```
pip install wandb 
```

**For Walker environments**, MuJoCo131 is required.
To install it, follow the same procedure as for MuJoCo200. To switch between different MuJoCo versions, you can use the following steps:
```
export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mjpro${VERSION_NUM}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro${VERSION_NUM}/bin
```

In addition to the aforementioned steps, you will also need to download the meta-world.

## Data Generation
Example of training policies and generating trajectories on multiple tasks:
For point-robot and cheetah-vel:
```
CUDA_VISIBLE_DEVICES=0 python policy_train.py ./configs/sparse-point-robot.json 
CUDA_VISIBLE_DEVICES=0 python policy_train.py ./configs/cheetah-vel.json
```

For Meta-World ML1 tasks:
```
python data_collection_ml1.py  ./configs/ml1.json
```
you can modify the task in `./configs/ml1.json`

data will be saved in `./data/`

## Offline RL Experiments
For Meta-World ML1 experiment, run: 
```
run_ml1.sh
```
To run different tasks, modify "env_name" in `./configs/cpearl-ml1.json` as well as "datadirs" in `run_ml1.sh`.

For cheetah-vel:
```
run_cheetah.sh
```
