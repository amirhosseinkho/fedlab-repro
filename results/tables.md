### E1: final test accuracy (%), mean ± std over seeds

| dataset | partition | FedAvg | FedProx (mu=0.01) | SCAFFOLD | seeds |
|---|---|---|---|---|---|
| cifar10 | Dirichlet alpha=0.1 | 55.1 ± 1.4 | 55.0 ± 1.4 | 26.1 ± 27.8 (2 of 3 diverged) | 3 |
| cifar10 | Dirichlet alpha=0.5 | 65.2 ± 1.1 | 65.2 ± 1.0 | 66.0 ± 0.7 | 3 |
| femnist | writers (natural) | 79.8 ± 0.4 | 79.8 ± 0.4 | 4.8 ± 0.0 (3 of 3 diverged) | 3 |

### E2: CIFAR-10 accuracy (%) vs Dirichlet alpha

| alpha | FedAvg | SCAFFOLD | SCAFFOLD - FedAvg | seeds |
|---|---|---|---|---|
| 0.1 | 55.1 ± 1.4 | 26.1 ± 27.8 (2 of 3 diverged) | -29.0 | 3 |
| 0.5 | 65.2 ± 1.1 | 66.0 ± 0.7 | +0.7 | 3 |
| 1 | 67.5 ± 1.0 | 68.8 ± 0.5 | +1.4 | 3 |
| 100 | 69.6 ± 0.4 | 69.7 ± 0.3 | +0.1 | 3 |

### E3a: communication (no DP), CIFAR-10 alpha=0.5, 100 rounds

| algorithm | E | clients/round | accuracy (%) | total MB | MB to 40% | MB to 50% | seeds |
|---|---|---|---|---|---|---|---|
| FedAvg | 1 | 5 | 41.8 | 4,273 | 3,846 | not reached | 1 |
| FedAvg | 1 | 10 | 44.7 | 8,546 | 5,128 | not reached | 1 |
| FedAvg | 1 | 20 | 45.4 | 17,093 | 8,546 | not reached | 1 |
| FedAvg | 5 | 5 | 63.9 ± 1.0 | 4,273 | 783 | 1,638 | 3 |
| FedAvg | 5 | 10 | 67.2 | 8,546 | 855 | 2,564 | 1 |
| FedAvg | 5 | 20 | 67.9 | 17,093 | 2,564 | 4,273 | 1 |
| SCAFFOLD | 1 | 5 | 45.0 | 8,546 | 5,555 | not reached | 1 |
| SCAFFOLD | 1 | 10 | 46.1 | 17,093 | 8,546 | not reached | 1 |
| SCAFFOLD | 1 | 20 | 10.0 (1 of 1 diverged) | 34,186 | 18,802 | not reached | 1 |
| SCAFFOLD | 5 | 5 | 67.0 ± 0.8 | 8,546 | 1,567 | 2,849 | 3 |
| SCAFFOLD | 5 | 10 | 69.2 ± 0.3 | 17,093 | 2,564 | 4,843 | 3 |
| SCAFFOLD | 5 | 20 | 70.4 ± 0.5 | 34,186 | 4,558 | 7,977 | 3 |

### E3b: user-level central DP (DP-FedAvg), delta=1e-5, 100 rounds

| clients/round | noise z | accuracy (%) | epsilon, no amplification | epsilon, Poisson-assumed | seeds |
|---|---|---|---|---|---|
| 5 | 0 | 40.4 | inf | inf | 1 |
| 5 | 0.1 | 39.7 | 5,612 | 2,416 | 1 |
| 5 | 0.3 | 14.8 | 721.5 | 89.8 | 1 |
| 5 | 0.5 | 10.0 (1 of 1 diverged) | 294.9 | 21.0 | 1 |
| 5 | 1 | 10.0 ± 0.0 (3 of 3 diverged) | 96.1 | 4.0 | 3 |
| 10 | 0 | 43.5 | inf | inf | 1 |
| 10 | 0.1 | 44.4 | 5,612 | 3,124 | 1 |
| 10 | 0.3 | 32.9 | 721.5 | 171.5 | 1 |
| 10 | 0.5 | 19.8 | 294.9 | 37.0 | 1 |
| 10 | 1 | 10.0 (1 of 1 diverged) | 96.1 | 7.9 | 1 |
| 20 | 0 | 44.7 | inf | inf | 1 |
| 20 | 0.1 | 45.2 | 5,612 | 3,860 | 1 |
| 20 | 0.3 | 46.0 ± 0.5 | 721.5 | 312.1 | 3 |
| 20 | 0.5 | 38.8 ± 0.7 | 294.9 | 79.8 | 3 |
| 20 | 1 | 20.1 ± 1.3 | 96.1 | 16.1 | 3 |

### E3c: sample-level DP (DP-SGD in each client), delta=1e-5, 100 rounds

| clients/round | noise sigma | accuracy (%) | max client epsilon | mean client epsilon | seeds |
|---|---|---|---|---|---|
| 5 | 0.5 | 35.1 | 50.9 | 25.7 | 1 |
| 5 | 1 | 25.4 | 14.9 | 6.0 | 1 |
| 5 | 1.5 | 15.9 ± 2.1 | 7.7 | 3.0 | 3 |
| 10 | 0.5 | 37.9 ± 1.3 | 61.5 | 35.6 | 3 |
| 10 | 1 | 33.8 ± 1.1 | 17.7 | 8.2 | 3 |
| 10 | 1.5 | 24.6 ± 0.7 | 9.4 | 4.1 | 3 |
| 20 | 0.5 | 39.3 ± 1.3 | 84.9 | 51.0 | 3 |
| 20 | 1 | 37.2 ± 1.3 | 23.7 | 11.5 | 3 |
| 20 | 1.5 | 31.3 ± 1.4 | 12.6 | 5.8 | 3 |

