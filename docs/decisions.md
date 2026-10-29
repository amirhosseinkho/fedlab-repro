# Engineering decisions and deviations

Each entry says what was decided, why, and what it affects. Entries are in the
order they were made.

## Environment

**D1. Python 3.13 on native Windows, no Ray.** The only CUDA build of PyTorch
already on the machine was `torch 2.11.0+cu128` for CPython 3.13. Flower 1.41.0
declares `ray==2.55.1; python_full_version >= '3.13' and sys_platform != 'win32'`
for its `simulation` extra, so Flower's simulation engine (Ray is its only
backend, `flwr/server/superlink/fleet/vce/backend/raybackend.py`) cannot run on
Windows with Python 3.13. The alternatives were a 2.75 GB download of the cp312
wheel (plus Ray on Windows, which is the configuration the brief already flagged
as fragile) or a CUDA stack inside WSL2. The author chose to stay on Windows
with Python 3.13. CI runs the same code on Python 3.12 and 3.13 (CPU).

**D2. `SequentialGrid` replaces only Flower's transport.** `fedlab/grid.py`
implements Flower's abstract `Grid` (the object a `ServerApp` uses to send and
receive `Message`s). Everything above it is Flower's own code: `ServerApp`,
`Strategy.start`, `FedAvg`, `FedProx`, `DifferentialPrivacyServerSideFixedClipping`,
node sampling (`sample_nodes`), aggregation (`aggregate_arrayrecords`) and
`ClientApp`. Clients run one at a time in one process. Messages are deep-copied
in both directions so that no client shares memory with the server, and each
node keeps a persistent `Context` across rounds, as a SuperNode does. What is
not exercised: gRPC serialisation, the SuperLink state machine, and Ray actors.

**D3. fp32 everywhere.** Mixed precision is broken on the GTX 1650 Ti, so it
is never enabled.

## Data

**D4. Partitions are computed on labels only.** flwr-datasets partitioners only
read the label (or writer) column, so they run on a label-only
`datasets.Dataset` with an `idx` column; the images stay in a uint8 tensor on
the GPU and each client batch is an index into it. This keeps the
flwr-datasets algorithms (`DirichletPartitioner`, `ShardPartitioner`,
`NaturalIdPartitioner`, `IidPartitioner`) and avoids per-batch PIL decoding.

**D5. Dirichlet partitioner settings.** `self_balancing=False` (Flower's
balancing step is not part of Hsu et al. 2019), `min_partition_size=2` so that
alpha = 0.1 with 100 clients can be sampled, and the partition seed equals the
run seed, so seed-to-seed variance includes the partition draw.

**D6. Shards.** `ShardPartitioner` with 2 shards per client and shard size
`len(train) // (2 * num_clients)`: 300 for MNIST with 100 clients, as in
McMahan et al. (2017).

**D7. FEMNIST.** `flwrlabs/femnist` (BSD-2-Clause) has a single `train` split.
A fixed random subset of writers (subset seed 0, writers with at least 20
samples) is the client population. Each writer keeps 90% of its samples for
training; the other 10% of every chosen writer form the central test set,
which mirrors LEAF's per-writer split. The subset size is set by the time
budget and stated in the results.

**D8. CIFAR-10 preprocessing.** Random 24x24 crop and horizontal flip for
training, centre 24x24 crop for testing, per-channel normalisation with
statistics of the training images. The FedAvg paper also adjusts contrast and
brightness and whitens each image; that is not done here.

## Models

**D9. CIFAR-10 model.** The TensorFlow tutorial CNN that McMahan et al. cite
(conv5x5-64, max-pool 3x3/2, LRN, conv5x5-64, LRN, max-pool 3x3/2, FC 384,
FC 192, linear). TensorFlow's `SAME` pooling pads asymmetrically; PyTorch's
`padding=1` pads symmetrically, which gives the same 12x12 and 6x6 output
sizes. Initialisation is PyTorch's default, not the tutorial's truncated
normal. 1,068,298 parameters.

**D10. FEMNIST model.** LEAF's CNN (conv5x5-32, pool, conv5x5-64, pool,
FC 2048, 62 classes): 6,603,710 parameters.

## Algorithms

**D11. FedProx proximal term.** Flower's `FedProx` strategy only adds
`"proximal-mu"` to the config; the client must add the term. Flower's
docstring example adds `(mu/2) * ||w - w_t||` (not squared). This repo uses
`(mu/2) * ||w - w_t||^2`, which is Eq. (2) of Li et al. (2020). A unit test
checks the resulting gradient.

**D12. SCAFFOLD aggregation is size-weighted (deviation from Algorithm 1).**
SCAFFOLD is a subclass of Flower's `FedAvg` (sampling and message construction
inherited). Algorithm 1 averages the sampled clients with equal weights,
`x += eta_g * mean_i(y_i - x)` and `c += (1/N) * sum_i delta c_i`. In the
phase-2 check (CIFAR-10, alpha = 0.1, 100 clients, 10% sampled, E = 2, 60
rounds, lr 0.05, seed 0) that version ended below FedAvg. Last-3 accuracy,
files in `results/phase_checks/`:

| run | partial, 60 rounds, lr 0.05 | full participation, 30 rounds, lr 0.05 |
|---|---|---|
| FedAvg, size-weighted (Flower default) | 0.371 | 0.614 |
| FedAvg, uniform weights | 0.357 | 0.549 |
| FedProx, mu = 0.01 | 0.372 | not run |
| SCAFFOLD, uniform (Algorithm 1 as written) | 0.337 | 0.442 |
| SCAFFOLD, size-weighted x and c | 0.412 | diverged (0.613 at round 20, NaN by round 30) |

The full-participation diagnostic uses 10 clients, alpha = 0.1, E = 1. Client
sizes there range from 796 to 13,568, so the number of local steps K_i ranges
from 16 to 272. Averaging equally weighted clients that take very different
numbers of steps optimises a different objective than the size-weighted one
that the balanced test set corresponds to, and the uniform control variate
keeps pulling the clients towards it. The repo therefore uses weights
p_i = n_i / n for both updates:
`x += eta_g * sum_{i in S} (n_i / n_S) (y_i - x)` and
`c += sum_{i in S} p_i * delta c_i`. With equal client sizes this is exactly
Algorithm 1 (unit test `test_weighted_scaffold_equals_paper_for_equal_client_sizes`),
and the hand-computed test checks the unweighted version against the paper's
equations. The server needs the total number of training examples `n`; in this
simulation it is known exactly. Option II is used for the client control
variate, with `K` the actual number of local steps the client took;
`eta_g = 1`. The divergence at lr 0.05 is the reason for the learning rate in
D19.

**D13. SCAFFOLD replies carry two ArrayRecords.** Flower's
`validate_message_reply_consistency` requires exactly one ArrayRecord, so the
SCAFFOLD strategy calls `_check_and_log_replies(..., validate=False)` and checks
the keys itself.

**D14. Evaluation.** Federated evaluation is disabled (`fraction_evaluate=0`);
the global model is evaluated centrally on the test set every `eval_every`
rounds and at the last round. Reported "final accuracy" is the mean of the
last three evaluations, to reduce the effect of a single noisy round.

**D15. No learning-rate decay, no momentum, no weight decay.** Local training
is plain minibatch SGD with a constant step size for every algorithm.

**D19. Learning rate 0.03 on CIFAR-10, 0.05 on FEMNIST.** One constant lr
per dataset, shared by all algorithms, chosen by pilots with seed 100 (not used
by any reported run) and by SCAFFOLD's stability. CIFAR-10 FedAvg pilot
(alpha = 0.5, E = 2, 30 rounds): lr 0.02 / 0.05 / 0.1 gave last-3 accuracy
0.293 / 0.349 / 0.374. At lr 0.1 both SCAFFOLD variants reached NaN within 35
rounds in the partial-participation check of D12 (FedAvg 0.402, FedProx 0.406);
at lr 0.05 size-weighted SCAFFOLD diverged in the full-participation
diagnostic. SCAFFOLD's analysis bounds the local step size by roughly
1 / (K L eta_g), so a client with K = 272 steps is the binding one. Repeating
the full-participation diagnostic for 60 rounds: lr 0.03 gave SCAFFOLD 0.682
vs FedAvg 0.680, lr 0.02 gave 0.648 vs 0.636, both stable. lr 0.03 is the
largest tested value at which SCAFFOLD did not diverge, so it is used for
every CIFAR-10 run. The cost is that FedAvg is run below its best pilot lr;
the README reports this. FEMNIST pilot (FedAvg, E = 2, 30 rounds): lr 0.05 /
0.1 / 0.2 gave 0.681 / 0.708 / 0.692; FEMNIST client sizes vary much less, and
lr 0.05 was kept after a 60-round SCAFFOLD pilot at that lr (see
`femnist_scaffold_lr0p05_s100.json`).

## Privacy

**D16. Central DP uses equal client weights.** DP-FedAvg's sensitivity
analysis assumes the clipped updates are averaged with a fixed denominator.
Flower's FedAvg weights by `num-examples`, so DP runs set
`weighted_by_key="dp-weight"` and every client reports `dp-weight = 1`.
Flower's accountant enforces equal weights.

**D17. Two central epsilons.** Flower 1.41's in-strategy accountant only
accepts `SamplingMethod.NO_AMPLIFICATION`; that value is attached to every DP
run and is valid for our simulation. The DP-FedAvg paper's accounting
(subsampled Gaussian with Poisson sampling, q = m/N) is also reported, computed
by Flower's `RdpAccountant` configured for Poisson sampling. Our simulation
samples exactly m clients without replacement (`random.sample`), so the Poisson
number is not a formal guarantee for these runs.

**D18. DP-SGD.** Opacus `PrivacyEngine(accountant="rdp")`, Poisson sampling
with rate `1 / len(loader)` (Opacus's default for `make_private`), flat
per-sample clipping. Each client's accountant history is saved in its
`context.state` and restored when it is sampled again, so its epsilon covers
all rounds it took part in. The Opacus noise generator is seeded per client and
round. DP-SGD is combined with FedAvg only.

**D20. DP hyperparameters.** Pilots with seed 100, CIFAR-10, alpha = 0.5,
FedAvg, E = 1, 10 of 100 clients per round, 40 rounds, lr 0.03 (files
`results/phase_checks/phase3_*`). Without DP the last-3 accuracy is 0.309.

* Central clip C. The median L2 norm of a client update over 20 rounds is
  0.32 at E = 1 (10th-90th percentile 0.25-0.39) and 0.56 at E = 5. C = 0.3,
  about the E = 1 median, so roughly half the updates are clipped. An earlier
  pilot at lr 0.05 used C = 0.5 and collapsed to about 13% at z = 0.5; its
  files were discarded with the lr change.
* Central noise z, with C = 0.3: z = 0 / 0.01 / 0.05 / 0.1 / 0.5 gave
  0.304 / 0.303 / 0.305 / 0.311 / 0.221. Clipping alone costs nothing
  measurable here; z = 0.5 costs about 9 points. E3 therefore adds z = 0.1,
  0.2 and 0.3 to the planned grid {0, 0.5, 1.0, 1.5}.
* DP-SGD lr. With per-sample clip 1.0 and lr 0.03, sigma = 0.5 / 1.0 / 1.5
  all gave about 0.146 (0.146 / 0.145 / 0.146): the curves barely depend on
  sigma, so the limit was the step size after clipping, not the noise. At
  sigma = 1.0, lr 0.1 / 0.3 / 1.0 gave 0.195 / 0.275 / 0.136 (lr 1.0 rises
  to 0.172 and then degrades). DP-SGD runs use lr 0.3; the other E3 runs keep
  lr 0.03, so DP-SGD and non-DP points differ in lr as well as in privacy.
* DP-SGD clip 1.0 and sigma in {0.5, 1.0, 1.5} are not tuned.

## Found while building

**D21. Near-equal minibatches.** Local epochs are split into
`ceil(n / B)` batches whose sizes differ by at most one, instead of full
batches of B plus a remainder. In a 5-round MNIST check (alpha = 100, nearly
IID) one client with 1,154 samples (= 36 x 32 + 2) ended its epoch with a
2-sample batch: loss 5.27, gradient norm 50.9 against about 2 for the full
batches, and its update norm jumped from 0.42 to 2.51. Averaging that update
dropped global accuracy from 93.3% to 56.5% in one round. With near-equal
batches every sample is still used once per epoch and the same run reaches
95.4%. (These figures are from a development log, not a file in
`results/`; the old batching code no longer exists.) All runs in `results/`
use near-equal batches; the pilot runs made before the fix were discarded and
repeated with `experiments/phase_checks.py`.

**D22. DP-SGD batches are drawn on the GPU.** Opacus's `make_private` still
wraps the model and optimizer and attaches the RDP accountant with
`sample_rate = 1 / len(loader)`; the batches are drawn with the same rule as
Opacus's `UniformWithReplacementSampler` (each example independently with
probability `sample_rate`, `int(1 / sample_rate)` batches per epoch) but on
device tensors, which cut the cost of a DP-SGD round on CIFAR-10 from about
11 s to about 3 s in development timing (not a file in `results/`). The epsilon of each client is computed from its accountant
history at evaluation time, not after every local update, and the RDP of one
step is cached per (sample rate, noise multiplier); the result is identical to
Opacus's `RDPAccountant` (`tests/test_privacy.py`). After each client update
the per-sample gradients are freed and Opacus's hooks removed: before that
fix, GPU memory reserved by PyTorch grew past the card's 4 GB within about 20
rounds and DP-SGD rounds slowed from about 2.6 s to over 20 s. The fix does
not change any computation; the wall times stored in the DP-SGD pilot files
in `results/phase_checks/` predate it.

**D23. GPU runs are not bitwise reproducible.** The CIFAR-10 model's
`LocalResponseNorm` uses `avg_pool3d`, whose CUDA backward pass has no
deterministic implementation (PyTorch warns about it under
`use_deterministic_algorithms(True, warn_only=True)`). Every seed is fixed, the
CPU path is bitwise reproducible (`tests/test_determinism.py`), but repeating a
CIFAR-10 run on the GPU can differ in the last digits.

**D24. Final accuracy excludes round 0.** "Final accuracy" is the mean of the
last three evaluations after round 0, so short runs never average in the
untrained model.

**D25. Quick mode.** `--quick` replaces the dataset by the first 12,000 MNIST
training images (2,000 test images in E1/E2; 2,000/1,000 in E3), 10 clients,
CPU, and a few rounds. E1/E2 quick runs use lr 0.1 so that FedAvg passes the
phase-1 acceptance check (>= 95% on the full MNIST test set in 5 rounds:
`results/phase_checks/phase1_mnist_quick_fedavg_s0.json`). In quick mode the
FEMNIST group of E1 is skipped (MNIST stands in for both datasets).
Quick-mode numbers are smoke tests and never reported.
