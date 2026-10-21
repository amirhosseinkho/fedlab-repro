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
rounds, seed 0) that version ended below FedAvg (mean of the last three
evaluations 0.321 vs 0.367). A full-participation diagnostic (10 clients,
alpha = 0.1, E = 1, 30 rounds) separated the cause:

| run (results/phase_checks/) | last-3 accuracy |
|---|---|
| FedAvg, size-weighted (Flower default) | 0.611 |
| FedAvg, uniform weights | 0.543 |
| SCAFFOLD, uniform (Algorithm 1 as written) | 0.444 |
| SCAFFOLD, size-weighted x and c | 0.620 |

The client sizes in that run range from 796 to 13,568, so the number of local
steps K_i ranges from 16 to 272. Averaging equally weighted clients that take
very different numbers of steps optimises a different objective than the
size-weighted one that the balanced test set corresponds to, and the uniform
control variate keeps pulling the clients towards it. The repo therefore uses
weights p_i = n_i / n for both updates:
`x += eta_g * sum_{i in S} (n_i / n_S) (y_i - x)` and
`c += sum_{i in S} p_i * delta c_i`. With equal client sizes this is exactly
Algorithm 1 (unit test `test_weighted_scaffold_equals_paper_for_equal_client_sizes`),
and the hand-computed test checks the unweighted version against the paper's
equations. With the weighted version the phase-2 check gave 0.384 vs FedAvg's
0.367. The server needs the total number of training examples `n`; in this
simulation it is known exactly. Option II is used for the client control
variate, with `K` the actual number of local steps the client took;
`eta_g = 1`.

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
