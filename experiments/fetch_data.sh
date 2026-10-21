#!/usr/bin/env bash
# Optional: fetch the Hub's parquet files with resumable curl and check them
# against the SHA-256 the Hub reports (LFS oid). Only needed on slow or
# unstable connections; otherwise prepare_data.py downloads from the Hub.
set -euo pipefail
R="${FEDLAB_DATA:-data}/raw"
mkdir -p "$R/cifar10" "$R/femnist"
get() { # url dest sha256
  for i in $(seq 1 60); do
    curl -sS -L -C - --retry 5 --retry-delay 3 --speed-limit 2000 --speed-time 60 -o "$2" "$1" && break
    echo "retry $i for $2"; sleep 3
  done
  echo "$3  $2" | sha256sum -c -
}
HF=https://huggingface.co/datasets
get "$HF/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet" "$R/cifar10/train-00000-of-00001.parquet" 8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883
get "$HF/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet" "$R/cifar10/test-00000-of-00001.parquet" 841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de
get "$HF/flwrlabs/femnist/resolve/main/data/train-00000-of-00001.parquet" "$R/femnist/train-00000-of-00001.parquet" cdb389c8d41d5facab79a4a8d452724c30d9c629a1c1e0b1e9ae9de2c633522b
