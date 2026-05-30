# Fraud Detection API

Backend for the [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) challenge.

The service receives card transaction payloads, converts them into 14-dimensional vectors, and classifies fraud risk using an IVF-style vector search over the provided reference dataset.

## Stack

- Go
- haproxy (Load Balancer)
- Docker / Docker Compose

## Architecture

The application follows the required Rinha topology: one load balancer in front of two API instances.

```txt
Client ──> haproxy:9999 (round-robin)
            ├── api1:8080
            └── api2:8080
```

haproxy only distributes requests between the two APIs. It does not inspect payloads or apply any fraud-detection logic. The API instances are stateless and load the same index files, so any request can be handled by either instance.

### Vector Search

The fraud detector is based on approximate nearest-neighbor (ANN) search over the official reference dataset. During preprocessing, the dataset is converted into an IVF-style index:

- `centroids.bin`: the trained cluster centroids.
- `offsets.bin`: the byte ranges for each cluster.
- `vectors.bin`: the quantized reference vectors and labels.

At runtime, the request flow is:

1. Parses the transaction JSON payload.
2. Converts it into a 14-dimensional vector.
3. Finds the closest IVF centroid.
4. Scans that cluster using quantized vectors.
5. Computes the fraud score from the top 5 nearest references (KNN).

### Resource Distribution

The Docker Compose setup stays within the challenge limit of `1 CPU` and `350MB`.

|   Service |   CPU |    Memory |
| --------: | ----: | --------: |
|      api1 |   0.4 |     160Mb |
|      api2 |   0.4 |     160Mb |
|   haproxy |   0.2 |      30Mb |
| **Total** | **1** | **350Mb** |

## Optimizations

- Use of IVF search instead of brute-force and HSNW for better memory usage / performance with the avaliable resources of the challenge
- Pre-processing of the reference vectors to binary (.bin) file
- In-memory loading of the vectors/IVF indexes from the .bin files with `mmap` to avoid heap pressuring
- Vector dimensions Quantization from `float32` to `int16` for a smaller vectors binary while keeping precision
- `sonic/encoding` for json incoding instead of `encoding/json` for faster serializing & deserializing
- `nProbe` = 12 / `nCentroids` = 1024 for best precision x performance balance
- Heap-like TopK to avoid sorting all vectors on a cluster
- Unroll the loop of the distance function for better performance
- Trade nginx for Haproxy to handle more requests/sec
- Use tcp on Haproxy configuration for lower overhead than http mode
- Adaptative nProbe to treat edge cases

## Pre-requisites

- Docker / Docker compose
- Go `>1.25.9`

## Running Locally

- Run using the source on the `main` branch

```bash
# Generate the IVF binary files
go run ./tools/preprocess.go
# Build and Run the Docker container
docker compose up --build
```

- Run using the ghcr hosted Docker container on the `submission` branch

```bash
# Switch to the submission branch
git checkout submission
docker compose up
```

> The service will be avaliable at http://localhost:9999/

## API endpoints

### `GET /ready`

- API status/health endpoint

### `POST /fraud-score`

- Recieves a transaction payload and gives a fraud score

Example:

```json
{
  "id": "tx-1329056812",
  "transaction": {
    "amount": 41.12,
    "installments": 2,
    "requested_at": "2026-03-11T18:45:53Z"
  },
  "customer": {
    "avg_amount": 82.24,
    "tx_count_24h": 3,
    "known_merchants": ["MERC-003", "MERC-016"]
  },
  "merchant": {
    "id": "MERC-016",
    "mcc": "5411",
    "avg_amount": 60.25
  },
  "terminal": {
    "is_online": false,
    "card_present": true,
    "km_from_home": 29.23
  },
  "last_transaction": null
}
```

```json
{
  "approved": true,
  "fraud_score": 0
}
```

## Docker Image

Published image:

```txt
ghcr.io/alexoliveiramartins/fraud-detection:latest
```

The `submission` branch uses this image directly instead of building from source.

## Rinha Compliance

- Two API instances behind an Haproxy round-robin load balancer.
- Public API exposed on port `9999`.
- `linux/amd64` compatible image.
- Docker bridge network.
- No privileged containers.
- Resource limits sum to `1 CPU` and `350MB` memory.
- Reference files are preprocessed only from the official dataset.
- Test payloads are not used as lookup data.

## License

MIT
