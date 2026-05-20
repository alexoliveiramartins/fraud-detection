# Fraud Detection API

Backend for the [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) challenge.

The service receives card transaction payloads, converts them into 14-dimensional vectors, and classifies fraud risk using an IVF-style vector search over the provided reference dataset.

## Stack

- Go
- Nginx load balancer
- Docker Compose
- Preprocessed binary IVF index

## Running Locally

Generate the IVF binary files:

```bash
go run ./tools/preprocess.go
```

Run the services:

```bash
docker compose up --build
```

API entrypoint:

```txt
http://localhost:9999
```

Health check:

```bash
curl http://localhost:9999/ready
```

## Docker Image

Published image:

```txt
ghcr.io/alexoliveiramartins/fraud-detection:latest
```

The `submission` branch uses this image directly instead of building from source.

## Rinha Compliance

- Two API instances behind an Nginx round-robin load balancer.
- Public API exposed on port `9999`.
- `linux/amd64` compatible image.
- Docker bridge network.
- No privileged containers.
- Resource limits sum to `1 CPU` and `350MB` memory.
- Reference files are preprocessed only from the official dataset.
- Test payloads are not used as lookup data.

## License

MIT
