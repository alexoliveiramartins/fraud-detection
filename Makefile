BENCH_PATTERN := Benchmark(SonicDecode|MakeVector|ClosestCentroids|IVFSearch|FraudPipeline|FraudHandler|PrecomputedResponseBody|IndexShape)
GO_CACHE := /tmp/fraud-detection-go-cache

up-dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build --force-recreate

bench:
	CGO_ENABLED=0 GOCACHE=$(GO_CACHE) go test ./internal/app -run '^$$' -bench '$(BENCH_PATTERN)' -benchmem -benchtime=10000x -count=3

bench-once:
	CGO_ENABLED=0 GOCACHE=$(GO_CACHE) go test ./internal/app -run '^$$' -bench '$(BENCH_PATTERN)' -benchmem -benchtime=10000x -count=1

eval:
	go run ./cmd/evaluate-test -test test/v3/test-data.json -nprobe 12

calibrate:
	go run ./cmd/calibrate-nprobe -test test/v3/test-data.json -base 12
