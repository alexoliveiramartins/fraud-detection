BENCH_PATTERN := Benchmark(01_JSONDecode|02_VectorizePayload|03_IVF_ClosestCentroids|04_IVF_Search|05_IVF_Search|06_FraudPipeline|07_HTTPHandler|08_PrecomputedResponseLookup|09_IndexClusterShape)
GO_CACHE := /tmp/fraud-detection-go-cache

define BENCH_LEGEND
@printf '%s\n' 'Benchmark legend:'
@printf '%s\n' '  01_JSONDecode_Sonic: parse request JSON into Payload using sonic.'
@printf '%s\n' '  02_VectorizePayload: convert an already-decoded Payload into the 14-dim vector.'
@printf '%s\n' '  03_IVF_ClosestCentroids: choose the MaxNProbe nearest centroids.'
@printf '%s\n' '  04_IVF_Search_SyntheticPayloads: IVF search on two fixed representative payloads.'
@printf '%s\n' '  05_IVF_Search_FullTestData: IVF search cycling through test/v3/test-data.json.'
@printf '%s\n' '  06_FraudPipeline_VectorizeSearchResponse: vectorize + IVF search + response lookup, no HTTP/JSON decode.'
@printf '%s\n' '  07_HTTPHandler_DecodeSearchWrite: full net/http handler with JSON decode, search, and response write.'
@printf '%s\n' '  08_PrecomputedResponseLookup: cost of selecting an already-rendered response body.'
@printf '%s\n' '  09_IndexClusterShape: reports cluster-size metrics; timing is not meaningful.'
@printf '%s\n' '  payload_last_tx_null: representative request where last_transaction is null.'
@printf '%s\n' '  payload_last_tx_present: representative request with last_transaction object.'
@printf '\n'
endef

up-dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build --force-recreate

bench:
	$(BENCH_LEGEND)
	CGO_ENABLED=0 GOCACHE=$(GO_CACHE) go test ./internal/app -run '^$$' -bench '$(BENCH_PATTERN)' -benchmem -benchtime=10000x -count=3

bench-once:
	$(BENCH_LEGEND)
	CGO_ENABLED=0 GOCACHE=$(GO_CACHE) go test ./internal/app -run '^$$' -bench '$(BENCH_PATTERN)' -benchmem -benchtime=10000x -count=1

eval:
	go run ./cmd/evaluate-test -test test/v3/test-data.json -nprobe 12

calibrate:
	go run ./cmd/calibrate-nprobe -test test/v3/test-data.json -base 12
