FROM golang:1.25-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./

COPY resources/references.json.gz ./resources/references.json.gz
COPY tools/preprocess.go ./tools/preprocess.go
COPY internal/vectorsearch ./internal/vectorsearch

RUN go run ./tools/preprocess.go

COPY cmd ./cmd
COPY internal/app ./internal/app

RUN GOAMD64=v3 GOMAXPROCS=1 CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o fraud-detection ./cmd/api

FROM scratch

WORKDIR /app

COPY --from=builder /app/fraud-detection .
COPY --from=builder /app/resources/ivf ./resources/ivf
COPY resources/mcc_risk.json ./resources/mcc_risk.json
COPY resources/normalization.json ./resources/normalization.json

EXPOSE 8080

CMD ["./fraud-detection"]
