FROM golang:1.25-alpine AS builder

WORKDIR /app

COPY go.mod ./

COPY resources/references.json.gz ./resources/references.json.gz
COPY tools/preprocess.go ./tools/preprocess.go
COPY internal/vectorsearch ./internal/vectorsearch

RUN go run ./tools/preprocess.go

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o fraud-detection .

FROM scratch

WORKDIR /app

COPY --from=builder /app/fraud-detection .
COPY --from=builder /app/resources/ivf ./resources/ivf
COPY --from=builder /app/resources/mcc_risk.json ./resources/mcc_risk.json
COPY --from=builder /app/resources/normalization.json ./resources/normalization.json

EXPOSE 8080

CMD ["./fraud-detection"]
