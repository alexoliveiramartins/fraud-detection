FROM golang:1.25-alpine AS builder

WORKDIR /app

COPY go.mod ./
# RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o fraud-detection .

FROM scratch

WORKDIR /app

COPY --from=builder /app/fraud-detection .
COPY --from=builder /app/resources ./resources

EXPOSE 8080

CMD ["./fraud-detection"]