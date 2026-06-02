package app

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/bytedance/sonic"

	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

var (
	benchmarkID      int
	benchmarkPayload vs.Payload
	benchmarkScore   float32
	benchmarkStatus  int
	benchmarkBody    []byte
)

type benchmarkPayloadFixture struct {
	name string
	json []byte
	body vs.Payload
}

type benchmarkTestData struct {
	Entries []struct {
		Request vs.Payload `json:"request"`
	} `json:"entries"`
}

func loadBenchmarkApp(b *testing.B) *App {
	b.Helper()

	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		b.Fatal("find benchmark file path")
	}

	repoRoot := filepath.Clean(filepath.Join(filepath.Dir(currentFile), "../.."))
	previousDir, err := os.Getwd()
	if err != nil {
		b.Fatalf("get working directory: %v", err)
	}
	if err := os.Chdir(repoRoot); err != nil {
		b.Fatalf("change to repository root: %v", err)
	}
	b.Cleanup(func() {
		if err := os.Chdir(previousDir); err != nil {
			b.Errorf("restore working directory: %v", err)
		}
	})

	a := &App{}
	if err := a.LoadMccRisk(); err != nil {
		b.Fatalf("load mcc risk: %v", err)
	}
	if err := a.LoadNormalization(); err != nil {
		b.Fatalf("load normalization: %v", err)
	}
	if err := a.LoadCentroids(); err != nil {
		b.Fatalf("load centroids: %v", err)
	}
	if err := a.LoadOffsets(); err != nil {
		b.Fatalf("load offsets and vectors: %v", err)
	}
	if err := a.LoadBBoxes(); err != nil {
		b.Fatalf("load bboxes: %v", err)
	}
	if len(a.IVF.Centroids) == 0 || len(a.IVF.Offsets) == 0 {
		b.Fatal("IVF benchmark data is empty")
	}

	return a
}

func benchmarkPayloads() []benchmarkPayloadFixture {
	requestedAt := time.Date(2026, time.March, 11, 20, 23, 35, 0, time.UTC)

	return []benchmarkPayloadFixture{
		{
			name: "without_last_transaction",
			json: []byte(`{
				"id":"tx-bench-null-last",
				"transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},
				"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},
				"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},
				"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},
				"last_transaction":null
			}`),
			body: vs.Payload{
				ID: "tx-bench-null-last",
				Transaction: vs.Transaction{
					Amount:       41.12,
					Installments: 2,
					RequestedAt:  time.Date(2026, time.March, 11, 18, 45, 53, 0, time.UTC),
				},
				Customer: vs.Customer{
					AvgAmount:      82.24,
					TxCount24h:     3,
					KnownMerchants: []string{"MERC-003", "MERC-016"},
				},
				Merchant: vs.Merchant{
					ID:        "MERC-016",
					Mcc:       "5411",
					AvgAmount: 60.25,
				},
				Terminal: vs.Terminal{
					IsOnline:    false,
					CardPresent: true,
					KmFromHome:  29.23,
				},
			},
		},
		{
			name: "with_last_transaction",
			json: []byte(`{
				"id":"tx-bench-last",
				"transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},
				"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001","MERC-001"]},
				"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},
				"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7090520965},
				"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8626479774}
			}`),
			body: vs.Payload{
				ID: "tx-bench-last",
				Transaction: vs.Transaction{
					Amount:       384.88,
					Installments: 3,
					RequestedAt:  requestedAt,
				},
				Customer: vs.Customer{
					AvgAmount:      769.76,
					TxCount24h:     3,
					KnownMerchants: []string{"MERC-009", "MERC-001", "MERC-001"},
				},
				Merchant: vs.Merchant{
					ID:        "MERC-001",
					Mcc:       "5912",
					AvgAmount: 298.95,
				},
				Terminal: vs.Terminal{
					IsOnline:    false,
					CardPresent: true,
					KmFromHome:  13.7090520965,
				},
				LastTransaction: &vs.LastTransaction{
					Timestamp:     time.Date(2026, time.March, 11, 14, 58, 35, 0, time.UTC),
					KmFromCurrent: 18.8626479774,
				},
			},
		},
	}
}

func loadBenchmarkVectors(b *testing.B, a *App, path string) []vs.Vector {
	b.Helper()

	file, err := os.Open(path)
	if err != nil {
		b.Fatalf("open benchmark test data: %v", err)
	}
	defer file.Close()

	var data benchmarkTestData
	if err := json.NewDecoder(file).Decode(&data); err != nil {
		b.Fatalf("decode benchmark test data: %v", err)
	}

	vectors := make([]vs.Vector, len(data.Entries))
	for i, entry := range data.Entries {
		vectors[i] = a.MakeVector(entry.Request)
	}
	return vectors
}

func benchmarkQueries(b *testing.B, ivf vs.IVFFile) []vs.Vector {
	b.Helper()

	return []vs.Vector{
		ivf.Centroids[0],
		ivf.Centroids[len(ivf.Centroids)/4],
		ivf.Centroids[len(ivf.Centroids)/2],
		ivf.Centroids[(len(ivf.Centroids)*3)/4],
	}
}

func reportPayloadMetrics(b *testing.B, payload benchmarkPayloadFixture) {
	b.Helper()
	b.ReportMetric(float64(len(payload.json)), "payload_B/op")
}

func scoreFromFraudCount(fraudCount int) float32 {
	return float32(fraudCount) / float32(topK)
}

func BenchmarkSonicDecode(b *testing.B) {
	for _, payload := range benchmarkPayloads() {
		b.Run(payload.name, func(b *testing.B) {
			var reader bytes.Reader

			b.ReportAllocs()
			reportPayloadMetrics(b, payload)
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				var body vs.Payload
				reader.Reset(payload.json)
				if err := sonic.ConfigDefault.NewDecoder(&reader).Decode(&body); err != nil {
					b.Fatal(err)
				}
				benchmarkPayload = body
			}
		})
	}
}

func BenchmarkMakeVector(b *testing.B) {
	a := loadBenchmarkApp(b)

	for _, payload := range benchmarkPayloads() {
		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			reportPayloadMetrics(b, payload)
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				benchmarkScore = a.MakeVector(payload.body)[0]
			}
		})
	}
}

func BenchmarkClosestCentroids(b *testing.B) {
	a := loadBenchmarkApp(b)
	queries := benchmarkQueries(b, a.IVF)

	b.ReportAllocs()
	b.ReportMetric(float64(vs.MaxNProbe), "max_nprobe")
	b.ReportMetric(float64(len(a.IVF.Centroids)), "centroids")
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		var ids [vs.MaxNProbe]int
		a.IVF.ClosestCentroids(&queries[i%len(queries)], &ids)
		benchmarkID = ids[0]
	}
}

func BenchmarkIVFSearchPayloads(b *testing.B) {
	a := loadBenchmarkApp(b)

	for _, payload := range benchmarkPayloads() {
		vec := a.MakeVector(payload.body)

		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			reportPayloadMetrics(b, payload)
			b.ReportMetric(float64(nProbe), "base_nprobe")
			b.ReportMetric(float64(vs.MaxNProbe), "max_nprobe")
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				fraudCount := a.IVF.IvfSearch(vec, topK, nProbe)
				benchmarkScore = scoreFromFraudCount(fraudCount)
			}
		})
	}
}

func BenchmarkIVFSearchTestData(b *testing.B) {
	a := loadBenchmarkApp(b)
	vectors := loadBenchmarkVectors(b, a, "test/v3/test-data.json")

	b.ReportAllocs()
	b.ReportMetric(float64(len(vectors)), "vectors")
	b.ReportMetric(float64(nProbe), "base_nprobe")
	b.ReportMetric(float64(vs.MaxNProbe), "max_nprobe")
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		fraudCount := a.IVF.IvfSearch(vectors[i%len(vectors)], topK, nProbe)
		benchmarkScore = scoreFromFraudCount(fraudCount)
	}
}

func BenchmarkFraudPipeline(b *testing.B) {
	a := loadBenchmarkApp(b)

	for _, payload := range benchmarkPayloads() {
		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			reportPayloadMetrics(b, payload)
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				vec := a.MakeVector(payload.body)
				fraudCount := a.IVF.IvfSearch(vec, topK, nProbe)
				benchmarkBody = fraudResponseBodies[fraudCount]
			}
		})
	}
}

func BenchmarkFraudHandler(b *testing.B) {
	a := loadBenchmarkApp(b)

	for _, payload := range benchmarkPayloads() {
		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			reportPayloadMetrics(b, payload)
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				req := httptest.NewRequest(http.MethodPost, "/fraud-score", bytes.NewReader(payload.json))
				req.Header.Set("Content-Type", "application/json")
				res := httptest.NewRecorder()

				a.FraudScoreHandler(res, req)
				benchmarkStatus = res.Code
				if res.Code != http.StatusOK {
					b.Fatalf("status=%d body=%s", res.Code, res.Body.String())
				}
			}
		})
	}
}

func BenchmarkPrecomputedResponseBody(b *testing.B) {
	body := fraudResponseBodies[2]

	b.ReportAllocs()
	b.ReportMetric(float64(len(body)), "response_B/op")
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchmarkBody = body
	}
}

func BenchmarkIndexShape(b *testing.B) {
	a := loadBenchmarkApp(b)

	counts := make([]int, len(a.IVF.Offsets))
	totalRefs := 0
	emptyClusters := 0

	for i, offset := range a.IVF.Offsets {
		count := int(offset.Count)
		counts[i] = count
		totalRefs += count
		if count == 0 {
			emptyClusters++
		}
	}
	sort.Ints(counts)

	percentile := func(p float64) int {
		idx := int(float64(len(counts)-1) * p)
		return counts[idx]
	}

	avgRefs := float64(totalRefs) / float64(len(counts))
	b.ReportAllocs()
	b.ReportMetric(float64(len(a.IVF.Centroids)), "centroids")
	b.ReportMetric(float64(len(a.IVF.Offsets)), "clusters")
	b.ReportMetric(float64(totalRefs), "refs_total")
	b.ReportMetric(avgRefs, "refs/cluster")
	b.ReportMetric(float64(emptyClusters), "empty_clusters")
	b.ReportMetric(float64(counts[0]), "refs_min")
	b.ReportMetric(float64(percentile(0.50)), "refs_p50")
	b.ReportMetric(float64(percentile(0.95)), "refs_p95")
	b.ReportMetric(float64(percentile(0.99)), "refs_p99")
	b.ReportMetric(float64(counts[len(counts)-1]), "refs_max")
	b.ReportMetric(float64(len(a.IVF.VectorsData))/(1024*1024), "vectors_MiB")
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchmarkStatus = totalRefs
	}
}
