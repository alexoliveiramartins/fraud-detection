package app

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

var (
	benchmarkNeighbors []vs.Neighbor
	benchmarkIDs       []int
	benchmarkID        int
	benchmarkResponse  vs.Response
	benchmarkStatus    int
)

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
	if len(a.IVF.Centroids) == 0 || len(a.IVF.Offsets) == 0 {
		b.Fatal("IVF benchmark data is empty")
	}

	return a
}

type benchmarkPayloadFixture struct {
	name string
	json []byte
	body vs.Payload
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

func benchmarkClusterExtremes(b *testing.B, ivf vs.IVFFile) (smallest, largest int) {
	b.Helper()

	if len(ivf.Offsets) == 0 {
		b.Fatal("IVF offsets are empty")
	}

	for i := 1; i < len(ivf.Offsets); i++ {
		if ivf.Offsets[i].Count < ivf.Offsets[smallest].Count {
			smallest = i
		}
		if ivf.Offsets[i].Count > ivf.Offsets[largest].Count {
			largest = i
		}
	}

	return smallest, largest
}

func benchmarkQueries(b *testing.B, ivf vs.IVFFile) []vs.Vector {
	b.Helper()

	smallest, largest := benchmarkClusterExtremes(b, ivf)
	return []vs.Vector{
		ivf.Centroids[0],
		ivf.Centroids[len(ivf.Centroids)/4],
		ivf.Centroids[len(ivf.Centroids)/2],
		ivf.Centroids[(len(ivf.Centroids)*3)/4],
		ivf.Centroids[smallest],
		ivf.Centroids[largest],
	}
}

func BenchmarkIVFSearchMixedClusters(b *testing.B) {
	a := loadBenchmarkApp(b)
	queries := benchmarkQueries(b, a.IVF)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		neighbors, err := a.IVF.IvfSearch(queries[i%len(queries)], 5, 1)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkNeighbors = neighbors
	}
}

func BenchmarkIVFSearchSmallestCluster(b *testing.B) {
	a := loadBenchmarkApp(b)
	smallest, _ := benchmarkClusterExtremes(b, a.IVF)
	query := a.IVF.Centroids[smallest]

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		neighbors, err := a.IVF.IvfSearch(query, 5, 1)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkNeighbors = neighbors
	}
}

func BenchmarkIVFSearchLargestCluster(b *testing.B) {
	a := loadBenchmarkApp(b)
	_, largest := benchmarkClusterExtremes(b, a.IVF)
	query := a.IVF.Centroids[largest]

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		neighbors, err := a.IVF.IvfSearch(query, 5, 1)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkNeighbors = neighbors
	}
}

func BenchmarkFraudSearch(b *testing.B) {
	a := loadBenchmarkApp(b)

	for _, payload := range benchmarkPayloads() {
		vec := a.MakeVector(payload.body)

		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				neighbors, err := a.IVF.IvfSearch(vec, topK, 1)
				if err != nil {
					b.Fatal(err)
				}

				benchmarkNeighbors = neighbors
			}
		})
	}
}

func BenchmarkFraudPipeline(b *testing.B) {
	a := loadBenchmarkApp(b)

	for _, payload := range benchmarkPayloads() {
		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				vec := a.MakeVector(payload.body)
				neighbors, err := a.IVF.IvfSearch(vec, topK, 1)
				if err != nil {
					b.Fatal(err)
				}

				benchmarkResponse = MakeResponse(neighbors)
			}
		})
	}
}

func BenchmarkFraudHandler(b *testing.B) {
	a := loadBenchmarkApp(b)

	for _, payload := range benchmarkPayloads() {
		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
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

func BenchmarkClosestCentroidsNProbe1(b *testing.B) {
	a := loadBenchmarkApp(b)
	queries := benchmarkQueries(b, a.IVF)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchmarkIDs = a.IVF.ClosestCentroids(queries[i%len(queries)], 1)
	}
}

func BenchmarkClosestCentroid(b *testing.B) {
	a := loadBenchmarkApp(b)
	queries := benchmarkQueries(b, a.IVF)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchmarkID = a.IVF.ClosestCentroid(queries[i%len(queries)])
	}
}
