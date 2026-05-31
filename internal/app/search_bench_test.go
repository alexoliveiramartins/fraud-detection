package app

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
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
	benchmarkScore    float32
	benchmarkIDs      []int
	benchmarkID       int
	benchmarkPayload  vs.Payload
	benchmarkResponse vs.Response
	benchmarkStatus   int
	benchmarkBody     []byte
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
	if err := a.LoadBBoxes(); err != nil {
		b.Fatalf("load bboxes: %v", err)
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

type benchmarkTestData struct {
	Entries []struct {
		Request          vs.Payload `json:"request"`
		ExpectedApproved bool       `json:"expected_approved"`
	} `json:"entries"`
}

type benchmarkDetectionData struct {
	vectors          []vs.Vector
	expectedApproved []bool
}

type benchmarkDetectionStats struct {
	tp              int
	tn              int
	fp              int
	fn              int
	scoreChanged    int
	approvedChanged int
	improved        int
	worsened        int
	weightedErrors  int
	detectionScore  float64
}

func loadBenchmarkTestVectors(b *testing.B, a *App, path string) []vs.Vector {
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

func loadBenchmarkDetectionData(b *testing.B, a *App, path string) benchmarkDetectionData {
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

	detectionData := benchmarkDetectionData{
		vectors:          make([]vs.Vector, len(data.Entries)),
		expectedApproved: make([]bool, len(data.Entries)),
	}
	for i, entry := range data.Entries {
		detectionData.vectors[i] = a.MakeVector(entry.Request)
		detectionData.expectedApproved[i] = entry.ExpectedApproved
	}

	return detectionData
}

func benchmarkScoresForNProbe(b *testing.B, a *App, vectors []vs.Vector, nProbe int) []float32 {
	b.Helper()

	scores := make([]float32, len(vectors))
	for i, vec := range vectors {
		fraudCount := a.IVF.IvfSearch(vec, topK, nProbe)
		scores[i] = benchmarkScoreFromCount(fraudCount)
	}

	return scores
}

func benchmarkScoreFromCount(fraudCount int) float32 {
	return float32(fraudCount) / float32(topK)
}

func benchmarkApproval(score float32) bool {
	return score < 0.6
}

func benchmarkWeightedError(expectedApproved, approved bool) int {
	if expectedApproved == approved {
		return 0
	}
	if approved {
		return 3
	}
	return 1
}

func benchmarkDetectionScore(weightedErrors, total int) float64 {
	const (
		k          = 1000
		epsilonMin = 0.001
		beta       = 300
	)

	epsilon := float64(weightedErrors) / float64(total)
	rateComponent := k * math.Log10(1/math.Max(epsilon, epsilonMin))
	absolutePenalty := -beta * math.Log10(1+float64(weightedErrors))

	return rateComponent + absolutePenalty
}

func benchmarkDetectionStatsForScores(expectedApproved []bool, baselineScores, scores []float32) benchmarkDetectionStats {
	stats := benchmarkDetectionStats{}

	for i, score := range scores {
		expected := expectedApproved[i]
		approved := benchmarkApproval(score)

		switch {
		case expected && approved:
			stats.tn++
		case !expected && !approved:
			stats.tp++
		case expected && !approved:
			stats.fp++
		case !expected && approved:
			stats.fn++
		}

		currentErr := benchmarkWeightedError(expected, approved)
		stats.weightedErrors += currentErr

		if baselineScores == nil {
			continue
		}

		baselineScore := baselineScores[i]
		baselineApproved := benchmarkApproval(baselineScore)
		if score != baselineScore {
			stats.scoreChanged++
		}
		if approved != baselineApproved {
			stats.approvedChanged++
		}

		baselineErr := benchmarkWeightedError(expected, baselineApproved)
		if currentErr < baselineErr {
			stats.improved++
		}
		if currentErr > baselineErr {
			stats.worsened++
		}
	}

	stats.detectionScore = benchmarkDetectionScore(stats.weightedErrors, len(scores))
	return stats
}

func reportDetectionStats(b *testing.B, stats benchmarkDetectionStats, total int) {
	b.Helper()

	b.ReportMetric(float64(stats.tp), "tp")
	b.ReportMetric(float64(stats.tn), "tn")
	b.ReportMetric(float64(stats.fp), "fp")
	b.ReportMetric(float64(stats.fn), "fn")
	b.ReportMetric(float64(stats.weightedErrors), "weighted_E")
	b.ReportMetric(stats.detectionScore, "det_score")
	b.ReportMetric(float64(stats.scoreChanged), "score_changed")
	b.ReportMetric(float64(stats.approvedChanged), "approved_changed")
	b.ReportMetric(float64(stats.improved), "improved")
	b.ReportMetric(float64(stats.worsened), "worsened")
	b.ReportMetric((float64(stats.scoreChanged)/float64(total))*100, "score_changed_%")
	b.ReportMetric((float64(stats.approvedChanged)/float64(total))*100, "approved_changed_%")
}

func benchmarkSelectiveTriggerCount(b *testing.B, a *App, vectors []vs.Vector) int {
	b.Helper()

	triggered := 0
	for _, vec := range vectors {
		fraudCount := a.IVF.IvfSearch(vec, topK, 1)
		if fraudCount == 2 || fraudCount == 3 {
			triggered++
		}
	}

	return triggered
}

func reportSelectiveTriggerMetrics(b *testing.B, vectors []vs.Vector, triggered int) {
	b.Helper()

	b.ReportMetric(float64(triggered), "selective_hits")
	b.ReportMetric((float64(triggered)/float64(len(vectors)))*100, "selective_%")
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

func BenchmarkClusterMetrics(b *testing.B) {
	a := loadBenchmarkApp(b)

	counts := make([]int, len(a.IVF.Offsets))
	total := 0
	empty := 0

	for i, offset := range a.IVF.Offsets {
		count := int(offset.Count)
		counts[i] = count
		total += count
		if count == 0 {
			empty++
		}
	}

	sort.Ints(counts)

	percentile := func(p float64) int {
		if len(counts) == 0 {
			return 0
		}
		idx := int(float64(len(counts)-1) * p)
		return counts[idx]
	}

	avg := float64(total) / float64(len(counts))
	min := counts[0]
	p50 := percentile(0.50)
	p75 := percentile(0.75)
	p90 := percentile(0.90)
	p95 := percentile(0.95)
	p99 := percentile(0.99)
	max := counts[len(counts)-1]

	b.ReportMetric(float64(len(counts)), "clusters")
	b.ReportMetric(float64(total), "refs")
	b.ReportMetric(avg, "avg_refs/cluster")
	b.ReportMetric(float64(empty), "empty_clusters")
	b.ReportMetric(float64(min), "min_refs")
	b.ReportMetric(float64(p50), "p50_refs")
	b.ReportMetric(float64(p75), "p75_refs")
	b.ReportMetric(float64(p90), "p90_refs")
	b.ReportMetric(float64(p95), "p95_refs")
	b.ReportMetric(float64(p99), "p99_refs")
	b.ReportMetric(float64(max), "max_refs")
	b.ReportMetric(float64(p99)/avg, "p99/avg")
	b.ReportMetric(float64(max)/avg, "max/avg")

	b.Logf("clusters=%d refs=%d avg=%.1f empty=%d", len(counts), total, avg, empty)
	b.Logf("min=%d p50=%d p75=%d p90=%d p95=%d p99=%d max=%d", min, p50, p75, p90, p95, p99, max)
	b.Logf("p99/avg=%.2f max/avg=%.2f", float64(p99)/avg, float64(max)/avg)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkID = max
	}
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

func benchmarkRefsScanned(b *testing.B, ivf vs.IVFFile, query vs.Vector, nProbe int) int {
	b.Helper()

	if nProbe <= 1 {
		centroidID := ivf.ClosestCentroid(query)
		return int(ivf.Offsets[centroidID].Count)
	}

	refs := 0
	var centroidIDs [vs.MaxNProbe]int
	ivf.ClosestCentroids(query, nProbe, &centroidIDs)
	for i := 0; i < nProbe; i++ {
		centroidID := centroidIDs[i]
		refs += int(ivf.Offsets[centroidID].Count)
	}
	return refs
}

func benchmarkAvgRefsScanned(b *testing.B, ivf vs.IVFFile, queries []vs.Vector, nProbe int) float64 {
	b.Helper()

	total := 0
	for _, query := range queries {
		total += benchmarkRefsScanned(b, ivf, query, nProbe)
	}
	return float64(total) / float64(len(queries))
}

func reportSearchMetrics(b *testing.B, ivf vs.IVFFile, query vs.Vector, nProbe int) {
	b.Helper()
	b.ReportMetric(float64(nProbe), "nprobe")
	b.ReportMetric(float64(len(ivf.Centroids)), "centroids")
	b.ReportMetric(float64(benchmarkRefsScanned(b, ivf, query, nProbe)), "refs/op")
}

func reportPayloadMetrics(b *testing.B, payload benchmarkPayloadFixture) {
	b.Helper()
	b.ReportMetric(float64(len(payload.json)), "payload_B/op")
}

func BenchmarkIVFSearchMixedClusters(b *testing.B) {
	a := loadBenchmarkApp(b)
	queries := benchmarkQueries(b, a.IVF)

	b.ReportAllocs()
	b.ResetTimer()
	b.ReportMetric(float64(1), "nprobe")
	b.ReportMetric(float64(len(a.IVF.Centroids)), "centroids")
	b.ReportMetric(benchmarkAvgRefsScanned(b, a.IVF, queries, 1), "refs/op")

	for i := 0; i < b.N; i++ {
		fraudCount := a.IVF.IvfSearch(queries[i%len(queries)], 5, 1)
		benchmarkScore = benchmarkScoreFromCount(fraudCount)
	}
}

func BenchmarkIVFSearchSmallestCluster(b *testing.B) {
	a := loadBenchmarkApp(b)
	smallest, _ := benchmarkClusterExtremes(b, a.IVF)
	query := a.IVF.Centroids[smallest]

	b.ReportAllocs()
	b.ResetTimer()
	reportSearchMetrics(b, a.IVF, query, 1)

	for i := 0; i < b.N; i++ {
		fraudCount := a.IVF.IvfSearch(query, 5, 1)
		benchmarkScore = benchmarkScoreFromCount(fraudCount)
	}
}

func BenchmarkIVFSearchLargestCluster(b *testing.B) {
	a := loadBenchmarkApp(b)
	_, largest := benchmarkClusterExtremes(b, a.IVF)
	query := a.IVF.Centroids[largest]

	b.ReportAllocs()
	b.ResetTimer()
	reportSearchMetrics(b, a.IVF, query, 1)

	for i := 0; i < b.N; i++ {
		fraudCount := a.IVF.IvfSearch(query, 5, 1)
		benchmarkScore = benchmarkScoreFromCount(fraudCount)
	}
}

func BenchmarkFraudSearch(b *testing.B) {
	a := loadBenchmarkApp(b)

	for _, payload := range benchmarkPayloads() {
		vec := a.MakeVector(payload.body)

		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			reportPayloadMetrics(b, payload)
			reportSearchMetrics(b, a.IVF, vec, 1)

			for i := 0; i < b.N; i++ {
				fraudCount := a.IVF.IvfSearch(vec, topK, 1)
				benchmarkScore = benchmarkScoreFromCount(fraudCount)
			}
		})
	}
}

func BenchmarkFraudSearchSelectiveNProbe3(b *testing.B) {
	a := loadBenchmarkApp(b)

	for _, payload := range benchmarkPayloads() {
		vec := a.MakeVector(payload.body)

		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			reportPayloadMetrics(b, payload)
			reportSearchMetrics(b, a.IVF, vec, 3)

			for i := 0; i < b.N; i++ {
				fraudCount := a.IVF.IvfSearch(vec, topK, 3)
				benchmarkScore = benchmarkScoreFromCount(fraudCount)
			}
		})
	}
}

func BenchmarkFraudSearchTestDataNProbe1(b *testing.B) {
	a := loadBenchmarkApp(b)
	vectors := loadBenchmarkTestVectors(b, a, "test/v3/test-data.json")
	triggered := benchmarkSelectiveTriggerCount(b, a, vectors)

	b.ReportAllocs()
	b.ResetTimer()
	b.ReportMetric(float64(len(vectors)), "vectors")
	reportSelectiveTriggerMetrics(b, vectors, triggered)

	for i := 0; i < b.N; i++ {
		fraudCount := a.IVF.IvfSearch(vectors[i%len(vectors)], topK, 1)
		benchmarkScore = benchmarkScoreFromCount(fraudCount)
	}
}

func BenchmarkFraudSearchTestDataSelectiveNProbe3(b *testing.B) {
	a := loadBenchmarkApp(b)
	vectors := loadBenchmarkTestVectors(b, a, "test/v3/test-data.json")
	triggered := benchmarkSelectiveTriggerCount(b, a, vectors)

	b.ReportAllocs()
	b.ResetTimer()
	b.ReportMetric(float64(len(vectors)), "vectors")
	reportSelectiveTriggerMetrics(b, vectors, triggered)

	for i := 0; i < b.N; i++ {
		fraudCount := a.IVF.IvfSearch(vectors[i%len(vectors)], topK, 3)
		benchmarkScore = benchmarkScoreFromCount(fraudCount)
	}
}

func BenchmarkNProbeDetectionComparison(b *testing.B) {
	a := loadBenchmarkApp(b)
	data := loadBenchmarkDetectionData(b, a, "test/v3/test-data.json")
	baselineScores := benchmarkScoresForNProbe(b, a, data.vectors, 1)

	for _, nProbe := range []int{1, 2, 3} {
		b.Run(fmt.Sprintf("selective_nprobe_%d", nProbe), func(b *testing.B) {
			scores := baselineScores
			if nProbe != 1 {
				scores = benchmarkScoresForNProbe(b, a, data.vectors, nProbe)
			}
			stats := benchmarkDetectionStatsForScores(data.expectedApproved, baselineScores, scores)

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				fraudCount := a.IVF.IvfSearch(data.vectors[i%len(data.vectors)], topK, nProbe)
				benchmarkScore = benchmarkScoreFromCount(fraudCount)
			}
			b.StopTimer()

			b.ReportMetric(float64(nProbe), "nprobe")
			b.ReportMetric(float64(len(data.vectors)), "vectors")
			reportDetectionStats(b, stats, len(data.vectors))
		})
	}
}

func BenchmarkFraudPipeline(b *testing.B) {
	a := loadBenchmarkApp(b)

	for _, payload := range benchmarkPayloads() {
		metricVec := a.MakeVector(payload.body)

		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			reportPayloadMetrics(b, payload)
			reportSearchMetrics(b, a.IVF, metricVec, 1)

			for i := 0; i < b.N; i++ {
				vec := a.MakeVector(payload.body)
				fraudCount := a.IVF.IvfSearch(vec, topK, 1)

				benchmarkResponse = MakeResponse(benchmarkScoreFromCount(fraudCount))
			}
		})
	}
}

func BenchmarkSonicDecode(b *testing.B) {
	for _, payload := range benchmarkPayloads() {
		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			reportPayloadMetrics(b, payload)

			var reader bytes.Reader
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

func BenchmarkSonicUnmarshalBytes(b *testing.B) {
	for _, payload := range benchmarkPayloads() {
		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			reportPayloadMetrics(b, payload)

			for i := 0; i < b.N; i++ {
				var body vs.Payload
				if err := sonic.Unmarshal(payload.json, &body); err != nil {
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
			b.ResetTimer()
			reportPayloadMetrics(b, payload)

			for i := 0; i < b.N; i++ {
				benchmarkScore = a.MakeVector(payload.body)[0]
			}
		})
	}
}

func BenchmarkMakeResponse(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchmarkResponse = MakeResponse(0.4)
	}
}

func BenchmarkSonicEncode(b *testing.B) {
	resp := vs.Response{
		Approved:   true,
		FraudScore: 0.4,
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := sonic.ConfigDefault.NewEncoder(io.Discard).Encode(resp); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPrecomputedResponseBody(b *testing.B) {
	body := []byte(`{"approved":true,"fraud_score":0.4}` + "\n")
	b.ReportAllocs()
	b.ResetTimer()
	b.ReportMetric(float64(len(body)), "response_B/op")

	for i := 0; i < b.N; i++ {
		benchmarkBody = body
	}
}

func BenchmarkHTTPTestHarness(b *testing.B) {
	payload := benchmarkPayloads()[0]

	b.ReportAllocs()
	b.ResetTimer()
	reportPayloadMetrics(b, payload)

	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest(http.MethodPost, "/fraud-score", bytes.NewReader(payload.json))
		req.Header.Set("Content-Type", "application/json")
		res := httptest.NewRecorder()

		benchmarkStatus = res.Code + len(req.Header)
	}
}

func BenchmarkFraudHandler(b *testing.B) {
	a := loadBenchmarkApp(b)

	for _, payload := range benchmarkPayloads() {
		b.Run(payload.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			reportPayloadMetrics(b, payload)

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
	b.ReportMetric(float64(1), "nprobe")
	b.ReportMetric(float64(len(a.IVF.Centroids)), "centroids")

	for i := 0; i < b.N; i++ {
		var ids [vs.MaxNProbe]int
		a.IVF.ClosestCentroids(queries[i%len(queries)], 1, &ids)
		benchmarkID = ids[0]
	}
}

func BenchmarkClosestCentroidsNProbe3(b *testing.B) {
	a := loadBenchmarkApp(b)
	queries := benchmarkQueries(b, a.IVF)

	b.ReportAllocs()
	b.ResetTimer()
	b.ReportMetric(float64(3), "nprobe")
	b.ReportMetric(float64(len(a.IVF.Centroids)), "centroids")

	for i := 0; i < b.N; i++ {
		var ids [vs.MaxNProbe]int
		a.IVF.ClosestCentroids(queries[i%len(queries)], 3, &ids)
		benchmarkID = ids[0]
	}
}

func BenchmarkClosestCentroid(b *testing.B) {
	a := loadBenchmarkApp(b)
	queries := benchmarkQueries(b, a.IVF)

	b.ReportAllocs()
	b.ResetTimer()
	b.ReportMetric(float64(1), "nprobe")
	b.ReportMetric(float64(len(a.IVF.Centroids)), "centroids")

	for i := 0; i < b.N; i++ {
		benchmarkID = a.IVF.ClosestCentroid(queries[i%len(queries)])
	}
}

func BenchmarkIndexShape(b *testing.B) {
	a := loadBenchmarkApp(b)

	minRefs := int(a.IVF.Offsets[0].Count)
	maxRefs := 0
	totalRefs := 0
	emptyClusters := 0

	for _, offset := range a.IVF.Offsets {
		count := int(offset.Count)
		totalRefs += count
		if count == 0 {
			emptyClusters++
		}
		if count < minRefs {
			minRefs = count
		}
		if count > maxRefs {
			maxRefs = count
		}
	}

	avgRefs := float64(totalRefs) / float64(len(a.IVF.Offsets))
	b.ReportAllocs()
	b.ResetTimer()
	b.ReportMetric(float64(len(a.IVF.Centroids)), "centroids")
	b.ReportMetric(float64(len(a.IVF.Offsets)), "clusters")
	b.ReportMetric(float64(emptyClusters), "empty_clusters")
	b.ReportMetric(float64(totalRefs), "refs_total")
	b.ReportMetric(avgRefs, "refs/cluster")
	b.ReportMetric(float64(minRefs), "refs_min")
	b.ReportMetric(float64(maxRefs), "refs_max")
	b.ReportMetric(float64(len(a.IVF.VectorsData))/(1024*1024), "vectors_MiB")

	for i := 0; i < b.N; i++ {
		benchmarkStatus = totalRefs
	}
}
