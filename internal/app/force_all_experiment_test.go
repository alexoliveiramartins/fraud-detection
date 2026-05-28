package app

import (
	"math"
	"testing"

	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

type forceAllTop struct {
	dist     [topK]int64
	label    [topK]bool
	size     int
	worstIdx int
}

func BenchmarkForceAllNProbeDetectionComparison(b *testing.B) {
	a := loadBenchmarkApp(b)
	data := loadBenchmarkDetectionData(b, a, "test/v3/test-data.json")
	baselineScores := benchmarkScoresForNProbe(b, a, data.vectors, 1)

	for _, nProbe := range []int{2, 3, 4, 6, 8, 10, 12} {
		b.Run("force_all_nprobe_"+itoa(nProbe), func(b *testing.B) {
			scores := make([]float32, len(data.vectors))
			for i, vec := range data.vectors {
				scores[i] = forceAllSearch(a, vec, nProbe)
			}

			stats := benchmarkDetectionStatsForScores(data.expectedApproved, baselineScores, scores)
			reportDetectionStats(b, stats, len(data.vectors))
			b.ReportMetric(float64(nProbe), "nprobe")
			b.ReportMetric(float64(len(data.vectors)), "vectors")
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				benchmarkScore = forceAllSearch(a, data.vectors[i%len(data.vectors)], nProbe)
			}
		})
	}
}

func forceAllSearch(a *App, query vs.Vector, nProbe int) float32 {
	queryQ := vs.QuantizeVector(query)
	var top forceAllTop

	var centroidIDs [vs.MaxNProbe]int
	a.IVF.ClosestCentroids(query, nProbe, &centroidIDs)
	for i := 0; i < nProbe; i++ {
		forceAllScanCluster(a, &top, queryQ, centroidIDs[i])
	}

	switch top.fraudCount() {
	case 0:
		return 0
	case 1:
		return 0.2
	case 2:
		return 0.4
	case 3:
		return 0.6
	case 4:
		return 0.8
	default:
		return 1
	}
}

func forceAllScanCluster(a *App, top *forceAllTop, queryQ vs.QuantizedVector, centroidID int) {
	cluster := a.IVF.Offsets[centroidID]
	start := int(cluster.Offset)
	end := start + int(cluster.Count)*vs.Int16ReferenceSize
	buf := a.IVF.VectorsData[start:end]

	for i := 0; i < int(cluster.Count); i++ {
		base := i * vs.Int16ReferenceSize
		worst := top.worst()
		dist := vs.DistQuantizedFromBuffer(queryQ, buf, base, worst)
		if dist < worst {
			top.push(dist, buf[base+28] == 1)
		}
	}
}

func (t *forceAllTop) fraudCount() int {
	count := 0
	for i := 0; i < t.size; i++ {
		if t.label[i] {
			count++
		}
	}
	return count
}

func (t *forceAllTop) worst() int64 {
	if t.size < topK {
		return math.MaxInt64
	}
	return t.dist[t.worstIdx]
}

func (t *forceAllTop) push(dist int64, label bool) {
	if t.size < topK {
		idx := t.size
		t.dist[idx] = dist
		t.label[idx] = label
		if t.size == 0 || dist > t.dist[t.worstIdx] {
			t.worstIdx = idx
		}
		t.size++
		return
	}

	if dist >= t.dist[t.worstIdx] {
		return
	}

	t.dist[t.worstIdx] = dist
	t.label[t.worstIdx] = label
	t.recomputeWorst()
}

func (t *forceAllTop) recomputeWorst() {
	worst := 0
	for i := 1; i < t.size; i++ {
		if t.dist[i] > t.dist[worst] {
			worst = i
		}
	}
	t.worstIdx = worst
}

func itoa(value int) string {
	if value == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for value > 0 {
		i--
		buf[i] = byte('0' + value%10)
		value /= 10
	}
	return string(buf[i:])
}
