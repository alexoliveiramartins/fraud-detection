package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/alexoliveiramartins/fraud-detection/internal/app"
	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

const topK = 5

type testFile struct {
	Entries []testEntry `json:"entries"`
}

type testEntry struct {
	Request          vs.Payload `json:"request"`
	ExpectedApproved bool       `json:"expected_approved"`
}

type top struct {
	dist     [topK]int64
	label    [topK]bool
	size     int
	worstIdx int
}

type classStats struct {
	tp       int
	tn       int
	fp       int
	fn       int
	weighted int
}

func main() {
	testPath := flag.String("test", "test/v3/test-data.json", "path to test-data.json")
	baseNProbe := flag.Int("base", 12, "base nProbe used before per-class escalation")
	limit := flag.Int("limit", 0, "max entries to evaluate; 0 means all")
	flag.Parse()

	if *baseNProbe <= 0 || *baseNProbe > vs.MaxNProbe {
		log.Fatalf("base nProbe must be between 1 and MaxNProbe=%d", vs.MaxNProbe)
	}

	a := loadApp()
	entries := loadEntries(*testPath, *limit)
	result := calibrate(a, entries, *baseNProbe, vs.MaxNProbe)

	fmt.Printf("entries: %d\n", len(entries))
	fmt.Printf("centroids: %d\n", len(a.IVF.Centroids))
	fmt.Printf("base_nprobe: %d\n", *baseNProbe)
	fmt.Printf("max_nprobe: %d\n", vs.MaxNProbe)
	for class := 0; class < 6; class++ {
		stats := result.stats[class][result.scaling[class]-*baseNProbe]
		fmt.Printf(
			"class_%d: initial=%d nprobe=%d FP=%d FN=%d weighted_E=%d TP=%d TN=%d\n",
			class,
			result.initial[class],
			result.scaling[class],
			stats.fp,
			stats.fn,
			stats.weighted,
			stats.tp,
			stats.tn,
		)
	}
	fmt.Printf("scaling: [%d, %d, %d, %d, %d, %d]\n",
		result.scaling[0], result.scaling[1], result.scaling[2],
		result.scaling[3], result.scaling[4], result.scaling[5],
	)
	fmt.Printf("total_FP: %d\n", result.totalFP)
	fmt.Printf("total_FN: %d\n", result.totalFN)
	fmt.Printf("total_weighted_E: %d\n", result.totalWeighted)
}

type calibrationResult struct {
	scaling       [6]int
	initial       [6]int
	stats         [6][]classStats
	totalFP       int
	totalFN       int
	totalWeighted int
}

func calibrate(a *app.App, entries []testEntry, baseNProbe int, maxNProbe int) calibrationResult {
	candidateCount := maxNProbe - baseNProbe + 1
	result := calibrationResult{}
	for class := range result.stats {
		result.stats[class] = make([]classStats, candidateCount)
	}

	var centroidIDs [vs.MaxNProbe]int

	for _, entry := range entries {
		query := a.MakeVector(entry.Request)
		queryQ := vs.QuantizeVector(query)
		a.IVF.ClosestCentroids(&query, &centroidIDs)

		var currentTop top
		initialClass := -1

		for rank := 1; rank <= maxNProbe; rank++ {
			scanCluster(a, &currentTop, queryQ, centroidIDs[rank-1])

			if rank == baseNProbe {
				initialClass = currentTop.fraudCount()
				result.initial[initialClass]++
			}
			if rank >= baseNProbe {
				result.stats[initialClass][rank-baseNProbe].add(entry.ExpectedApproved, currentTop.fraudCount())
			}
		}
	}

	for class := 0; class < 6; class++ {
		best := -1
		for nProbe := baseNProbe; nProbe <= maxNProbe; nProbe++ {
			if result.stats[class][nProbe-baseNProbe].weighted == 0 {
				best = nProbe
				break
			}
		}

		if best == -1 {
			bestIdx := 0
			for i := 1; i < candidateCount; i++ {
				if result.stats[class][i].weighted < result.stats[class][bestIdx].weighted {
					bestIdx = i
				}
			}
			best = baseNProbe + bestIdx
		}

		result.scaling[class] = best
		stats := result.stats[class][best-baseNProbe]
		result.totalFP += stats.fp
		result.totalFN += stats.fn
		result.totalWeighted += stats.weighted
	}

	return result
}

func loadApp() *app.App {
	a := &app.App{}
	if err := a.LoadMccRisk(); err != nil {
		log.Fatalf("load mcc risk: %v", err)
	}
	if err := a.LoadNormalization(); err != nil {
		log.Fatalf("load normalization: %v", err)
	}
	if err := a.LoadCentroids(); err != nil {
		log.Fatalf("load centroids: %v", err)
	}
	if err := a.LoadOffsets(); err != nil {
		log.Fatalf("load offsets: %v", err)
	}
	if err := a.LoadBBoxes(); err != nil {
		log.Fatalf("load bboxes: %v", err)
	}
	return a
}

func loadEntries(path string, limit int) []testEntry {
	file, err := os.Open(path)
	if err != nil {
		log.Fatalf("open test data: %v", err)
	}
	defer file.Close()

	var data testFile
	if err := json.NewDecoder(file).Decode(&data); err != nil {
		log.Fatalf("decode test data: %v", err)
	}

	if limit > 0 && limit < len(data.Entries) {
		return data.Entries[:limit]
	}
	return data.Entries
}

func scanCluster(a *app.App, top *top, queryQ vs.QuantizedVector, centroidID int) {
	worst := top.worst()
	if len(a.IVF.BBoxMin) > 0 && worst != math.MaxInt64 {
		lb := bboxLowerBound(a, queryQ, centroidID, worst)
		if lb >= worst {
			return
		}
	}

	cluster := a.IVF.Offsets[centroidID]
	start := int(cluster.Offset)
	end := start + int(cluster.Count)*vs.Int16ReferenceSize
	buf := a.IVF.VectorsData[start:end]

	for i := 0; i < int(cluster.Count); i++ {
		base := i * vs.Int16ReferenceSize
		_ = buf[base+28]
		worst := top.worst()
		dist := vs.DistQuantizedFromBuffer(queryQ, buf, base, worst)
		if dist < worst {
			top.push(dist, buf[base+28] == 1)
		}
	}
}

func bboxLowerBound(a *app.App, query vs.QuantizedVector, centroidID int, worst int64) int64 {
	min := a.IVF.BBoxMin[centroidID]
	max := a.IVF.BBoxMax[centroidID]

	var sum int64
	for d := 0; d < 14; d++ {
		q := query[d]
		var diff int64
		if q < min[d] {
			diff = int64(min[d]) - int64(q)
		} else if q > max[d] {
			diff = int64(q) - int64(max[d])
		}
		sum += diff * diff
		if sum >= worst {
			return sum
		}
	}
	return sum
}

func (s *classStats) add(expectedApproved bool, fraudCount int) {
	approved := fraudCount < 3
	switch {
	case expectedApproved && approved:
		s.tn++
	case !expectedApproved && !approved:
		s.tp++
	case expectedApproved && !approved:
		s.fp++
		s.weighted++
	case !expectedApproved && approved:
		s.fn++
		s.weighted += 3
	}
}

func (t *top) worst() int64 {
	if t.size < topK {
		return math.MaxInt64
	}
	return t.dist[t.worstIdx]
}

func (t *top) push(dist int64, label bool) {
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

	worst := 0
	for i := 1; i < t.size; i++ {
		if t.dist[i] > t.dist[worst] {
			worst = i
		}
	}
	t.worstIdx = worst
}

func (t *top) fraudCount() int {
	count := 0
	for i := 0; i < t.size; i++ {
		if t.label[i] {
			count++
		}
	}
	return count
}
