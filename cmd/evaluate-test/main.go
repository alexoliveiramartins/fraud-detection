package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/alexoliveiramartins/fraud-detection/internal/app"
	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

type testFile struct {
	Entries []testEntry `json:"entries"`
}

type testEntry struct {
	Request          vs.Payload `json:"request"`
	ExpectedApproved bool       `json:"expected_approved"`
}

type stats struct {
	tp, tn, fp, fn int
	totalScoreTime time.Duration
}

func main() {
	testPath := flag.String("test", "test/v3/test-data.json", "path to test-data.json")
	nprobesFlag := flag.String("nprobe", "1,2,3", "comma-separated nProbe values to evaluate")
	limit := flag.Int("limit", 0, "max entries to evaluate; 0 means all")
	flag.Parse()

	nprobes, err := parseNProbes(*nprobesFlag)
	if err != nil {
		log.Fatalf("parse nprobe: %v", err)
	}

	a := loadApp()
	entries := loadEntries(*testPath, *limit)
	fmt.Printf("entries: %d\n", len(entries))
	fmt.Println("nprobe\tFP\tFN\tTP\tTN\tweighted_E\tfailure_%\tdetection_score\tavg_score_us")

	for _, nprobe := range nprobes {
		result := evaluate(a, entries, nprobe)
		weightedErrors := result.fp + result.fn*3
		failureRate := float64(result.fp+result.fn) / float64(result.total())
		detectionScore := scoreDetection(weightedErrors, result.total())
		avgScoreUS := float64(result.totalScoreTime.Nanoseconds()) / float64(result.total()) / 1000

		fmt.Printf(
			"%d\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.2f\t%.2f\n",
			nprobe,
			result.fp,
			result.fn,
			result.tp,
			result.tn,
			weightedErrors,
			failureRate*100,
			detectionScore,
			avgScoreUS,
		)
	}
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

func evaluate(a *app.App, entries []testEntry, nprobe int) stats {
	var result stats
	for _, entry := range entries {
		vec := a.MakeVector(entry.Request)

		started := time.Now()
		score, err := a.IVF.IvfSearch(vec, 5, nprobe)
		result.totalScoreTime += time.Since(started)
		if err != nil {
			log.Fatalf("ivf search: %v", err)
		}

		approved := score < 0.6
		switch {
		case entry.ExpectedApproved && approved:
			result.tn++
		case !entry.ExpectedApproved && !approved:
			result.tp++
		case entry.ExpectedApproved && !approved:
			result.fp++
		case !entry.ExpectedApproved && approved:
			result.fn++
		}
	}
	return result
}

func (s stats) total() int {
	return s.tp + s.tn + s.fp + s.fn
}

func scoreDetection(weightedErrors, total int) float64 {
	const (
		k          = 1000
		epsilonMin = 0.001
		beta       = 300
	)

	if total == 0 {
		return 0
	}
	epsilon := float64(weightedErrors) / float64(total)
	rateComponent := k * math.Log10(1/math.Max(epsilon, epsilonMin))
	absolutePenalty := -beta * math.Log10(1+float64(weightedErrors))
	return rateComponent + absolutePenalty
}

func parseNProbes(raw string) ([]int, error) {
	parts := strings.Split(raw, ",")
	nprobes := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			return nil, err
		}
		if value <= 0 {
			return nil, fmt.Errorf("nprobe must be positive: %d", value)
		}
		nprobes = append(nprobes, value)
	}
	if len(nprobes) == 0 {
		return nil, fmt.Errorf("at least one nprobe is required")
	}
	return nprobes, nil
}
