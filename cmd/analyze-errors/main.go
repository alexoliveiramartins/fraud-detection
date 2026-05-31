package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/alexoliveiramartins/fraud-detection/internal/app"
	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

const topK = 5

var featureNames = [14]string{
	"amount",
	"installments",
	"amount_vs_customer_avg",
	"hour",
	"weekday",
	"minutes_since_last",
	"km_from_last",
	"km_from_home",
	"tx_count_24h",
	"is_online",
	"card_present",
	"unknown_merchant",
	"mcc_risk",
	"merchant_avg_amount",
}

type testFile struct {
	Entries []testEntry `json:"entries"`
}

type testEntry struct {
	Request            vs.Payload `json:"request"`
	ExpectedApproved   bool       `json:"expected_approved"`
	ExpectedFraudScore float32    `json:"expected_fraud_score"`
}

type neighbor struct {
	Dist    int64
	Label   bool
	Cluster int
}

type traceTop struct {
	items    [topK]neighbor
	size     int
	worstIdx int
}

type searchTrace struct {
	Score             float32
	FraudCount        int
	InitialFraudCount int
	ClosestCentroid   int
	Selective         bool
	CentroidIDs       []int
	ScannedClusters   []int
	Neighbors         []neighbor
}

type confusion struct {
	tp int
	tn int
	fp int
	fn int
}

type errorCase struct {
	Index         int
	Kind          string
	Expected      bool
	Approved      bool
	ExpectedScore float32
	Vector        vs.Vector
	Trace         searchTrace
	Payload       vs.Payload
}

type aggregate struct {
	count          int
	amount         float64
	amountRatio    float64
	tx24           float64
	kmHome         float64
	merchantAvg    float64
	mccRisk        float64
	lastCount      int
	lastMinutes    float64
	lastKm         float64
	knownMerchant  int
	online         int
	cardPresent    int
	featureSum     [14]float64
	scoreBuckets   map[string]int
	mccBuckets     map[string]int
	topLabelBucket map[string]int
}

func main() {
	testPath := flag.String("test", "test/v3/test-data.json", "path to test-data.json")
	nProbe := flag.Int("nprobe", 12, "nProbe value to evaluate")
	forceAll := flag.Bool("force-all", false, "scan nProbe clusters for every query instead of only score-borderline queries")
	outPath := flag.String("out", "", "markdown report output path")
	limit := flag.Int("limit", 0, "max entries to evaluate; 0 means all")
	flag.Parse()

	if *nProbe <= 0 {
		log.Fatalf("nprobe must be positive")
	}
	if *nProbe > vs.MaxNProbe {
		log.Fatalf("nprobe=%d exceeds vectorsearch.MaxNProbe=%d", *nProbe, vs.MaxNProbe)
	}

	a := loadApp()
	entries := loadEntries(*testPath, *limit)
	errors, counts := analyze(a, entries, *nProbe, *forceAll)

	reportPath := *outPath
	if reportPath == "" {
		suffix := ""
		if *forceAll {
			suffix = "-force-all"
		}
		reportPath = fmt.Sprintf("test/v3/error-report-nprobe%d%s.md", *nProbe, suffix)
	}

	if err := writeReport(reportPath, *testPath, *nProbe, *forceAll, len(entries), counts, errors); err != nil {
		log.Fatalf("write report: %v", err)
	}

	weightedErrors := counts.fp + counts.fn*3
	fmt.Printf("wrote %s\n", reportPath)
	fmt.Printf("entries=%d FP=%d FN=%d TP=%d TN=%d weighted_E=%d detection_score=%.2f\n",
		len(entries), counts.fp, counts.fn, counts.tp, counts.tn, weightedErrors, detectionScore(weightedErrors, len(entries)))
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

func analyze(a *app.App, entries []testEntry, nProbe int, forceAll bool) ([]errorCase, confusion) {
	var errors []errorCase
	var counts confusion

	for i, entry := range entries {
		vec := a.MakeVector(entry.Request)
		trace := traceSearch(a, vec, nProbe, forceAll)
		approved := trace.Score < 0.6

		switch {
		case entry.ExpectedApproved && approved:
			counts.tn++
		case !entry.ExpectedApproved && !approved:
			counts.tp++
		case entry.ExpectedApproved && !approved:
			counts.fp++
			errors = append(errors, errorCase{
				Index:         i,
				Kind:          "FP",
				Expected:      entry.ExpectedApproved,
				Approved:      approved,
				ExpectedScore: entry.ExpectedFraudScore,
				Vector:        vec,
				Trace:         trace,
				Payload:       entry.Request,
			})
		case !entry.ExpectedApproved && approved:
			counts.fn++
			errors = append(errors, errorCase{
				Index:         i,
				Kind:          "FN",
				Expected:      entry.ExpectedApproved,
				Approved:      approved,
				ExpectedScore: entry.ExpectedFraudScore,
				Vector:        vec,
				Trace:         trace,
				Payload:       entry.Request,
			})
		}
	}

	return errors, counts
}

func traceSearch(a *app.App, query vs.Vector, nProbe int, forceAll bool) searchTrace {
	queryQ := vs.QuantizeVector(query)
	var top traceTop

	closest := a.IVF.ClosestCentroid(query)
	scanCluster(a, &top, queryQ, closest)

	trace := searchTrace{
		ClosestCentroid: closest,
		ScannedClusters: []int{closest},
	}

	initialFraudCount := top.fraudCount()
	trace.InitialFraudCount = initialFraudCount

	if nProbe > 1 && (forceAll || initialFraudCount == 2 || initialFraudCount == 3) {
		trace.Selective = true
		var centroidIDs [vs.MaxNProbe]int
		a.IVF.ClosestCentroids(&query, &centroidIDs)
		trace.CentroidIDs = centroidIDs[:nProbe:nProbe]

		for i := 0; i < nProbe; i++ {
			centroidID := centroidIDs[i]
			if centroidID == closest {
				continue
			}
			scanCluster(a, &top, queryQ, centroidID)
			trace.ScannedClusters = append(trace.ScannedClusters, centroidID)
		}
	}

	trace.FraudCount = top.fraudCount()
	trace.Score = scoreFromFraudCount(trace.FraudCount)
	trace.Neighbors = top.sorted()
	return trace
}

func scanCluster(a *app.App, top *traceTop, queryQ vs.QuantizedVector, centroidID int) {
	cluster := a.IVF.Offsets[centroidID]
	start := int(cluster.Offset)
	end := start + int(cluster.Count)*vs.Int16ReferenceSize
	buf := a.IVF.VectorsData[start:end]

	for i := 0; i < int(cluster.Count); i++ {
		base := i * vs.Int16ReferenceSize
		worst := top.worst()
		dist := vs.DistQuantizedFromBuffer(queryQ, buf, base, worst)
		if dist < worst {
			top.push(neighbor{
				Dist:    dist,
				Label:   buf[base+28] == 1,
				Cluster: centroidID,
			})
		}
	}
}

func (t *traceTop) fraudCount() int {
	count := 0
	for i := 0; i < t.size; i++ {
		if t.items[i].Label {
			count++
		}
	}
	return count
}

func (t *traceTop) worst() int64 {
	if t.size < topK {
		return math.MaxInt64
	}
	return t.items[t.worstIdx].Dist
}

func (t *traceTop) push(item neighbor) {
	if t.size < topK {
		idx := t.size
		t.items[idx] = item
		if t.size == 0 || item.Dist > t.items[t.worstIdx].Dist {
			t.worstIdx = idx
		}
		t.size++
		return
	}

	if item.Dist >= t.items[t.worstIdx].Dist {
		return
	}

	t.items[t.worstIdx] = item
	t.recomputeWorst()
}

func (t *traceTop) recomputeWorst() {
	worst := 0
	for i := 1; i < t.size; i++ {
		if t.items[i].Dist > t.items[worst].Dist {
			worst = i
		}
	}
	t.worstIdx = worst
}

func (t *traceTop) sorted() []neighbor {
	items := make([]neighbor, t.size)
	copy(items, t.items[:t.size])
	sort.Slice(items, func(i, j int) bool {
		return items[i].Dist < items[j].Dist
	})
	return items
}

func writeReport(path string, testPath string, nProbe int, forceAll bool, total int, counts confusion, errors []errorCase) error {
	if err := os.MkdirAll(dir(path), 0o755); err != nil {
		return err
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	weightedErrors := counts.fp + counts.fn*3
	fmt.Fprintf(file, "# Error report\n\n")
	fmt.Fprintf(file, "- test: `%s`\n", testPath)
	fmt.Fprintf(file, "- nProbe: `%d`\n", nProbe)
	fmt.Fprintf(file, "- force_all: `%t`\n", forceAll)
	fmt.Fprintf(file, "- entries: `%d`\n", total)
	fmt.Fprintf(file, "- FP: `%d`\n", counts.fp)
	fmt.Fprintf(file, "- FN: `%d`\n", counts.fn)
	fmt.Fprintf(file, "- TP: `%d`\n", counts.tp)
	fmt.Fprintf(file, "- TN: `%d`\n", counts.tn)
	fmt.Fprintf(file, "- weighted_E: `%d`\n", weightedErrors)
	fmt.Fprintf(file, "- detection_score: `%.2f`\n\n", detectionScore(weightedErrors, total))

	writeScoreDistribution(file, errors)
	writeAggregate(file, errors)
	writeFeatureAverages(file, errors)
	writeCaseTable(file, errors)
	writeVectorTable(file, errors)

	return nil
}

func writeScoreDistribution(file *os.File, errors []errorCase) {
	scoreByKind := map[string]map[string]int{
		"FP": {},
		"FN": {},
	}

	for _, item := range errors {
		scoreByKind[item.Kind][scoreKey(item.Trace.Score)]++
	}

	fmt.Fprintf(file, "## Score distribution\n\n")
	fmt.Fprintf(file, "| score | FP | FN |\n")
	fmt.Fprintf(file, "|---:|---:|---:|\n")
	for _, score := range []string{"0.0", "0.2", "0.4", "0.6", "0.8", "1.0"} {
		fmt.Fprintf(file, "| %s | %d | %d |\n", score, scoreByKind["FP"][score], scoreByKind["FN"][score])
	}
	fmt.Fprintf(file, "\n")
}

func writeAggregate(file *os.File, errors []errorCase) {
	aggregates := map[string]*aggregate{
		"FP": newAggregate(),
		"FN": newAggregate(),
	}

	for _, item := range errors {
		aggregates[item.Kind].add(item)
	}

	fmt.Fprintf(file, "## Aggregate raw signals\n\n")
	fmt.Fprintf(file, "| type | count | avg_amount | avg_amount_ratio | avg_tx24 | avg_km_home | last_tx%% | avg_last_min | avg_last_km | known_merch%% | online%% | card%% | avg_mcc_risk |\n")
	fmt.Fprintf(file, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, kind := range []string{"FP", "FN"} {
		agg := aggregates[kind]
		fmt.Fprintf(file, "| %s | %d | %.2f | %.3f | %.2f | %.2f | %.1f | %.2f | %.2f | %.1f | %.1f | %.1f | %.3f |\n",
			kind,
			agg.count,
			agg.avg(agg.amount),
			agg.avg(agg.amountRatio),
			agg.avg(agg.tx24),
			agg.avg(agg.kmHome),
			agg.percent(agg.lastCount),
			agg.avgPresent(agg.lastMinutes, agg.lastCount),
			agg.avgPresent(agg.lastKm, agg.lastCount),
			agg.percent(agg.knownMerchant),
			agg.percent(agg.online),
			agg.percent(agg.cardPresent),
			agg.avg(agg.mccRisk),
		)
	}
	fmt.Fprintf(file, "\n")

	for _, kind := range []string{"FP", "FN"} {
		agg := aggregates[kind]
		fmt.Fprintf(file, "### Top MCCs %s\n\n", kind)
		fmt.Fprintf(file, "| mcc | count |\n")
		fmt.Fprintf(file, "|---|---:|\n")
		for _, bucket := range sortedBuckets(agg.mccBuckets, 8) {
			fmt.Fprintf(file, "| %s | %d |\n", bucket.key, bucket.value)
		}
		fmt.Fprintf(file, "\n")
	}
}

func writeFeatureAverages(file *os.File, errors []errorCase) {
	aggregates := map[string]*aggregate{
		"FP": newAggregate(),
		"FN": newAggregate(),
	}
	for _, item := range errors {
		aggregates[item.Kind].add(item)
	}

	fmt.Fprintf(file, "## Vector feature averages\n\n")
	fmt.Fprintf(file, "| dim | feature | FP_avg | FN_avg |\n")
	fmt.Fprintf(file, "|---:|---|---:|---:|\n")
	for dim, name := range featureNames {
		fmt.Fprintf(file, "| %d | %s | %.4f | %.4f |\n",
			dim,
			name,
			aggregates["FP"].avg(aggregates["FP"].featureSum[dim]),
			aggregates["FN"].avg(aggregates["FN"].featureSum[dim]),
		)
	}
	fmt.Fprintf(file, "\n")
}

func writeCaseTable(file *os.File, errors []errorCase) {
	sort.Slice(errors, func(i, j int) bool {
		if errors[i].Kind == errors[j].Kind {
			return errors[i].Index < errors[j].Index
		}
		return errors[i].Kind < errors[j].Kind
	})

	fmt.Fprintf(file, "## Error cases\n\n")
	fmt.Fprintf(file, "| idx | id | type | expected_score | score | initial_fc | final_fc | selective | top_labels | top_dists | clusters | mcc | amount | ratio | tx24 | km_home | last_min | last_km | known | online | card |\n")
	fmt.Fprintf(file, "|---:|---|---|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|\n")
	for _, item := range errors {
		p := item.Payload
		fmt.Fprintf(file, "| %d | %s | %s | %.1f | %.1f | %d | %d | %t | %s | %s | %s | %s | %.2f | %.3f | %d | %.2f | %s | %s | %t | %t | %t |\n",
			item.Index,
			escapeCell(p.ID),
			item.Kind,
			item.ExpectedScore,
			item.Trace.Score,
			item.Trace.InitialFraudCount,
			item.Trace.FraudCount,
			item.Trace.Selective,
			topLabels(item.Trace.Neighbors),
			topDists(item.Trace.Neighbors),
			ints(item.Trace.ScannedClusters),
			p.Merchant.Mcc,
			p.Transaction.Amount,
			amountRatio(p),
			p.Customer.TxCount24h,
			p.Terminal.KmFromHome,
			floatCell(lastMinutes(p)),
			floatCell(lastKm(p)),
			knownMerchant(p),
			p.Terminal.IsOnline,
			p.Terminal.CardPresent,
		)
	}
	fmt.Fprintf(file, "\n")
}

func writeVectorTable(file *os.File, errors []errorCase) {
	fmt.Fprintf(file, "## Error vectors\n\n")
	fmt.Fprintf(file, "| idx | type | %s |\n", strings.Join(featureNames[:], " | "))
	fmt.Fprintf(file, "|---:|---|%s|\n", strings.Repeat("---:|", len(featureNames)))
	for _, item := range errors {
		fmt.Fprintf(file, "| %d | %s", item.Index, item.Kind)
		for _, value := range item.Vector {
			fmt.Fprintf(file, " | %.4f", value)
		}
		fmt.Fprintf(file, " |\n")
	}
}

func newAggregate() *aggregate {
	return &aggregate{
		scoreBuckets:   make(map[string]int),
		mccBuckets:     make(map[string]int),
		topLabelBucket: make(map[string]int),
	}
}

func (a *aggregate) add(item errorCase) {
	p := item.Payload
	a.count++
	a.amount += float64(p.Transaction.Amount)
	a.amountRatio += float64(amountRatio(p))
	a.tx24 += float64(p.Customer.TxCount24h)
	a.kmHome += float64(p.Terminal.KmFromHome)
	a.merchantAvg += float64(p.Merchant.AvgAmount)
	a.mccRisk += float64(item.Vector[12])
	a.scoreBuckets[scoreKey(item.Trace.Score)]++
	a.mccBuckets[p.Merchant.Mcc]++
	a.topLabelBucket[topLabels(item.Trace.Neighbors)]++

	if p.LastTransaction != nil {
		a.lastCount++
		a.lastMinutes += float64(*lastMinutes(p))
		a.lastKm += float64(*lastKm(p))
	}
	if knownMerchant(p) {
		a.knownMerchant++
	}
	if p.Terminal.IsOnline {
		a.online++
	}
	if p.Terminal.CardPresent {
		a.cardPresent++
	}
	for dim, value := range item.Vector {
		a.featureSum[dim] += float64(value)
	}
}

func (a *aggregate) avg(sum float64) float64 {
	if a.count == 0 {
		return 0
	}
	return sum / float64(a.count)
}

func (a *aggregate) avgPresent(sum float64, count int) float64 {
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func (a *aggregate) percent(count int) float64 {
	if a.count == 0 {
		return 0
	}
	return float64(count) / float64(a.count) * 100
}

type bucket struct {
	key   string
	value int
}

func sortedBuckets(values map[string]int, limit int) []bucket {
	buckets := make([]bucket, 0, len(values))
	for key, value := range values {
		buckets = append(buckets, bucket{key: key, value: value})
	}
	sort.Slice(buckets, func(i, j int) bool {
		if buckets[i].value == buckets[j].value {
			return buckets[i].key < buckets[j].key
		}
		return buckets[i].value > buckets[j].value
	})
	if limit > 0 && limit < len(buckets) {
		return buckets[:limit]
	}
	return buckets
}

func scoreFromFraudCount(count int) float32 {
	switch count {
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

func detectionScore(weightedErrors int, total int) float64 {
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

func scoreKey(score float32) string {
	return fmt.Sprintf("%.1f", score)
}

func topLabels(items []neighbor) string {
	labels := make([]string, len(items))
	for i, item := range items {
		if item.Label {
			labels[i] = "F"
		} else {
			labels[i] = "L"
		}
	}
	return strings.Join(labels, "")
}

func topDists(items []neighbor) string {
	parts := make([]string, len(items))
	for i, item := range items {
		parts[i] = fmt.Sprint(item.Dist)
	}
	return strings.Join(parts, ",")
}

func ints(values []int) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = fmt.Sprint(value)
	}
	return strings.Join(parts, ",")
}

func amountRatio(p vs.Payload) float32 {
	if p.Customer.AvgAmount == 0 {
		return 0
	}
	return p.Transaction.Amount / p.Customer.AvgAmount
}

func lastMinutes(p vs.Payload) *float64 {
	if p.LastTransaction == nil {
		return nil
	}
	value := p.Transaction.RequestedAt.Sub(p.LastTransaction.Timestamp).Minutes()
	return &value
}

func lastKm(p vs.Payload) *float64 {
	if p.LastTransaction == nil {
		return nil
	}
	value := float64(p.LastTransaction.KmFromCurrent)
	return &value
}

func knownMerchant(p vs.Payload) bool {
	for _, id := range p.Customer.KnownMerchants {
		if id == p.Merchant.ID {
			return true
		}
	}
	return false
}

func floatCell(value *float64) string {
	if value == nil {
		return ""
	}
	return fmt.Sprintf("%.2f", *value)
}

func escapeCell(value string) string {
	return strings.ReplaceAll(value, "|", "\\|")
}

func dir(path string) string {
	idx := strings.LastIndex(path, "/")
	if idx == -1 {
		return "."
	}
	return path[:idx]
}
