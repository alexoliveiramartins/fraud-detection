package vectorsearch

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// const Float32ReferenceSize = 57
const (
	Int16ReferenceSize = 29
	fixedTopK          = 5
	QuantScale         = 10000
	MaxNProbe          = 64
)

var nProbeScaling = [6]int{12, 16, 28, 28, 28, 12}

type QuantizedVector [14]int16

func SampleReferences(items []Reference, sampleSize int, seed int64) []Reference {
	if sampleSize >= len(items) {
		return items
	}

	rng := rand.New(rand.NewSource(seed))

	ids := make([]int, len(items))
	for i := range ids {
		ids[i] = i
	}

	for i := 0; i < sampleSize; i++ {
		j := i + rng.Intn(len(items)-i)
		ids[i], ids[j] = ids[j], ids[i]
	}

	sample := make([]Reference, sampleSize)
	for i := 0; i < sampleSize; i++ {
		sample[i] = items[ids[i]]
	}

	return sample
}

func (ivf *IVF) Build(items []Reference, nCentroids int) {
	sampleSize := 65536
	var seed int64 = 42
	sample := SampleReferences(items, sampleSize, seed)

	ivf.Centroids = TrainCentroids(sample, nCentroids)
	ivf.Lists = make([][]Reference, nCentroids)

	// para cada item, acha o centroide mais perto
	// e guarda o item no indice do centroide
	for _, item := range items {
		centroid := ivf.ClosestCentroid(item.Vector)
		ivf.Lists[centroid] = append(ivf.Lists[centroid], item)
	}
}

func (ivf *IVF) ClosestCentroids(query Vector, nProbe int) []int {
	ids := make([]int, len(ivf.Centroids))

	for i := range ivf.Centroids {
		ids[i] = i
	}

	sort.Slice(ids, func(i, j int) bool {
		distI := Dist(query, ivf.Centroids[ids[i]])
		distJ := Dist(query, ivf.Centroids[ids[j]])

		return distI < distJ
	})

	if nProbe > len(ids) {
		nProbe = len(ids)
	}

	return ids[:nProbe]
}

func (ivf *IVF) ClosestCentroid(vec Vector) int {
	closest := Dist(vec, ivf.Centroids[0])
	centroid := 0

	for j := 1; j < len(ivf.Centroids); j++ {
		d := Dist(vec, ivf.Centroids[j])
		if d < closest {
			closest = d
			centroid = j
		}
	}
	return centroid
}

func InitCentroidsKmeansPlus(items []Reference, nCentroids int, seed int64) []Vector {
	rng := rand.New(rand.NewSource(seed))

	centroids := make([]Vector, nCentroids)
	minDists := make([]float32, len(items))

	first := rng.Intn(len(items))
	centroids[0] = items[first].Vector

	for i := range items {
		minDists[i] = Dist(items[i].Vector, centroids[0])
	}

	// escolhe o centroide mais distante (soma) dos centroides escolhidos
	for c := 1; c < nCentroids; c++ {
		// fmt.Printf("Inicializando centroide %d...\r", c)
		var total float64
		for _, d := range minDists {
			total += float64(d)
		}

		if total == 0 {
			centroids[c] = items[rng.Intn(len(items))].Vector
			continue
		}

		target := rng.Float64() * total
		var acc float64
		chosen := len(items) - 1

		for i, d := range minDists {
			acc += float64(d)
			if acc >= target {
				chosen = i
				break
			}
		}

		centroids[c] = items[chosen].Vector

		for i := range items {
			d := Dist(items[i].Vector, centroids[c])
			if d < minDists[i] {
				minDists[i] = d
			}
		}
	}
	// fmt.Println(" ")
	return centroids
}

func TrainCentroids(items []Reference, nCentroids int) []Vector {
	// init com kmeans++
	fmt.Printf("Inicializando centroides com kmeans++...\n")
	centroids := InitCentroidsKmeansPlus(items, nCentroids, 80)
	fmt.Printf("Treinando centroides...\n")
	maxIterations := 8
	for iter := 0; iter < maxIterations; iter++ {
		sums := make([]Vector, nCentroids)
		counts := make([]int, nCentroids)

		for _, item := range items {
			// fmt.Printf("Iteração numero: %d | Item (3M): %d \r", iter, i)
			closest := 0
			bestDist := Dist(item.Vector, centroids[0])

			// para cada (item), acha o centroide mais perto
			for i := 1; i < nCentroids; i++ {
				d := Dist(item.Vector, centroids[i])
				if d < bestDist {
					bestDist = d
					closest = i
				}
			}

			counts[closest]++
			for j := 0; j < 14; j++ {
				sums[closest][j] += item.Vector[j]
			}
		}

		for i := 0; i < nCentroids; i++ {
			if counts[i] == 0 {
				continue
			}

			inv := 1 / float32(counts[i])
			for j := 0; j < 14; j++ {
				centroids[i][j] = sums[i][j] * inv
			}
		}
	}
	return centroids
}

// distancia euclidiana ao quadrado (d2)
func Dist(a, b Vector) float32 {
	var sum float32

	diff1 := a[1] - b[1]
	sum += diff1 * diff1

	diff2 := a[2] - b[2]
	sum += diff2 * diff2

	diff3 := a[3] - b[3]
	sum += diff3 * diff3

	diff4 := a[4] - b[4]
	sum += diff4 * diff4

	diff5 := a[5] - b[5]
	sum += diff5 * diff5

	diff6 := a[6] - b[6]
	sum += diff6 * diff6

	diff7 := a[7] - b[7]
	sum += diff7 * diff7

	diff8 := a[8] - b[8]
	sum += diff8 * diff8

	diff9 := a[9] - b[9]
	sum += diff9 * diff9

	diff10 := a[10] - b[10]
	sum += diff10 * diff10

	diff11 := a[11] - b[11]
	sum += diff11 * diff11

	diff12 := a[12] - b[12]
	sum += diff12 * diff12

	diff13 := a[13] - b[13]
	sum += diff13 * diff13

	return sum
}

// especificas do IVF por arquivos .bin com offsets

func (ivf *IVFFile) ClosestCentroid(query Vector) int {
	bestID := 0
	bestDist := Dist(query, ivf.Centroids[0])

	for id := 1; id < len(ivf.Centroids); id++ {
		dist := Dist(query, ivf.Centroids[id])

		if dist < bestDist {
			bestID = id
			bestDist = dist
		}
	}

	return bestID
}

// hot path da aplicacao
func (ivf *IVFFile) IvfSearch(query Vector, k int, nProbe int) (float32, error) {
	queryQ := QuantizeVector(query)
	var top fixedTop

	var centroidIDs [MaxNProbe]int
	ivf.ClosestCentroids(query, MaxNProbe, &centroidIDs)

	// closestCentroidID := ivf.ClosestCentroid(query)

	ivf.searchIntoAdditionalTop(&top, queryQ, &centroidIDs, 0, nProbe)

	fraudCount := top.fraudCount()

	// busca em mais clusters dinamicamente
	additionalClusters := nProbeScaling[fraudCount] - nProbe
	if additionalClusters > 0 {
		ivf.searchIntoAdditionalTop(&top, queryQ, &centroidIDs, nProbe, nProbeScaling[fraudCount])
		fraudCount = top.fraudCount()
	}

	// evita fazer calculo da divisao ja que topK = 5 sempre na rinha (micro-otimizacao)
	switch fraudCount {
	case 0:
		return 0, nil
	case 1:
		return 0.2, nil
	case 2:
		return 0.4, nil
	case 3:
		return 0.6, nil
	case 4:
		return 0.8, nil
	default:
		return 1, nil
	}
}

// preenche "ids" com os ids dos (nProbe) centroides mais proximos
func (ivf *IVFFile) ClosestCentroids(
	query Vector,
	nProbe int,
	ids *[MaxNProbe]int,
) {
	var dists [MaxNProbe]float32
	size := 0
	worstIdx := 0
	var worstDist float32

	for id := 0; id < len(ivf.Centroids); id++ {
		dist := Dist(query, ivf.Centroids[id])

		// preenche os nProbe primeiros
		if size < nProbe {
			ids[size] = id
			dists[size] = dist

			// escolhe a pior distancia e o indice dela
			if size == 0 || dist > worstDist {
				worstDist = dist
				worstIdx = size
			}

			size++
			continue
		}
		// (early exit) pula se a distancia ja for pior do que a pior distancia no topK
		if dist >= worstDist {
			continue
		}

		// substitui a pior distancia atual pela nova
		ids[worstIdx] = id
		dists[worstIdx] = dist

		// escolhe novamente a pior distancia
		worstIdx = 0
		worstDist = dists[0]
		for i := 1; i < nProbe; i++ {
			if dists[i] > worstDist {
				worstDist = dists[i]
				worstIdx = i
			}
		}
	}

	for i := 1; i < nProbe; i++ {
		id := ids[i]
		dist := dists[i]
		j := i - 1

		for j >= 0 && dists[j] > dist {
			ids[j+1] = ids[j]
			dists[j+1] = dists[j]
			j--
		}

		ids[j+1] = id
		dists[j+1] = dist
	}
}

func (ivf *IVFFile) searchIntoAdditionalTop(
	top *fixedTop,
	queryQ QuantizedVector,
	centroidIDs *[MaxNProbe]int,
	// skipID int,
	start int,
	end int,
) {
	for i := start; i < end; i++ {
		// if centroidIDs[i] == skipID {
		// 	continue
		// }
		ivf.scanCluster(top, queryQ, centroidIDs[i])
	}
}

func (ivf *IVFFile) scanCluster(top *fixedTop, queryQ QuantizedVector, centroidID int) {
	cluster := ivf.Offsets[centroidID]

	start := int(cluster.Offset)
	end := start + int(cluster.Count)*Int16ReferenceSize
	buf := ivf.VectorsData[start:end]

	for i := 0; i < int(cluster.Count); i++ {
		base := i * Int16ReferenceSize

		worst := top.worst()
		dist := DistQuantizedFromBuffer(queryQ, buf, base, worst)

		if dist < worst {
			top.push(dist, buf[base+28] == 1)
		}
	}
}

func (t *fixedTop) fraudCount() int {
	count := 0
	for i := 0; i < t.size; i++ {
		if t.label[i] {
			count++
		}
	}
	return count
}

func (t *fixedTop) worst() int64 {
	if t.size < fixedTopK {
		return math.MaxInt64
	}
	return t.dist[t.worstIdx]
}

// insercao no topK
func (t *fixedTop) push(dist int64, label bool) {
	// se o tamanho da array for menor que o topK, adiciona os (topK) primeiros
	// e incrementa o tamanho da array
	if t.size < fixedTopK {
		idx := t.size
		t.dist[idx] = dist
		t.label[idx] = label

		if t.size == 0 || dist > t.dist[t.worstIdx] {
			t.worstIdx = idx
		}
		t.size++
		return
	}

	// se a distancia for maior do que a pior (maior ditancia no topK),
	// ignora essa distancia
	if dist >= t.dist[t.worstIdx] {
		return
	}

	// se nao, entao adiciona no topK como pior distancia
	t.dist[t.worstIdx] = dist
	t.label[t.worstIdx] = label
	// depois reordena o topK para caso essa que foi inserida
	// nao for a pior mesmo, e deixa a pior em worstIdx
	t.recomputeWorst()
}

func (t *fixedTop) recomputeWorst() {
	worst := 0
	// percorre o topK para encontrar a >real< pior distancia (maior distancia)
	for i := 1; i < t.size; i++ {
		if t.dist[i] > t.dist[worst] {
			worst = i
		}
	}
	// worstIdx guarda a pior distancia
	t.worstIdx = worst
}

func DistQuantizedFromBuffer(query QuantizedVector, buf []byte, base int, worstDist int64) int64 {
	var sum int64

	// for loop unrolled para melhor performance e menos alloc/req
	diff0 := int64(query[0]) - int64(int16(uint16(buf[base])|uint16(buf[base+1])<<8))
	sum += diff0 * diff0
	if sum >= worstDist {
		return sum
	}

	diff1 := int64(query[1]) - int64(int16(uint16(buf[base+2])|uint16(buf[base+3])<<8))
	sum += diff1 * diff1
	if sum >= worstDist {
		return sum
	}

	diff2 := int64(query[2]) - int64(int16(uint16(buf[base+4])|uint16(buf[base+5])<<8))
	sum += diff2 * diff2
	if sum >= worstDist {
		return sum
	}

	diff3 := int64(query[3]) - int64(int16(uint16(buf[base+6])|uint16(buf[base+7])<<8))
	sum += diff3 * diff3
	if sum >= worstDist {
		return sum
	}

	diff4 := int64(query[4]) - int64(int16(uint16(buf[base+8])|uint16(buf[base+9])<<8))
	sum += diff4 * diff4
	if sum >= worstDist {
		return sum
	}

	diff5 := int64(query[5]) - int64(int16(uint16(buf[base+10])|uint16(buf[base+11])<<8))
	sum += diff5 * diff5
	if sum >= worstDist {
		return sum
	}

	diff6 := int64(query[6]) - int64(int16(uint16(buf[base+12])|uint16(buf[base+13])<<8))
	sum += diff6 * diff6
	if sum >= worstDist {
		return sum
	}

	diff7 := int64(query[7]) - int64(int16(uint16(buf[base+14])|uint16(buf[base+15])<<8))
	sum += diff7 * diff7
	if sum >= worstDist {
		return sum
	}

	diff8 := int64(query[8]) - int64(int16(uint16(buf[base+16])|uint16(buf[base+17])<<8))
	sum += diff8 * diff8
	if sum >= worstDist {
		return sum
	}

	diff9 := int64(query[9]) - int64(int16(uint16(buf[base+18])|uint16(buf[base+19])<<8))
	sum += diff9 * diff9
	if sum >= worstDist {
		return sum
	}

	diff10 := int64(query[10]) - int64(int16(uint16(buf[base+20])|uint16(buf[base+21])<<8))
	sum += diff10 * diff10
	if sum >= worstDist {
		return sum
	}

	diff11 := int64(query[11]) - int64(int16(uint16(buf[base+22])|uint16(buf[base+23])<<8))
	sum += diff11 * diff11
	if sum >= worstDist {
		return sum
	}

	diff12 := int64(query[12]) - int64(int16(uint16(buf[base+24])|uint16(buf[base+25])<<8))
	sum += diff12 * diff12
	if sum >= worstDist {
		return sum
	}

	diff13 := int64(query[13]) - int64(int16(uint16(buf[base+26])|uint16(buf[base+27])<<8))
	sum += diff13 * diff13
	if sum >= worstDist {
		return sum
	}

	return sum
}

// transforma float32 -> int16 em [-1, 1]
func EncodeFloat(v float32) int16 {
	if v < -1 {
		v = -1
	}
	if v > 1 {
		v = 1
	}
	return int16(math.Round(float64(v * QuantScale)))
}

func QuantizeVector(vec Vector) QuantizedVector {
	var q QuantizedVector

	for i := range 14 {
		q[i] = EncodeFloat(vec[i])
	}

	return q
}
