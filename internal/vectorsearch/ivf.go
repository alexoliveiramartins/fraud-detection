package vectorsearch

import (
	"encoding/binary"
	"math"
	"math/rand"
	"sort"
)

// const Float32ReferenceSize = 57
const (
	Int16ReferenceSize = 29
	fixedTopK          = 5
	QuantScale         = 10000
)

type QuantizedVector [14]int16

func (ivf *IVF) Build(items []Reference, nCentroids int) {
	ivf.Centroids = TrainCentroids(items, nCentroids)
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

func TrainCentroids(items []Reference, nCentroids int) []Vector {
	centroids := make([]Vector, nCentroids)

	// inicializa os centroids com os n primeiros vetores aleatorios (seed fixa)
	rng := rand.New(rand.NewSource(42))
	perm := rng.Perm(len(items))

	for i := 0; i < nCentroids; i++ {
		centroids[i] = items[perm[i]].Vector
	}

	maxIterations := 20
	for iter := 0; iter < maxIterations; iter++ {
		// para cada centroide (m), um array de vetores (n) -> matriz (m)x(n)
		groups := make([][]Vector, nCentroids)

		for _, item := range items {
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
			// adiciona cada vetor na array do seu centroide mais perto (closest)
			groups[closest] = append(groups[closest], item.Vector)
		}

		// centroid[i] = MÉDIA dos vetores de cada centroide no grupo (k-means)
		for i := 0; i < nCentroids; i++ {
			if len(groups[i]) == 0 {
				continue
			}

			var newCentroid Vector
			// calcula a media somando cada dimensao de todos os vetores dos grupos...
			for _, vec := range groups[i] {
				for j := 0; j < 14; j++ {
					newCentroid[j] += vec[j]
				}
			}
			// ... e divide pelo tamanho do grupo
			for j := 0; j < 14; j++ {
				newCentroid[j] /= float32(len(groups[i]))
			}
			// o vetor médio vira o centroide
			centroids[i] = newCentroid
		}
	}
	return centroids
}

// distancia euclidiana ao quadrado (d2)
func Dist(a, b Vector) float32 {
	var sum float32

	for i := 0; i < 14; i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}

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

func (ivf *IVFFile) ClosestCentroids(query Vector, nProbe int) []int {
	ids := make([]int, 0, nProbe)

	for i := range nProbe {
		ids = append(ids, i)
	}

	sort.Slice(ids, func(c, j int) bool {
		distI := Dist(query, ivf.Centroids[ids[c]])
		distJ := Dist(query, ivf.Centroids[ids[j]])

		return distI < distJ
	})

	for i := nProbe; i < len(ivf.Centroids); i++ {
		// preenche os n primeiros elementos
		if Dist(query, ivf.Centroids[i]) < Dist(query, ivf.Centroids[ids[nProbe-1]]) {
			ids[nProbe-1] = i
			sort.Slice(ids, func(c, j int) bool {
				distI := Dist(query, ivf.Centroids[ids[c]])
				distJ := Dist(query, ivf.Centroids[ids[j]])

				return distI < distJ
			})
		}
	}
	return ids
}

func (ivf *IVFFile) IvfSearch(query Vector, k int, nProbe int) (float32, error) {
	queryQ := QuantizeVector(query)

	var top fixedTop
	ivf.searchIntoTop(&top, query, queryQ, nProbe)

	var score float32
	for _, label := range top.label {
		if label == true {
			score += 1
		}
	}

	return score / float32(top.size), nil
}

func (ivf *IVFFile) searchIntoTop(top *fixedTop, query Vector, queryQ QuantizedVector, nProbe int) {
	if nProbe <= 1 {
		centroidID := ivf.ClosestCentroid(query)
		ivf.scanCluster(top, queryQ, centroidID)
		return
	}

	centroidIDs := ivf.ClosestCentroids(query, nProbe)
	for _, centroidID := range centroidIDs {
		ivf.scanCluster(top, queryQ, centroidID)
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
	for dim := 0; dim < 14; dim++ {
		pos := base + dim*2
		refValue := int16(binary.LittleEndian.Uint16(buf[pos : pos+2]))

		diff := int64(query[dim]) - int64(refValue)
		sum += diff * diff
		if sum >= worstDist {
			return sum
		}
	}
	return sum
}

func insertTopK(top *[]Neighbor, candidate Neighbor, k int) {
	if len(*top) < k {
		*top = append(*top, candidate)
		return
	}

	worst := 0
	for i := 1; i < len(*top); i++ {
		if (*top)[i].Dist > (*top)[worst].Dist {
			worst = i
		}
	}

	if candidate.Dist < (*top)[worst].Dist {
		(*top)[worst] = candidate
	}
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
