package vectorsearch

import (
	"math"
	"math/rand"
	"sort"
)

// const Float32ReferenceSize = 57
const (
	Int16ReferenceSize = 29
	fixedTopK          = 5
	QuantScale         = 10000
	MaxNProbe          = 12
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

// hot path da aplicacao
func (ivf *IVFFile) IvfSearch(query Vector, k int, nProbe int) (float32, error) {
	queryQ := QuantizeVector(query)
	var top fixedTop

	closestCentroidID := ivf.ClosestCentroid(query)
	ivf.scanCluster(&top, queryQ, closestCentroidID)

	fraudCount := top.fraudCount()

	// busca em mais clusters para casos de borda (fraudscore = 0.4 e 0.6)
	if nProbe > 1 {
		var centroidIDs [MaxNProbe]int
		ivf.ClosestCentroids(query, nProbe, &centroidIDs)
		ivf.searchIntoAdditionalTop(&top, queryQ, nProbe, centroidIDs, closestCentroidID)
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
func (ivf *IVFFile) ClosestCentroids(query Vector, nProbe int, ids *[MaxNProbe]int) {
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
}

func (ivf *IVFFile) searchIntoAdditionalTop(
	top *fixedTop,
	queryQ QuantizedVector,
	nProbe int,
	centroidIDs [MaxNProbe]int,
	skipID int,
) {
	for i := 0; i < nProbe; i++ {
		if centroidIDs[i] == skipID {
			continue
		}
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
