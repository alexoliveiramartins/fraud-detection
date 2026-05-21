package vectorsearch

import (
	"encoding/binary"
	"math"
	"sort"
)

// const Float32ReferenceSize = 57
const Int16ReferenceSize = 29

const QuantScale = 10000

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

	// inicializa os centroids apenas com os 'i' primeiros vetores
	for i := 0; i < nCentroids; i++ {
		centroids[i] = items[i].Vector
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

func (ivf *IVFFile) ClosestCentroids(query Vector, nProbe int) []int {
	ids := make([]int, len(ivf.Centroids))

	for i := range ivf.Centroids {
		ids[i] = i
	}

	// ordena o vetor de centroids por distancia ate o vetor da query
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

func (ivf *IVFFile) IvfSearch(query Vector, k int, nProbe int) ([]Neighbor, error) {
	queryQ := QuantizeVector(query)

	// encontra os centroids proximos
	centroidIDs := ivf.ClosestCentroids(query, nProbe)

	top := make([]Neighbor, 0, k)

	for _, centroidID := range centroidIDs {
		cluster := ivf.Offsets[centroidID]

		start := int(cluster.Offset)
		end := start + int(cluster.Count)*Int16ReferenceSize

		buf := ivf.VectorsData[start:end]

		for i := 0; i < int(cluster.Count); i++ {
			base := i * Int16ReferenceSize

			neighbor := Neighbor{
				Dist:  DistQuantizedFromBuffer(queryQ, buf, base),
				Label: buf[base+28] == 1,
			}

			insertTopK(&top, neighbor, k)
		}
	}

	return top, nil
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

func DistQuantizedFromBuffer(query QuantizedVector, buf []byte, base int) int64 {
	var sum int64
	for dim := 0; dim < 14; dim++ {
		pos := base + dim*2
		refValue := int16(binary.LittleEndian.Uint16(buf[pos : pos+2]))

		diff := int64(query[dim]) - int64(refValue)
		sum += diff * diff
	}
	return sum
}
