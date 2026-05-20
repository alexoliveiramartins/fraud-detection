package vectorsearch

import (
	"encoding/binary"
	"math"
	"os"
	"sort"
)

const Float32ReferenceSize = 57

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

// acha o centroide mais perto e busca apenas nos vetores daquele centroide
func (ivf *IVF) IvfSearch(query Vector, k int, nProbe int) []Neighbor {
	centroidIDs := ivf.ClosestCentroids(query, nProbe)

	var candidates []Neighbor

	for _, centroidID := range centroidIDs {
		for _, reference := range ivf.Lists[centroidID] {
			candidates = append(candidates, Neighbor{
				Dist:  Dist(query, reference.Vector),
				Label: reference.Label,
			})
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Dist < candidates[j].Dist
	})

	if k > len(candidates) {
		k = len(candidates)
	}

	return candidates[:k]
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
	// encontra os centroids proximos
	centroidIDs := ivf.ClosestCentroids(query, nProbe)

	file, err := os.Open(ivf.VectorsPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	top := make([]Neighbor, 0, k)

	for _, centroidID := range centroidIDs {
		cluster := ivf.Offsets[centroidID]
		size := int(cluster.Count) * Float32ReferenceSize
		buf := make([]byte, size)

		if _, err := file.ReadAt(buf, int64(cluster.Offset)); err != nil {
			return nil, err
		}

		for i := 0; i < int(cluster.Count); i++ {
			base := i * Float32ReferenceSize
			var ref Vector

			for dim := 0; dim < 14; dim++ {
				pos := base + dim*4
				bits := binary.LittleEndian.Uint32(buf[pos : pos+4])
				ref[dim] = math.Float32frombits(bits)
			}

			neighbor := Neighbor{
				Dist:  Dist(query, ref),
				Label: buf[base+56] == 1,
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
