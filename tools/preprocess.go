package main

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"os"

	. "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

var ivfIndex IVF

const nCentroids = 1024

func toReferences(rawRefs []RawReference) []Reference {
	refs := make([]Reference, len(rawRefs))
	for i, raw := range rawRefs {
		refs[i] = Reference{
			Vector: raw.Vector,
			Label:  raw.Label == "fraud",
		}
	}

	return refs
}

func loadReferences() []RawReference {
	// if _, err := os.Stat("resources/references.bin"); err == nil {
	// 	fmt.Println("resources/references.bin already exists, skipping preprocess")
	// 	return
	// }

	input, err := os.Open("resources/references.json.gz")
	if err != nil {
		panic(err)
	}
	defer input.Close()

	gz, err := gzip.NewReader(input)
	if err != nil {
		panic(err)
	}
	defer gz.Close()

	decoder := json.NewDecoder(gz)

	var refs []RawReference
	err = decoder.Decode(&refs)
	if err != nil {
		panic(err)
	}
	return refs
}

// funcao para salvar os centroides em formato binario
func writeCentroids(path string, centroids []Vector) {
	// cria o arquivo binario centroids.bin
	output, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer output.Close()

	// os primeiros 4 bits = count do tamanho da lista de centroids
	count := uint32(len(centroids))
	if err := binary.Write(output, binary.LittleEndian, count); err != nil {
		panic(err)
	}

	// escreve cada centroide no binario
	for _, centroid := range centroids { // em cada centroide
		for _, value := range centroid { // grava as 14 dimensoes
			if err := binary.Write(output, binary.LittleEndian, value); err != nil {
				panic(err)
			}
		}
	}

	// estrutura final fica assim:
	// uint32 count
	// centroid 0
	// 		14 float32
	// centroid 1
	// 		14 float32
	// ...
	// centroid nCentroids
	// 		14 float32
}

func writeClusters(vectorsPath, offsetsPath string, clusters [][]Reference) {
	// cria o arquivo vectors.bin
	vectorsFile, err := os.Create(vectorsPath)
	if err != nil {
		panic(err)
	}
	defer vectorsFile.Close()

	// cria o arquivo offsets.bin
	offsetsFile, err := os.Create(offsetsPath)
	if err != nil {
		panic(err)
	}
	defer offsetsFile.Close()

	// escreve o numero de clusters (= numero de centroides) no arquivo offsets.bin
	clusterCount := uint32(len(clusters))
	if err := binary.Write(offsetsFile, binary.LittleEndian, clusterCount); err != nil {
		panic(err)
	}

	var currentOffset uint64 = 0
	// const refSize uint64 = 57 // float32 + 1byte
	const refSize uint64 = 29 // 14 int16 + 1 byte label

	for _, cluster := range clusters { // para cada cluster
		// escreve o tamanho do cluster (individual) e o offset atual no arquivo de offsets.bin
		count := uint32(len(cluster))
		if err := binary.Write(offsetsFile, binary.LittleEndian, currentOffset); err != nil {
			panic(err)
		}
		if err := binary.Write(offsetsFile, binary.LittleEndian, count); err != nil {
			panic(err)
		}

		// para cada ref (vetor) no cluster
		for _, ref := range cluster {
			// escreve as dimensoes do vetor (ref) no arquivo vectors.bin
			for _, value := range ref.Vector {
				// quantiza o vetor de float32 -> int16 (metade do tamanho)
				q := EncodeFloat(value)
				if err := binary.Write(vectorsFile, binary.LittleEndian, q); err != nil {
					panic(err)
				}
			}
			// no final escreve o byte da label (1/0)(fraude/legit)
			var label byte = 0
			if ref.Label {
				label = 1
			}
			if _, err := vectorsFile.Write([]byte{label}); err != nil {
				panic(err)
			}
		}

		// incrementa o offset pelo tamanho do cluster * tamanho do dado
		// (14 * refSize + 1 bit da label)
		currentOffset += uint64(len(cluster)) * refSize
	}
}

func main() {
	rawRefs := loadReferences()
	refs := toReferences(rawRefs)

	ivfIndex.Build(refs, nCentroids)

	if err := os.MkdirAll("resources/ivf", 0755); err != nil {
		panic(err)
	}

	writeCentroids("resources/ivf/centroids.bin", ivfIndex.Centroids)
	// fmt.Printf("wrote %d centroids\n", len(ivfIndex.Centroids))
	writeClusters("resources/ivf/vectors.bin", "resources/ivf/offsets.bin", ivfIndex.Lists)
}
