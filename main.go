package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

func loadReferences() error {
	file, err := os.Open("resources/references.bin")
	if err != nil {
		return err
	}
	defer file.Close()

	var fileReferences []vs.Reference

	for {
		var ref vs.Reference

		for i := range 14 {
			err := binary.Read(file, binary.LittleEndian, &ref.Vector[i])
			if err != nil {
				if err == io.EOF {
					references = fileReferences
					return nil
				}
				return err
			}
		}

		var label [1]byte
		_, err := file.Read(label[:])
		if err != nil {
			return err
		}

		ref.Label = label[0] == 1

		fileReferences = append(fileReferences, ref)
	}
}

func loadCentroids() error {
	file, err := os.Open("resources/ivf/centroids.bin")
	if err != nil {
		return err
	}
	defer file.Close()

	var count uint32
	if err := binary.Read(file, binary.LittleEndian, &count); err != nil {
		return err
	}

	centroids := make([]vs.Vector, count)

	for i := range centroids {
		for j := range 14 {
			if err := binary.Read(file, binary.LittleEndian, &centroids[i][j]); err != nil {
				return err
			}
		}
	}

	ivfIndexes.Centroids = centroids

	return nil
}

func loadMccRisk() error {
	file, err := os.Open("resources/mcc_risk.json")
	if err != nil {
		return err
	}
	defer file.Close()

	var risks map[string]float32

	err = json.NewDecoder(file).Decode(&risks)
	if err != nil {
		return err
	}

	mccRisk = risks
	return nil
}

func loadNormalization() error {
	file, err := os.Open("resources/normalization.json")
	if err != nil {
		return err
	}
	defer file.Close()

	var norm map[string]float32

	err = json.NewDecoder(file).Decode(&norm)
	if err != nil {
		return err
	}

	normalization = norm
	return nil
}

func loadOffsets() error {
	file, err := os.Open("resources/ivf/offsets.bin")
	if err != nil {
		return err
	}
	defer file.Close()

	var count uint32
	if err := binary.Read(file, binary.LittleEndian, &count); err != nil {
		return err
	}

	offsets := make([]vs.ClusterOffset, count)

	for i := range offsets {
		if err := binary.Read(file, binary.LittleEndian, &offsets[i].Offset); err != nil {
			return err
		}
		if err := binary.Read(file, binary.LittleEndian, &offsets[i].Count); err != nil {
			return err
		}
	}

	ivfIndexes.Offsets = offsets
	ivfIndexes.VectorsPath = "resources/ivf/vectors.bin"

	return nil
}

var mccRisk map[string]float32
var normalization map[string]float32
var references []vs.Reference

var ivfIndexes vs.IVFFile

var topK int = 5

func main() {
	fmt.Printf("Carregando mcc_risk.json...\n")
	err := loadMccRisk()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Carregando normalization.json...\n")
	err = loadNormalization()
	if err != nil {
		fmt.Println(err)
		return
	}
	// fmt.Printf("Carregando references.bin...\n")
	// err = loadReferences()
	// if err != nil {
	// 	fmt.Println(err)
	// 	return
	// }

	err = loadCentroids()
	if err != nil {
		panic(err)
	}

	err = loadOffsets()
	if err != nil {
		panic(err)
	}

	// fmt.Printf("Treinando centroides do indice IVF... ")
	// ivfIndexes.Build(references, 512)
	// fmt.Printf("Concluído!\n")

	http.HandleFunc("/ready", readyHandler)
	http.HandleFunc("/fraud-score", fraudScoreHandler)

	fmt.Println("Server running on http://localhost:8080")
	err = http.ListenAndServe(":8080", nil)
	if err != nil {
		fmt.Println(err)
	}
}
