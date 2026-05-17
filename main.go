package main

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

func loadReferences() error {
	file, err := os.Open("resources/references.json.gz")
	if err != nil {
		return err
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return err
	}
	defer gz.Close()

	var fileReferences []Reference

	err = json.NewDecoder(gz).Decode(&fileReferences)
	if err != nil {
		return err
	}

	references = fileReferences
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

var mccRisk map[string]float32
var normalization map[string]float32
var references []Reference

func main() {
	err := loadMccRisk()
	if err != nil {
		fmt.Println(err)
		return
	}
	err = loadNormalization()
	if err != nil {
		fmt.Println(err)
		return
	}
	err = loadReferences()
	if err != nil {
		fmt.Println(err)
		return
	}

	http.HandleFunc("/ready", readyHandler)
	http.HandleFunc("/fraud-score", fraudScoreHandler)

	fmt.Println("Server running on http://localhost:8080")
	err = http.ListenAndServe(":8080", nil)
	if err != nil {
		fmt.Println(err)
	}
}
