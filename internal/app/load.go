package app

import (
	"encoding/binary"
	"encoding/json"
	"os"

	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
	"golang.org/x/sys/unix"
)

type Normalization struct {
	MaxAmount       float32
	MaxInstallments float32
	AmountVsAvg     float32
	MaxMinutes      float32
	MaxKm           float32
	MaxTxCount      float32
	MaxMerchantAvg  float32
}

type App struct {
	MccRisk       map[string]float32
	Normalization Normalization
	IVF           vs.IVFFile
}

func (a *App) Warmup() {
	const step = 4

	for i := 0; i < len(a.IVF.Centroids); i += step {
		_, _ = a.IVF.IvfSearch(a.IVF.Centroids[i], topK, nProbe)
	}
}

func (a *App) LoadCentroids() error {
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

	a.IVF.Centroids = centroids

	return nil
}

func (a *App) LoadMccRisk() error {
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

	a.MccRisk = risks
	return nil
}

func (a *App) LoadNormalization() error {
	file, err := os.Open("resources/normalization.json")
	if err != nil {
		return err
	}
	defer file.Close()

	var data map[string]float32
	err = json.NewDecoder(file).Decode(&data)
	if err != nil {
		return err
	}

	a.Normalization = Normalization{
		MaxAmount:       data["max_amount"],
		MaxInstallments: data["max_installments"],
		AmountVsAvg:     data["amount_vs_avg_ratio"],
		MaxMinutes:      data["max_minutes"],
		MaxKm:           data["max_km"],
		MaxTxCount:      data["max_tx_count_24h"],
		MaxMerchantAvg:  data["max_merchant_avg_amount"],
	}

	return nil
}

func (a *App) LoadOffsets() error {
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
	a.IVF.Offsets = offsets

	file, err = os.Open("resources/ivf/vectors.bin")
	if err != nil {
		return err
	}
	info, err := file.Stat()
	if err != nil {
		file.Close()
		return err
	}
	size := info.Size()
	data, err := unix.Mmap(int(file.Fd()), 0, int(size), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		file.Close()
		return err
	}
	a.IVF.VectorsData = data
	a.IVF.VectorsFile = file

	return nil
}
