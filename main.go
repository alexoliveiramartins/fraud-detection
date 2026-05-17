package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

func limit(n float32) float32 {
	lim := max(0, n)
	lim = min(1, lim)
	return lim
}

func makeVector(p Payload) [14]float32 {
	var vec [14]float32

	vec[0] = limit(p.Transaction.Amount / normalization["max_amount"])
	vec[1] = limit(float32(p.Transaction.Installments) / normalization["max_installments"])
	vec[2] = limit((p.Transaction.Amount / p.Customer.AvgAmount) / normalization["amount_vs_avg_ratio"])
	vec[3] = float32(p.Transaction.RequestedAt.Hour()) / 23
	weekDay := (int(p.Transaction.RequestedAt.Weekday()) - 1) % 6
	vec[4] = float32(weekDay) / float32(6)
	if p.LastTransaction == nil {
		vec[5] = -1
		vec[6] = -1
	} else {
		vec[5] = limit(float32(p.LastTransaction.Timestamp.Minute()) / normalization["max_minutes"])
		vec[6] = limit(p.LastTransaction.KmFromCurrent / normalization["max_km"])
	}
	vec[7] = limit(p.Terminal.KmFromHome / normalization["max_km"])
	vec[8] = limit(float32(p.Customer.TxCount24h) / normalization["max_tx_count_24h"])
	if p.Terminal.IsOnline {
		vec[9] = 1
	} else {
		vec[9] = 0
	}
	if p.Terminal.CardPresent {
		vec[10] = 1
	} else {
		vec[10] = 0
	}
	if p.Merchant.ID == "" {
		vec[11] = 1
	} else {
		vec[11] = 0
	}

	risk, ok := mccRisk[p.Merchant.Mcc]
	if ok {
		vec[12] = risk
	} else {
		vec[12] = 0.5
	}

	vec[13] = limit(p.Merchant.AvgAmount / normalization["max_merchant_avg_amount"])

	return vec
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

	http.HandleFunc("/ready", readyHandler)
	http.HandleFunc("/fraud-score", fraudScoreHandler)

	fmt.Println("Server running on http://localhost:8080")
	err = http.ListenAndServe(":8080", nil)
	if err != nil {
		fmt.Println(err)
	}
}
