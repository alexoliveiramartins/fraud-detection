package main

import (
	"slices"

	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

type Neighbour struct {
	Index int
	Dist  float32
}

func MakeResponse(neighbours []vs.Neighbor) vs.Response {
	var score float32
	for i := range len(neighbours) {
		if neighbours[i].Label == true {
			score += 1
		}
	}
	score /= 5

	appr := false
	if score < 0.6 {
		appr = true
	}

	return vs.Response{
		Approved:   appr,
		FraudScore: score,
	}
}

func distEuclid(vec1 vs.Vector, vec2 vs.Vector) float32 {
	var sum float32
	for i := range 14 {
		diff := vec1[i] - vec2[i]
		sum += diff * diff
	}
	return sum
}

func knn(vec vs.Vector) []Neighbour {
	knn := make([]Neighbour, 5)

	worst := 0
	for i := range 5 {
		knn[i] = Neighbour{
			Index: i,
			Dist:  distEuclid(vec, references[i].Vector),
		}
		if knn[i].Dist > knn[worst].Dist {
			worst = i
		}
	}

	for i := 5; i < len(references); i++ {
		neighbour := Neighbour{
			Index: i,
			Dist:  distEuclid(vec, references[i].Vector),
		}
		if neighbour.Dist < knn[worst].Dist {
			knn[worst] = neighbour

			worst = 0
			for j := range 5 {
				if knn[j].Dist > knn[worst].Dist {
					worst = j
				}
			}
		}
	}
	return knn
}

func limit(n float32) float32 {
	lim := max(0, n)
	lim = min(1, lim)
	return lim
}

func makeVector(p vs.Payload) vs.Vector {
	var vec vs.Vector

	vec[0] = limit(p.Transaction.Amount / normalization["max_amount"])
	vec[1] = limit(float32(p.Transaction.Installments) / normalization["max_installments"])
	vec[2] = limit((p.Transaction.Amount / p.Customer.AvgAmount) / normalization["amount_vs_avg_ratio"])
	vec[3] = float32(p.Transaction.RequestedAt.Hour()) / 23
	weekDay := (int(p.Transaction.RequestedAt.Weekday()) + 6) % 7
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
	if slices.Contains(p.Customer.KnownMerchants, p.Merchant.ID) {
		vec[11] = 0
	} else {
		vec[11] = 1
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
