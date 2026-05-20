package app

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

func limit(n float32) float32 {
	lim := max(0, n)
	lim = min(1, lim)
	return lim
}

func (a *App) MakeVector(p vs.Payload) vs.Vector {
	var vec vs.Vector

	vec[0] = limit(p.Transaction.Amount / a.Normalization["max_amount"])
	vec[1] = limit(float32(p.Transaction.Installments) / a.Normalization["max_installments"])
	vec[2] = limit((p.Transaction.Amount / p.Customer.AvgAmount) / a.Normalization["amount_vs_avg_ratio"])
	vec[3] = float32(p.Transaction.RequestedAt.Hour()) / 23
	weekDay := (int(p.Transaction.RequestedAt.Weekday()) + 6) % 7
	vec[4] = float32(weekDay) / float32(6)
	if p.LastTransaction == nil {
		vec[5] = -1
		vec[6] = -1
	} else {
		vec[5] = limit(float32(p.LastTransaction.Timestamp.Minute()) / a.Normalization["max_minutes"])
		vec[6] = limit(p.LastTransaction.KmFromCurrent / a.Normalization["max_km"])
	}
	vec[7] = limit(p.Terminal.KmFromHome / a.Normalization["max_km"])
	vec[8] = limit(float32(p.Customer.TxCount24h) / a.Normalization["max_tx_count_24h"])
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

	risk, ok := a.MccRisk[p.Merchant.Mcc]
	if ok {
		vec[12] = risk
	} else {
		vec[12] = 0.5
	}

	vec[13] = limit(p.Merchant.AvgAmount / a.Normalization["max_merchant_avg_amount"])

	return vec
}
