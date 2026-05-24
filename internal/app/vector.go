package app

import (
	"slices"

	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

type Neighbour struct {
	Index int
	Dist  float32
}

func MakeResponse(score float32) vs.Response {
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

	vec[0] = limit(p.Transaction.Amount / a.Normalization.MaxAmount)
	vec[1] = limit(float32(p.Transaction.Installments) / a.Normalization.MaxInstallments)
	vec[2] = limit((p.Transaction.Amount / p.Customer.AvgAmount) / a.Normalization.AmountVsAvg)
	vec[3] = float32(p.Transaction.RequestedAt.Hour()) / 23
	weekDay := (int(p.Transaction.RequestedAt.Weekday()) + 6) % 7
	vec[4] = float32(weekDay) / float32(6)
	if p.LastTransaction == nil {
		vec[5] = -1
		vec[6] = -1
	} else {
		minutesSinceLast := p.Transaction.RequestedAt.Sub(p.LastTransaction.Timestamp).Minutes()
		vec[5] = limit(float32(minutesSinceLast) / a.Normalization.MaxMinutes)
		vec[6] = limit(p.LastTransaction.KmFromCurrent / a.Normalization.MaxKm)
	}
	vec[7] = limit(p.Terminal.KmFromHome / a.Normalization.MaxKm)
	vec[8] = limit(float32(p.Customer.TxCount24h) / a.Normalization.MaxTxCount)
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

	vec[13] = limit(p.Merchant.AvgAmount / a.Normalization.MaxMerchantAvg)

	return vec
}
