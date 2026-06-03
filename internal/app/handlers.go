package app

import (
	"fmt"
	"net/http"

	"github.com/bytedance/sonic"

	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

const (
	topK   int = 5
	nProbe int = 12
)

var fraudResponseBodies = [6][]byte{
	[]byte(`{"approved":true,"fraud_score":0}`),
	[]byte(`{"approved":true,"fraud_score":0.2}`),
	[]byte(`{"approved":true,"fraud_score":0.4}`),
	[]byte(`{"approved":false,"fraud_score":0.6}`),
	[]byte(`{"approved":false,"fraud_score":0.8}`),
	[]byte(`{"approved":false,"fraud_score":1}`),
}

func (a *App) FraudScoreHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case http.MethodPost:
		var body vs.Payload

		err := sonic.ConfigDefault.NewDecoder(r.Body).Decode(&body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		vec := a.MakeVector(body)

		fraudCount := a.IVF.IvfSearch(vec, topK, nProbe)

		resp := fraudResponseBodies[fraudCount]

		_, _ = w.Write(resp)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (a *App) ReadyHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case http.MethodGet:
		fmt.Fprintln(w, "Hello World")
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}
