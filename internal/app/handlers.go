package app

import (
	"fmt"
	"net/http"

	"github.com/bytedance/sonic"

	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

const (
	topK   int = 5
	nProbe int = 8
)

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

		ivf, err := a.IVF.IvfSearch(vec, topK, nProbe)
		if err != nil {
			sonic.ConfigDefault.NewEncoder(w).Encode(vs.Response{
				Approved:   false,
				FraudScore: 1,
			})
			return
		}

		resp := MakeResponse(ivf)

		err = sonic.ConfigDefault.NewEncoder(w).Encode(resp)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
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
