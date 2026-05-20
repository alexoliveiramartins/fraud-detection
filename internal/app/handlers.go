package app

import (
	"encoding/json"
	"fmt"
	"net/http"

	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

var topK int = 5

func (a *App) FraudScoreHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case http.MethodPost:
		var body vs.Payload
		err := json.NewDecoder(r.Body).Decode(&body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		vec := a.MakeVector(body)
		// knn := knn(vec)
		ivf, err := a.IVF.IvfSearch(vec, topK, 1)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		resp := MakeResponse(ivf)
		err = json.NewEncoder(w).Encode(resp)
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
