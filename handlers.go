package main

import (
	"encoding/json"
	"fmt"
	"net/http"

	vs "github.com/alexoliveiramartins/fraud-detection/internal/vectorsearch"
)

func fraudScoreHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case http.MethodPost:
		var body vs.Payload
		err := json.NewDecoder(r.Body).Decode(&body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		vec := makeVector(body)
		// knn := knn(vec)
		ivf, err := ivfIndexes.IvfSearch(vec, topK, 3)
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

func readyHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case http.MethodGet:
		fmt.Fprintln(w, "Hello World")
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}
