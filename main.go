package main

import (
	"fmt"
	"net/http"
)

func fraudScore(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case http.MethodPost:
		fmt.Fprintln(w, "WIP")
	default:
		fmt.Fprintln(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func ready(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case http.MethodGet:
		fmt.Fprintln(w, "Hello World")
	default:
		fmt.Fprintln(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func main() {
	fmt.Println("Hello World")

	http.HandleFunc("/ready", ready)
	http.HandleFunc("/fraud-score", fraudScore)

	fmt.Println("Server running on http://localhost:8080")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		fmt.Println(err)
	}
}
