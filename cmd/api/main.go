package main

import (
	"fmt"
	"net/http"

	"github.com/alexoliveiramartins/fraud-detection/internal/app"
)

func main() {
	app := &app.App{}

	fmt.Printf("Carregando mcc_risk.json...\n")
	err := app.LoadMccRisk()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Carregando normalization.json...\n")
	err = app.LoadNormalization()
	if err != nil {
		fmt.Println(err)
		return
	}
	err = app.LoadCentroids()
	if err != nil {
		panic(err)
	}

	err = app.LoadOffsets()
	if err != nil {
		panic(err)
	}

	app.Warmup()

	http.HandleFunc("/ready", app.ReadyHandler)
	http.HandleFunc("/fraud-score", app.FraudScoreHandler)

	fmt.Println("Server running on http://localhost:8080")
	err = http.ListenAndServe(":8080", nil)
	if err != nil {
		fmt.Println(err)
	}
}
