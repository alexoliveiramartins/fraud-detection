package main

import (
	"fmt"
	"net"
	"net/http"
	"os"

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

	err = app.LoadBBoxes()
	if err != nil {
		panic(err)
	}

	app.Warmup()

	http.HandleFunc("/ready", app.ReadyHandler)
	http.HandleFunc("/fraud-score", app.FraudScoreHandler)
	
	socketPath := os.Getenv("API_SOCKET")
	if socketPath != "" {
		_ = os.Remove(socketPath)
		ln, err := net.Listen("unix", socketPath)
		if err != nil {
			panic(err)
		}
		_ = os.Chmod(socketPath, 0666)
		defer ln.Close()

		err = http.Serve(ln, nil)
	} else {
		err = http.ListenAndServe(":8080", nil)
	}
	
	// err = http.ListenAndServe(":8080", nil)
	// if err != nil {
	// 	fmt.Println(err)
	// }
	fmt.Println("Server running on http://localhost:8080")
}
