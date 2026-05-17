package main

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
)

type RawReference struct {
	Vector [14]float32 `json:"vector"`
	Label  string      `json:"label"`
}

func main() {
	input, err := os.Open("resources/references.json.gz")
	if err != nil {
		panic(err)
	}
	defer input.Close()

	gz, err := gzip.NewReader(input)
	if err != nil {
		panic(err)
	}
	defer gz.Close()

	output, err := os.Create("resources/references.bin")
	if err != nil {
		panic(err)
	}
	defer output.Close()

	decoder := json.NewDecoder(gz)

	var refs []RawReference
	err = decoder.Decode(&refs)
	if err != nil {
		panic(err)
	}

	for _, ref := range refs {
		for _, value := range ref.Vector {
			err := binary.Write(output, binary.LittleEndian, value)
			if err != nil {
				panic(err)
			}
		}

		var label byte = 0
		if ref.Label == "fraud" {
			label = 1
		}

		if _, err := output.Write([]byte{label}); err != nil {
			panic(err)
		}
	}

	fmt.Printf("wrote %d references\n", len(refs))
}
