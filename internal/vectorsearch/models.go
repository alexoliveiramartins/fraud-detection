package vectorsearch

import (
	"os"
	"time"
)

// ----------- references

type Reference struct {
	Vector Vector `json:"vector"`
	Label  bool   `json:"label"` // true se fraude
}

type RawReference struct {
	Vector Vector `json:"vector"`
	Label  string `json:"label"`
}

// ----------- payload

type Payload struct {
	ID              string           `json:"id"`
	Transaction     Transaction      `json:"transaction"`
	Customer        Customer         `json:"customer"`
	Merchant        Merchant         `json:"merchant"`
	Terminal        Terminal         `json:"terminal"`
	LastTransaction *LastTransaction `json:"last_transaction"`
}

type Transaction struct {
	Amount       float32   `json:"amount"`
	Installments int       `json:"installments"`
	RequestedAt  time.Time `json:"requested_at"`
}

type Customer struct {
	AvgAmount      float32  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type Merchant struct {
	ID        string  `json:"id"`
	Mcc       string  `json:"mcc"`
	AvgAmount float32 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float32 `json:"km_from_home"`
}

type LastTransaction struct {
	Timestamp     time.Time `json:"timestamp"`
	KmFromCurrent float32   `json:"km_from_current"`
}

// ----------- response

type Response struct {
	Approved   bool    `json:"approved"`
	FraudScore float32 `json:"fraud_score"`
}

// ---------- IVF Search

type Vector [14]float32

type IVF struct {
	Centroids []Vector
	Lists     [][]Reference
}

type IVFFile struct {
	Centroids   []Vector
	Offsets     []ClusterOffset
	VectorsData []byte
	VectorsFile *os.File
	BBoxMin     []QuantizedVector
	BBoxMax     []QuantizedVector
}

type ClusterOffset struct {
	Offset uint64
	Count  uint32
}

type fixedTop struct {
	dist     [fixedTopK]int64
	label    [fixedTopK]bool
	size     int
	worstIdx int
}
