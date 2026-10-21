package schemas

import "encoding/json"

// BifrostExtractionRequest asks a GLiNER-style encoder to extract structured data from Text.
//
// Schema is the unified GLiNER2 schema: an object with any of "entities", "classifications",
// "structures" and "relations". It travels to the provider unchanged, so every schema feature
// a GLiNER backend supports works without a matching Go type here.
type BifrostExtractionRequest struct {
	Provider       ModelProvider         `json:"provider"`
	Model          string                `json:"model"`
	Text           string                `json:"text"`
	Schema         json.RawMessage       `json:"schema"`
	Params         *ExtractionParameters `json:"params,omitempty"`
	Fallbacks      []Fallback            `json:"fallbacks,omitempty"`
	RawRequestBody []byte                `json:"-"`
}

// GetRawRequestBody returns the raw request body for the extraction request.
func (r *BifrostExtractionRequest) GetRawRequestBody() []byte {
	return r.RawRequestBody
}

// ExtractionParameters contains optional parameters for an extraction request.
// Nil fields take the backend's defaults (GLiNER2: threshold 0.5, confidence and spans on).
type ExtractionParameters struct {
	Threshold         *float64               `json:"threshold,omitempty"`
	IncludeConfidence *bool                  `json:"include_confidence,omitempty"`
	IncludeSpans      *bool                  `json:"include_spans,omitempty"`
	ExtraParams       map[string]interface{} `json:"-"`
}

// BifrostExtractionResponse holds the backend's result object as returned: entity labels map to
// span lists, each classification task to its label, each structure to its records, and
// "relation_extraction" to head/tail pairs per relation.
type BifrostExtractionResponse struct {
	ID          string                     `json:"id,omitempty"`
	Model       string                     `json:"model"`
	Result      json.RawMessage            `json:"result"`
	Usage       *BifrostLLMUsage           `json:"usage,omitempty"`
	ExtraFields BifrostResponseExtraFields `json:"extra_fields"`
}
