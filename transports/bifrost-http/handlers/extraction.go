package handlers

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/bytedance/sonic"
	"github.com/google/uuid"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/lib"
	"github.com/valyala/fasthttp"
)

var extractionParamsKnownFields = map[string]bool{
	"model":              true,
	"fallbacks":          true,
	"text":               true,
	"schema":             true,
	"threshold":          true,
	"include_confidence": true,
	"include_spans":      true,
	"targets":            true,
}

// ExtractionHandlerRequest is a bifrost extraction request: GLiNER-style schema extraction.
type ExtractionHandlerRequest struct {
	Text    string          `json:"text"`
	Schema  json.RawMessage `json:"schema"`
	Targets []string        `json:"targets,omitempty"` // compare route only: "provider/model" or a chain name per target
	BifrostParams
	*schemas.ExtractionParameters
}

// prepareExtractionRequest prepares a BifrostExtractionRequest from the HTTP request body.
func prepareExtractionRequest(ctx *fasthttp.RequestCtx, config *lib.Config) (*ExtractionHandlerRequest, *schemas.BifrostExtractionRequest, error) {
	req, base, err := prepareRequest[ExtractionHandlerRequest](ctx, config, extractionParamsKnownFields)
	if err != nil {
		return nil, nil, err
	}
	if strings.TrimSpace(req.Text) == "" {
		return nil, nil, fmt.Errorf("text is required for extraction")
	}
	if trimmed := strings.TrimSpace(string(req.Schema)); !strings.HasPrefix(trimmed, "{") {
		return nil, nil, fmt.Errorf("schema must be an object with entities, classifications, structures or relations")
	}
	if req.ExtractionParameters == nil {
		req.ExtractionParameters = &schemas.ExtractionParameters{}
	}
	if t := req.ExtractionParameters.Threshold; t != nil && (*t < 0 || *t > 1) {
		return nil, nil, fmt.Errorf("threshold must be between 0 and 1")
	}
	req.ExtractionParameters.ExtraParams = base.ExtraParams
	return req, &schemas.BifrostExtractionRequest{
		Provider:  base.Provider,
		Model:     base.ModelName,
		Text:      req.Text,
		Schema:    req.Schema,
		Params:    req.ExtractionParameters,
		Fallbacks: base.Fallbacks,
	}, nil
}

// extraction handles POST /v1/extract - GLiNER-style schema extraction
func (h *CompletionHandler) extraction(ctx *fasthttp.RequestCtx) {
	_, bifrostReq, err := prepareExtractionRequest(ctx, h.config)
	if err != nil {
		SendError(ctx, fasthttp.StatusBadRequest, err.Error())
		return
	}
	bifrostCtx, cancel := lib.ConvertToBifrostContext(ctx, h.config)
	defer cancel()
	if bifrostCtx == nil {
		SendError(ctx, fasthttp.StatusBadRequest, "Failed to convert context")
		return
	}
	resp, bifrostErr := h.client.ExtractionRequest(bifrostCtx, bifrostReq)
	if bifrostErr != nil {
		forwardProviderHeadersFromContext(ctx, bifrostCtx)
		SendBifrostError(ctx, bifrostErr)
		return
	}
	if resp != nil {
		lib.ApplyBifrostResponseHeaders(ctx, bifrostCtx, resp.ExtraFields)
	}
	SendJSON(ctx, resp)
}

// ExtractionCompareResult is one target's answer in a compare call.
type ExtractionCompareResult struct {
	Target    string          `json:"target"`
	Provider  string          `json:"provider,omitempty"`
	Model     string          `json:"model,omitempty"`
	LatencyMs int64           `json:"latency_ms"`
	Result    json.RawMessage `json:"result,omitempty"`
	Error     string          `json:"error,omitempty"`
}

// ExtractionCompareResponse holds every target's answer and the diff of each against the first.
type ExtractionCompareResponse struct {
	Results []ExtractionCompareResult `json:"results"`
	Diffs   []ExtractionDiff          `json:"diffs"`
}

// extractionCompare handles POST /v1/extract/compare: the same text and schema go to every
// target, one after the other, each as its own logged request without fallbacks.
// ponytail: targets run in sequence, so the call takes the sum of their latencies; run them
// in parallel if compare calls ever sit on a hot path.
func (h *CompletionHandler) extractionCompare(ctx *fasthttp.RequestCtx) {
	req, bifrostReq, err := prepareExtractionRequest(ctx, h.config)
	if err != nil {
		SendError(ctx, fasthttp.StatusBadRequest, err.Error())
		return
	}
	if len(req.Targets) < 2 {
		SendError(ctx, fasthttp.StatusBadRequest, "targets needs at least two entries, e.g. [\"fastino/fastino/gliner2.5-base-v1\", \"gliner-local/gliner2.5-base-v1\"]")
		return
	}
	// The transport keeps one context per HTTP request, so each target runs in a child of it with
	// its own request ID (one log row each) and billing nonce (governance counts every call).
	bifrostCtx, cancel := lib.ConvertToBifrostContext(ctx, h.config)
	defer cancel()
	if bifrostCtx == nil {
		SendError(ctx, fasthttp.StatusBadRequest, "Failed to convert context")
		return
	}
	out := ExtractionCompareResponse{Results: make([]ExtractionCompareResult, 0, len(req.Targets))}
	for _, target := range req.Targets {
		row := ExtractionCompareResult{Target: target}
		provider, model, resolveErr := resolveModelAndProvider(ctx, h.config, target)
		if resolveErr != nil {
			row.Error = resolveErr.Error()
			out.Results = append(out.Results, row)
			continue
		}
		one := *bifrostReq
		one.Provider, one.Model, one.Fallbacks = provider, model, nil
		child := schemas.NewBifrostContext(bifrostCtx, schemas.NoDeadline)
		child.SetValue(schemas.BifrostContextKeyRequestID, uuid.New().String())
		child.SetValue(schemas.BifrostContextKeyBillingNonce, uuid.New().String())
		start := time.Now()
		resp, bifrostErr := h.client.ExtractionRequest(child, &one)
		child.Cancel()
		row.LatencyMs = time.Since(start).Milliseconds()
		switch {
		case bifrostErr != nil:
			row.Error = bifrostErrorMessage(bifrostErr)
		case resp != nil:
			row.Provider, row.Model, row.Result = string(resp.ExtraFields.Provider), resp.Model, resp.Result
		}
		out.Results = append(out.Results, row)
	}
	for i := 1; i < len(out.Results); i++ {
		a, b := out.Results[0], out.Results[i]
		if a.Result == nil || b.Result == nil {
			continue
		}
		diff, diffErr := DiffExtractionResults(a.Result, b.Result)
		if diffErr != nil {
			continue
		}
		diff.A, diff.B = a.Target, b.Target
		out.Diffs = append(out.Diffs, *diff)
	}
	SendJSON(ctx, out)
}

func bifrostErrorMessage(e *schemas.BifrostError) string {
	if e.Error != nil && e.Error.Message != "" {
		return e.Error.Message
	}
	return "extraction failed"
}

// SetDiff lists what two answers share and what only one of them found.
type SetDiff struct {
	Both  []string `json:"both"`
	OnlyA []string `json:"only_a"`
	OnlyB []string `json:"only_b"`
}

// LabelDiff compares one classification task.
type LabelDiff struct {
	A    string `json:"a"`
	B    string `json:"b"`
	Same bool   `json:"same"`
}

// ExtractionDiff compares two GLiNER2 result objects task by task. Agreement is the share of
// matched items (entity spans, relation pairs, classification labels) over all distinct items.
type ExtractionDiff struct {
	A               string               `json:"a"`
	B               string               `json:"b"`
	Agreement       float64              `json:"agreement"`
	Entities        map[string]SetDiff   `json:"entities,omitempty"`
	Classifications map[string]LabelDiff `json:"classifications,omitempty"`
	Relations       map[string]SetDiff   `json:"relations,omitempty"`
	OtherSame       map[string]bool      `json:"other_same,omitempty"` // structures and unknown keys: equal as JSON
}

// DiffExtractionResults diffs two result objects in GLiNER2's output format: "entities" maps a
// label to spans, "relation_extraction" maps a relation to head/tail pairs, an object with a
// "label" (or a bare string) is a classification task, and anything else is compared as JSON.
// Texts are compared case-insensitively, so spans and confidences do not count.
func DiffExtractionResults(a, b json.RawMessage) (*ExtractionDiff, error) {
	var ma, mb map[string]json.RawMessage
	if err := sonic.Unmarshal(a, &ma); err != nil {
		return nil, err
	}
	if err := sonic.Unmarshal(b, &mb); err != nil {
		return nil, err
	}
	d := &ExtractionDiff{}
	matched, total := 0, 0
	count := func(s SetDiff) {
		matched += len(s.Both)
		total += len(s.Both) + len(s.OnlyA) + len(s.OnlyB)
	}
	for _, key := range unionKeys(ma, mb) {
		va, vb := ma[key], mb[key]
		switch {
		case key == "entities":
			d.Entities = map[string]SetDiff{}
			ga, gb := groupTexts(va, spanText), groupTexts(vb, spanText)
			for _, label := range unionKeys(ga, gb) {
				s := diffSets(ga[label], gb[label])
				d.Entities[label] = s
				count(s)
			}
		case key == "relation_extraction":
			d.Relations = map[string]SetDiff{}
			ga, gb := groupTexts(va, pairText), groupTexts(vb, pairText)
			for _, rel := range unionKeys(ga, gb) {
				s := diffSets(ga[rel], gb[rel])
				d.Relations[rel] = s
				count(s)
			}
		default:
			la, okA := classificationLabel(va)
			lb, okB := classificationLabel(vb)
			if okA || okB {
				if d.Classifications == nil {
					d.Classifications = map[string]LabelDiff{}
				}
				same := okA && okB && strings.EqualFold(la, lb)
				d.Classifications[key] = LabelDiff{A: la, B: lb, Same: same}
				total++
				if same {
					matched++
				}
				continue
			}
			if d.OtherSame == nil {
				d.OtherSame = map[string]bool{}
			}
			d.OtherSame[key] = jsonEqual(va, vb)
		}
	}
	if total > 0 {
		d.Agreement = float64(matched) / float64(total)
	} else {
		d.Agreement = 1
	}
	return d, nil
}

func unionKeys[V any](a, b map[string]V) []string {
	seen := map[string]bool{}
	for k := range a {
		seen[k] = true
	}
	for k := range b {
		seen[k] = true
	}
	keys := make([]string, 0, len(seen))
	for k := range seen {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// groupTexts reads {"label": [item, ...]} and maps each label to the item texts.
func groupTexts(raw json.RawMessage, text func(json.RawMessage) string) map[string][]string {
	var groups map[string][]json.RawMessage
	if len(raw) == 0 || sonic.Unmarshal(raw, &groups) != nil {
		return nil
	}
	out := make(map[string][]string, len(groups))
	for label, items := range groups {
		for _, item := range items {
			if t := text(item); t != "" {
				out[label] = append(out[label], t)
			}
		}
	}
	return out
}

// spanText reads an entity as a bare string or an object with "text".
func spanText(raw json.RawMessage) string {
	var s string
	if sonic.Unmarshal(raw, &s) == nil {
		return normalizeText(s)
	}
	var obj struct {
		Text string `json:"text"`
	}
	if sonic.Unmarshal(raw, &obj) == nil {
		return normalizeText(obj.Text)
	}
	return ""
}

// pairText reads a relation as {"head": span, "tail": span} or a [head, tail] pair.
func pairText(raw json.RawMessage) string {
	var obj struct {
		Head json.RawMessage `json:"head"`
		Tail json.RawMessage `json:"tail"`
	}
	if sonic.Unmarshal(raw, &obj) == nil && obj.Head != nil {
		return spanText(obj.Head) + " -> " + spanText(obj.Tail)
	}
	var pair []json.RawMessage
	if sonic.Unmarshal(raw, &pair) == nil && len(pair) == 2 {
		return spanText(pair[0]) + " -> " + spanText(pair[1])
	}
	return ""
}

// classificationLabel reads a task answer as {"label": ...} or a bare string.
func classificationLabel(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 {
		return "", false
	}
	var s string
	if sonic.Unmarshal(raw, &s) == nil {
		return s, true
	}
	var obj struct {
		Label *string `json:"label"`
	}
	if sonic.Unmarshal(raw, &obj) == nil && obj.Label != nil {
		return *obj.Label, true
	}
	return "", false
}

func normalizeText(s string) string {
	return strings.ToLower(strings.Join(strings.Fields(s), " "))
}

func diffSets(a, b []string) SetDiff {
	inA, inB := map[string]bool{}, map[string]bool{}
	for _, x := range a {
		inA[x] = true
	}
	for _, x := range b {
		inB[x] = true
	}
	d := SetDiff{Both: []string{}, OnlyA: []string{}, OnlyB: []string{}}
	for _, x := range unionKeys(inA, inB) {
		switch {
		case inA[x] && inB[x]:
			d.Both = append(d.Both, x)
		case inA[x]:
			d.OnlyA = append(d.OnlyA, x)
		default:
			d.OnlyB = append(d.OnlyB, x)
		}
	}
	return d
}

func jsonEqual(a, b json.RawMessage) bool {
	var va, vb interface{}
	if sonic.Unmarshal(a, &va) != nil || sonic.Unmarshal(b, &vb) != nil {
		return false
	}
	ja, _ := sonic.ConfigStd.Marshal(va)
	jb, _ := sonic.ConfigStd.Marshal(vb)
	return string(ja) == string(jb)
}
