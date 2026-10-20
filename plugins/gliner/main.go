// Package gliner serves zero-shot named-entity recognition (GLiNER) through Bifrost on
// Fastino's own API, so a Fastino client only changes its base URL to <bifrost>/fastino:
//
//	POST /fastino/v1/gliner-2
//
// Bifrost has no NER request type, so the HTTP transport owns this route (handlers/gliner.go):
// it settles the caller, asks governance for the verdict (virtual key, access, budgets, rate
// limits), and only then calls Serve. The plugin holds the config and does the work.
//
// Requests go to the hosted Fastino API first, unchanged and under a daily call cap. When that
// fails, entity requests fall back to a self-hosted GLiNER server (POST <local_url>/v1/gliner,
// body {text, entity_types, threshold}), and the plugin rebuilds Fastino's response shape:
// {result: {entities: {label: [{text, confidence, start, end}]}}, token_usage}.
// Classifications, structures, relations and batched text have no local fallback.
package gliner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

const PluginName = "gliner"

// Route is the Fastino-compatible path; the HTTP transport registers it.
const Route = "/fastino/v1/gliner-2"

const (
	defaultFastinoURL      = "https://api.fastino.ai/v1/gliner-2"
	defaultFastinoDailyCap = 5000 // $0.03 per million input tokens
	hostedTimeout          = 30 * time.Second
	localTimeout           = 120 * time.Second // a cold local model loads first
	maxUpstreamBody        = 8 << 20
)

// Config is the plugin's config.json block. A missing Fastino key skips the hosted tier,
// and a missing local_url skips the self-hosted one.
type Config struct {
	FastinoURL      string             `json:"fastino_url,omitempty"`
	FastinoAPIKey   *schemas.SecretVar `json:"fastino_api_key,omitempty"`
	FastinoDailyCap int                `json:"fastino_daily_cap,omitempty"`
	LocalURL        string             `json:"local_url,omitempty"`
}

// fastinoRequest holds the fields the local fallback needs; Fastino gets the original body.
type fastinoRequest struct {
	Text              json.RawMessage `json:"text"`
	Schema            json.RawMessage `json:"schema"`
	Threshold         *float64        `json:"threshold"`
	IncludeConfidence *bool           `json:"include_confidence"`
	IncludeSpans      *bool           `json:"include_spans"`
}

type localEntity struct {
	Start *int     `json:"start"`
	End   *int     `json:"end"`
	Text  string   `json:"text"`
	Label string   `json:"label"`
	Score *float64 `json:"score"`
}

type GlinerPlugin struct {
	fastinoURL string
	fastinoKey string
	fastinoCap int
	localURL   string
	client     *http.Client
	logger     schemas.Logger

	mu           sync.Mutex
	day          string
	fastinoCalls int
}

func Init(config *Config, logger schemas.Logger) (*GlinerPlugin, error) {
	if config == nil {
		config = &Config{}
	}
	p := &GlinerPlugin{
		fastinoURL: config.FastinoURL,
		fastinoKey: config.FastinoAPIKey.GetValue(), // resolved value; SecretVarAsString would give "env.NAME"
		fastinoCap: config.FastinoDailyCap,
		localURL:   strings.TrimRight(config.LocalURL, "/"),
		client:     &http.Client{},
		logger:     logger,
	}
	if p.fastinoURL == "" {
		p.fastinoURL = defaultFastinoURL
	}
	if p.fastinoCap <= 0 {
		p.fastinoCap = defaultFastinoDailyCap
	}
	if p.fastinoKey == "" && p.localURL == "" {
		return nil, errors.New("gliner: set fastino_api_key, local_url, or both")
	}
	return p, nil
}

func (p *GlinerPlugin) GetName() string { return PluginName }

func (p *GlinerPlugin) Cleanup() error { return nil }

// Serve answers one already-authorized request body with a status code and a JSON body.
func (p *GlinerPlugin) Serve(ctx context.Context, body []byte) (int, []byte) {
	var r fastinoRequest
	if err := json.Unmarshal(body, &r); err != nil || len(r.Text) == 0 || len(r.Schema) == 0 {
		return reply(http.StatusBadRequest, map[string]string{"detail": "expected a JSON body with text and schema"})
	}

	var errs []error
	if p.fastinoKey != "" && p.takeFastino() {
		out, err := p.post(ctx, hostedTimeout, p.fastinoURL, map[string]string{"X-API-Key": p.fastinoKey}, body)
		if err == nil {
			return http.StatusOK, out
		}
		errs = append(errs, fmt.Errorf("fastino: %w", err))
	}
	if p.localURL != "" {
		out, err := p.local(ctx, r)
		if err == nil {
			return reply(http.StatusOK, out)
		}
		errs = append(errs, fmt.Errorf("local: %w", err))
	}
	if len(errs) == 0 {
		errs = append(errs, errors.New("fastino daily cap reached and no local_url"))
	}
	err := errors.Join(errs...)
	p.logger.Warn("gliner: %v", err)
	return reply(http.StatusBadGateway, map[string]string{"detail": "every upstream failed: " + err.Error()})
}

// takeFastino counts a hosted call against today's cap (UTC). It counts requests, not successes.
func (p *GlinerPlugin) takeFastino() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if day := time.Now().UTC().Format(time.DateOnly); day != p.day {
		p.day, p.fastinoCalls = day, 0
	}
	if p.fastinoCalls >= p.fastinoCap {
		return false
	}
	p.fastinoCalls++
	return true
}

// local answers an entity-only request from the self-hosted server in Fastino's response shape.
func (p *GlinerPlugin) local(ctx context.Context, r fastinoRequest) (map[string]any, error) {
	var text string
	if err := json.Unmarshal(r.Text, &text); err != nil {
		return nil, errors.New("only a single text string has a local fallback")
	}
	labels, err := entityLabels(r.Schema)
	if err != nil {
		return nil, err
	}
	threshold := 0.5
	if r.Threshold != nil {
		threshold = *r.Threshold
	}
	payload, _ := json.Marshal(map[string]any{"text": text, "entity_types": labels, "threshold": threshold})
	body, err := p.post(ctx, localTimeout, p.localURL+"/v1/gliner", nil, payload)
	if err != nil {
		return nil, err
	}
	var out struct {
		Entities *[]localEntity `json:"entities"`
	}
	if err := json.Unmarshal(body, &out); err != nil || out.Entities == nil {
		return nil, errors.New("response has no entities")
	}

	withConfidence := r.IncludeConfidence == nil || *r.IncludeConfidence
	withSpans := r.IncludeSpans == nil || *r.IncludeSpans
	byLabel := make(map[string][]map[string]any, len(labels))
	for _, label := range labels {
		byLabel[label] = []map[string]any{}
	}
	for _, e := range *out.Entities {
		if _, asked := byLabel[e.Label]; !asked {
			continue
		}
		span := map[string]any{"text": e.Text}
		if withConfidence && e.Score != nil {
			span["confidence"] = *e.Score
		}
		if withSpans && e.Start != nil && e.End != nil {
			span["start"], span["end"] = *e.Start, *e.End
		}
		byLabel[e.Label] = append(byLabel[e.Label], span)
	}
	return map[string]any{"result": map[string]any{"entities": byLabel}, "token_usage": 0}, nil
}

// entityLabels reads the entity labels from a Fastino schema: the deprecated flat label list,
// or {"entities": [labels] | {label: description}} with no other task.
func entityLabels(raw json.RawMessage) ([]string, error) {
	var flat []string
	if json.Unmarshal(raw, &flat) == nil {
		return checkLabels(flat)
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(raw, &schema); err != nil {
		return nil, errors.New("schema is neither a label list nor an object")
	}
	for key := range schema {
		if key != "entities" {
			return nil, fmt.Errorf("schema %q has no local fallback", key)
		}
	}
	if json.Unmarshal(schema["entities"], &flat) == nil {
		return checkLabels(flat)
	}
	var described map[string]json.RawMessage
	if err := json.Unmarshal(schema["entities"], &described); err != nil {
		return nil, errors.New("schema entities is neither a label list nor an object")
	}
	labels := make([]string, 0, len(described))
	for label := range described {
		labels = append(labels, label)
	}
	slices.Sort(labels)
	return checkLabels(labels)
}

func checkLabels(labels []string) ([]string, error) {
	if len(labels) == 0 || slices.Contains(labels, "") {
		return nil, errors.New("schema has no entity labels")
	}
	return labels, nil
}

func (p *GlinerPlugin) post(ctx context.Context, timeout time.Duration, url string, headers map[string]string, payload []byte) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxUpstreamBody))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	return body, nil
}

func reply(status int, v any) (int, []byte) {
	body, _ := json.Marshal(v)
	return status, body
}
