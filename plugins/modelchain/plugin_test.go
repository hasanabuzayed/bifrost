package modelchain

import (
	"encoding/json"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

func newPlugin(t *testing.T) *ModelChainPlugin {
	t.Helper()
	p, err := Init(&Config{Chains: map[string][]Target{"gorse-rerank": {
		{Provider: "jina", Model: "jina-reranker-v3.5"},
		{Provider: "openrouter-rerank", Model: "free-model"},
		{Provider: "llama-rerank", Model: "bge-reranker-v2-m3:latest"},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestPreRequestHookSetsStrictChain(t *testing.T) {
	p := newPlugin(t)
	req := &schemas.BifrostRequest{RequestType: schemas.RerankRequest, RerankRequest: &schemas.BifrostRerankRequest{Model: "gorse-rerank"}}
	if err := p.PreRequestHook(nil, req); err != nil {
		t.Fatal(err)
	}
	r := req.RerankRequest
	if r.Provider != "jina" || r.Model != "jina-reranker-v3.5" || len(r.Fallbacks) != 2 ||
		r.Fallbacks[0].Provider != "openrouter-rerank" || r.Fallbacks[1].Model != "bge-reranker-v2-m3:latest" {
		t.Fatalf("unexpected routing: %+v", r)
	}

	decision := &schemas.BifrostRequest{RequestType: schemas.DecisionRequest, DecisionRequest: &schemas.BifrostDecisionRequest{Model: "gorse-rerank"}}
	_ = p.PreRequestHook(nil, decision)
	if d := decision.DecisionRequest; d.Provider != "jina" || len(d.Fallbacks) != 2 {
		t.Fatalf("decision request not chained: %+v", d)
	}

	other := &schemas.BifrostRequest{RequestType: schemas.RerankRequest, RerankRequest: &schemas.BifrostRerankRequest{Provider: "jina", Model: "jina-reranker-v3.5"}}
	_ = p.PreRequestHook(nil, other)
	if len(other.RerankRequest.Fallbacks) != 0 {
		t.Fatal("a model without a chain was rerouted")
	}
}

func post(p *ModelChainPlugin, model string, status int, body string) *schemas.HTTPResponse {
	reqBody, _ := json.Marshal(map[string]any{"model": model, "query": "q", "documents": []string{"a", "b", "c", "d"}})
	resp := &schemas.HTTPResponse{StatusCode: status, Body: []byte(body)}
	_ = p.HTTPTransportPostHook(nil, &schemas.HTTPRequest{Path: route, Body: reqBody}, resp)
	return resp
}

func TestPostHookIdentityOnlyForProviderFailures(t *testing.T) {
	p := newPlugin(t)
	resp := post(p, "gorse-rerank", 502, `{"type":"provider_connection_failed"}`)
	var out struct {
		Model   string `json:"model"`
		Results []struct {
			Index int     `json:"index"`
			Score float64 `json:"relevance_score"`
		} `json:"results"`
	}
	if err := json.Unmarshal(resp.Body, &out); err != nil || resp.StatusCode != 200 || out.Model != "identity" || len(out.Results) != 4 {
		t.Fatalf("want identity 200, got %d %s", resp.StatusCode, resp.Body)
	}
	for i, r := range out.Results {
		if r.Index != i || (i > 0 && r.Score >= out.Results[i-1].Score) {
			t.Fatalf("want original order with falling scores, got %+v", out.Results)
		}
	}
	if resp := post(p, "gorse-rerank", 429, `{"type":"rate_limited"}`); resp.StatusCode != 200 {
		t.Fatalf("a capped chain must still answer 200, got %d", resp.StatusCode)
	}

	for _, c := range []struct {
		model  string
		status int
		body   string
	}{
		{"gorse-rerank", 401, `{"type":"virtual_key_required"}`},
		{"gorse-rerank", 401, `{"type":"access_not_found"}`},
		{"gorse-rerank", 400, `{"type":"invalid_request_error"}`},
		{"jina-reranker-v3.5", 502, `{}`},
		{"gorse-rerank", 200, `{"results":[]}`},
	} {
		if resp := post(p, c.model, c.status, c.body); resp.StatusCode != c.status {
			t.Fatalf("%s %d %s: want the response unchanged, got %d", c.model, c.status, c.body, resp.StatusCode)
		}
	}
}

func TestInitRejectsBadConfig(t *testing.T) {
	for _, cfg := range []*Config{nil, {}, {Chains: map[string][]Target{"x": {}}}, {Chains: map[string][]Target{"x": {{Provider: "jina"}}}}} {
		if _, err := Init(cfg); err == nil {
			t.Fatalf("want an error for %+v", cfg)
		}
	}
}
