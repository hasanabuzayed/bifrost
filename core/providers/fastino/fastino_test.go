package fastino_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/maximhq/bifrost/core/providers/fastino"
	"github.com/maximhq/bifrost/core/schemas"
)

func TestExtraction(t *testing.T) {
	var got map[string]json.RawMessage
	var auth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, `{"error":{"message":"wrong path"}}`, http.StatusNotFound)
			return
		}
		auth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		if string(got["model"]) == `"warming"` {
			w.WriteHeader(425)
			_, _ = w.Write([]byte(`{"error":{"type":"invalid_request_error","code":"model_warming","message":"Model is warming up."}}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"fastino/gliner2.5-base-v1",
			"choices":[{"message":{"role":"assistant","content":"{\"entities\":{\"company\":[{\"text\":\"Apple\",\"confidence\":1.0,\"start\":0,\"end\":5}]}}"}}],
			"usage":{"prompt_tokens":7,"completion_tokens":30,"total_tokens":37}}`))
	}))
	defer server.Close()

	provider, err := fastino.NewFastinoProvider(&schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL, DefaultRequestTimeoutInSeconds: 30, AllowPrivateNetwork: true},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	key := schemas.Key{Value: *schemas.NewSecretVar("fk-test")}
	threshold := 0.4
	req := &schemas.BifrostExtractionRequest{
		Provider: schemas.Fastino,
		Model:    "fastino/gliner2.5-base-v1",
		Text:     "Apple hired Tim.",
		Schema:   json.RawMessage(`{"entities":["company"]}`),
		Params:   &schemas.ExtractionParameters{Threshold: &threshold, ExtraParams: map[string]interface{}{"store": false, "model": "overridden?"}},
	}

	resp, bifrostErr := provider.Extraction(ctx, key, req)
	if bifrostErr != nil {
		t.Fatalf("extraction failed: %+v", bifrostErr.Error)
	}
	if auth != "Bearer fk-test" {
		t.Errorf("authorization = %q", auth)
	}
	if string(got["model"]) != `"fastino/gliner2.5-base-v1"` || string(got["threshold"]) != `0.4` || string(got["store"]) != `false` {
		t.Errorf("body model/threshold/store = %s %s %s", got["model"], got["threshold"], got["store"])
	}
	if string(got["schema"]) != `{"entities":["company"]}` {
		t.Errorf("schema = %s", got["schema"])
	}
	if string(got["messages"]) != `[{"role":"user","content":"Apple hired Tim."}]` {
		t.Errorf("messages = %s", got["messages"])
	}
	var result map[string]map[string][]map[string]interface{}
	if err := json.Unmarshal(resp.Result, &result); err != nil || result["entities"]["company"][0]["text"] != "Apple" {
		t.Errorf("result = %s (%v)", resp.Result, err)
	}
	if resp.Usage == nil || resp.Usage.PromptTokens != 7 || resp.Model != "fastino/gliner2.5-base-v1" {
		t.Errorf("usage/model = %+v %s", resp.Usage, resp.Model)
	}

	// A provider error keeps its status, so a chain can fall back to the next tier.
	req.Model = "warming"
	if _, bifrostErr = provider.Extraction(ctx, key, req); bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != 425 {
		t.Errorf("warming error = %+v", bifrostErr)
	}

	if provider.GetProviderKey() != schemas.Fastino {
		t.Errorf("provider key = %s", provider.GetProviderKey())
	}
}
