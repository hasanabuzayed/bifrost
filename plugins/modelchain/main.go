// Package modelchain gives a logical model name a strict provider chain, and rerank a safe last answer.
//
// A caller that cannot send Bifrost fallbacks (Gorse posts one Jina-style /v1/rerank body, n8n
// posts TypeSafe System One bodies) names a logical model, say "gorse-rerank". The plugin:
//
//   - PreRequestHook (rerank and decision requests): sets the first chain entry as the
//     provider/model and the rest as explicit fallbacks, so Bifrost tries them in order (its own
//     key-weight fallbacks pick the first provider at random). Each attempt still runs
//     governance, so per-model rate limits in governance.model_configs work as caps per tier.
//   - HTTPTransportPostHook (/v1/rerank only): when every attempt failed, answers 200 with the
//     documents in their original order, so a caller that indexes its list with the results never
//     loses the list. Refusals of the caller itself (no or unknown virtual key) and malformed
//     requests keep their error.
package modelchain

import (
	"encoding/json"
	"errors"
	"net/http"
	"slices"

	"github.com/maximhq/bifrost/core/schemas"
)

const PluginName = "model-chain"

const route = "/v1/rerank"

// Target is one provider/model attempt.
type Target struct {
	Provider schemas.ModelProvider `json:"provider"`
	Model    string                `json:"model"`
}

// Config maps each logical model name to its ordered chain.
type Config struct {
	Chains map[string][]Target `json:"chains"`
}

// Error types and statuses that stay errors: they describe the caller, not the providers.
var (
	callerRefusals = []string{"virtual_key_required", "access_not_found", "access_blocked", "access_unresolved"}
	callerFaults   = []int{http.StatusBadRequest, http.StatusRequestEntityTooLarge, http.StatusUnsupportedMediaType, http.StatusUnprocessableEntity}
)

type ModelChainPlugin struct {
	chains map[string][]Target
}

func Init(config *Config) (*ModelChainPlugin, error) {
	if config == nil || len(config.Chains) == 0 {
		return nil, errors.New("model-chain: set at least one chain")
	}
	for name, chain := range config.Chains {
		if name == "" || len(chain) == 0 {
			return nil, errors.New("model-chain: every chain needs a name and at least one target")
		}
		for _, t := range chain {
			if t.Provider == "" || t.Model == "" {
				return nil, errors.New("model-chain: chain " + name + " has a target without provider or model")
			}
		}
	}
	return &ModelChainPlugin{chains: config.Chains}, nil
}

func (p *ModelChainPlugin) GetName() string { return PluginName }

func (p *ModelChainPlugin) Cleanup() error { return nil }

// chainFor returns the chain for a request's model, accepting it with or without a provider prefix.
func (p *ModelChainPlugin) chainFor(provider schemas.ModelProvider, model string) []Target {
	if chain, ok := p.chains[model]; ok {
		return chain
	}
	if provider != "" {
		return p.chains[string(provider)+"/"+model]
	}
	return nil
}

// PreRequestHook routes a logical rerank or decision model through its chain.
func (p *ModelChainPlugin) PreRequestHook(_ *schemas.BifrostContext, req *schemas.BifrostRequest) error {
	if req == nil || (req.RequestType != schemas.RerankRequest && req.RequestType != schemas.DecisionRequest) {
		return nil
	}
	provider, model, _ := req.GetRequestFields()
	chain := p.chainFor(provider, model)
	if chain == nil {
		return nil
	}
	req.SetProvider(chain[0].Provider)
	req.SetModel(chain[0].Model)
	fallbacks := make([]schemas.Fallback, 0, len(chain)-1)
	for _, t := range chain[1:] {
		fallbacks = append(fallbacks, schemas.Fallback{Provider: t.Provider, Model: t.Model})
	}
	req.SetFallbacks(fallbacks)
	return nil
}

func (p *ModelChainPlugin) PreLLMHook(_ *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	return req, nil, nil
}

func (p *ModelChainPlugin) PostLLMHook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	return resp, bifrostErr, nil
}

// HTTPTransportPreAuthHook is a no-op.
func (*ModelChainPlugin) HTTPTransportPreAuthHook(_ *schemas.BifrostContext, _ *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}

func (*ModelChainPlugin) HTTPTransportPreHook(_ *schemas.BifrostContext, _ *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}

// HTTPTransportPostHook replaces a failed chained rerank with the original document order.
func (p *ModelChainPlugin) HTTPTransportPostHook(_ *schemas.BifrostContext, req *schemas.HTTPRequest, resp *schemas.HTTPResponse) error {
	if req == nil || resp == nil || req.Path != route || resp.StatusCode == http.StatusOK || slices.Contains(callerFaults, resp.StatusCode) {
		return nil
	}
	var body struct {
		Model     string            `json:"model"`
		Documents []json.RawMessage `json:"documents"`
	}
	if json.Unmarshal(req.Body, &body) != nil || len(body.Documents) == 0 || p.chainFor("", body.Model) == nil {
		return nil
	}
	var failure struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(resp.Body, &failure) == nil && slices.Contains(callerRefusals, failure.Type) {
		return nil
	}

	n := len(body.Documents)
	results := make([]map[string]any, n)
	for i := range results {
		results[i] = map[string]any{"index": i, "relevance_score": 1 - float64(i)/float64(n)}
	}
	out, err := json.Marshal(map[string]any{"model": "identity", "usage": map[string]int{"total_tokens": 0}, "results": results})
	if err != nil {
		return nil
	}
	resp.StatusCode = http.StatusOK
	resp.Body = out
	if resp.Headers == nil {
		resp.Headers = map[string]string{}
	}
	resp.Headers["Content-Type"] = "application/json"
	return nil
}

func (*ModelChainPlugin) HTTPTransportResponseHeadersHook(_ *schemas.BifrostContext, _ *schemas.HTTPRequest, _ *schemas.HTTPResponseMetadata) error {
	return nil
}

func (*ModelChainPlugin) HTTPTransportStreamChunkHook(_ *schemas.BifrostContext, _ *schemas.HTTPRequest, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
	return chunk, nil
}
