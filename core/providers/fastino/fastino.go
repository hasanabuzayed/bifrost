// Package fastino implements the Fastino provider for Bifrost.
//
// Fastino serves three model kinds behind one key: GLiNER encoders (entities, classifications,
// structures, relations) through POST /v1/chat/completions with a top-level "schema", GLiDE
// decisions through POST /v1/systemone, and decoder LLMs through the same chat endpoint.
// Extraction is implemented here. Decisions go through the Typesafe adapter, whose wire shape
// Fastino's /v1/systemone shares. Chat, streaming and model listing go through the embedded
// OpenAI provider, which Fastino's endpoints are compatible with.
//
// As a custom provider base type, it lets a self-hosted GLiNER2 server that copies Fastino's
// GLiNER contract sit behind the same adapter, so cloud and local answers have one shape.
package fastino

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/bytedance/sonic"

	"github.com/maximhq/bifrost/core/providers/openai"
	"github.com/maximhq/bifrost/core/providers/typesafe"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

const (
	fastinoDefaultBaseURL = "https://api.fastino.ai"
	fastinoChatPath       = "/v1/chat/completions"
)

// FastinoProvider implements the Provider interface for Fastino's API.
// ponytail: operations Fastino does not serve (embeddings, images, speech, ...) reach Fastino
// through the embedded OpenAI provider and fail there; add explicit stubs if the upstream error
// ever needs to be a Bifrost "unsupported operation" instead.
type FastinoProvider struct {
	*openai.OpenAIProvider
	decisions            *typesafe.TypesafeProvider
	logger               schemas.Logger
	client               *fasthttp.Client
	networkConfig        schemas.NetworkConfig
	sendBackRawRequest   bool
	sendBackRawResponse  bool
	customProviderConfig *schemas.CustomProviderConfig
}

// NewFastinoProvider creates a new Fastino provider instance.
func NewFastinoProvider(config *schemas.ProviderConfig, logger schemas.Logger) (*FastinoProvider, error) {
	config.CheckAndSetDefaults()
	if config.NetworkConfig.BaseURL == "" {
		config.NetworkConfig.BaseURL = fastinoDefaultBaseURL
	}
	config.NetworkConfig.BaseURL = strings.TrimRight(config.NetworkConfig.BaseURL, "/")

	requestTimeout := time.Second * time.Duration(config.NetworkConfig.DefaultRequestTimeoutInSeconds)
	client := &fasthttp.Client{
		ReadTimeout:         requestTimeout,
		WriteTimeout:        requestTimeout,
		MaxConnsPerHost:     config.NetworkConfig.MaxConnsPerHost,
		MaxIdleConnDuration: time.Second * time.Duration(config.NetworkConfig.KeepAliveTimeoutInSeconds),
		MaxConnWaitTimeout:  requestTimeout,
		MaxConnDuration:     time.Second * time.Duration(schemas.DefaultMaxConnDurationInSeconds),
		ConnPoolStrategy:    fasthttp.FIFO,
	}
	client = providerUtils.ConfigureProxy(client, config.ProxyConfig, logger)
	client = providerUtils.ConfigureDialer(client, config.NetworkConfig.AllowPrivateNetwork)
	client = providerUtils.ConfigureTLS(client, config.NetworkConfig, logger)

	decisions, err := typesafe.NewTypesafeProvider(delegateConfig(config, schemas.Typesafe), logger)
	if err != nil {
		return nil, err
	}
	return &FastinoProvider{
		OpenAIProvider:       openai.NewOpenAIProvider(delegateConfig(config, schemas.OpenAI), logger),
		decisions:            decisions,
		logger:               logger,
		client:               client,
		networkConfig:        config.NetworkConfig,
		sendBackRawRequest:   config.SendBackRawRequest,
		sendBackRawResponse:  config.SendBackRawResponse,
		customProviderConfig: config.CustomProviderConfig,
	}, nil
}

// delegateConfig copies the provider config for a delegate adapter, keeping this provider's
// name (so errors, logs and metrics say "fastino" or the custom name) and its request gating.
func delegateConfig(config *schemas.ProviderConfig, base schemas.ModelProvider) *schemas.ProviderConfig {
	copied := *config
	custom := schemas.CustomProviderConfig{CustomProviderKey: string(schemas.Fastino), BaseProviderType: base}
	if config.CustomProviderConfig != nil {
		custom = *config.CustomProviderConfig
		custom.BaseProviderType = base
	}
	copied.CustomProviderConfig = &custom
	return &copied
}

// GetProviderKey returns "fastino", or the custom provider name when Fastino backs one.
func (provider *FastinoProvider) GetProviderKey() schemas.ModelProvider {
	return providerUtils.GetProviderName(schemas.Fastino, provider.customProviderConfig)
}

// Decision runs a GLiDE decision on POST /v1/systemone through the Typesafe adapter.
func (provider *FastinoProvider) Decision(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostDecisionRequest) (*schemas.BifrostDecisionResponse, *schemas.BifrostError) {
	return provider.decisions.Decision(ctx, key, request)
}

// fastinoExtractionResponse is the part of Fastino's chat-shaped GLiNER answer Bifrost reads.
type fastinoExtractionResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// Extraction runs a GLiNER schema extraction: the text goes in a user message and the schema
// at the top level, and the result is the JSON object in choices[0].message.content.
func (provider *FastinoProvider) Extraction(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostExtractionRequest) (*schemas.BifrostExtractionResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.Fastino, provider.customProviderConfig, schemas.ExtractionRequest); err != nil {
		return nil, err
	}
	body, err := buildExtractionBody(request)
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderRequestMarshal, err)
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
	path, isCompleteURL := providerUtils.GetRequestPath(ctx, fastinoChatPath, provider.customProviderConfig, schemas.ExtractionRequest)
	if !isCompleteURL {
		path = provider.networkConfig.BaseURL + path
	}
	req.SetRequestURI(path)
	req.Header.SetMethod(http.MethodPost)
	req.Header.SetContentType("application/json")
	if key.Value.GetValue() != "" {
		req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	}
	req.SetBody(body)

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))
	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, providerUtils.SetErrorLatency(openai.ParseOpenAIError(resp), latency)
	}

	respBody, err := providerUtils.CheckAndDecodeBody(resp)
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseDecode, err)
	}
	var native fastinoExtractionResponse
	if err := sonic.Unmarshal(respBody, &native); err != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseUnmarshal, err)
	}
	if len(native.Choices) == 0 {
		return nil, providerUtils.NewBifrostOperationError("fastino returned no choices for the extraction", nil)
	}
	content := strings.TrimSpace(native.Choices[0].Message.Content)
	if !sonic.ValidString(content) || !strings.HasPrefix(content, "{") {
		return nil, providerUtils.NewBifrostOperationError(fmt.Sprintf("fastino extraction content is not a JSON object: %.200s", content), nil)
	}

	model := native.Model
	if model == "" {
		model = request.Model
	}
	response := &schemas.BifrostExtractionResponse{
		ID:     native.ID,
		Model:  model,
		Result: []byte(content),
	}
	if native.Usage != nil {
		response.Usage = &schemas.BifrostLLMUsage{
			PromptTokens:     native.Usage.PromptTokens,
			CompletionTokens: native.Usage.CompletionTokens,
			TotalTokens:      native.Usage.TotalTokens,
		}
	}
	response.ExtraFields.Latency = latency.Milliseconds()
	if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
		response.ExtraFields.RawRequest = sonic.NoCopyRawMessage(body)
	}
	if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
		response.ExtraFields.RawResponse = sonic.NoCopyRawMessage(respBody)
	}
	return response, nil
}

// fastinoMessage keeps role before content on the wire.
type fastinoMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// buildExtractionBody writes Fastino's GLiNER request: extra params first, so the modelled
// fields always win, then model, messages, schema and the optional knobs.
func buildExtractionBody(request *schemas.BifrostExtractionRequest) ([]byte, error) {
	body := map[string]interface{}{}
	if request.Params != nil {
		for k, v := range request.Params.ExtraParams {
			body[k] = v
		}
		if request.Params.Threshold != nil {
			body["threshold"] = *request.Params.Threshold
		}
		if request.Params.IncludeConfidence != nil {
			body["include_confidence"] = *request.Params.IncludeConfidence
		}
		if request.Params.IncludeSpans != nil {
			body["include_spans"] = *request.Params.IncludeSpans
		}
	}
	body["model"] = request.Model
	body["messages"] = []fastinoMessage{{Role: "user", Content: request.Text}}
	body["schema"] = sonic.NoCopyRawMessage(request.Schema)
	return sonic.Marshal(body)
}
