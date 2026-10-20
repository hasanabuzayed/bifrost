package handlers

import (
	"github.com/fasthttp/router"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/plugins/gliner"
	"github.com/maximhq/bifrost/plugins/governance"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/lib"
	"github.com/valyala/fasthttp"
)

// GlinerHandler serves the gliner plugin's Fastino-compatible NER route. Bifrost has no NER
// request type, so the request never enters core, where governance normally runs: this handler
// settles the caller and asks governance for the verdict before the plugin does any work.
type GlinerHandler struct {
	config *lib.Config
}

func NewGlinerHandler(config *lib.Config) *GlinerHandler {
	return &GlinerHandler{config: config}
}

func (h *GlinerHandler) RegisterRoutes(r *router.Router, middlewares ...schemas.BifrostHTTPMiddleware) {
	r.POST(gliner.Route, lib.ChainMiddlewares(h.serve, middlewares...))
}

func (h *GlinerHandler) serve(ctx *fasthttp.RequestCtx) {
	plugin, err := lib.FindPluginAs[*gliner.GlinerPlugin](h.config, gliner.PluginName)
	if err != nil {
		SendError(ctx, fasthttp.StatusNotFound, "the gliner plugin is not enabled")
		return
	}
	// Unlike the realtime minting path, a missing governance plugin refuses the request:
	// this route spends a hosted API key and must never run ungoverned.
	governancePlugin, err := lib.FindPluginAs[governance.BaseGovernancePlugin](h.config, governance.PluginName)
	if err != nil {
		SendError(ctx, fasthttp.StatusServiceUnavailable, "governance is not loaded")
		return
	}

	bifrostCtx, cancel := lib.ConvertToBifrostContext(ctx, h.config)
	defer cancel()
	if userID, ok := ctx.UserValue(schemas.BifrostContextKeyUserID).(string); ok && userID != "" {
		bifrostCtx.SetValue(schemas.BifrostContextKeyUserID, userID)
	}
	if userName, ok := ctx.UserValue(schemas.BifrostContextKeyUserName).(string); ok && userName != "" {
		bifrostCtx.SetValue(schemas.BifrostContextKeyUserName, userName)
	}
	// The credential and the user travel on the context, which is where evaluation reads them.
	if _, denied := governancePlugin.Evaluate(bifrostCtx, &governance.EvaluationRequest{
		RequestType: schemas.PassthroughRequest,
		Provider:    "fastino",
		Model:       "gliner-2",
	}); denied != nil {
		SendBifrostError(ctx, denied)
		return
	}

	status, body := plugin.Serve(bifrostCtx, ctx.PostBody())
	ctx.SetStatusCode(status)
	ctx.SetContentType("application/json")
	ctx.SetBody(body)
}
