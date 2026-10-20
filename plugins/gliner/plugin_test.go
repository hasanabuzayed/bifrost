package gliner

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
)

var testLogger = bifrost.NewDefaultLogger(schemas.LogLevelError)

// upstream records raw request bodies and API keys, and answers with the given status and body.
type upstream struct {
	mu     sync.Mutex
	bodies []string
	keys   []string
	status int
	body   string
}

func (u *upstream) server(t *testing.T) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		u.mu.Lock()
		u.bodies = append(u.bodies, string(data))
		u.keys = append(u.keys, r.Header.Get("X-API-Key"))
		status, body := u.status, u.body
		u.mu.Unlock()
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(s.Close)
	return s
}

func (u *upstream) calls() int {
	u.mu.Lock()
	defer u.mu.Unlock()
	return len(u.bodies)
}

func (u *upstream) set(status int) {
	u.mu.Lock()
	u.status = status
	u.mu.Unlock()
}

// fastinoLive is the shape the live API returns: entities keyed by label.
const fastinoLive = `{"result":{"entities":{"person":[{"text":"Hasan","confidence":1.0,"start":0,"end":5}]}},"token_usage":12,"x_fastino":null}`

const localOK = `{"entities":[{"start":7,"end":13,"text":"Riyadh","label":"city","score":0.8},
{"start":0,"end":5,"text":"Hasan","label":"person","score":0.9},{"text":"x","label":"not-asked","score":0.9}],"model":"gliner-x-base"}`

func newPlugin(t *testing.T, fastinoURL, key, localURL string, cap int) *GlinerPlugin {
	t.Helper()
	var secret *schemas.SecretVar
	if key != "" {
		secret = schemas.NewSecretVar(key)
	}
	p, err := Init(&Config{FastinoURL: fastinoURL, FastinoAPIKey: secret, FastinoDailyCap: cap, LocalURL: localURL}, testLogger)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func call(t *testing.T, p *GlinerPlugin, body string) (int, []byte, map[string]any) {
	t.Helper()
	status, resp := p.Serve(context.Background(), []byte(body))
	var out map[string]any
	_ = json.Unmarshal(resp, &out)
	return status, resp, out
}

func entities(out map[string]any) map[string]any {
	return out["result"].(map[string]any)["entities"].(map[string]any)
}

func TestFastinoFirstUnchanged(t *testing.T) {
	fastino, local := &upstream{status: 200, body: fastinoLive}, &upstream{status: 200, body: localOK}
	fs, ls := fastino.server(t), local.server(t)
	p := newPlugin(t, fs.URL, "k", ls.URL, 0)
	req := `{"text":"Hasan, Riyadh","schema":{"entities":["person","city"]},"threshold":0.4}`

	status, raw, _ := call(t, p, req)
	if status != 200 || string(raw) != fastinoLive {
		t.Fatalf("want Fastino's body unchanged, got %d %s", status, raw)
	}
	if fastino.bodies[0] != req || fastino.keys[0] != "k" || local.calls() != 0 {
		t.Fatalf("want the request unchanged with the key, got %q key %q, local calls %d", fastino.bodies[0], fastino.keys[0], local.calls())
	}
}

func TestLocalFallbackInFastinoShape(t *testing.T) {
	fastino, local := &upstream{status: 500}, &upstream{status: 200, body: localOK}
	fs, ls := fastino.server(t), local.server(t)
	p := newPlugin(t, fs.URL, "k", ls.URL, 0)

	status, raw, out := call(t, p, `{"text":"Hasan, Riyadh","schema":{"entities":{"person":"a name","city":"a city"}},"threshold":0.3}`)
	if status != 200 {
		t.Fatalf("want local fallback 200, got %d %s", status, raw)
	}
	got := entities(out)
	city := got["city"].([]any)[0].(map[string]any)
	if len(got) != 2 || city["text"] != "Riyadh" || city["confidence"] != 0.8 || city["start"] != 7.0 {
		t.Fatalf("want only asked labels in Fastino's shape, got %v", got)
	}
	var sent map[string]any
	_ = json.Unmarshal([]byte(local.bodies[0]), &sent)
	if sent["threshold"] != 0.3 || len(sent["entity_types"].([]any)) != 2 {
		t.Fatalf("local request: %v", sent)
	}

	// The deprecated flat label list, without confidence or spans.
	_, _, out = call(t, p, `{"text":"Hasan","schema":["person"],"include_confidence":false,"include_spans":false}`)
	if person := entities(out)["person"].([]any)[0].(map[string]any); len(person) != 1 || person["text"] != "Hasan" {
		t.Fatalf("want text only, got %v", person)
	}

	// No local fallback for other tasks or batched text, and every upstream failing is a 502.
	for _, body := range []string{
		`{"text":"x","schema":{"entities":["city"],"classifications":[{"task":"t","labels":["a"]}]}}`,
		`{"text":["a","b"],"schema":["city"]}`,
	} {
		if status, _, _ := call(t, p, body); status != 502 {
			t.Fatalf("%s: want 502, got %d", body, status)
		}
	}
	local.set(500)
	if status, _, _ := call(t, p, `{"text":"x","schema":["city"]}`); status != 502 {
		t.Fatalf("want 502 when every upstream fails, got %d", status)
	}
}

func TestFastinoDailyCap(t *testing.T) {
	fastino, local := &upstream{status: 200, body: fastinoLive}, &upstream{status: 200, body: localOK}
	fs, ls := fastino.server(t), local.server(t)
	p := newPlugin(t, fs.URL, "k", ls.URL, 2)
	for i := 0; i < 3; i++ {
		call(t, p, `{"text":"x","schema":["city"]}`)
	}
	if fastino.calls() != 2 || local.calls() != 1 {
		t.Fatalf("want 2 Fastino + 1 local, got %d + %d", fastino.calls(), local.calls())
	}
}

func TestFastinoKeyFromEnv(t *testing.T) {
	t.Setenv("GLINER_TEST_FASTINO_KEY", "from-env")
	fastino := &upstream{status: 200, body: fastinoLive}
	fs := fastino.server(t)
	var cfg Config
	if err := json.Unmarshal([]byte(`{"fastino_url":"`+fs.URL+`","fastino_api_key":"env.GLINER_TEST_FASTINO_KEY"}`), &cfg); err != nil {
		t.Fatal(err)
	}
	p, err := Init(&cfg, testLogger)
	if err != nil {
		t.Fatal(err)
	}
	call(t, p, `{"text":"x","schema":["city"]}`)
	if len(fastino.keys) != 1 || fastino.keys[0] != "from-env" {
		t.Fatalf("want the resolved env value sent, got %q", fastino.keys)
	}
}

func TestBadInput(t *testing.T) {
	p := newPlugin(t, "", "", "http://127.0.0.1:9", 0)
	for _, body := range []string{`not json`, `{"schema":["city"]}`, `{"text":"x"}`} {
		if status, _, _ := call(t, p, body); status != 400 {
			t.Fatalf("%s: want 400, got %d", body, status)
		}
	}
	if _, err := Init(&Config{}, testLogger); err == nil {
		t.Fatal("want an error without any upstream")
	}
}
