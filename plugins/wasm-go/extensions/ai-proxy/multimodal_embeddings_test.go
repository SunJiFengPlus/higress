package main

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	wasmhost "github.com/higress-group/wasm-go/pkg/test"
	"github.com/stretchr/testify/require"
)

// These tests use the registered plugin and SDK host emulator, so routing,
// body buffering, provider transforms and response handling all run together.
func TestMultimodalEmbeddings(t *testing.T) {
	wasmhost.RunTest(t, func(t *testing.T) {
		qwenInputs := []string{
			`{"contents":[{"text":"blue sky"}]}`,
			`{"contents":[{"image":"data:image/png;base64,aGVsbG8="}]}`,
			`{"contents":[{"text":"blue sky"},{"image":"https://example.com/image.jpg"}]}`,
			`{"contents":[{"video":"https://example.com/video.mp4"}]}`,
		}
		for _, backend := range []struct {
			name, typ, path, domain string
			compatible              bool
			inputs                  []string
			responses               []string
		}{
			{"qwen", "qwen", "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding", "dashscope.aliyuncs.com", false,
				qwenInputs,
				[]string{`{"output":{"embeddings":[{"embedding":[0.1,0.2],"index":3,"type":"image","extra":{"x":1}}]},"usage":{"input_tokens":7,"image_tokens":4},"request_id":"qwen-id","unknown":true}`}},
			{"qwen-compatible", "qwen", "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding", "dashscope.aliyuncs.com", true,
				qwenInputs,
				[]string{`{"output":{"embeddings":[{"embedding":[0.1],"index":0,"type":"mixed"}]},"usage":{"input_tokens":7}}`}},
			{"doubao", "doubao", "/api/v3/embeddings/multimodal", "ark.cn-beijing.volces.com", false,
				[]string{`[{"type":"text","text":"blue sky"}]`, `[{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}]`, `[{"type":"text","text":"blue sky"},{"type":"image_url","image_url":{"url":"https://example.com/image.jpg"}}]`, `[{"type":"video_url","video_url":{"url":"https://example.com/video.mp4","fps":1}}]`},
				[]string{`{"id":"ark-id","data":{"embedding":[0.1,0.2],"object":"embedding","extra":{"x":1}},"usage":{"prompt_tokens":7,"total_tokens":7},"unknown":true}`, `{"id":"ark-base64","data":{"embedding":"AACAPw==","sparse_embedding":{"indices":[1],"values":[0.5]}},"usage":{"total_tokens":2}}`}},
		} {
			t.Run(backend.name, func(t *testing.T) {
				for i, input := range backend.inputs {
					t.Run([]string{"text", "image", "mixed", "video"}[i], func(t *testing.T) {
						cfg := map[string]interface{}{"type": backend.typ, "protocol": "original", "basePath": "/gateway", "apiTokens": []string{"test-token"}, "qwenEnableCompatible": backend.compatible, "modelMapping": map[string]string{"test-model": "must-not-map"}, "customSettings": []map[string]interface{}{{"name": "extra.added", "value": "must-not-add", "mode": "raw"}}}
						// Fields and formatting stay opaque despite modelMapping
						// and customSettings, including chat-shaped unknown fields.
						body := []byte(`{ "model":"test-model", "input":` + input + `, "parameters":{"enable_fusion":true,"dimension":1024,"fps":1},"encoding_format":"float","unknown":{"keep":[1,"x"]},"messages":[{"role":"assistant","reasoning_content":"opaque"}] }`)
						host := newMultimodalHost(t, cfg, "/gateway"+backend.path+"?trace=1", backend.path+"?trace=1", backend.domain)
						host.CallOnHttpRequestBody(body)
						require.Nil(t, host.GetLocalResponse())
						require.Equal(t, body, host.GetRequestBody())
						response := backend.responses[i%len(backend.responses)]
						assertMultimodalResponse(t, host, "200", response)
					})
				}
				t.Run("opaque-model-and-error", func(t *testing.T) {
					host := newMultimodalHost(t, map[string]interface{}{"type": backend.typ, "protocol": "original", "apiTokens": []string{"test-token"}, "qwenEnableCompatible": backend.compatible}, backend.path, backend.path, backend.domain)
					body := `{"model":"ep-opaque","input":` + backend.inputs[0] + `,"extra":{"preserve":true}}`
					host.CallOnHttpRequestBody([]byte(body))
					require.Equal(t, []byte(body), host.GetRequestBody())
					assertMultimodalResponse(t, host, "400", `{"code":"InvalidParameter","message":"provider error","request_id":"error-id","extra":{"preserve":true}}`)
				})
			})
		}
	})
}

func TestMultimodalKeepsLegacyRoutes(t *testing.T) {
	wasmhost.RunTest(t, func(t *testing.T) {
		for _, tc := range []struct {
			name, typ, path, upstream, domain string
			compatible                        bool
		}{
			{"qwen-text", "qwen", "/v1/embeddings", "/api/v1/services/embeddings/text-embedding/text-embedding", "dashscope.aliyuncs.com", false},
			{"qwen-compatible-text", "qwen", "/v1/embeddings", "/compatible-mode/v1/embeddings", "dashscope.aliyuncs.com", true},
			{"doubao-text", "doubao", "/v1/embeddings", "/api/v3/embeddings", "ark.cn-beijing.volces.com", false},
		} {
			t.Run(tc.name, func(t *testing.T) {
				host := newMultimodalHost(t, map[string]interface{}{"type": tc.typ, "apiTokens": []string{"test-token"}, "qwenEnableCompatible": tc.compatible, "modelMapping": map[string]string{"alias": "mapped"}}, tc.path, tc.upstream, tc.domain)
				body := `{"model":"alias","input":["text"],"messages":[{"role":"user","content":"hello"}]}`
				host.CallOnHttpRequestBody([]byte(body))
				require.Nil(t, host.GetLocalResponse())
				if tc.name == "qwen-text" {
					require.JSONEq(t, `{"model":"mapped","input":{"texts":["text"]},"parameters":{}}`, string(host.GetRequestBody()))
				} else {
					require.JSONEq(t, `{"model":"mapped","input":["text"],"messages":[{"role":"user","content":"hello"}]}`, string(host.GetRequestBody()))
				}
			})
		}
	})
}

func newMultimodalHost(t *testing.T, cfg map[string]interface{}, requestPath, upstreamPath, domain string, extraHeaders ...[2]string) wasmhost.TestHost {
	t.Helper()
	data, err := json.Marshal(map[string]interface{}{"provider": cfg})
	require.NoError(t, err)
	host, status := wasmhost.NewTestHost(data)
	t.Cleanup(host.Reset)
	require.Equal(t, types.OnPluginStartStatusOK, status)
	headers := [][2]string{{":authority", "gateway.example.com"}, {":method", "POST"}, {":path", requestPath}, {"content-type", "application/json"}, {"authorization", "Bearer client-token"}}
	host.CallOnHttpRequestHeaders(append(headers, extraHeaders...))
	for name, want := range map[string]string{":path": upstreamPath, ":authority": domain} {
		got, ok := wasmhost.GetHeaderValue(host.GetRequestHeaders(), name)
		require.True(t, ok, name)
		require.Equal(t, want, got, name)
	}
	auth, ok := wasmhost.GetHeaderValue(host.GetRequestHeaders(), "authorization")
	require.True(t, ok)
	require.True(t, strings.HasPrefix(auth, "Bearer "))
	require.Contains(t, cfg["apiTokens"], strings.TrimPrefix(auth, "Bearer "))
	return host
}

func assertMultimodalResponse(t *testing.T, host wasmhost.TestHost, status, body string) {
	t.Helper()
	require.NoError(t, host.SetProperty([]string{"response", "code_details"}, []byte("via_upstream")))
	require.Equal(t, types.ActionContinue, host.CallOnHttpResponseHeaders([][2]string{{":status", status}, {"content-type", "application/json"}, {"x-provider-request-id", "test-id"}}))
	host.CallOnHttpResponseBody([]byte(body))
	require.Equal(t, []byte(body), host.GetResponseBody())
	for name, want := range map[string]string{":status": status, "content-type": "application/json", "x-provider-request-id": "test-id"} {
		got, ok := wasmhost.GetHeaderValue(host.GetResponseHeaders(), name)
		require.True(t, ok, name)
		require.Equal(t, want, got, name)
	}
}

func TestMultimodalResponseOptions(t *testing.T) {
	wasmhost.RunTest(t, func(t *testing.T) {
		for _, option := range []string{"promoteThinkingOnEmpty", "hiclawMode"} {
			for _, compatible := range []bool{false, true} {
				name := option
				if compatible {
					name += "-compatible"
				}
				t.Run(name, func(t *testing.T) {
					host := newMultimodalHost(t, map[string]interface{}{"type": "qwen", "protocol": "original", "apiTokens": []string{"test-token"}, "qwenEnableCompatible": compatible, option: true}, "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding", "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding", "dashscope.aliyuncs.com")
					host.CallOnHttpRequestBody([]byte(`{"model":"test-model","input":{"contents":[{"text":"hello"}]}}`))
					assertMultimodalResponse(t, host, "200", `{"output":{"embeddings":[{"embedding":[0.1],"index":0}]},"usage":{"input_tokens":3},"request_id":"test-id","choices":[{"message":{"content":"","reasoning_content":"opaque"}}],"extra":{"keep":true}}`)
				})
			}
		}
	})
}

func TestMultimodalRetry(t *testing.T) {
	wasmhost.RunTest(t, func(t *testing.T) {
		for _, backend := range []struct {
			name, typ, path, upstreamPath, input, response string
			compatible                                     bool
		}{
			{"qwen", "qwen", "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding", "/tenant/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding?trace=1&opaque=a%2Fb", `{"contents":[{"text":"hello"},{"image":"data:image/png;base64,aGVsbG8="}]}`, `{ "output":{"embeddings":[{"embedding":[0.1,0.2],"index":3,"type":"image","extra":{"keep":true}}]},"usage":{"input_tokens":7,"image_tokens":4},"request_id":"test-id","unknown":"AACAPw==" }`, false},
			{"qwen-compatible", "qwen", "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding", "/tenant/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding?trace=1&opaque=a%2Fb", `{"contents":[{"image":"data:image/png;base64,aGVsbG8="}]}`, `{ "output":{"embeddings":[{"embedding":[0.1],"index":0,"type":"mixed","unknown":true}]},"usage":{"input_tokens":7},"request_id":"test-id","unknown":"AACAPw==" }`, true},
			{"doubao", "doubao", "/api/v3/embeddings/multimodal", "/tenant/api/v3/embeddings/multimodal?trace=1&opaque=a%2Fb", `[{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}]`, `{ "id":"test-id","data":{"embedding":"AACAPw==","sparse_embedding":{"indices":[1],"values":[0.5]},"extra":{"keep":true}},"usage":{"total_tokens":2},"unknown":true }`, false},
		} {
			t.Run(backend.name, func(t *testing.T) {
				for _, tc := range []struct {
					name        string
					enabled     bool
					statuses    []string
					maxRetries  int
					wantRetries int
					succeed     bool
				}{
					{"success", true, []string{"503"}, 1, 1, true},
					{"disabled", false, []string{"503"}, 1, 0, false},
					{"unmatched-status", true, []string{"429"}, 1, 0, false},
					{"max-retries", true, []string{"503"}, 2, 2, false},
				} {
					t.Run(tc.name, func(t *testing.T) {
						cfg := map[string]interface{}{
							"type": backend.typ, "protocol": "original", "apiTokens": []string{"test-token", "second-token", "third-token", "fourth-token"}, "qwenEnableCompatible": backend.compatible,
							"basePath": "/gateway", "basePathHandling": "removePrefix", "providerBasePath": "/tenant", "providerDomain": "example.com",
							"modelMapping": map[string]string{"test-model": "must-not-map"}, "customSettings": []map[string]interface{}{{"name": "extra.added", "value": "must-not-add", "mode": "raw"}},
							"retryOnFailure": map[string]interface{}{"enabled": tc.enabled, "retryOnStatus": tc.statuses, "maxRetries": tc.maxRetries},
						}
						body := []byte(`{ "model":"test-model", "input":` + backend.input + `, "unknown":{"keep":[1,"x"]},"messages":[{"role":"assistant","reasoning_content":"opaque"}] }`)
						host := newMultimodalHost(t, cfg, "/gateway"+backend.path+"?trace=1&opaque=a%2Fb", backend.upstreamPath, "example.com", [2]string{"content-length", strconv.Itoa(len(body))}, [2]string{"x-private-canary", "must-not-replay"})
						require.NoError(t, host.SetClusterName("test-cluster"))
						host.CallOnHttpRequestBody(body)
						require.Equal(t, body, host.GetRequestBody())
						firstAuth, ok := wasmhost.GetHeaderValue(host.GetRequestHeaders(), "authorization")
						require.True(t, ok)
						usedTokens := []string{firstAuth}

						originalError := []byte(`{ "code":"ServiceUnavailable", "message":"original provider error", "extra":{"keep":true} }`)
						require.NoError(t, host.SetProperty([]string{"response", "code_details"}, []byte("via_upstream")))
						action := host.CallOnHttpResponseHeaders([][2]string{{":status", "503"}, {"content-type", "application/json"}, {"x-provider-request-id", "original-id"}})
						if tc.wantRetries == 0 {
							require.Equal(t, types.ActionContinue, action)
						} else {
							require.Equal(t, types.HeaderStopAllIterationAndWatermark, action)
						}
						for i := 0; i < tc.wantRetries; i++ {
							calls := host.GetHttpCalloutAttributes()
							require.Len(t, calls, 1, "retry %d", i+1)
							call := calls[0]
							require.Equal(t, "test-cluster", call.Upstream)
							require.Equal(t, body, call.Body, "retry must preserve body bytes")
							for name, want := range map[string]string{":method": "POST", ":path": backend.upstreamPath, ":authority": "example.com", "content-type": "application/json"} {
								got, ok := wasmhost.GetHeaderValue(call.Headers, name)
								require.True(t, ok, name)
								require.Equal(t, want, got, name)
							}
							for _, header := range call.Headers {
								require.Contains(t, []string{":method", ":path", ":authority", "content-type", "authorization"}, strings.ToLower(header[0]), "retry headers must exclude caller Content-Length and private headers")
							}
							auth, ok := wasmhost.GetHeaderValue(call.Headers, "authorization")
							require.True(t, ok)
							require.True(t, strings.HasPrefix(auth, "Bearer "))
							require.Contains(t, cfg["apiTokens"], strings.TrimPrefix(auth, "Bearer "))
							require.NotContains(t, usedTokens, auth, "retry must rotate the failed token")
							usedTokens = append(usedTokens, auth)
							if tc.succeed {
								host.CallOnHttpCall([][2]string{{":status", "200"}, {"content-type", "application/json"}, {"x-provider-request-id", "retry-id"}}, []byte(backend.response))
							} else {
								host.CallOnHttpCall([][2]string{{":status", "503"}, {"content-type", "application/json"}}, []byte(`{"message":"retry provider error"}`))
							}
						}
						require.Empty(t, host.GetHttpCalloutAttributes())
						if tc.succeed {
							response := host.GetLocalResponse()
							require.NotNil(t, response)
							require.Equal(t, uint32(200), response.StatusCode)
							require.Equal(t, []byte(backend.response), response.Data)
							requestID, ok := wasmhost.GetHeaderValue(response.Headers, "x-provider-request-id")
							require.True(t, ok)
							require.Equal(t, "retry-id", requestID)
						} else {
							require.Equal(t, types.ActionContinue, host.GetHttpStreamAction())
							require.Nil(t, host.GetLocalResponse())
							host.CallOnHttpResponseBody(originalError)
							require.Equal(t, originalError, host.GetResponseBody())
							status, ok := wasmhost.GetHeaderValue(host.GetResponseHeaders(), ":status")
							require.True(t, ok)
							require.Equal(t, "503", status)
						}
					})
				}
			})
		}
	})
}
