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

func TestMultimodalNativePathsAreNotGatewayAliases(t *testing.T) {
	for _, path := range []string{
		"/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
		"/api/v3/embeddings/multimodal",
	} {
		t.Run(path, func(t *testing.T) { require.Empty(t, getApiName(path)) })
	}
}

// These tests use the registered plugin and SDK host emulator, so routing,
// body buffering, provider transforms and response handling all run together.
func TestNativeMultimodalEmbeddings(t *testing.T) {
	wasmhost.RunTest(t, func(t *testing.T) {
		qwenInputs := []string{
			`{"contents":[{"text":"blue sky"}]}`,
			`{"contents":[{"image":"data:image/png;base64,aGVsbG8="}]}`,
			`{"contents":[{"text":"blue sky"},{"image":"https://example.com/image.jpg"}]}`,
			`{"contents":[{"video":"https://example.com/video.mp4"}]}`,
		}
		for _, backend := range []struct {
			name, typ, alias, nativePath, domain, mappedModel string
			compatible                                        bool
			inputs                                            []string
			responses                                         []string
		}{
			{"qwen", "qwen", "/v1/multimodal-embedding", "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding", "dashscope.aliyuncs.com", "qwen3-vl-embedding", false,
				qwenInputs,
				[]string{`{"output":{"embeddings":[{"embedding":[0.1,0.2],"index":3,"type":"image","extra":{"x":1}}]},"usage":{"input_tokens":7,"image_tokens":4},"request_id":"qwen-id","unknown":true}`}},
			{"qwen-compatible", "qwen", "/v1/multimodal-embedding", "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding", "dashscope.aliyuncs.com", "qwen3.6-plus", true,
				qwenInputs,
				[]string{`{"output":{"embeddings":[{"embedding":[0.1],"index":0,"type":"mixed"}]},"usage":{"input_tokens":7}}`}},
			{"doubao", "doubao", "/v1/embeddings/multimodal", "/api/v3/embeddings/multimodal", "ark.cn-beijing.volces.com", "ep-20261109000000-opaque", false,
				[]string{`[{"type":"text","text":"blue sky"}]`, `[{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}]`, `[{"type":"text","text":"blue sky"},{"type":"image_url","image_url":{"url":"https://example.com/image.jpg"}}]`, `[{"type":"video_url","video_url":{"url":"https://example.com/video.mp4","fps":1}}]`},
				[]string{`{"id":"ark-id","data":{"embedding":[0.1,0.2],"object":"embedding","extra":{"x":1}},"usage":{"prompt_tokens":7,"total_tokens":7},"unknown":true}`, `{"id":"ark-base64","data":{"embedding":"AACAPw==","sparse_embedding":{"indices":[1],"values":[0.5]}},"usage":{"total_tokens":2}}`}},
		} {
			t.Run(backend.name, func(t *testing.T) {
				for i, input := range backend.inputs {
					for _, original := range []bool{false, true} {
						name := []string{"text", "image", "mixed", "video"}[i]
						if original {
							name += "-original"
						}
						t.Run(name, func(t *testing.T) {
							cfg := map[string]interface{}{"type": backend.typ, "apiTokens": []string{"native-token"}, "qwenEnableCompatible": backend.compatible, "modelMapping": map[string]string{"alias": backend.mappedModel}}
							requestPath := "/gateway" + backend.alias + "?trace=1"
							upstreamPath := backend.nativePath + "?trace=1"
							if original {
								cfg["protocol"] = "original"
								cfg["basePath"] = "/gateway"
								requestPath = "/gateway" + backend.nativePath + "?trace=1"
							}
							// Chat-shaped unknown fields must stay opaque, including for a
							// mapped model that would enable chat preserve_thinking.
							body := `{"model":"alias","input":` + input + `,"parameters":{"enable_fusion":true,"dimension":1024,"fps":1},"encoding_format":"float","unknown":{"keep":[1,"x"]},"messages":[{"role":"assistant","reasoning_content":"opaque"}]}`
							want := strings.Replace(body, `"model":"alias"`, `"model":"`+backend.mappedModel+`"`, 1)
							if original {
								want = body
							}
							host := newMultimodalHost(t, cfg, requestPath, upstreamPath, backend.domain)
							host.CallOnHttpRequestBody([]byte(body))
							require.Nil(t, host.GetLocalResponse())
							require.JSONEq(t, want, string(host.GetRequestBody()))
							if original {
								require.Equal(t, []byte(body), host.GetRequestBody())
							}
							response := backend.responses[i%len(backend.responses)]
							assertNativeResponse(t, host, "200", response)
						})
					}
				}
				t.Run("opaque-model-and-error", func(t *testing.T) {
					host := newMultimodalHost(t, map[string]interface{}{"type": backend.typ, "apiTokens": []string{"native-token"}, "qwenEnableCompatible": backend.compatible}, backend.alias, backend.nativePath, backend.domain)
					body := `{"model":"ep-opaque","input":` + backend.inputs[0] + `,"extra":{"preserve":true}}`
					host.CallOnHttpRequestBody([]byte(body))
					require.Equal(t, []byte(body), host.GetRequestBody())
					assertNativeResponse(t, host, "400", `{"code":"InvalidParameter","message":"native error","request_id":"error-id","extra":{"preserve":true}}`)
				})
				t.Run("capability-override", func(t *testing.T) {
					api := "qwen/v1/embeddings/multimodal"
					if backend.typ == "doubao" {
						api = "doubao/v3/embeddings/multimodal"
					}
					host := newMultimodalHost(t, map[string]interface{}{"type": backend.typ, "apiTokens": []string{"native-token"}, "qwenEnableCompatible": backend.compatible, "capabilities": map[string]string{api: "/custom/native?configured=1"}}, backend.alias+"?trace=1", "/custom/native?configured=1&trace=1", backend.domain)
					body := `{"model":"opaque","input":` + backend.inputs[0] + `}`
					host.CallOnHttpRequestBody([]byte(body))
					require.Nil(t, host.GetLocalResponse())
					require.Equal(t, []byte(body), host.GetRequestBody())
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
				host := newMultimodalHost(t, map[string]interface{}{"type": tc.typ, "apiTokens": []string{"native-token"}, "qwenEnableCompatible": tc.compatible, "modelMapping": map[string]string{"alias": "mapped"}}, tc.path, tc.upstream, tc.domain)
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

func assertNativeResponse(t *testing.T, host wasmhost.TestHost, status, body string) {
	t.Helper()
	require.NoError(t, host.SetProperty([]string{"response", "code_details"}, []byte("via_upstream")))
	host.CallOnHttpResponseHeaders([][2]string{{":status", status}, {"content-type", "application/json"}, {"x-native-request-id", "native-id"}})
	host.CallOnHttpResponseBody([]byte(body))
	require.Equal(t, []byte(body), host.GetResponseBody())
	for name, want := range map[string]string{":status": status, "content-type": "application/json", "x-native-request-id": "native-id"} {
		got, ok := wasmhost.GetHeaderValue(host.GetResponseHeaders(), name)
		require.True(t, ok, name)
		require.Equal(t, want, got, name)
	}
}

func TestMultimodalNativeResponseOptions(t *testing.T) {
	wasmhost.RunTest(t, func(t *testing.T) {
		for _, option := range []string{"promoteThinkingOnEmpty", "hiclawMode"} {
			for _, compatible := range []bool{false, true} {
				name := option
				if compatible {
					name += "-compatible"
				}
				t.Run(name, func(t *testing.T) {
					host := newMultimodalHost(t, map[string]interface{}{"type": "qwen", "apiTokens": []string{"native-token"}, "qwenEnableCompatible": compatible, option: true}, "/v1/multimodal-embedding", "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding", "dashscope.aliyuncs.com")
					host.CallOnHttpRequestBody([]byte(`{"model":"native","input":{"contents":[{"text":"hello"}]}}`))
					assertNativeResponse(t, host, "200", `{"output":{"embeddings":[{"embedding":[0.1],"index":0}]},"usage":{"input_tokens":3},"request_id":"native","choices":[{"message":{"content":"","reasoning_content":"opaque"}}],"extra":{"keep":true}}`)
				})
			}
		}
	})
}

func TestMultimodalUnconfiguredOpenAI(t *testing.T) {
	wasmhost.RunTest(t, func(t *testing.T) {
		for _, alias := range []string{"/v1/multimodal-embedding", "/v1/embeddings/multimodal"} {
			t.Run(alias, func(t *testing.T) {
				host := newMultimodalHost(t, map[string]interface{}{"type": "openai", "apiTokens": []string{"native-token"}, "modelMapping": map[string]string{"*": "chat-model"}, "responseJsonSchema": map[string]string{"type": "json_object"}}, "/gateway"+alias, "/gateway"+alias, "api.openai.com")
				body := []byte(`{"model":"native","input":{"contents":[{"text":"hello"}]},"unknown":[1,true]}`)
				host.CallOnHttpRequestBody(body)
				require.Nil(t, host.GetLocalResponse())
				require.Equal(t, body, host.GetRequestBody())
			})
		}
	})
}

func TestMultimodalNativeRetries(t *testing.T) {
	wasmhost.RunTest(t, func(t *testing.T) {
		for _, backend := range []struct {
			name, typ, alias, nativePath, input string
			compatible                          bool
		}{
			{"qwen", "qwen", "/v1/multimodal-embedding", "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding", `{"contents":[{"image":"data:image/png;base64,aGVsbG8="}]}`, false},
			{"qwen-compatible", "qwen", "/v1/multimodal-embedding", "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding", `{"contents":[{"image":"data:image/png;base64,aGVsbG8="}]}`, true},
			{"doubao", "doubao", "/v1/embeddings/multimodal", "/api/v3/embeddings/multimodal", `[{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}]`, false},
		} {
			for _, original := range []bool{false, true} {
				name := backend.name
				if original {
					name += "-original"
				}
				t.Run(name, func(t *testing.T) {
					cfg := map[string]interface{}{"type": backend.typ, "apiTokens": []string{"native-token", "other-token"}, "qwenEnableCompatible": backend.compatible, "providerDomain": "final.example.com", "providerBasePath": "/upstream", "modelMapping": map[string]string{"alias": "mapped", "mapped": "wrong-second-mapping"}, "retryOnFailure": map[string]interface{}{"enabled": true, "maxRetries": 1}}
					cfg["customSettings"] = []map[string]interface{}{{"name": "extra.added", "value": "once", "mode": "raw"}}
					requestPath := "/gateway" + backend.alias + "?trace=1"
					upstreamPath := "/upstream" + backend.nativePath + "?trace=1"
					if original {
						cfg["protocol"] = "original"
						cfg["basePath"] = "/gateway"
						requestPath = "/gateway" + backend.nativePath + "?trace=1"
					}
					body := []byte(`{ "model":"alias", "input":` + backend.input + `, "extra":{"keep":[1,true]} }`)
					host := newMultimodalHost(t, cfg, requestPath, upstreamPath, "final.example.com", [2]string{"content-length", strconv.Itoa(len(body))}, [2]string{"x-private-canary", "route-removes-this"})
					require.NoError(t, host.SetClusterName("native-cluster"))
					host.CallOnHttpRequestBody(body)
					require.Nil(t, host.GetLocalResponse())
					firstBody := append([]byte(nil), host.GetRequestBody()...)
					if original {
						require.Equal(t, body, firstBody)
					} else {
						require.Contains(t, string(firstBody), `"model":"mapped"`)
						require.Contains(t, string(firstBody), `"added":"once"`)
					}
					firstHeaders := host.GetRequestHeaders()
					firstAuth, _ := wasmhost.GetHeaderValue(firstHeaders, "authorization")
					require.NoError(t, host.SetProperty([]string{"response", "code_details"}, []byte("via_upstream")))
					require.Equal(t, types.HeaderStopAllIterationAndWatermark, host.CallOnHttpResponseHeaders([][2]string{{":status", "503"}, {"content-type", "application/json"}}))
					calls := host.GetHttpCalloutAttributes()
					require.Len(t, calls, 1)
					require.Equal(t, "native-cluster", calls[0].Upstream)
					require.Equal(t, firstBody, calls[0].Body)
					for _, header := range []string{":method", ":path", ":authority", "content-type"} {
						want, _ := wasmhost.GetHeaderValue(firstHeaders, header)
						got, ok := wasmhost.GetHeaderValue(calls[0].Headers, header)
						require.True(t, ok, header)
						require.Equal(t, want, got, header)
					}
					_, hasLength := wasmhost.GetHeaderValue(calls[0].Headers, "content-length")
					require.False(t, hasLength, "retry must not reuse caller length after model mapping")
					_, hasPrivateHeader := wasmhost.GetHeaderValue(calls[0].Headers, "x-private-canary")
					require.False(t, hasPrivateHeader, "direct-cluster retry must not forward headers a route may remove")
					retryAuth, ok := wasmhost.GetHeaderValue(calls[0].Headers, "authorization")
					require.True(t, ok)
					require.NotEqual(t, firstAuth, retryAuth)
					require.Contains(t, []string{"Bearer native-token", "Bearer other-token"}, retryAuth)
					response := []byte(`{"output":{"embeddings":[{"embedding":[0.1]}]},"data":{"embedding":"AACAPw=="},"extra":{"keep":true}}`)
					host.CallOnHttpCall([][2]string{{":status", "200"}, {"content-type", "application/json"}, {"x-native-request-id", "retry-id"}}, response)
					local := host.GetLocalResponse()
					require.NotNil(t, local)
					require.Equal(t, uint32(200), local.StatusCode)
					require.Equal(t, response, local.Data)
				})
			}
		}
	})
}
