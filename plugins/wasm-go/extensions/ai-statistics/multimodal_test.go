package main

import (
	"strconv"
	"testing"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	"github.com/higress-group/wasm-go/pkg/test"
	"github.com/higress-group/wasm-go/pkg/wrapper"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestMultimodalUsage(t *testing.T) {
	test.RunTest(t, func(t *testing.T) {
		for _, tc := range []struct {
			name, path, body     string
			input, output, total int64
			details              string
		}{
			{"qwen3-vl-embedding-usage", PathQwenMultimodalEmbeddings, `{"output":{"embeddings":[]},"usage":{"image_tokens":4,"input_tokens":43,"total_tokens":47}}`, 47, 0, 47, `{"text_tokens":43,"image_tokens":4}`},
			{"multimodal-embedding-v1-usage", PathQwenMultimodalEmbeddings, `{"output":{"embeddings":[]},"usage":{"duration":0,"image_count":1,"image_tokens":128,"input_tokens":4}}`, 132, 0, 132, `{"text_tokens":4,"image_tokens":128}`},
			{"qwen-image-only", PathQwenMultimodalEmbeddings, `{"output":{"embeddings":[]},"usage":{"input_tokens":0,"image_tokens":7}}`, 7, 0, 7, `{"text_tokens":0,"image_tokens":7}`},
			{"qwen-details", PathQwenMultimodalEmbeddings, `{"output":{"embeddings":[]},"usage":{"input_tokens":10,"input_tokens_details":{"text_tokens":3,"image_tokens":7},"output_tokens":2,"total_tokens":12}}`, 10, 2, 12, `{"text_tokens":3,"image_tokens":7}`},
			{"doubao", PathDoubaoMultimodalEmbeddings, `{"model":"embedding-model","data":{"embedding":[0.1]},"usage":{"prompt_tokens":10,"prompt_tokens_details":{"text_tokens":3,"image_tokens":7},"total_tokens":10}}`, 10, 0, 10, `{"text_tokens":3,"image_tokens":7}`},
			{"qwen-ocr", "/api/v1/services/aigc/multimodal-generation/generation", `{"output":{"choices":[]},"usage":{"input_tokens":3555,"image_tokens":3470,"output_tokens":1981,"total_tokens":5536}}`, 3555, 1981, 5536, ""},
			{"qwen-gateway-alias", "/v1/multimodal-embedding", `{"output":{"embeddings":[]},"usage":{"input_tokens":3,"image_tokens":7,"total_tokens":10}}`, 3, 0, 10, ""},
			{"explicit-zero", PathQwenMultimodalEmbeddings, `{"output":{"embeddings":[]},"usage":{"input_tokens":7,"image_tokens":4,"total_tokens":0}}`, 11, 0, 0, ""},
			{"invalid-usage", PathQwenMultimodalEmbeddings, `{"output":{"embeddings":[]},"usage":null}`, 0, 0, 0, ""},
			{"text-embedding", PathOpenAIEmbeddings, `{"data":[{"embedding":[0.1]}],"usage":{"prompt_tokens":11,"total_tokens":11}}`, 11, 0, 11, ""},
		} {
			t.Run(tc.name, func(t *testing.T) {
				host, status := test.NewTestHost([]byte(`{"use_default_response_attributes":true,"enable_path_suffixes":["*"]}`))
				defer host.Reset()
				require.Equal(t, types.OnPluginStartStatusOK, status)
				require.NoError(t, host.SetRouteName("api-v1"))
				require.NoError(t, host.SetClusterName("cluster-1"))
				host.CallOnHttpRequestHeaders([][2]string{{":authority", "example.com"}, {":method", "POST"}, {":path", "/gateway" + tc.path + "?trace=1"}, {"content-type", "application/json"}, {"x-mse-consumer", "user1"}})
				host.CallOnHttpRequestBody([]byte(`{"model":"embedding-model","input":{"contents":[{"text":"hello"}]}}`))
				host.CallOnHttpResponseHeaders([][2]string{{":status", "200"}, {"content-type", "application/json"}})
				host.CallOnHttpResponseBody([]byte(tc.body))
				require.Equal(t, []byte(tc.body), host.GetResponseBody())
				raw, err := host.GetProperty([]string{wrapper.AILogKey})
				require.NoError(t, err)
				attrs := gjson.Parse(wrapper.UnmarshalStr(`"` + string(raw) + `"`))
				for key, want := range map[string]int64{"input_token": tc.input, "output_token": tc.output, "total_token": tc.total} {
					require.True(t, attrs.Get(key).Exists(), key)
					require.Equal(t, want, attrs.Get(key).Int(), key)
				}
				require.Equal(t, "embedding-model", attrs.Get("model").String())
				if tc.total > 0 {
					for key, want := range map[string]int64{ArmsInputToken: tc.input, ArmsOutputToken: tc.output, ArmsTotalToken: tc.total} {
						value, ok := getSpanValue(host, key)
						require.True(t, ok)
						require.Equal(t, strconv.FormatInt(want, 10), value)
					}
				}
				for metric, want := range map[string]int64{"input_token": tc.input, "total_token": tc.total} {
					got, err := host.GetCounterMetric(streamingMetricName("embedding-model", metric))
					if want == 0 {
						require.Error(t, err, "zero token count must not create metric %s", metric)
					} else {
						require.NoError(t, err)
						require.Equal(t, uint64(want), got)
					}
				}
				if tc.details == "" {
					require.Empty(t, attrs.Get("input_token_details").String())
				} else {
					require.JSONEq(t, tc.details, attrs.Get("input_token_details").String())
				}
			})
		}
	})
}

func TestMultimodalExplicitPaths(t *testing.T) {
	test.RunGoTest(t, func(t *testing.T) {
		host, status := test.NewTestHost([]byte(`{"use_default_response_attributes":true,"enable_path_suffixes":["/messages"]}`))
		defer host.Reset()
		require.Equal(t, types.OnPluginStartStatusOK, status)
		config, err := host.GetMatchConfig()
		require.NoError(t, err)
		require.Equal(t, []string{"/messages"}, config.(*AIStatisticsConfig).enablePathSuffixes)
	})
}
