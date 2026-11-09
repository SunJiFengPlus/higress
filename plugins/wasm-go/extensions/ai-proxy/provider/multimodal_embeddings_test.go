package provider

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMultimodalKeepsOriginalTextPaths(t *testing.T) {
	qwen := &qwenProvider{}
	doubao := &doubaoProvider{}
	for _, tc := range []struct {
		provider   ApiNameHandler
		path, want string
	}{
		{qwen, "/api/v1/services/embeddings/text-embedding/text-embedding", "openai/v1/embeddings"},
		{doubao, "/api/v3/embeddings", "openai/v1/embeddings"},
	} {
		t.Run(tc.path, func(t *testing.T) { require.Equal(t, ApiName(tc.want), tc.provider.GetApiName(tc.path)) })
	}
}
