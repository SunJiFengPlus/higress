<!--
Copyright 2026 alibaba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Support native multimodal embeddings in ai-proxy

Add native multimodal embedding capabilities for Alibaba Cloud DashScope and
Volcengine Ark, including provider defaults, gateway aliases, and original-protocol
paths. Preserve native input,
vectors, usage, and errors while retaining existing text embedding behavior.

Add request-lifecycle regression tests and configuration examples in Chinese
and English.

Fixes [#3729](https://github.com/higress-group/higress/issues/3729).
