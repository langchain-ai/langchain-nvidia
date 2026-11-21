# NIM compatibility and test coverage

A built-in model ID is not a guarantee that the model is currently available
on NVIDIA's hosted API, downloadable, or compatible with every SDK feature.
Hosted and self-hosted deployments must be evaluated separately.

Use the deployment's actual served model ID and follow the
[adoption guide](adoption_guide.md). Check the features your application uses,
such as streaming, tool calling, structured output, embeddings or reranking.
Successful chat alone does not validate the other features.

Doctor checks configuration and reachability. It does not certify model
accuracy or replace deployment-specific tests. Unknown availability and
capability evidence must not be interpreted as supported or unsupported.

The package's offline [replay tests](bcb_replay_fixtures.md) check connector
parsing against reviewed, sanitized response shapes without live API calls.
They protect against parser regressions, not current endpoint availability.

Reviewed model identities and their source metadata are documented in the
[model registry](model_registry.md). Catalog metadata alone does not establish
runtime compatibility.

## Reviewed observations

The [public observation table](compatibility_state.md) is paired with a
[machine-readable snapshot](data/compatibility_public.json). It contains only
reviewed public output, not private test reports. An empty snapshot means no
observations have been approved for this publication; it is not evidence that
models are unsupported.

Each observation is limited to its exact SDK version, source revision,
deployment and test time. A past passing result is not a guarantee of current
endpoint availability, support for untested capabilities, or release readiness.
The [publication contract](compatibility_publication.md) defines the allowed
fields and review boundary.
