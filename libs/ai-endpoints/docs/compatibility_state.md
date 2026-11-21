# Reviewed SDK compatibility observations

These are dated test observations, not a live service status or a general
support certification. Hosted and self-hosted deployments are distinct.
Unknown means untested or insufficient evidence, not unsupported.

The table is derived from the paired
[public snapshot](data/compatibility_public.json). Normal PR tests check that
the table and snapshot agree. Private evidence collection, review and table
generation stay outside this repository.

<!-- BEGIN REVIEWED COMPATIBILITY OBSERVATIONS -->
| Model | Deployment | SDK version | SDK revision | Observed at (UTC) | Chat | Streaming | Tools | Structured output | Reasoning | VLM | Embedding | Rerank | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

No approved public observations have been published. No support claim is made.
<!-- END REVIEWED COMPATIBILITY OBSERVATIONS -->

## Reading the table

- `passed` and `failed` describe the named SDK capability at the recorded time.
- `unknown` does not establish either support or a defect.
- The SDK revision identifies the tested code, which may differ from the latest
  package release or repository main.
- A native endpoint probe, catalog entry or offline replay is not SDK execution.
- A passing capability does not accept a native baseline or approve a release.
- Observation age matters: revalidate the intended deployment before relying
  on historical results. This static page does not refresh status automatically.

See the [publication contract](compatibility_publication.md) for limitation
codes and the public-data boundary.
