# BCB replay fixtures

Replay fixtures are small, public-safe response captures used by unit tests to
exercise realistic NIM response shapes without calling live endpoints. They are a
fast guardrail for connector parsing behavior; BCB remains the live compatibility
validator and source of compatibility evidence.

## Current fixture state

The checked-in fixtures under `tests/data/bcb_replay_fixtures/` are synthetic
contract examples. They validate the replay harness and the connector parsing
paths for:

- `ChatNVIDIA.bind_tools()` completion and streaming response parsing.
- `ChatNVIDIA.with_structured_output()` response parsing.
- `AIMessage` reasoning-field round trips used by multi-turn messages.
- `NVIDIAEmbeddings.embed_query()` embedding response parsing.
- `NVIDIARerank.compress_documents()` ranking response parsing.
- Negative provider error envelopes under `negative-fixtures/`.

They are not real BCB evidence and should not be used to claim model
compatibility.

## Promotion criteria for real BCB captures

Only add a BCB capture to this fixture corpus after an evidence owner and a
connector maintainer review it. Before commit, remove or replace:

- API keys, bearer tokens, cookies, auth headers, signed URLs, user identifiers,
  request IDs, trace IDs, job IDs, timestamps, host-specific paths, and raw log
  fragments.
- Original prompts, uploaded image/audio/document payloads, private tool names,
  private endpoint URLs, internal model target IDs, and unstable generated text
  that is not needed for the parser contract.
- Large vectors, images, and raw streaming logs. Keep only the smallest response
  fields needed to replay the connector behavior.

Every promoted fixture should include metadata in `manifest.json` describing the
BCB profile name, pass/fail status, fingerprint hash, source artifact folder,
model family or sanitized model name, review status, and why the response shape
is worth preserving. If provenance cannot be reviewed, keep the fixture out of
the public repository and use a private review artifact instead.

## Follow-up automation path

Once the BCB output schema is stable, convert only passing BCB profiles into the
positive fixture corpus by default. Failed profiles should be skipped or exported
separately into `negative-fixtures/` with an explicit negative-contract reason.
The export should fail closed when required review metadata is missing, run the
same public-safety checks as the unit tests, and require a maintainer-reviewed
manifest diff before any capture is committed.

## Expected coverage

The first real corpus should include fixtures spanning at least three of these
categories:

- Tool-call completion responses, including nullable assistant content.
- Streaming tool-call chunks and usage-only final chunks.
- Structured-output responses, including reasoning fields or thinking tags.
- Message round-trip cases for reasoning fields returned by the API.
- Embedding response-shape cases for `NVIDIAEmbeddings`.
- Reranking response-shape cases for `NVIDIARerank`.
- Provider error envelopes that should render useful connector errors, stored as
  negative fixtures unless they came from an explicitly approved error profile.

Run the replay tests with:

```sh
poetry run pytest tests/unit_tests/test_bcb_replay_fixtures.py
```
