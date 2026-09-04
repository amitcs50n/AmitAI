# Aevon V1 UI/runtime integration

Starting revision: `b90abb0a30f730bccf644f9cf5484e9d6eb53e5f`, verified on a clean
`main` checkout before edits. V5.2 text runtime is frozen for this ticket.

## Audit before code changes

Inspected `AmitaiApp`, `ChatView`, `Composer`, `Message`, `AssetPreview`,
`RemoteVisionConsent`, `MemoryView`, Settings, Markdown/tool cards, the same-origin
API proxy, API/SSE/types/privacy helpers, backend routes/schemas/chat service,
memory and secure asset boundaries, runtime selection and public generator
interfaces, and existing frontend/backend/runtime/privacy/vision tests.

"Fully wired" below describes code integration, not real-weight quality or GPU
readiness. The production launcher defaults to deterministic mock responses until
an operator explicitly selects local Transformers or remote inference. Frontend
unit fixtures also mock transport; they are not evidence of live inference.

| Feature | Starting classification | Existing path / actual gap | Disposition |
| --- | --- | --- | --- |
| 1. Create conversation | Already fully wired | New chat starts a draft; first successful send atomically creates the conversation. Explicit POST create also exists. | Preserve |
| 2. Switch conversation | Already fully wired | Sidebar selection loads details; active sends lock switching; request IDs reject stale responses. | Preserve |
| 3. Load history | Already fully wired | GET conversation supplies persisted messages, metadata and assets. | Preserve |
| 4. Send text | Already fully wired | Composer → streaming API → ChatService → selected generator. | Preserve |
| 5. Streaming text | Already fully wired | POST fetch/SSE incrementally renders deltas; final/done ordering and reconstruction are checked. | Preserve |
| 6. Stop/cancel | Already fully wired | AbortController cancels fetch/reader; proxy forwards abort; backend signals generation and avoids partial commits. | Preserve |
| 7. Retry/regenerate | Partially wired | Failed/cancelled pending turns retry without duplicating the user turn; confirmed commits offer reload, not resend. No saved-answer regeneration endpoint exists. | Preserve V1 retry; defer saved-answer regeneration |
| 8. Reload persistence | Already fully wired | Server database stores turns; browser stores only selected ID/preferences and refetches history. | Preserve |
| 9. Memory management | Already fully wired | Create/search/edit/forget APIs, sensitivity selection, revision conflicts and tombstone UI. | Preserve |
| 10. Memory affects chat | Already fully wired | Backend retrieves bounded relevant memory before ProviderChatGenerator; local/remote projections apply. Image turns intentionally do not retrieve/mutate memory. | Preserve |
| 11. Calculator | Already fully wired | Existing bounded runtime tool loop invokes calculator; final metadata records execution. | Preserve |
| 12. Hidden tool protocol | Already fully wired | Runtime filters envelopes before deltas/final persistence; UI renders final Markdown and structured activity cards. | Preserve; verify through real backend with fake provider |
| 13. Single image attach | Broken/inconsistent | Composer permits four staged uploads although ChatService allows one; multiple uploads disable sending. | Enforce one upload in UI |
| 14. Preview/remove | Already fully wired | Authenticated same-origin image URL; explicit delete removes staged assets. | Preserve |
| 15. Image + text send | Partially wired | Existing asset IDs/consent travel through stream request; disabled local vision currently still permits sending a non-analysis placeholder. | Block unavailable vision for all scopes |
| 16. Local vision | Partially wired | Native image sessions already exist; UI fails to honor `enabled=false` for local scope. | Fix UI gate only |
| 17. Remote vision consent | Already fully wired | Per-image checkbox starts unchecked; every failed/cancelled retry requires new consent; backend grant enforced before decryption/transport. | Preserve |
| 18. Streaming vision | Already fully wired | Same stream endpoint dispatches native/remote vision; same SSE lifecycle renders output. | Preserve |
| 19. Vision cancellation | Already fully wired | Existing cancellation signal and grant/image cleanup; UI retains pending image for retry. | Preserve |
| 20. Image persistence/privacy | Already fully wired | Canonicalized encrypted local assets; successful commit promotes temporary assets. Removed staged assets are deleted; abandoned temporary assets expire after 24 hours. No browser image storage or optimizer. | Preserve |
| 21. Error states | Broken/inconsistent | Chat collapses privacy/consent failures into generic generation failure. Pending image retry cannot refresh unavailable capabilities. | Safe specific messages and capability refresh |
| 22. Loading/generation states | Already fully wired | Loading history, assistant waiting/deltas, Stop, finishing-after-commit, neutral stopped status and retry. | Preserve |
| 23. Backend/API failures | Partially wired | Network, malformed/incomplete SSE, saved-but-reload-failed paths work; proxy configuration/outage is misleadingly shown as generation failure. | Explain known connection/configuration failures safely |
| 24. Local/remote selection/status | Partially wired | Launcher/env explicitly selects mode; browser only shows Connected and vision scope. Mock responses are not identified in Settings. | Add read-only configured inference status; retain server-side selection |

## Narrow implementation plan

Use existing components, `/api/capabilities`, API client and SSE protocol. Extend
capabilities with only a safe inference-mode enum derived from the injected
generator (`None` is mock; declared execution/vision scope identifies local or
remote; otherwise unknown). This reports configuration, not loaded-model health,
and exposes no endpoint, credentials or internal settings. Settings will show
mode and image availability with existing styling; unsupported tool placeholders
will be replaced by the actual calculator capability description.

Fix one-image staging and enabled/scope checks in Composer/ChatView, including
retry. Map known backend privacy/consent/proxy errors to fixed product wording;
unknown errors remain generic and never echo arbitrary exception text. Add CPU
regressions and exercise actual UI/proxy/backend orchestration with fake inference.

No runtime model, initial/repair prompt, generation config, epistemic guard,
context compiler/layout, memory architecture, calculator semantics, tool registry
or vision model implementation changes are planned. No new integrations or UI
redesign. Existing failed/cancelled retry satisfies V1; saved-answer regeneration
would need a separate persistence contract and is deferred.

## Validation and remaining real-weight gate

Final disposition:

| Features | Result |
| --- | --- |
| 1–6, 8–12, 14, 17–20, 22 | Existing integration preserved and covered by frontend/backend tests. |
| 13 | Fixed: only one image can be staged; both visible controls and the selection handler enforce the limit. Removing/sending restores attachment availability. |
| 15–16 | Fixed: image send and retry require enabled local/remote vision. Unknown, disabled and missing capabilities block sending; text remains available. |
| 21, 23 | Fixed: safe privacy, consent, proxy-configuration, backend-outage and memory-conflict messages; known proxy failures report Disconnected. Pending-image retries can refresh capabilities. Unknown generation errors never echo internal details. |
| 24 | Fixed: authenticated no-store capabilities report only a configured inference enum; Settings shows mock/local/remote/unknown and image availability. Older backends fall back to unknown inference mode. Selection remains in the existing server launcher/configuration. Unsupported tool placeholders are replaced with the calculator description. |
| 7 | Existing failed/cancelled-message retry preserved. Regenerating a successfully saved answer remains deferred; no backend contract defines replacement/deletion of that saved turn. |

### Evidence and reproducibility

The new `frontend/components/RuntimeIntegration.test.tsx` mounts the actual
`AmitaiApp` in JSDOM. Browser requests use the existing API client and imported
Next route handlers, then real loopback HTTP to an authenticated FastAPI backend.
`tests/ui_runtime_fixture.py` injects a fake CPU engine into the unchanged
`TransformersChatGenerator`/`ProviderChatGenerator` orchestration. It uses a
temporary SQLite database and the real encrypted asset store; only database
encryption is disabled in this disposable test fixture. No production launcher
or authentication bypass is added.

The test verifies incremental text and vision before final completion, calculator
execution and result consumption, absence of raw tool protocol in the UI, memory
created in the UI and consumed in the compiled system context after remount,
authenticated image preview/removal, persisted image history, real producer
cancellation with no partial persistence, retry without duplicate turns, sanitized
vision failure, a second conversation and switching back to saved history.
Remote consent and renewed retry consent are exercised in the existing actual-app
component tests; remote transport/grants/native vision are exercised by the real
backend/runtime tests with fake engines and HTTP transports. This does not claim
real GPU, real remote connectivity, visual browser QA, or model semantic quality.

Run the normal frontend commands in `frontend/`:

```text
npm test
npm run test:integration
npm run lint
npm run typecheck
npm run build
```

The integration test uses `python` on PATH, or the explicit `AMITAI_TEST_PYTHON`
interpreter override, with the repository's Python dependencies available. It
spawns a hidden loopback fixture, forces offline model-hub settings, and cleans
up the child and its temporary files. No dependency installation is needed here.
For this Windows checkout, the existing ignored `.cache/v5_cpu_runner.py` and
`.cache/v5_cpu_support` bridge bundled Python 3.12 to the installed repository
dependencies because the checkout's original Python 3.11 launcher is unavailable.

| Validation | Result |
| --- | --- |
| Targeted Python: backend streaming, backend vision, remote vision, memory, backend API, runtime | 257 passed; one existing Starlette/httpx deprecation warning |
| Full Python: `pytest -q -rs --tb=short` | 1,924 passed, 1 skipped, 1 warning in 67.99 seconds |
| Frontend unit/component/API/proxy/SSE/privacy: `npm test` | 100 passed |
| UI/proxy/live-backend CPU integration: `npm run test:integration` | 1 passed |
| Frontend ESLint | Passed |
| Frontend TypeScript | Passed |
| Frontend production build | Passed; `/`, `/_not-found`, and dynamic `/api/[...path]` built |
| Diff whitespace check | Passed |

The sole Python skip is the existing POSIX mode check on Windows
(`tests/test_asset_encryption.py:327`); Windows ACL checks still execute. The
warning is the existing Starlette TestClient/httpx deprecation.

### Remaining gate before/at the final real A100 session

No V1 integration blocker remains in the tested CPU paths. Before that session,
review this one scoped commit and provision the authorized A100 runtime with the
existing frozen model/configuration and secure local/remote connection. Availability,
credentials, actual weight loading, device support and real deployment connectivity
were deliberately not probed in this ticket.

On that session, run the frozen twelve-case
`eval/aevon_epistemic_killer_v5_2.jsonl` and perform human semantic review of every
real output, including any mechanical repair. Then smoke-test the actual UI with
real text, calculator, memory recall, one-image analysis, consent, streaming,
Stop/retry and reload in the deployed inference mode. Real image accuracy,
latency, resource use, cancellation timing and semantic quality remain unmeasured.
These are validation gates, not evidence that another runtime change is needed.
No GPU, A100, RunPod, model download, real inference or training occurred here.
