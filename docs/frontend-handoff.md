# Front-end handoff

The UI and controller can now be developed independently against API v1.

## Ownership

- Kimi owns `src/researchlab/static/index.html`, `app.js`, and `styles.css`.
- Codex owns Python controller code, persistence, execution, tests, API docs,
  OpenAPI, and fixtures.
- Changes that cross this boundary should first update the API contract and its
  tests; the UI should not depend on controller internals or SQLite layout.

## Integration rules

1. Use only `/api/v1/*` for new UI code.
2. Read `/api/v1/capabilities` before presenting optional actions.
3. Render loading, empty, partial, error, and success states. Show the stable
   error `message`; put `details` behind a disclosure rather than dumping raw data.
4. Poll run events with the `after` cursor. Keep the latest cursor per run and
   tolerate an empty event page.
5. Treat paths as opaque strings so Windows and Linux paths both render correctly.
6. Never request, display, or persist private-key contents or passwords. The server
   form accepts an SSH config alias only.
7. Do not infer whether a device is free. Use its `busy` value and show process/user
   details when supplied.
8. Keep dependencies local to the repository; the UI must work without a CDN.

## Development inputs

- Contract: [`openapi-v1.json`](openapi-v1.json)
- Response examples: [`fixtures`](fixtures)
- Live API docs while the app runs: `http://127.0.0.1:8765/api/v1/openapi.json`

The fixtures contain fictional Linux paths, device names, and IDs. They may be
used directly in mock mode and are safe to publish.

## Suggested UI acceptance path

The first integrated slice should let a researcher register a server alias,
inspect its hardware/environments, import a local project, create an immutable
version, start a run, follow its log/metrics, and open emitted artifacts. Use the
synthetic example for local acceptance; remote NVIDIA and Ascend acceptance stays
pending until suitable servers are available.
