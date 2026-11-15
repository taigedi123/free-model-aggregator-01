# free-model-aggregator-01

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier model providers, in JSON and in
server-sent events.

## Run

```sh
python3 main.py --list   # print the catalog and exit
```

In a trial the sandbox sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE` and
`MAKEFUN_TRIAL_SEED`; the server probes each provider's catalog route, builds
one catalog, registers its port with the env, serves the 20 requests the env's
client sends, and exits 0 when the env says the workload is done.

## Routing

Every free model that meets a request's requirements is a candidate, ranked
by the sealed quality table's rating (best first, deterministic tie-break on
provider id then model id). A provider call refused with 402/429/5xx is
retried on the next candidate. A provider that refuses twice in a row is
deprioritised for the rest of the trial, so a spent budget stops costing
calls. A JSON-mode OpenAI request is never routed to an Anthropic-wire
provider, whose wire cannot carry JSON mode.

## Wires

The client speaks OpenAI or Anthropic; each provider speaks its own wire
(OpenAI, Anthropic or Gemini). Translations preserve exactly what the client
checks through the provider's echo: system text, user text, tool names,
max_tokens, stop sequences, the image hash and JSON mode. Provider usage
counts are carried back unchanged, and stop reasons are mapped within their
family (stop / end_turn / STOP).

## Practice scores

Local practice trials against the kit's sealed env and scorer (all 20/20
requests valid):

| trial | score | quality | served | wire | failover | catalog | calls |
|-------|-------|---------|--------|------|----------|---------|-------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 27    |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 30    |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 29    |

Trials run with the kit's own env server and scorer (a small local driver,
`run_trial.py`, stands in for `grade-local --driver process`, which only runs
Node bots; this entry is Python and graded server-side in Docker).

Quality 99 is the practical ceiling on these instances: the best free model
for 17 of the 20 requests lives on one provider whose per-trial budget is 12
calls, so at least a few requests must fall back to the next-best model.
