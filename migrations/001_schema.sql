-- Unravel schema. Runs once, as the database owner, when the Postgres volume is created.
--
-- Two boundaries are enforced here rather than in application code:
--   * unravel_agent can read transactions and nothing else. It has no grant on ground_truth.
--   * no table or role represents filing. A case can only end approved or rejected by an analyst.

CREATE TABLE transactions (
    tx_id               text PRIMARY KEY,
    minute              bigint NOT NULL,
    src                 text NOT NULL,
    dst                 text NOT NULL,
    amount_paid         double precision NOT NULL,
    payment_currency    text NOT NULL,
    amount_received     double precision NOT NULL,
    receiving_currency  text NOT NULL,
    payment_format      text NOT NULL,
    usd                 double precision NOT NULL
);
CREATE INDEX transactions_src_minute ON transactions (src, minute);
CREATE INDEX transactions_dst_minute ON transactions (dst, minute);

-- Ground truth is only for offline evaluation. The live pipeline never needs it.
CREATE SCHEMA ground_truth;
CREATE TABLE ground_truth.labels (
    tx_id            text PRIMARY KEY,
    is_laundering    boolean NOT NULL,
    pattern          text,
    pattern_instance integer
);

CREATE TABLE scores (
    tx_id         text PRIMARY KEY,
    minute        bigint NOT NULL,
    raw_score     double precision NOT NULL,
    probability   double precision NOT NULL,
    model_version text NOT NULL,
    scored_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE alerts (
    alert_id      text PRIMARY KEY,
    tx_id         text NOT NULL,
    minute        bigint NOT NULL,
    raw_score     double precision NOT NULL,
    probability   double precision NOT NULL,
    band          text NOT NULL,
    model_version text NOT NULL,
    reasons       jsonb NOT NULL DEFAULT '[]',
    state         text NOT NULL DEFAULT 'new' CHECK (state IN ('new', 'investigating', 'drafted')),
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX alerts_queue ON alerts (state, raw_score DESC);

CREATE TABLE cases (
    alert_id          text PRIMARY KEY REFERENCES alerts (alert_id),
    status            text NOT NULL
        CHECK (status IN ('pending_review', 'incomplete', 'approved', 'rejected')),
    draft             jsonb,
    findings          jsonb NOT NULL DEFAULT '[]',
    incomplete_reason text,
    model             text NOT NULL,
    usage             jsonb NOT NULL DEFAULT '{}',
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);

-- An append-only audit trail of analyst actions, including edits as before-and-after pairs.
CREATE TABLE review_events (
    id           bigserial PRIMARY KEY,
    alert_id     text NOT NULL REFERENCES cases (alert_id),
    analyst      text NOT NULL,
    action       text NOT NULL CHECK (action IN ('approve', 'reject')),
    reason       text,
    draft_before jsonb,
    draft_after  jsonb,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE ROLE unravel_agent LOGIN PASSWORD 'agent_read_only_local';
GRANT USAGE ON SCHEMA public TO unravel_agent;
GRANT SELECT ON transactions TO unravel_agent;
REVOKE ALL ON SCHEMA ground_truth FROM unravel_agent;
REVOKE ALL ON ALL TABLES IN SCHEMA ground_truth FROM unravel_agent;
