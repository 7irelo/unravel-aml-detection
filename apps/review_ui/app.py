"""Thin analyst review UI. All logic lives behind the API."""

from __future__ import annotations

import json
import os

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("UNRAVEL_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Unravel", layout="wide")
st.title("Unravel: analyst review")
st.caption(
    "Drafts are machine-written and need a human decision. Approving records your decision; "
    "nothing is filed from this system. Synthetic data."
)


def get(path: str, **params: object) -> object:
    response = httpx.get(f"{API}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


status = st.sidebar.selectbox(
    "Status", ["pending_review", "incomplete", "approved", "rejected", "all"]
)
cases = get("/cases", **({} if status == "all" else {"status": status}))
assert isinstance(cases, list)
st.sidebar.metric("Cases", len(cases))
if not cases:
    st.info("No cases with this status.")
    st.stop()

st.dataframe(
    pd.DataFrame(cases)[["alert_id", "status", "band", "probability", "recommendation", "pattern"]],
    hide_index=True,
    use_container_width=True,
)
alert_id = st.selectbox("Open case", [c["alert_id"] for c in cases])
case = get(f"/cases/{alert_id}")
assert isinstance(case, dict)
draft = case.get("draft")

left, right = st.columns([3, 2])
with left:
    if case["status"] == "incomplete":
        st.error(f"Incomplete: {case.get('incomplete_reason')}")
    if case.get("findings"):
        st.warning("The verifier found problems with this draft.")
        st.dataframe(pd.DataFrame(case["findings"]), hide_index=True)
    if draft:
        st.subheader(f"{draft['recommendation'].replace('_', ' ').title()}: {draft['pattern']}")
        for section in ("who", "what", "when", "where", "why_suspicious"):
            st.markdown(f"**{section.replace('_', ' ').title()}**  \n{draft['narrative'][section]}")
        st.subheader("Claims")
        for claim in draft["claims"]:
            st.markdown(f"- `{claim['id']}` ({claim['kind']}) {claim['text']}")
            if claim["tx_ids"]:
                st.caption(", ".join(claim["tx_ids"][:20]))
        if draft["contrary_evidence"]:
            st.subheader("Contrary evidence")
            for claim in draft["contrary_evidence"]:
                st.markdown(f"- {claim['text']}")
        if draft["open_questions"]:
            st.subheader("Open questions")
            for question in draft["open_questions"]:
                st.markdown(f"- {question}")

with right:
    st.metric("Calibrated probability", f"{case['probability']:.4f}", case["band"])
    st.caption(f"Model {case['model_version']}. Drafted by {case['model']}.")
    if case.get("reasons"):
        st.subheader("Why the alert fired")
        st.caption("SHAP explains the model's score, not the cause of the behaviour.")
        for reason in case["reasons"]:
            st.markdown(f"- {reason['description']} (+{reason['contribution']:.2f})")
    if case["cited_transactions"]:
        st.subheader("Cited transactions")
        st.dataframe(pd.DataFrame(case["cited_transactions"]), hide_index=True)

st.divider()
if case["status"] in ("pending_review", "incomplete"):
    analyst = st.text_input("Your name")
    reason = st.text_area("Reason (required to reject)")
    edited = st.text_area(
        "Edited draft JSON (optional)", json.dumps(draft, indent=1) if draft else ""
    )
    approve, reject = st.columns(2)
    action = None
    if approve.button("Approve draft", type="primary", disabled=not draft):
        action = "approve"
    if reject.button("Reject"):
        action = "reject"
    if action:
        body = {"analyst": analyst or "unknown", "action": action, "reason": reason or None}
        if draft and edited.strip() and json.loads(edited) != draft:
            body["edited_draft"] = json.loads(edited)
        result = httpx.post(f"{API}/cases/{alert_id}/decision", json=body, timeout=60)
        if result.is_success:
            st.success(f"Recorded: {result.json()['status']}")
            st.rerun()
        else:
            st.error(result.text)
else:
    st.info(f"Decided: {case['status']}")
    st.dataframe(pd.DataFrame(case["review_events"]), hide_index=True)
