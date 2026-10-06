"""
Dashboard: `streamlit run app/dashboard.py`

Read-only. Four pages over the files the commands write to output/ (no betting logic lives here, and
nothing on these pages can place or change a bet). Needs the `streamlit` package, which is NOT in
requirements.txt until the owner approves it: `pip install streamlit`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # so `streamlit run app/dashboard.py` finds cfbmodel

import pandas as pd
import streamlit as st

from app import data

st.set_page_config(page_title="CFB Edge", layout="wide")
st.sidebar.title("CFB Edge")
page = st.sidebar.radio("Page", ["Edge Board", "Validation", "CLV & Bankroll", "Data Health"])
st.sidebar.caption("Research tool, not betting advice. Most weeks most rows should be PASS.")


def nothing_yet(cmd: str):
    st.info(f"Nothing here yet. Run `{cmd}` first.")


if page == "Edge Board":
    st.header("Edge Board")
    boards = data.list_boards()
    if not boards:
        nothing_yet("python -m cfbmodel board --season 2026 --week N")
    else:
        label = st.selectbox("Board", [f"{s} week {w}" for s, w, _ in boards])
        season, week = [(s, w) for s, w, _ in boards if f"{s} week {w}" == label][0]
        df, meta = data.load_board(season, week)
        for banner in meta.get("banners", []):
            st.warning(banner)
        counts = df["status"].value_counts()
        cols = st.columns(5)
        for col, name in zip(cols, ["BET", "INFO", "PASS", "FCS", "NO_LINE"]):
            col.metric(name, int(counts.get(name, 0)))
        keep = st.multiselect("Show", ["BET", "INFO", "PASS", "FCS", "NO_LINE"], default=["BET", "INFO"])
        shown = df[df["status"].isin(keep)]
        if shown.empty:
            st.success("Nothing to show for those statuses. That is a normal result: an honest no-bet is a correct answer.")
        else:
            st.dataframe(shown, use_container_width=True, hide_index=True)
        st.caption("BET = clears the bar AND the model has been proven out-of-sample for that market. "
                   "INFO = the model sees something nobody has proven it may be trusted on.")

elif page == "Validation":
    st.header("Validation: may the model influence a bet?")
    v = data.validation()
    if not v["status"]:
        nothing_yet("python -m cfbmodel validate --seasons 2022 2023 2024 2025 2026")
    else:
        stt = v["status"]
        if stt["status"] == "SUSPECTED_LEAK":
            st.error("SUSPECTED LEAK: a model coefficient is above 0.5. Do not trust anything until this is explained.")
        if v["age_days"] is not None and v["age_days"] > 14:
            st.warning(f"This validation is {v['age_days']:.0f} days old: re-run `validate`.")
        passed = [k for k, m in stt["markets"].items() if m["passed"]]
        (st.success if passed else st.info)(
            f"Markets that passed: {', '.join(passed)}" if passed else
            "Nothing passed. The model has not been shown to beat the betting line, so the app prices off the market. "
            "That is a valid, expected result.")
        st.subheader("Markets")
        st.dataframe(data.markets_table(stt), use_container_width=True, hide_index=True)
        st.subheader("Segments that passed")
        sp = data.segments_table(stt, only_passed=True)
        st.dataframe(sp, use_container_width=True, hide_index=True) if len(sp) else st.write("None.")
        st.subheader("Looked good on the train seasons but failed the holdout (the guard working)")
        near = data.segments_table(stt, near_misses=True)
        st.dataframe(near, use_container_width=True, hide_index=True) if len(near) else st.write("None.")
        with st.expander("Full report"):
            st.markdown(v["report"] or "")

elif page == "CLV & Bankroll":
    st.header("CLV & Bankroll")
    settled = data.settled_bets()
    text = data.clv_report_text()
    if settled is None:
        nothing_yet("python -m cfbmodel log-bet ...` and then `python -m cfbmodel clv")
    else:
        st.markdown(text or "")
        curve = data.bankroll_curve(settled)
        if len(curve):
            st.subheader("Cumulative profit (units, in the order bets were logged)")
            st.line_chart(curve.set_index("bet_id")["cumulative_profit"])
        st.subheader("Every bet")
        st.dataframe(settled, use_container_width=True, hide_index=True)

else:
    st.header("Data Health")
    h = data.data_health()
    st.subheader("CFBD call budget this month")
    st.progress(min(h["calls_used"] / h["limit"], 1.0), text=f"{h['calls_used']} of {h['limit']} used; {h['calls_remaining']} left")
    if h["calls_used"] >= h["refuse_at"]:
        st.error(f"New calls are refused from {h['refuse_at']} unless you pass --force.")
    elif h["calls_used"] >= h["warn_at"]:
        st.warning(f"Past the {h['warn_at']}-call warning line.")
    st.subheader("Cache freshness")
    st.dataframe(h["freshness"], use_container_width=True, hide_index=True) if len(h["freshness"]) else st.write("Cache is empty.")
    st.subheader("Names that are not FBS teams")
    if len(h["unmatched"]):
        st.caption("Mostly FCS opponents. If an FBS team is here under another spelling, add it to data/team_aliases.csv.")
        st.dataframe(h["unmatched"], use_container_width=True, hide_index=True)
    else:
        st.write("None logged.")
    st.subheader("Model parameters")
    st.write(f"Active version: **{h['params_version']}**")
    if len(h["models"]):
        st.dataframe(h["models"], use_container_width=True, hide_index=True)
    st.write(f"Last learning run: {h['last_learn'] or 'never'}")
    drift = h["drift"]
    (st.error if drift and drift.get("active") else st.write)(
        f"Drift alarm ACTIVE: {drift['reason']}. Model weight is forced to 0." if drift and drift.get("active")
        else "No drift alarm.")
