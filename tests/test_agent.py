import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skuagent import Agent, accuracy, diagnose, run_pipeline, simulate  # noqa: E402
from skuagent.jira import build_tickets  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return simulate(n_skus=900, seed=5)


def test_contribution_reconciles(data):
    skus, orders = data
    f = run_pipeline(skus, orders)
    expected = (orders.gross - orders.cogs - orders.freight_total - orders.refund - orders.marketing).sum()
    assert abs(f.contribution.sum() - expected) < 1.0   # separating return freight moves cost, never loses it


def test_return_freight_fix_removes_false_freight_flags(data):
    skus, orders = data
    naive = accuracy(diagnose(run_pipeline(skus, orders, separate_return_freight=False)), skus)
    fixed = accuracy(diagnose(run_pipeline(skus, orders)), skus)
    assert fixed["false_freight_flags"] < naive["false_freight_flags"]


def test_remorse_not_sent_to_listing_fix(data):
    skus, orders = data
    f = run_pipeline(skus, orders)
    assert accuracy(diagnose(f, remorse_routing=False), skus)["remorse_sent_to_listing_fix"] > 0.5
    assert accuracy(diagnose(f), skus)["remorse_sent_to_listing_fix"] < 0.05


def test_low_confidence_never_gets_individual_action(data):
    skus, orders = data
    d = diagnose(run_pipeline(skus, orders))
    low = d[(d.confidence == "Low") & d.action_id.notna()]
    assert (low.action_id == "SPA-900").all()


def test_tickets_route_to_owner_project(data):
    skus, orders = data
    d = diagnose(run_pipeline(skus, orders))
    for t in build_tickets(d[d.confidence != "Low"], "test"):
        assert t["summary"].startswith(f"[{t['action_id']}]")
        assert t["project"] in {"PRC", "OPS", "VND", "CNT", "MKT", "CS", "ANA"}


def test_agent_refuses_held_and_pattern_actions(data, tmp_path):
    skus, orders = data
    a = Agent(skus, orders, "test", tmp_path)
    a.run_pipeline(); a.validate_sample()
    assert "refused" in a.create_jira_ticket("SPA-900")
    a.gate["SPA-501"] = "HOLD"
    assert "refused" in a.create_jira_ticket("SPA-501")


# ---------------------------------------------------------------- hallucination guardrails (docs/LLM_DESIGN.md)
@pytest.fixture(scope="module")
def ran(data, tmp_path_factory):
    skus, orders = data
    a = Agent(skus, orders, "2026-W42", tmp_path_factory.mktemp("g"))
    a.run_offline()
    return a


def test_template_insights_pass_grounding(ran):
    assert ran.check_insights(ran.template_insights()).ok
    assert ran.insights_source == "template"


@pytest.mark.parametrize("fabrication, expected", [
    ("Recovering this would add $4.5M to EBITDA.", "$4.5M"),               # invented figure
    ("Margin improved to 14.8% this week.", "14.8%"),                      # invented rate
    ("Logistics should action SPA-777 first.", "SPA-777"),                 # invented action ID
    ("See ticket PRC-999 for details.", "PRC-999"),                        # Jira key never created
    ("Worst offender is S99999.", "S99999"),                               # SKU that does not exist
])
def test_fabricated_insights_are_rejected(ran, fabrication, expected):
    rep = ran.check_insights(ran.template_insights() + "\n" + fabrication)
    assert not rep.ok and expected in rep.feedback()


def test_write_report_rejects_then_falls_back(data, tmp_path):
    skus, orders = data
    a = Agent(skus, orders, "2026-W42", tmp_path)
    a.run_offline()
    bad = "Margin gap is $9.9M across SPA-777."
    for i in range(1, 3):
        r = a.write_report(bad)
        assert r["rejected"] and r["attempt"] == i and "$9.9M" in r["fix_these"]
    r = a.write_report(bad)                                                  # third strike: model text withheld
    assert r["insights_source"] == "template_fallback"
    text = (tmp_path / "insights_2026-W42.md").read_text()
    assert "$9.9M" not in text and "withheld" in text
