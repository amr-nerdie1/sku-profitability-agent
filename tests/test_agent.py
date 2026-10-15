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
