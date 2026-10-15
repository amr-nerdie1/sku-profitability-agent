"""Run one weekly cycle of the SKU profitability agent.

    python run_agent.py            # offline plan (no API key needed), Jira dry-run
    ANTHROPIC_API_KEY=... python run_agent.py --live          # Claude drives the tool loop
    JIRA_URL=... JIRA_EMAIL=... JIRA_API_TOKEN=... python run_agent.py --live   # real tickets
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from skuagent import Agent, accuracy, diagnose, run_pipeline, simulate  # noqa: E402

NAVY, TEAL, GREY, RED, AMBER = "#0B1F3A", "#0E8C8C", "#9AA5B1", "#B42318", "#D97706"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13})


def figures(agent: Agent, skus, orders):
    fig_dir = ROOT / "figures"; fig_dir.mkdir(exist_ok=True)
    d = agent.diag

    # 1: where the money goes (margin gap by driver)
    g = d[d.primary_driver != "HEALTHY"].groupby("primary_driver").margin_gap.sum().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(g.index, g.values / 1000, color=TEAL)
    for i, v in enumerate(g.values):
        ax.text(v / 1000, i, f" ${v/1000:,.0f}K", va="center", fontsize=10)
    ax.set_xlabel("Margin gap to target ($K)"); ax.set_title("Where the margin goes, by primary driver")
    fig.tight_layout(); fig.savefig(fig_dir / "01_margin_gap_by_driver.png", dpi=160); plt.close(fig)

    # 2: accuracy fixes, before vs after
    naive = accuracy(diagnose(run_pipeline(skus, orders, separate_return_freight=False), remorse_routing=False), skus)
    fixed = accuracy(d, skus)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, key, title, fmt in [(axes[0], "false_freight_flags", "False freight flags\n(return freight in freight column)", "{:.0f}"),
                                (axes[1], "remorse_sent_to_listing_fix", "Buyer-remorse SKUs sent\nto 'fix the listing'", "{:.0%}"),
                                (axes[2], "action_accuracy", "Right action for the\nright team", "{:.0%}")]:
        vals = [naive[key], fixed[key]]
        bars = ax.bar(["Before fix", "After fix"], vals, color=[GREY, TEAL])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=11); ax.set_yticks([])
    fig.suptitle("Diagnostic accuracy was the real product problem", fontweight="bold")
    fig.tight_layout(); fig.savefig(fig_dir / "02_accuracy_fixes.png", dpi=160); plt.close(fig)

    # 3: tickets by team
    t = pd.DataFrame(agent.tickets)
    if len(t):
        by = t.groupby("owner").agg(tickets=("key", "count"), gap=("margin_gap", "sum")).sort_values("gap")
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.barh(by.index, by.gap / 1000, color=NAVY)
        for i, (n, v) in enumerate(zip(by.tickets, by.gap)):
            ax.text(v / 1000, i, f" {n} ticket{'s' if n > 1 else ''}, ${v/1000:,.0f}K", va="center", fontsize=10)
        ax.set_xlabel("Margin gap routed ($K)"); ax.set_title("Jira tickets raised this week, by owning team")
        fig.tight_layout(); fig.savefig(fig_dir / "03_tickets_by_team.png", dpi=160); plt.close(fig)
    return naive, fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="let Claude drive the tool loop (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--week", default="2026-W42")
    args = ap.parse_args()

    skus, orders = simulate()
    agent = Agent(skus, orders, args.week, ROOT / "output")
    log = agent.run_live() if args.live else agent.run_offline()

    for step in log:
        out = step["output"]
        short = out if len(str(out)) < 160 else str(out)[:160] + "..."
        print(f"> {step['tool']}({step['input'] or ''})\n  {short}")
    naive, fixed = figures(agent, skus, orders)
    print("\nAccuracy vs hidden truth (synthetic data):")
    print("  before fixes:", naive)
    print("  after fixes: ", fixed)
    print("\nTickets:")
    for t in agent.tickets:
        print(f"  {t['key']:<8} {t['priority']:<8} {t['owner']:<18} {t['summary']}")


if __name__ == "__main__":
    main()
