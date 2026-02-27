#!/usr/bin/env python3
"""A/B test: does inference_priority actually affect scheduling?

Runs the SAME 72 tasks twice:
  A (control):    no decorator — all tasks use default priority (1)
  B (experiment): @inference_priority with tiered priorities (1,3,5,7,10)

If the staircase looks identical in both, inference_priority does nothing
and the pattern is purely from max_tokens differences.

Usage:
    cd libs/ai-endpoints
    poetry run python docs/chat/priority_tuning.py
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_nvidia_ai_endpoints import ChatNVIDIADynamo, inference_priority

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMETERS                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

NIM_BASE_URL = "http://localhost:8099/v1"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

NUM_RUNS = 5
NUM_WARMUP_CALLS = 20
WARMUP_OSL = 500

TIERS: list[dict[str, Any]] = [
    {"priority": 1,  "label": "CRITICAL",   "osl": 128,  "max_tokens": 128,  "count": 2},
    {"priority": 3,  "label": "HIGH",       "osl": 512,  "max_tokens": 512,  "count": 5},
    {"priority": 5,  "label": "MEDIUM",     "osl": 1024, "max_tokens": 1024, "count": 10},
    {"priority": 7,  "label": "LOW",        "osl": 2048, "max_tokens": 2048, "count": 20},
    {"priority": 10, "label": "BACKGROUND", "osl": 4096, "max_tokens": 4096, "count": 35},
]

CUSTOMER_QUERY = (
    "I've been a loyal customer for 5 years, but my last three orders "
    "have all arrived damaged. I was charged $149.99 and still haven't "
    "received a refund."
)

# System prompts pool
SYSTEM_PROMPTS = [
    "Analyze customer sentiment in detail.",
    "Determine if this needs manager escalation. Justify in detail.",
    "Categorize into a support category with full reasoning.",
    "Determine the correct department and explain why in detail.",
    "Analyze the customer's communication tone and frustration level.",
    "Summarize the customer's issue for an agent handoff.",
    "Provide a detailed breakdown of all customer intents.",
    "Assess the risk level of this customer interaction.",
    "Identify all relevant company policies for this query.",
    "Review the customer's account history in detail.",
    "Check for similar past support tickets and outcomes.",
    "Identify potential product defects mentioned and their severity.",
    "Review the terms and conditions relevant to this case.",
    "Check SLA requirements for this customer tier.",
    "Search the knowledge base for relevant articles.",
    "Find FAQ entries that match this customer issue.",
    "Identify response templates for this issue type.",
    "Find precedent cases and how they were resolved.",
    "Research shipping carrier reliability data with statistics.",
    "Provide a comprehensive review of warranty and return policies.",
    "Analyze the full financial impact of this complaint.",
    "Review recent quality control reports and trend analysis.",
    "Conduct a full regulatory compliance review for this case.",
    "Check inventory status for replacement items.",
    "Research supplier quality metrics for this product.",
    "Analyze logistics chain for delivery issues.",
    "Review packaging standards for fragile items.",
    "Check shipping insurance coverage details.",
    "Analyze seasonal patterns in damage reports.",
    "Review warehouse handling procedures.",
    "Compare carrier damage rates across providers.",
    "Analyze regional delivery performance data.",
    "Calculate vendor reliability scores for this product line.",
    "Write a detailed industry benchmark report for resolution times.",
    "Write a comprehensive competitor refund policy analysis.",
    "Conduct a thorough root cause analysis of recurring damage.",
    "Draft multiple compensation options with cost-benefit analysis.",
    "Write a detailed process improvement proposal to prevent recurrence.",
    "Identify cross-sell or retention opportunities.",
    "Assess churn risk for this customer.",
    "Estimate impact on net promoter score.",
    "Identify agent training opportunities from this case.",
    "Identify automation opportunities for this issue type.",
    "Forecast volume of similar complaints next quarter.",
    "Monitor social media for related complaints.",
    "Draft internal documentation for this case pattern.",
    "Suggest A/B tests for resolution strategies.",
    "Draft knowledge base article from this resolution.",
    "Review customer feedback trends across all channels.",
    "Analyze cost of customer acquisition vs retention for this segment.",
]


@dataclass
class StageResult:
    name: str
    start: float
    end: float
    tier_label: str


async def main() -> None:
    # Create one LLM per tier (different max_completion_tokens)
    llms: dict[int, ChatNVIDIADynamo] = {}
    for tier in TIERS:
        llms[tier["priority"]] = ChatNVIDIADynamo(
            base_url=NIM_BASE_URL,
            model=MODEL,
            max_completion_tokens=tier["max_tokens"],
        )

    # ── Build chains ───────────────────────────────────────────────────────
    chains: dict[str, tuple[Any, str, int]] = {}  # name -> (chain, tier_label, priority)
    prompt_idx = 0
    for tier in TIERS:
        tier_llm = llms[tier["priority"]]
        for i in range(tier["count"]):
            name = f"{tier['label'].lower()}_{i+1:02d}"
            sys_prompt = SYSTEM_PROMPTS[prompt_idx % len(SYSTEM_PROMPTS)]
            prompt_idx += 1
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", f"{sys_prompt} Be thorough."),
                    ("user", "{query}"),
                ])
                | tier_llm.bind(osl=tier["osl"])
                | StrOutputParser()
            )
            chains[name] = (chain, tier["label"], tier["priority"])

    total_tasks = len(chains)
    print(f"Built {total_tasks} chains across {len(TIERS)} tiers:")
    for tier in TIERS:
        print(f"  {tier['label']:12s}  priority={tier['priority']:2d}  "
              f"count={tier['count']:2d}  osl={tier['osl']}  "
              f"max_tokens={tier['max_tokens']}")
    print()

    # ── Timed helper ──────────────────────────────────────────────────────
    async def _timed(chain: Any, name: str, query: str,
                     t0: float, tier_label: str) -> StageResult:
        start = time.perf_counter()
        await chain.ainvoke({"query": query})
        end = time.perf_counter()
        return StageResult(name=name, start=start - t0, end=end - t0,
                           tier_label=tier_label)

    # ── Decorated wrappers (for experiment run) ───────────────────────────
    @inference_priority(priority=1)
    async def p1_call(chain: Any, name: str, query: str,
                      t0: float, tier_label: str) -> StageResult:
        return await _timed(chain, name, query, t0, tier_label)

    @inference_priority(priority=3)
    async def p3_call(chain: Any, name: str, query: str,
                      t0: float, tier_label: str) -> StageResult:
        return await _timed(chain, name, query, t0, tier_label)

    @inference_priority(priority=5)
    async def p5_call(chain: Any, name: str, query: str,
                      t0: float, tier_label: str) -> StageResult:
        return await _timed(chain, name, query, t0, tier_label)

    @inference_priority(priority=7)
    async def p7_call(chain: Any, name: str, query: str,
                      t0: float, tier_label: str) -> StageResult:
        return await _timed(chain, name, query, t0, tier_label)

    @inference_priority(priority=10)
    async def p10_call(chain: Any, name: str, query: str,
                       t0: float, tier_label: str) -> StageResult:
        return await _timed(chain, name, query, t0, tier_label)

    priority_wrappers = {
        1: p1_call, 3: p3_call, 5: p5_call, 7: p7_call, 10: p10_call,
    }

    # ── No-decorator wrapper (for control run) ───────────────────────────
    # No @inference_priority — uses ChatNVIDIADynamo default (priority=1)
    async def no_priority_call(chain: Any, name: str, query: str,
                               t0: float, tier_label: str) -> StageResult:
        return await _timed(chain, name, query, t0, tier_label)

    # ── Warmup ─────────────────────────────────────────────────────────────
    warmup_llm = llms[TIERS[0]["priority"]]
    print(f"Warming up GPU ({NUM_WARMUP_CALLS} calls, osl={WARMUP_OSL})...")
    warmup = [warmup_llm.ainvoke("Warm up.", osl=WARMUP_OSL)
              for _ in range(NUM_WARMUP_CALLS)]
    await asyncio.gather(*warmup)
    print("Warmup done.\n")

    # ── Run both modes ─────────────────────────────────────────────────────
    tier_labels = [t["label"] for t in TIERS]

    def collect_stats(
        results: list[StageResult],
    ) -> dict[str, dict[str, float]]:
        """Per-tier max and mean end times."""
        stats: dict[str, dict[str, float]] = {}
        for label in tier_labels:
            tier_r = [r for r in results if r.tier_label == label]
            if tier_r:
                stats[label] = {
                    "max": max(r.end for r in tier_r),
                    "mean": statistics.mean(r.end for r in tier_r),
                }
        return stats

    for mode_name, use_priority in [("A: NO PRIORITY (control)", False),
                                     ("B: WITH @inference_priority", True)]:
        print("=" * 70)
        print(f"  {mode_name}")
        print("=" * 70)

        all_max: dict[str, list[float]] = {t["label"]: [] for t in TIERS}
        all_mean: dict[str, list[float]] = {t["label"]: [] for t in TIERS}

        for run_idx in range(1, NUM_RUNS + 1):
            print(f"  Run {run_idx}/{NUM_RUNS} — {total_tasks} calls...", end=" ")
            t0 = time.perf_counter()
            tasks = []
            for name, (chain, tier_label, pri) in chains.items():
                if use_priority:
                    wrapper = priority_wrappers[pri]
                else:
                    wrapper = no_priority_call
                tasks.append(wrapper(chain, name, CUSTOMER_QUERY, t0, tier_label))
            results = await asyncio.gather(*tasks)

            wall = max(r.end for r in results)
            print(f"done in {wall:.1f}s")

            stats = collect_stats(results)
            for label in tier_labels:
                s = stats[label]
                all_max[label].append(s["max"])
                all_mean[label].append(s["mean"])
                print(f"    {label:12s}: max={s['max']:6.1f}s  mean={s['mean']:6.1f}s")
            print()

        # Median summary for this mode
        print(f"  MEDIAN ({mode_name}):")
        print(f"    {'Tier':12s}  {'Med Max':>8s}  {'Med Mean':>9s}")
        print(f"    {'─'*12}  {'─'*8}  {'─'*9}")
        for label in tier_labels:
            mm = statistics.median(all_max[label])
            mn = statistics.median(all_mean[label])
            print(f"    {label:12s}  {mm:>7.1f}s  {mn:>8.1f}s")
        print()

    # ── Side-by-side comparison ────────────────────────────────────────────
    # Re-run a quick 3-run comparison and store both
    print("=" * 70)
    print("  SIDE-BY-SIDE COMPARISON (3 fresh runs each)")
    print("=" * 70)

    comparison: dict[str, dict[str, list[float]]] = {}
    for mode_name, use_priority in [("NO_PRIORITY", False),
                                     ("WITH_PRIORITY", True)]:
        max_times: dict[str, list[float]] = {t["label"]: [] for t in TIERS}
        mean_times: dict[str, list[float]] = {t["label"]: [] for t in TIERS}

        for run_idx in range(3):
            t0 = time.perf_counter()
            tasks = []
            for name, (chain, tier_label, pri) in chains.items():
                if use_priority:
                    wrapper = priority_wrappers[pri]
                else:
                    wrapper = no_priority_call
                tasks.append(wrapper(chain, name, CUSTOMER_QUERY, t0, tier_label))
            results = await asyncio.gather(*tasks)

            for label in tier_labels:
                tier_r = [r for r in results if r.tier_label == label]
                if tier_r:
                    max_times[label].append(max(r.end for r in tier_r))
                    mean_times[label].append(statistics.mean(r.end for r in tier_r))

        comparison[mode_name] = {
            f"{label}_max": statistics.median(max_times[label])
            for label in tier_labels
        }
        comparison[mode_name].update({
            f"{label}_mean": statistics.median(mean_times[label])
            for label in tier_labels
        })

    print()
    print(f"  {'Tier':12s}  {'NO_PRI Max':>11s}  {'PRI Max':>11s}  {'Delta':>8s}  "
          f"{'NO_PRI Mean':>12s}  {'PRI Mean':>10s}  {'Delta':>8s}")
    print(f"  {'─'*12}  {'─'*11}  {'─'*11}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*8}")
    for label in tier_labels:
        no_max = comparison["NO_PRIORITY"][f"{label}_max"]
        pri_max = comparison["WITH_PRIORITY"][f"{label}_max"]
        no_mean = comparison["NO_PRIORITY"][f"{label}_mean"]
        pri_mean = comparison["WITH_PRIORITY"][f"{label}_mean"]
        d_max = pri_max - no_max
        d_mean = pri_mean - no_mean
        print(f"  {label:12s}  {no_max:>10.1f}s  {pri_max:>10.1f}s  {d_max:>+7.1f}s  "
              f"{no_mean:>11.1f}s  {pri_mean:>9.1f}s  {d_mean:>+7.1f}s")

    print()
    print("If deltas are ~0 across all tiers, inference_priority has NO effect.")
    print("If CRITICAL delta is negative and BACKGROUND delta is positive,")
    print("  inference_priority IS working (speeding up critical, slowing background).")


if __name__ == "__main__":
    asyncio.run(main())
