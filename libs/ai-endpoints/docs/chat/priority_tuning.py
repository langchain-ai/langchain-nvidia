#!/usr/bin/env python3
"""Priority-only A/B test: does priority actually affect Dynamo scheduling?

Saturates the GPU with 200 diverse, realistic queries to force queueing.
Holds max_tokens, osl, latency_sensitivity, iat constant across all requests.
Only priority differs.

  A (control):    all 200 requests at default priority (1)
  B (experiment): same 200 requests with tiered priorities (1,3,5,7,10)

If CRITICAL tasks finish faster in B than A, priority scheduling works.
If completion times are identical, priority is a no-op.

Usage:
    cd libs/ai-endpoints
    .venv/bin/python docs/chat/priority_tuning.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_nvidia_ai_endpoints import ChatNVIDIADynamo

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

NIM_BASE_URL = "http://localhost:8099/v1"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

NUM_RUNS = 5
NUM_WARMUP_CALLS = 20

# Fixed for ALL requests — only priority changes
MAX_TOKENS = 2048
OSL = 2048
LATENCY_SENSITIVITY = 1.0
IAT = 250
DEFAULT_PRIORITY = 1  # control mode: all requests use this

# Heavy bottom distribution — flood the queue so priority has something to sort
TIERS: list[dict[str, Any]] = [
    {"priority": 1,  "label": "CRITICAL",   "count": 5},
    {"priority": 3,  "label": "HIGH",       "count": 10},
    {"priority": 5,  "label": "MEDIUM",     "count": 20},
    {"priority": 7,  "label": "LOW",        "count": 40},
    {"priority": 10, "label": "BACKGROUND", "count": 125},
]
# Total: 200 concurrent requests

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DIVERSE QUERY POOL                                                        ║
# ║  Each request gets a unique (system, user) pair — realistic multi-agent     ║
# ║  workload with varying complexity and domain.                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

QUERIES: list[dict[str, str]] = [
    # ── Customer support ────────────────────────────────────────────────────
    {
        "system": "You are a senior customer support analyst. Provide a thorough, "
                  "detailed analysis with specific recommendations.",
        "user": "I've been a loyal customer for 5 years, but my last three orders "
                "have all arrived damaged. I was charged $149.99 and still haven't "
                "received a refund. I've called support 4 times and keep getting "
                "transferred. I'm considering switching to a competitor.",
    },
    {
        "system": "You are a customer escalation specialist. Analyze the situation "
                  "and draft a comprehensive resolution plan.",
        "user": "My subscription was auto-renewed at the wrong price ($89/mo instead "
                "of the $49/mo promotional rate I was promised). I have the original "
                "email confirmation showing the promotional rate. I've been overcharged "
                "for 6 months totaling $240 in excess charges.",
    },
    {
        "system": "You are a customer retention expert. Analyze churn risk and "
                  "propose a detailed retention strategy.",
        "user": "I've been a premium member since 2019 but the recent UI redesign "
                "has made the product nearly unusable for my workflow. Three features "
                "I rely on daily were removed without notice. My team of 12 is "
                "evaluating alternatives. Our annual spend is $15,000.",
    },
    {
        "system": "You are a warranty claims processor. Review this case thoroughly "
                  "and determine the appropriate resolution.",
        "user": "My high-end laptop purchased 11 months ago has a screen defect — "
                "there's a growing dead pixel cluster in the center. I've documented "
                "it with photos over the past 3 weeks showing it's spreading. The "
                "laptop cost $2,499 and I use it for professional photo editing.",
    },
    {
        "system": "You are a fraud detection analyst. Investigate this account "
                  "activity and provide a detailed risk assessment.",
        "user": "I noticed 3 charges I didn't make on my account: $299 from an "
                "electronics store in a city I've never visited, $150 from an online "
                "marketplace, and $75 from a streaming service I don't use. All "
                "happened within 2 hours yesterday while I was at work.",
    },
    # ── Financial analysis ──────────────────────────────────────────────────
    {
        "system": "You are a financial analyst. Provide a detailed market analysis "
                  "with supporting reasoning and risk factors.",
        "user": "Analyze the potential impact of rising interest rates on the "
                "technology sector over the next 12 months. Consider both large-cap "
                "and growth companies, debt levels, and how different sub-sectors "
                "(SaaS, semiconductors, AI infrastructure) might be affected.",
    },
    {
        "system": "You are a portfolio risk manager. Conduct a comprehensive risk "
                  "assessment with quantitative reasoning.",
        "user": "Our portfolio has 60% US equities, 20% international, 15% bonds, "
                "and 5% alternatives. Given current geopolitical tensions, inflation "
                "trends, and the inverted yield curve, what are the top 5 risks and "
                "what hedging strategies would you recommend?",
    },
    {
        "system": "You are a startup valuation expert. Provide a detailed valuation "
                  "analysis with multiple methodologies.",
        "user": "A Series B SaaS startup has $8M ARR growing 120% YoY, 85% gross "
                "margins, $2M net burn, 130% net dollar retention, and 18 months of "
                "runway. They're raising at $200M pre-money. Is this reasonable? "
                "Compare using revenue multiples, growth-adjusted metrics, and comps.",
    },
    # ── Legal analysis ──────────────────────────────────────────────────────
    {
        "system": "You are a contract law specialist. Review this scenario and "
                  "provide detailed legal analysis with citations to common law "
                  "principles.",
        "user": "A software vendor's SLA promises 99.99% uptime but the service "
                "was down for 8 hours last month (99.0% uptime). The contract's "
                "liability clause caps damages at 'fees paid in the affected month' "
                "($10,000) but our actual losses from the outage were $500,000 in "
                "lost sales. What are our legal options?",
    },
    {
        "system": "You are an employment law attorney. Analyze this workplace "
                  "situation thoroughly and advise on legal exposure.",
        "user": "An employee was terminated 2 weeks after filing an internal "
                "complaint about safety violations. The company says it was part "
                "of a planned restructuring, but the employee was the only person "
                "let go from their department. They had positive performance reviews "
                "for the past 3 years. What's the retaliation risk?",
    },
    # ── Technical/engineering ───────────────────────────────────────────────
    {
        "system": "You are a senior software architect. Provide a detailed technical "
                  "analysis with trade-offs and recommendations.",
        "user": "We're migrating a monolithic Django application (500K LOC, 200 "
                "database tables, 50M daily requests) to microservices. The team is "
                "debating between event-driven architecture with Kafka vs. REST-based "
                "service mesh. We have 15 engineers and need to maintain zero downtime "
                "during migration. What's your recommended approach?",
    },
    {
        "system": "You are a database performance expert. Diagnose this issue and "
                  "provide a comprehensive optimization plan.",
        "user": "Our PostgreSQL database has grown to 2TB and query performance has "
                "degraded significantly. The main table has 500M rows and receives "
                "10K writes/second. Key queries that used to take 50ms now take 5s. "
                "We're seeing high WAL write latency and vacuum is falling behind. "
                "Replication lag to read replicas is growing. What should we do?",
    },
    {
        "system": "You are a cloud infrastructure architect. Design a comprehensive "
                  "solution with cost analysis and scaling strategy.",
        "user": "We need to process 10 million images per day through an ML pipeline "
                "(object detection + classification + quality scoring). Each image is "
                "5-15MB. Results must be available within 30 seconds of upload. Current "
                "cost budget is $50K/month. We're on AWS. Design the infrastructure.",
    },
    {
        "system": "You are a cybersecurity incident responder. Analyze this security "
                  "event and provide a detailed response plan.",
        "user": "Our SIEM detected unusual outbound traffic from 3 application "
                "servers: 2GB of data exfiltrated to an unknown IP over port 443 "
                "during off-hours. The servers host our customer database (2M records "
                "with PII). Preliminary analysis shows a compromised service account "
                "with elevated privileges. What's the immediate response plan?",
    },
    # ── Research and analysis ───────────────────────────────────────────────
    {
        "system": "You are a medical research analyst. Provide a thorough literature "
                  "review with critical analysis of methodology.",
        "user": "Summarize the current state of research on mRNA vaccine technology "
                "for cancer treatment. Cover the major clinical trials, compare "
                "approaches (personalized neoantigen vs shared tumor antigens), "
                "discuss the key challenges, and assess the timeline for broad "
                "clinical availability.",
    },
    {
        "system": "You are a climate science policy advisor. Provide a detailed "
                  "analysis with quantitative data and policy recommendations.",
        "user": "Analyze the feasibility of achieving net-zero emissions by 2050 "
                "for a mid-sized industrial nation. Consider the energy mix "
                "transition timeline, required infrastructure investment, impact on "
                "GDP, job displacement and creation, and the role of carbon capture "
                "technology vs renewable expansion.",
    },
    {
        "system": "You are an education technology researcher. Provide a detailed "
                  "evidence-based analysis with specific recommendations.",
        "user": "Evaluate the effectiveness of AI tutoring systems compared to "
                "traditional classroom instruction for K-12 mathematics education. "
                "Cover learning outcomes, engagement metrics, equity implications, "
                "teacher workload impact, and long-term retention. Include specific "
                "study results where possible.",
    },
    # ── Creative and strategic ──────────────────────────────────────────────
    {
        "system": "You are a brand strategy consultant. Develop a comprehensive "
                  "go-to-market strategy with detailed reasoning.",
        "user": "A direct-to-consumer electric bicycle company wants to launch in "
                "the US market. They have strong brand recognition in Europe with "
                "premium positioning ($3K-$5K price range). Their differentiator is "
                "integrated GPS tracking and theft recovery. Budget is $2M for the "
                "first year. Develop the US launch strategy.",
    },
    {
        "system": "You are a supply chain optimization expert. Analyze the situation "
                  "and provide a detailed restructuring plan.",
        "user": "Our manufacturing company sources components from 12 suppliers "
                "across 6 countries. Recent disruptions (port congestion, chip "
                "shortages, geopolitical tensions) have caused 3 production "
                "shutdowns this year costing $5M each. Current inventory model is "
                "just-in-time. Redesign our supply chain for resilience while "
                "minimizing cost increase.",
    },
    {
        "system": "You are a product strategy director. Conduct a thorough "
                  "competitive analysis and recommend a product roadmap.",
        "user": "Our B2B project management SaaS tool ($50M ARR, 5000 customers) "
                "is losing enterprise deals to a competitor who just launched AI "
                "features (auto-scheduling, risk prediction, resource optimization). "
                "Our R&D team is 40 engineers. We have $10M in the bank. What's "
                "the 12-month product strategy to compete?",
    },
    # ── Healthcare ──────────────────────────────────────────────────────────
    {
        "system": "You are a clinical informatics specialist. Analyze this "
                  "patient workflow problem and recommend improvements.",
        "user": "Our hospital's emergency department sees 300 patients per day "
                "but average wait time has grown to 4.5 hours. Triage nurses "
                "report the EHR system requires 15 minutes of data entry per "
                "patient. Lab results take 90 minutes to return. Bed turnover "
                "averages 6 hours. Propose a comprehensive optimization plan.",
    },
    {
        "system": "You are a pharmaceutical regulatory affairs expert. Provide "
                  "a detailed regulatory pathway analysis.",
        "user": "We have a novel gene therapy for sickle cell disease that "
                "showed 85% efficacy in Phase II with 120 patients. We want to "
                "pursue accelerated approval from the FDA. The therapy requires "
                "myeloablative conditioning and has a 5% serious adverse event "
                "rate. Outline the regulatory strategy and timeline.",
    },
    # ── Education ───────────────────────────────────────────────────────────
    {
        "system": "You are a curriculum design expert. Create a detailed "
                  "learning pathway with assessment strategies.",
        "user": "Design a 16-week introductory data science course for "
                "business professionals with no coding experience. They need to "
                "be able to clean data, create visualizations, run basic "
                "statistical analyses, and present findings to stakeholders. "
                "Include weekly topics, projects, and assessment rubrics.",
    },
    {
        "system": "You are an educational equity researcher. Analyze this "
                  "achievement gap scenario and propose interventions.",
        "user": "A school district with 25,000 students has a persistent 30-point "
                "gap in math proficiency between students from high-income and "
                "low-income neighborhoods. The district has tried tutoring programs "
                "and summer school with minimal impact. Budget for new initiatives "
                "is $2M annually. What evidence-based interventions would help?",
    },
    # ── Real estate and urban planning ──────────────────────────────────────
    {
        "system": "You are an urban planning consultant. Provide a detailed "
                  "feasibility analysis with community impact assessment.",
        "user": "A mid-size city (population 350,000) wants to convert a "
                "defunct 50-acre industrial site near downtown into a mixed-use "
                "development. The site has known soil contamination from a former "
                "chemical plant. The city wants affordable housing, green space, "
                "and commercial amenities. Analyze the feasibility and trade-offs.",
    },
    {
        "system": "You are a commercial real estate analyst. Conduct a thorough "
                  "market analysis and investment recommendation.",
        "user": "Evaluate a $15M acquisition opportunity for a 100,000 sq ft "
                "Class B office building in a secondary market. Current occupancy "
                "is 72%, average lease term remaining is 2.3 years, and the "
                "largest tenant (40% of revenue) has indicated they may not renew. "
                "Cap rate is 7.2%. Is this a good investment?",
    },
    # ── Logistics and operations ────────────────────────────────────────────
    {
        "system": "You are a logistics optimization specialist. Design an "
                  "efficient distribution network with cost modeling.",
        "user": "An e-commerce company ships 50,000 packages daily from 3 "
                "warehouses on the East Coast. They want to expand to same-day "
                "delivery in 10 major metro areas. Current average delivery time "
                "is 3.2 days. Shipping costs are $4.50 per package. Design the "
                "expanded fulfillment network and estimate the cost impact.",
    },
    {
        "system": "You are a manufacturing process engineer. Analyze the "
                  "production bottleneck and propose solutions.",
        "user": "Our automotive parts factory runs 3 shifts producing 10,000 "
                "units daily but defect rate has climbed from 1.2% to 3.8% over "
                "6 months. CNC machines are running at 92% utilization. Quality "
                "inspection is manual with 6 inspectors per shift. We suspect "
                "tool wear and operator fatigue. Propose a comprehensive fix.",
    },
    # ── Environmental science ───────────────────────────────────────────────
    {
        "system": "You are an environmental impact assessment specialist. "
                  "Conduct a thorough analysis with mitigation recommendations.",
        "user": "A proposed offshore wind farm with 80 turbines would be built "
                "15 miles off the coast in a known migratory bird corridor and "
                "near a commercial fishing ground. The project would generate "
                "800MW of clean energy for 200,000 homes. Analyze the "
                "environmental trade-offs and propose mitigation measures.",
    },
    {
        "system": "You are a water resource management expert. Analyze this "
                  "crisis scenario and develop a long-term sustainability plan.",
        "user": "A region dependent on a single aquifer for 80% of its water "
                "supply is seeing water table levels drop 3 feet per year. "
                "Agricultural irrigation accounts for 65% of usage, residential "
                "25%, and industrial 10%. Population is growing 2% annually. "
                "Current reserves will last approximately 15 years at current "
                "rates. Develop a comprehensive water management strategy.",
    },
    # ── Marketing and advertising ───────────────────────────────────────────
    {
        "system": "You are a digital marketing strategist. Design a "
                  "comprehensive multi-channel campaign with ROI projections.",
        "user": "A mid-market fitness equipment company ($30M revenue) wants to "
                "launch a premium connected rowing machine ($2,500) competing "
                "against Peloton and Hydrow. Their current customer base is "
                "commercial gyms but this is their first D2C product. Marketing "
                "budget is $3M for the launch year. Design the campaign.",
    },
    {
        "system": "You are a brand crisis communications expert. Develop a "
                  "comprehensive response strategy and messaging framework.",
        "user": "A food company's product was linked to a salmonella outbreak "
                "affecting 200 people across 12 states. The FDA has issued a "
                "voluntary recall. Social media mentions are up 5,000% with "
                "negative sentiment at 89%. A class-action lawsuit has been "
                "filed. Three major retailers have pulled the product. Develop "
                "the crisis response plan.",
    },
    # ── Human resources ─────────────────────────────────────────────────────
    {
        "system": "You are an organizational development consultant. Analyze "
                  "the situation and propose a transformation roadmap.",
        "user": "A 500-person tech company has 35% annual turnover (industry "
                "average is 15%). Exit interviews cite lack of career growth, "
                "poor management, and below-market compensation. The company "
                "has grown from 200 to 500 people in 18 months. Engineering "
                "turnover is 45%. Replacement cost averages $75K per hire. "
                "Design a comprehensive retention strategy.",
    },
    {
        "system": "You are a compensation and benefits analyst. Design a "
                  "competitive total rewards package with market benchmarking.",
        "user": "A Series C startup (400 employees) needs to redesign its "
                "compensation structure. Currently they pay 25th percentile "
                "cash with above-market equity. But as IPO timeline extends, "
                "equity is less attractive. They're losing senior engineers "
                "to FAANG companies. Annual budget for compensation adjustments "
                "is $5M. Design the new compensation framework.",
    },
    # ── Data science and AI ─────────────────────────────────────────────────
    {
        "system": "You are an ML systems architect. Design a production ML "
                  "pipeline with monitoring and governance.",
        "user": "We need to build a real-time fraud detection system processing "
                "50,000 credit card transactions per second. Current rule-based "
                "system catches 60% of fraud with 2% false positive rate. We "
                "want to use ML to improve to 90% detection with under 0.5% "
                "false positives. Latency requirement is under 100ms per "
                "transaction. Design the end-to-end system.",
    },
    {
        "system": "You are an AI ethics researcher. Conduct a thorough bias "
                  "audit framework analysis.",
        "user": "Our hiring platform uses an AI model to screen 500,000 resumes "
                "annually. An internal audit found that the model advances "
                "candidates from certain zip codes at 2x the rate of others, "
                "even controlling for qualifications. The model was trained on "
                "historical hiring decisions. Analyze the bias sources and "
                "propose a comprehensive remediation plan.",
    },
    # ── Energy and sustainability ───────────────────────────────────────────
    {
        "system": "You are a renewable energy project finance analyst. "
                  "Evaluate this investment opportunity with detailed financials.",
        "user": "A 200MW solar farm project requires $250M in capital. The site "
                "has excellent irradiance (5.5 kWh/m2/day). A 20-year PPA is "
                "offered at $0.035/kWh with 1.5% annual escalation. Estimated "
                "capacity factor is 28%. O&M costs are projected at $8/MWh. "
                "ITC is 30%. Debt is available at 5.5% for 60% of capital. "
                "Evaluate the project economics and key risk factors.",
    },
    {
        "system": "You are a corporate sustainability strategist. Develop a "
                  "comprehensive ESG roadmap.",
        "user": "A Fortune 500 manufacturing company emits 2M tons of CO2 "
                "annually. 40% comes from natural gas in production, 35% from "
                "electricity, and 25% from fleet operations. They've committed "
                "to net-zero by 2040. Current renewable energy is 12% of total. "
                "Develop a phased roadmap with cost estimates and timeline "
                "for each major decarbonization initiative.",
    },
    # ── Government and public policy ────────────────────────────────────────
    {
        "system": "You are a public policy analyst. Evaluate this proposed "
                  "regulation with economic impact analysis.",
        "user": "A state is considering requiring all employers with 50+ "
                "employees to provide 12 weeks of paid family leave at 80% of "
                "salary, funded by a 0.5% payroll tax split between employer "
                "and employee. Current voluntary adoption is 23%. Analyze the "
                "economic impact on businesses, workers, and the state budget.",
    },
    {
        "system": "You are a transportation policy expert. Design an "
                  "integrated transit solution with ridership projections.",
        "user": "A growing metro area (2M population, projected 2.8M by 2040) "
                "has severe congestion. Average commute is 45 minutes. The "
                "existing bus system covers 30% of the metro. There's no rail. "
                "A $4B bond measure for transit expansion is being considered. "
                "Options include light rail, BRT, or a hybrid approach. "
                "Analyze each option with cost-benefit analysis.",
    },
    # ── Nonprofit and social impact ─────────────────────────────────────────
    {
        "system": "You are a nonprofit strategy consultant. Develop a "
                  "comprehensive organizational growth plan.",
        "user": "A food bank serving 50,000 families annually wants to double "
                "capacity in 3 years. Current budget is $8M (60% donations, "
                "25% grants, 15% government). They have 30 staff and 500 "
                "regular volunteers. Warehouse capacity is at 90%. Distribution "
                "is through 45 partner sites. Develop the scaling strategy "
                "including fundraising, operations, and partnerships.",
    },
    {
        "system": "You are a social impact measurement specialist. Design "
                  "an evaluation framework with quantitative metrics.",
        "user": "A workforce development program trains 2,000 adults annually "
                "in digital skills. Completion rate is 68% and 45% of graduates "
                "find employment within 6 months. The program costs $5,000 per "
                "participant. Funders want rigorous impact evidence. Design a "
                "comprehensive evaluation framework including counterfactual "
                "analysis and long-term outcome tracking.",
    },
    # ── Aerospace and defense ───────────────────────────────────────────────
    {
        "system": "You are an aerospace systems engineer. Conduct a detailed "
                  "trade study with technical analysis.",
        "user": "A satellite communications company needs to choose between "
                "LEO constellation (500 satellites at 550km) vs MEO "
                "constellation (24 satellites at 8,000km) for global broadband. "
                "Target latency is under 50ms, throughput 100Mbps per user "
                "terminal, and 99.9% availability. Compare the architectures "
                "on cost, performance, and operational complexity.",
    },
    {
        "system": "You are a defense procurement analyst. Evaluate this "
                  "acquisition program and identify key risks.",
        "user": "A next-generation unmanned aerial system program has a $2B "
                "development budget over 7 years. The prime contractor is "
                "proposing a novel hybrid-electric propulsion system that has "
                "only been demonstrated at 1/3 scale. Schedule is already 8 "
                "months behind after 2 years. Two critical subsystem tests "
                "failed last quarter. Assess the program risks and recommend "
                "corrective actions.",
    },
    # ── Agriculture and food science ────────────────────────────────────────
    {
        "system": "You are a precision agriculture specialist. Design a "
                  "smart farming implementation plan.",
        "user": "A 5,000-acre corn and soybean farm wants to implement "
                "precision agriculture. Current yields are 180 bu/acre for corn "
                "and 55 bu/acre for soybeans, both below regional averages. "
                "Soil types vary across the property. They have basic GPS on "
                "equipment but no variable rate application or remote sensing. "
                "Budget is $500K over 3 years. Design the implementation plan.",
    },
    {
        "system": "You are a food safety and supply chain expert. Analyze "
                  "this traceability problem and design a solution.",
        "user": "A national grocery chain with 800 stores needs to implement "
                "farm-to-shelf traceability for all fresh produce within 18 "
                "months to comply with new FDA requirements. Currently only "
                "30% of suppliers provide lot-level tracking. The chain sources "
                "from 2,000 suppliers across 15 countries. Design the "
                "traceability system architecture and rollout plan.",
    },
    # ── Insurance and risk ──────────────────────────────────────────────────
    {
        "system": "You are an actuarial science expert. Analyze this risk "
                  "pool and recommend pricing adjustments.",
        "user": "A property insurance company covering coastal homes has seen "
                "claims increase 40% over 5 years due to intensifying hurricanes. "
                "Their current portfolio is 50,000 policies with $25B total "
                "insured value. Reinsurance costs have doubled. 15% of policies "
                "are in the highest-risk zones. Analyze the risk pool and "
                "recommend a sustainable pricing and underwriting strategy.",
    },
    {
        "system": "You are a cyber insurance underwriter. Evaluate this "
                  "enterprise's risk profile and propose coverage terms.",
        "user": "A mid-market healthcare company (5,000 employees, $500M "
                "revenue) is applying for $10M in cyber insurance. They handle "
                "2M patient records. Last year they had a ransomware incident "
                "that cost $2M to remediate. They've since implemented EDR and "
                "MFA but still run some legacy systems. Security audit score "
                "is 65/100. Evaluate the risk and propose terms.",
    },
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DATA                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass
class Result:
    name: str
    start_time: float  # seconds from batch start to first token sent
    end_time: float    # seconds from batch start to response received
    tier_label: str


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  GANTT CHART                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

TIER_COLORS = {
    "CRITICAL":   "#d62728",  # red
    "HIGH":       "#ff7f0e",  # orange
    "MEDIUM":     "#2ca02c",  # green
    "LOW":        "#1f77b4",  # blue
    "BACKGROUND": "#7f7f7f",  # gray
}


def save_gantt(
    results: list[Result],
    title: str,
    filepath: str,
) -> None:
    """Save a Gantt chart of request timings to disk."""
    # Sort: by tier order, then by end_time within tier
    tier_order = {
        "CRITICAL": 0, "HIGH": 1, "MEDIUM": 2,
        "LOW": 3, "BACKGROUND": 4,
    }
    sorted_results = sorted(
        results,
        key=lambda r: (tier_order.get(r.tier_label, 99), r.end_time),
    )

    n = len(sorted_results)
    fig_height = max(6, n * 0.12)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    y_labels = []
    for i, r in enumerate(sorted_results):
        color = TIER_COLORS.get(r.tier_label, "#333333")
        ax.barh(
            i, r.end_time - r.start_time,
            left=r.start_time, height=0.7,
            color=color, alpha=0.85, edgecolor="white",
            linewidth=0.3,
        )
        y_labels.append(r.name)

    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=4, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Time (seconds from batch start)")
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=TIER_COLORS[lb], label=lb)
        for lb in tier_order
    ]
    ax.legend(
        handles=legend_handles, loc="lower right",
        fontsize=8, framealpha=0.9,
    )

    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> Saved {filepath}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def main() -> None:
    # Single LLM — same config for every request
    llm = ChatNVIDIADynamo(
        base_url=NIM_BASE_URL,
        model=MODEL,
        max_completion_tokens=MAX_TOKENS,
    )

    total_tasks = sum(t["count"] for t in TIERS)
    tier_labels = [t["label"] for t in TIERS]

    print(f"Priority-only A/B test: {total_tasks} requests per trial")
    print(f"Fixed: max_tokens={MAX_TOKENS}, osl={OSL}, "
          f"latency_sensitivity={LATENCY_SENSITIVITY}, iat={IAT}")
    print(f"Query pool: {len(QUERIES)} unique (system, user) pairs")
    print()
    for t in TIERS:
        print(f"  {t['label']:12s}  priority={t['priority']:2d}  count={t['count']:3d}")
    print()

    # ── Build task list ──────────────────────────────────────────────────────
    # Each task: (name, tier_label, priority, messages)
    tasks_spec: list[tuple[str, str, int, list]] = []
    query_idx = 0
    for tier in TIERS:
        for i in range(tier["count"]):
            name = f"{tier['label'].lower()}_{i + 1:03d}"
            q = QUERIES[query_idx % len(QUERIES)]
            query_idx += 1
            msgs = [
                SystemMessage(content=q["system"]),
                HumanMessage(content=q["user"]),
            ]
            tasks_spec.append((name, tier["label"], tier["priority"], msgs))

    # ── Timed call — passes priority directly as a kwarg ─────────────────────
    async def timed_call(
        name: str,
        t0: float,
        tier_label: str,
        messages: list,
        priority: int = DEFAULT_PRIORITY,
    ) -> Result:
        start = time.perf_counter() - t0
        await llm.ainvoke(messages, osl=OSL, priority=priority)
        end = time.perf_counter() - t0
        return Result(
            name=name, start_time=start,
            end_time=end, tier_label=tier_label,
        )

    # ── Warmup ───────────────────────────────────────────────────────────────
    warmup_msgs = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain the concept of supply and demand in detail."),
    ]
    print(f"Warming up ({NUM_WARMUP_CALLS} calls, max_tokens={MAX_TOKENS})...")
    await asyncio.gather(
        *[llm.ainvoke(warmup_msgs, osl=OSL) for _ in range(NUM_WARMUP_CALLS)]
    )
    print("Warmup done.\n")

    # ── Output directory for Gantt charts ───────────────────────────────────
    out_dir = os.path.join(
        os.path.dirname(__file__), "priority_gantt_charts",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Gantt charts will be saved to: {out_dir}\n")

    # ── Helper to compute per-tier stats ─────────────────────────────────────
    def tier_stats(results: list[Result]) -> dict[str, dict[str, float]]:
        by_tier: dict[str, list[float]] = {lb: [] for lb in tier_labels}
        for r in results:
            by_tier[r.tier_label].append(r.end_time)
        out: dict[str, dict[str, float]] = {}
        for lb in tier_labels:
            times = by_tier[lb]
            if times:
                out[lb] = {
                    "median": statistics.median(times),
                    "mean": statistics.mean(times),
                    "min": min(times),
                    "max": max(times),
                }
        return out

    # ── Run both modes ───────────────────────────────────────────────────────
    mode_summaries: dict[str, dict[str, float]] = {}

    for mode_name, use_priority in [("A: CONTROL (all priority=1)", False),
                                     ("B: EXPERIMENT (tiered priority)", True)]:
        print("=" * 70)
        print(f"  {mode_name}")
        print("=" * 70)

        all_medians: dict[str, list[float]] = {lb: [] for lb in tier_labels}

        for run_idx in range(1, NUM_RUNS + 1):
            print(f"  Run {run_idx}/{NUM_RUNS} — {total_tasks} calls...",
                  end=" ", flush=True)
            t0 = time.perf_counter()

            coros = []
            for name, tier_label, pri, msgs in tasks_spec:
                p = pri if use_priority else DEFAULT_PRIORITY
                coros.append(timed_call(name, t0, tier_label, msgs, priority=p))

            results = list(await asyncio.gather(*coros))
            wall = max(r.end_time for r in results)
            print(f"done in {wall:.1f}s")

            stats = tier_stats(results)
            for lb in tier_labels:
                s = stats[lb]
                all_medians[lb].append(s["median"])
                n = sum(1 for r in results if r.tier_label == lb)
                print(f"    {lb:12s}: med={s['median']:6.1f}s  "
                      f"mean={s['mean']:6.1f}s  "
                      f"min={s['min']:5.1f}s  max={s['max']:5.1f}s  "
                      f"[n={n}]")

            # Save Gantt chart
            mode_tag = "control" if not use_priority else "experiment"
            chart_path = os.path.join(
                out_dir, f"{mode_tag}_run{run_idx:02d}.png",
            )
            save_gantt(
                results,
                title=(
                    f"{mode_name} - Run {run_idx}/{NUM_RUNS} "
                    f"({total_tasks} requests, {wall:.1f}s wall)"
                ),
                filepath=chart_path,
            )
            print()

        # Median-of-medians across runs
        final = {lb: statistics.median(all_medians[lb]) for lb in tier_labels}
        mode_summaries[mode_name] = final

        print(f"  MEDIAN OF MEDIANS ({mode_name}):")
        for lb in tier_labels:
            print(f"    {lb:12s}: {final[lb]:6.1f}s")
        print()

    # ── Side-by-side comparison ──────────────────────────────────────────────
    control_key = "A: CONTROL (all priority=1)"
    experiment_key = "B: EXPERIMENT (tiered priority)"
    ctrl = mode_summaries[control_key]
    expt = mode_summaries[experiment_key]

    print("=" * 70)
    print("  SIDE-BY-SIDE COMPARISON (median of medians)")
    print("=" * 70)
    print()
    print(f"  {'Tier':12s}  {'Control':>9s}  {'Experiment':>11s}  {'Delta':>8s}")
    print(f"  {'~' * 12}  {'~' * 9}  {'~' * 11}  {'~' * 8}")
    for lb in tier_labels:
        delta = expt[lb] - ctrl[lb]
        print(f"  {lb:12s}  {ctrl[lb]:>8.1f}s  {expt[lb]:>10.1f}s  {delta:>+7.1f}s")

    print()
    print("INTERPRETATION:")
    print("  Negative CRITICAL delta => priority IS speeding up critical tasks")
    print("  Positive BACKGROUND delta => priority IS deprioritizing background")
    print("  All deltas ~0 => priority has NO scheduling effect")


if __name__ == "__main__":
    asyncio.run(main())
