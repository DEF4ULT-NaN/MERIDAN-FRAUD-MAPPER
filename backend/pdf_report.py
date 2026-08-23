"""
Phase D — PDF fraud report generator.

Renders a 1–2 page exec-readable summary using reportlab.  Includes:
  - title + timestamp
  - summary statistics (n_nodes, n_edges, n_fraud_predicted, n_rings)
  - per-ring cluster summaries
  - top-10 highest-risk accounts with their top reasons
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_pdf(graph_payload: dict) -> bytes:
    """Return a PDF as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Fraud Network Report",
    )
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    small = ParagraphStyle(
        "small", parent=body, fontSize=8, leading=10, textColor=colors.grey
    )

    story: list[Any] = []

    # --- Title ---
    story.append(Paragraph("Fraud Network Analysis Report", h1))
    story.append(Paragraph(
        f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        small,
    ))
    story.append(Spacer(1, 0.2 * inch))

    # --- Summary ---
    stats = graph_payload.get("stats", {})
    scoring = graph_payload.get("scoring", {})
    method = scoring.get("method", "rule_based")

    story.append(Paragraph("Summary", h2))
    n_fraud_pred = sum(
        1 for n in graph_payload.get("nodes", []) if n.get("risk_score", 0) >= 0.5
    )
    summary_rows = [
        ["Total accounts", stats.get("n_nodes", 0)],
        ["Total shared-attribute edges", stats.get("n_edges", 0)],
        ["Planted fraud rings", stats.get("n_rings", 0)],
        ["Accounts flagged (risk >= 50%)", n_fraud_pred],
        ["Scoring method", method],
    ]
    t = Table(summary_rows, colWidths=[2.4 * inch, 3.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f4f8")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.25 * inch))

    # --- Per-ring cluster summaries ---
    rings = graph_payload.get("rings", [])
    if rings:
        story.append(Paragraph("Detected Fraud Clusters", h2))
        rows = [["Ring", "Size", "Tightness", "Avg risk"]]
        nodes_by_id = {n["id"]: n for n in graph_payload.get("nodes", [])}
        for r in rings:
            members = [nodes_by_id[m] for m in r.get("account_ids", [])
                       if m in nodes_by_id]
            if not members:
                continue
            avg = sum(float(n.get("risk_score", 0.0)) for n in members) / len(members)
            rows.append([
                r.get("ring_id", ""),
                str(len(members)),
                r.get("tightness", ""),
                f"{avg*100:.1f}%",
            ])
        t = Table(rows, colWidths=[1.2 * inch, 0.8 * inch, 1.2 * inch, 1.0 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.25 * inch))

    # --- Top-10 highest-risk accounts ---
    story.append(Paragraph("Top 10 Highest-Risk Accounts", h2))
    nodes_sorted = sorted(
        graph_payload.get("nodes", []),
        key=lambda n: -float(n.get("risk_score", 0.0)),
    )[:10]

    # Build reason lookup
    reason_lookup: dict[str, list[str]] = {n["id"]: [] for n in graph_payload.get("nodes", [])}
    for e in graph_payload.get("edges", []):
        reason_lookup.setdefault(e["source"], []).extend(e.get("reasons", [])[:2])
        reason_lookup.setdefault(e["target"], []).extend(e.get("reasons", [])[:2])

    rows = [["ID", "Name", "Risk", "Top reason"]]
    cell_style = ParagraphStyle("cell", parent=body, fontSize=8, leading=10)
    for n in nodes_sorted:
        top_reason = (reason_lookup.get(n["id"], ["(no shared attributes)"]) or ["(none)"])[0]
        rows.append([
            n["id"],
            Paragraph(str(n.get("name", ""))[:30], cell_style),
            f"{float(n.get('risk_score', 0))*100:.1f}%",
            Paragraph(str(top_reason), cell_style),
        ])
    t = Table(rows, colWidths=[1.0 * inch, 1.3 * inch, 0.7 * inch, 2.6 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)

    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph(
        "Risk scores reflect graph-derived signals (shared IPs, devices, "
        "phone numbers, bank accounts) and per-account behavioural "
        "features.  See the interactive dashboard for the full graph "
        "and per-node explanations.",
        small,
    ))

    doc.build(story)
    return buf.getvalue()
