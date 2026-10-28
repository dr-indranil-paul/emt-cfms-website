"""
PlastInMap knowledge-graph query and hypothesis generation module.

Loads pre-computed KG tables exported from the ChatEMT pipeline
(see README.md → "KG Lineage" for full data provenance).
Supports OpenCode, OpenRouter, and OpenAI as LLM backends.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
import pandas as pd

try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
PROJECT_ROOT = HERE.parent
EXPLORATORY_TABLES = HERE / "data" / "tables"
CROSS_ASSEMBLY_TABLES = EXPLORATORY_TABLES
KG_TABLES = EXPLORATORY_TABLES / "emt_cfms_exploratory_analysis"
MANUSCRIPT_TABLES = PROJECT_ROOT / "04_manuscripts" / "01_docs_and_tables"
CLUSTERONE_TABLES = EXPLORATORY_TABLES
GOCC_TABLES = EXPLORATORY_TABLES

# ── Neo4j credentials ──────────────────────────────────────────────────────────
LIT_KG_URI = os.getenv("LIT_KG_NEO4J_URI", "bolt://localhost:7688")
LIT_KG_USER = os.getenv("LIT_KG_NEO4J_USER", "neo4j")
LIT_KG_PWD = os.getenv("LIT_KG_NEO4J_PWD", "password")

# ── Provider / model catalogue (consumed by the Streamlit UI) ─────────────────
PROVIDER_MODELS: dict[str, list[str]] = {
    "OpenCode": [
        "mimo-v2.5-free",
        "north-mini-code-free",
        "nemotron-3-ultra-free",
        "deepseek-v4-flash-free",
        "big-pickle",
        "claude-sonnet-4-6",
        "claude-opus-4-8",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "kimi-k2.6",
        "qwen3.6-plus",
    ],
    "OpenRouter": [
        "meta-llama/llama-3.3-70b-instruct:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "qwen/qwen3-coder:free",
        "openai/gpt-oss-120b:free",
        "google/gemma-4-31b-it:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
    ],
    "OpenAI": [
        "gpt-6.1-terra",
        "gpt-5.6-terra",
        "gpt-5.6-terra-pro",
        "gpt-5.6-sol",
        "gpt-5.6-sol-pro",
        "gpt-5.6-luna",
        "gpt-5.6-luna-pro",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.2",
        "gpt-4.1",
        "gpt-4.1-mini",
    ],
}

_OR_SKIP_KEYWORDS = (
    "content-safety",
    "clip",
    "vision",
    "vl",
    "nano-9b",
    "nano-12b",
    "1.2b",
    "code",
    "safety",
    "dolphin",
    "laguna",
)


def refresh_openrouter_free_models() -> None:
    """Fetch live free models from OpenRouter, filter to general-purpose ones, sort by context size."""
    try:
        import urllib.request, json as _json

        with urllib.request.urlopen(
            "https://openrouter.ai/api/v1/models", timeout=8
        ) as r:
            data = _json.loads(r.read())
        free = [
            m
            for m in data["data"]
            if str(m.get("pricing", {}).get("prompt", "1")) == "0"
            and str(m.get("pricing", {}).get("completion", "1")) == "0"
            and m["id"].endswith(":free")
            and not any(kw in m["id"].lower() for kw in _OR_SKIP_KEYWORDS)
        ]
        # Sort by context length descending — larger context = more capable
        free.sort(key=lambda m: m.get("context_length", 0), reverse=True)
        ids = [m["id"] for m in free]
        if ids:
            PROVIDER_MODELS["OpenRouter"] = ids
    except Exception:
        pass  # silently keep the hardcoded fallback list


# Attempt live refresh at import time (non-blocking; fails silently on no network)
refresh_openrouter_free_models()

# Env-var name each provider reads its key from
PROVIDER_ENV_KEY: dict[str, str] = {
    "OpenCode": "OPENCODE_API_KEY",
    "OpenRouter": "OPENROUTER_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
}


# ── Pre-computed table loaders ─────────────────────────────────────────────────


def _load_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    paths = {
        "lit_edges": CROSS_ASSEMBLY_TABLES / "all_lit_causal_edges.csv",
        "asm_edges": CROSS_ASSEMBLY_TABLES / "cross_assembly_lit_graph.csv",
        "reactome_asm": CROSS_ASSEMBLY_TABLES / "reactome_cross_assembly_graph.csv",
        "asm_degree": CROSS_ASSEMBLY_TABLES / "assembly_causal_degree.csv",
        "hub_genes": EXPLORATORY_TABLES / "cao_probe/hub_ranking.csv",
        "ih_gates": EXPLORATORY_TABLES / "cao_probe/ih_and_gate_scores.csv",
        "patterns": CLUSTERONE_TABLES / "clusterone_pattern_memberships.csv",
        "gocc": GOCC_TABLES / "clusterone_gocc_annotated_full_reg.csv",
        "combined_gene_edges": KG_TABLES / "combined_kg_gene_edges.csv",
        "lit_ie_all": KG_TABLES / "lit_kg_all_edges.csv",
        "extracted_gene_links": KG_TABLES / "lit_kg_extracted_gene_links.csv",
    }
    for key, path in paths.items():
        if path.exists():
            tables[key] = pd.read_csv(path)

    names_path = HERE / "data/assembly_data/complex_names.csv"
    if names_path.exists():
        tables["names"] = pd.read_csv(names_path).rename(
            columns={"complex_name": "complex_id"}
        )

    s5_path = MANUSCRIPT_TABLES / "Table S4. ClusterOne Protein Assemblies.xlsx"
    if s5_path.exists():
        tables["s5"] = pd.read_excel(
            s5_path, sheet_name="ClusterOne_Protein_Assemblies"
        ).rename(columns={"complex_name": "complex_id"})

    return tables


def _tables_singleton() -> dict[str, pd.DataFrame]:
    if not hasattr(_tables_singleton, "_cache"):
        _tables_singleton._cache = _load_tables()
    return _tables_singleton._cache


# ── Lookup helpers ─────────────────────────────────────────────────────────────


def find_assemblies_for_gene(gene: str) -> list[dict]:
    t = _tables_singleton()
    results = []
    if "s5" not in t:
        return results
    s5 = t["s5"]
    mask = s5["component_genes"].str.contains(
        r"\b" + re.escape(gene) + r"\b", case=False, na=False, regex=True
    )
    for _, row in s5[mask].iterrows():
        cid = row["complex_id"]
        name = "(unassigned)"
        if "names" in t:
            n_row = t["names"][t["names"]["complex_id"] == cid]
            if not n_row.empty:
                name = n_row.iloc[0].get("assigned_name", "(unassigned)")
        pattern = "unclassified"
        if "patterns" in t:
            p_row = t["patterns"][t["patterns"]["complex_name"] == cid]
            if not p_row.empty:
                pattern = p_row.iloc[0].get("pattern", "unclassified")
        results.append({"complex_id": cid, "assigned_name": name, "pattern": pattern})
    return results


def find_lit_edges_for_gene(gene: str, max_edges: int = 20) -> pd.DataFrame:
    t = _tables_singleton()
    if "lit_edges" not in t:
        return pd.DataFrame()
    df = t["lit_edges"]
    mask = df["src_display"].str.contains(gene, case=False, na=False) | df[
        "tgt_display"
    ].str.contains(gene, case=False, na=False)
    return df[mask].head(max_edges)


def find_assembly_edges(complex_ids: list[str], max_edges: int = 20) -> pd.DataFrame:
    t = _tables_singleton()
    if "asm_edges" not in t:
        return pd.DataFrame()
    df = t["asm_edges"]
    mask = df["src_assembly"].isin(complex_ids) | df["tgt_assembly"].isin(complex_ids)
    return df[mask].head(max_edges)


def find_combined_gene_edges(gene: str, max_edges: int = 10) -> pd.DataFrame:
    """Query the unified Lit + Reactome + PathwayCommons combined gene->gene graph."""
    t = _tables_singleton()
    if "combined_gene_edges" not in t:
        return pd.DataFrame()
    df = t["combined_gene_edges"]
    mask = df["src_gene"].str.contains(gene, case=False, na=False) | df[
        "tgt_gene"
    ].str.contains(gene, case=False, na=False)
    return df[mask].head(max_edges)


def find_gene_term_associations(gene: str, max_terms: int = 5) -> list[dict]:
    """Find ExtractedTerm, Pathway, GO BP associations for a gene from the full Lit IE table."""
    t = _tables_singleton()
    results: list[dict] = []

    if "lit_ie_all" in t:
        ie = t["lit_ie_all"]
        # Gene as source -> term/pathway as target
        src_mask = ie["src_display"].str.contains(gene, case=False, na=False)
        for _, row in ie[src_mask].head(max_terms * 3).iterrows():
            tgt_label = str(row.get("tgt_label", ""))
            tgt_disp = str(row.get("tgt_display", ""))
            if (
                tgt_label not in ("ReferenceGeneProduct", "nan", "", "None")
                and tgt_disp
            ):
                results.append(
                    {
                        "entity": tgt_disp,
                        "type": tgt_label,
                        "rel": row.get("rel_type", ""),
                        "evidence": row.get("evidence_count", 1),
                        "direction": "gene->entity",
                    }
                )
        # Term/pathway as source -> gene as target
        tgt_mask = ie["tgt_display"].str.contains(gene, case=False, na=False)
        for _, row in ie[tgt_mask].head(max_terms * 3).iterrows():
            src_label = str(row.get("src_label", ""))
            src_disp = str(row.get("src_display", ""))
            if (
                src_label not in ("ReferenceGeneProduct", "nan", "", "None")
                and src_disp
            ):
                results.append(
                    {
                        "entity": src_disp,
                        "type": src_label,
                        "rel": row.get("rel_type", ""),
                        "evidence": row.get("evidence_count", 1),
                        "direction": "entity->gene",
                    }
                )

    if "extracted_gene_links" in t:
        eg = t["extracted_gene_links"]
        eg_mask = eg["gene_display"].str.contains(gene, case=False, na=False)
        for _, row in eg[eg_mask].head(max_terms).iterrows():
            results.append(
                {
                    "entity": row.get("term_name", ""),
                    "type": "ExtractedTerm",
                    "rel": "extractedGene",
                    "evidence": 1,
                    "direction": "term->gene",
                }
            )

    # Deduplicate by entity
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in results:
        key = r["entity"] + "|" + r["type"]
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped[:max_terms]


def get_assembly_profile(complex_id: str) -> dict:
    t = _tables_singleton()
    profile = {
        "complex_id": complex_id,
        "assigned_name": "(unassigned)",
        "pattern": "unclassified",
        "go_category": "",
        "go_term": "",
        "causal_out": 0,
        "causal_in": 0,
        "net_flow": 0,
        "members": [],
    }

    if "names" in t:
        n = t["names"][t["names"]["complex_id"] == complex_id]
        if not n.empty:
            profile["assigned_name"] = n.iloc[0].get("assigned_name", "(unassigned)")

    if "patterns" in t:
        p = t["patterns"][t["patterns"]["complex_name"] == complex_id]
        if not p.empty:
            profile["pattern"] = p.iloc[0].get("pattern", "unclassified")

    if "gocc" in t:
        g = (
            t["gocc"][t["gocc"]["complex_name"] == complex_id]
            .sort_values("p.adjust.log10", ascending=False)
            .drop_duplicates("last_go_term")
        )
        if not g.empty:
            profile["go_category"] = g.iloc[0].get("my_level1", "")
            profile["go_term"] = g.iloc[0].get("last_go_term", "")

    if "asm_degree" in t:
        d = t["asm_degree"][t["asm_degree"]["assembly"] == complex_id]
        if not d.empty:
            profile["causal_out"] = int(d.iloc[0].get("out_degree", 0) or 0)
            profile["causal_in"] = int(d.iloc[0].get("in_degree", 0) or 0)
            profile["net_flow"] = float(d.iloc[0].get("net_flow", 0) or 0)

    if "s5" in t:
        s = t["s5"][t["s5"]["complex_id"] == complex_id]
        if not s.empty:
            genes_str = str(s.iloc[0].get("component_genes", ""))
            profile["members"] = [g.strip() for g in genes_str.split(",") if g.strip()]

    return profile


# ── Optional live Neo4j ────────────────────────────────────────────────────────


def _neo4j_driver(uri: str, user: str, pwd: str):
    if not NEO4J_AVAILABLE:
        return None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        driver.verify_connectivity()
        return driver
    except Exception:
        return None


def query_lit_kg_for_gene(gene: str, max_results: int = 15) -> list[dict]:
    driver = _neo4j_driver(LIT_KG_URI, LIT_KG_USER, LIT_KG_PWD)
    if driver is None:
        return []
    cypher = """
    MATCH (a)-[r]->(b)
    WHERE (a.displayName CONTAINS $gene OR b.displayName CONTAINS $gene)
      AND type(r) IN ['POSITIVE_CAUSAL','NEGATIVE_CAUSAL','positive_causal','negative_causal',
                       'CAUSAL_POSITIVE','CAUSAL_NEGATIVE']
    RETURN a.displayName AS src, type(r) AS rel, b.displayName AS tgt,
           r.evidence_count AS evidence, r.contradicting AS contradicting
    LIMIT $limit
    """
    results = []
    try:
        with driver.session() as session:
            for rec in session.run(cypher, gene=gene, limit=max_results):
                results.append(dict(rec))
    except Exception:
        pass
    finally:
        driver.close()
    return results


# ── Context builder ────────────────────────────────────────────────────────────


def build_context(query: str) -> str:
    t = _tables_singleton()

    gene_tokens = re.findall(r"\b([A-Z][A-Z0-9]{1,9})\b", query)
    cx_tokens = [c.upper() for c in re.findall(r"\b(C\d{4})\b", query, re.IGNORECASE)]

    sections: list[str] = []

    all_cx_ids: set[str] = set(cx_tokens)
    for gene in gene_tokens:
        for asm in find_assemblies_for_gene(gene):
            all_cx_ids.add(asm["complex_id"])

    if all_cx_ids:
        sections.append("=== Assemblies containing queried genes ===")
        for cid in sorted(all_cx_ids)[:8]:
            p = get_assembly_profile(cid)
            top_members = ", ".join(p["members"][:12]) + (
                "…" if len(p["members"]) > 12 else ""
            )
            sections.append(
                f"• {cid} ({p['assigned_name']}) | Program: {p['pattern']} | "
                f"GO: {p['go_category']} – {p['go_term']} | "
                f"Causal out={p['causal_out']} in={p['causal_in']} net={p['net_flow']:+.0f}\n"
                f"  Members (sample): {top_members}"
            )

        for gene in gene_tokens[:4]:
            lit = find_lit_edges_for_gene(gene, max_edges=10)
            if not lit.empty:
                sections.append(f"\n=== Literature causal edges for {gene} ===")
                for _, row in lit.iterrows():
                    direction = (
                        "\u2192"
                        if "positive" in str(row.get("rel_type", "")).lower()
                        else "\u22a3"
                    )
                    sections.append(
                        f"  {row.get('src_display', '')} {direction} {row.get('tgt_display', '')} "
                        f"(evidence={row.get('evidence_count', 1)}, "
                        f"contradicting={row.get('contradicting', False)})"
                    )

            # Combined KG (Lit + Reactome) gene-level edges
            combined = find_combined_gene_edges(gene, max_edges=8)
            if not combined.empty:
                sections.append(
                    f"\n=== Combined KG (Lit + Reactome + PathwayCommons) edges for {gene} ==="
                )
                for _, row in combined.iterrows():
                    src = row.get("src_gene", "")
                    tgt = row.get("tgt_gene", "")
                    rel = row.get("rel_type", "")
                    src_kg = row.get("source", "")
                    sections.append(f"  {src} \u2192 {tgt} ({rel}, source: {src_kg})")

            # Term/pathway associations
            terms = find_gene_term_associations(gene, max_terms=5)
            if terms:
                sections.append(f"\n=== Term/pathway associations for {gene} ===")
                for tr in terms:
                    direction_arrow = (
                        "\u2192" if tr["direction"] == "gene->entity" else "\u2190"
                    )
                    sections.append(
                        f"  {tr['entity']} ({tr['type']}) {direction_arrow} "
                        f"rel: {tr['rel']}, evidence: {tr['evidence']}"
                    )

    if all_cx_ids:
        asm_edges = find_assembly_edges(list(all_cx_ids), max_edges=12)
        if not asm_edges.empty:
            names_map = {}
            if "names" in t:
                names_map = dict(
                    zip(t["names"]["complex_id"], t["names"]["assigned_name"])
                )
            sections.append("\n=== Cross-assembly causal edges ===")
            for _, row in asm_edges.iterrows():
                src_n = names_map.get(
                    row.get("src_assembly", ""), row.get("src_assembly", "")
                )
                tgt_n = names_map.get(
                    row.get("tgt_assembly", ""), row.get("tgt_assembly", "")
                )
                direction = (
                    "→" if "positive" in str(row.get("rel_type", "")).lower() else "⊣"
                )
                sections.append(
                    f"  {row.get('src_assembly', '')} [{src_n}] {direction} "
                    f"{row.get('tgt_assembly', '')} [{tgt_n}] "
                    f"via {row.get('src_gene', '')} "
                    f"(prog: {row.get('src_program', '')} → {row.get('tgt_program', '')})"
                )

    live_results = []
    for gene in gene_tokens[:2]:
        live_results.extend(query_lit_kg_for_gene(gene, max_results=8))
    if live_results:
        sections.append("\n=== Live knowledge graph relationships ===")
        for r in live_results[:12]:
            direction = "→" if "positive" in str(r.get("rel", "")).lower() else "⊣"
            sections.append(
                f"  {r.get('src', '')} {direction} {r.get('tgt', '')} "
                f"(evidence={r.get('evidence', '')})"
            )

    if "ih_gates" in t and all_cx_ids:
        ih_sub = t["ih_gates"][t["ih_gates"]["assembly"].isin(all_cx_ids)]
        if not ih_sub.empty:
            sections.append("\n=== IH hybrid AND-gate scores ===")
            for _, row in ih_sub.iterrows():
                sections.append(
                    f"  {row['assembly']}: EL inputs={row.get('n_el_input_edges', 0)}, "
                    f"ME inputs={row.get('n_me_input_edges', 0)}, "
                    f"AND-gate score={row.get('and_gate_score', 0):.3f}"
                )

    if not sections:
        return (
            "No specific assembly or causal context found for this query "
            "in the PlastInMap knowledge base."
        )
    return "\n".join(sections)


# ── System prompt (shared across all providers) ────────────────────────────────

SYSTEM_PROMPT = """\
You are a computational biologist specializing in epithelial–mesenchymal plasticity, macromolecular \
assemblies, and Barrett's esophagus progression. You have access to structured data from the EMT-InMap study — \
a native-state assembly atlas of 611 protein complexes, 9,503 co-complex associations, 537 high-confidence \
interface models, and 44,945 directed causal/regulatory edges across Epithelial Legacy (EL), Intermediate Hybrid (IH), \
and Mesenchymal Emerging (ME) assembly programs, derived from co-fractionation mass spectrometry \
of TGF-β–induced EMT in MCF10A cells, transferred via Blueprint–Projection to deep patient proteomes (27 patients). \
The study also includes spatial discovery imaging and an independent outcome-annotated cohort of 18 patients. \
CP-A is a supporting epithelial model whose released interaction data are available in Browse Data and Table S3; \
the assembly and causal context retrieved here remains the MCF10A reference. Do not infer CP-A-specific \
assemblies, structural interfaces or causal mechanisms from MCF10A context, or claim a validated progression predictor. \
The context also includes the CAO (Causal Assembly Ontology) \
combined knowledge graph: 87,475 unified gene→gene edges merging Lit KG (literature-mined causal, \
3,570 edges) with Reactome (reaction-ordering, 10,489 edges) and PathwayCommons (73,416 edges \
from 9+ curated databases), plus extracted gene→term associations.

EMT program definitions:
- EL (Epithelial Legacy): assemblies enriched in epithelial cells (Day 0), down-regulated during EMT
- IH (Intermediate Hybrid): assemblies peaking at transitional states (Days 2–6), chimeric
- ME (Mesenchymal Emerging): assemblies gained in mesenchymal state (Day 8)

When generating hypotheses:
1. Ground your reasoning in the specific assemblies, genes, and causal edges provided.
2. Propose mechanistic hypotheses about how assemblies drive or restrain state transitions.
3. Note causal directionality (positive = activation/induction; negative = inhibition).
4. Highlight the hybrid (IH) program when relevant — the paper's key finding is that IH \
is architecturally distinct, not merely transitional.
5. Be concise and scientific. Reference specific assembly IDs (C0XXX) and gene names.
6. Include one **Mermaid flowchart** (```mermaid ... ```) illustrating the proposed mechanism.
    Start with `flowchart TD` and use 4–8 short, biologically specific nodes.
    Use exactly one relationship per line. Activation syntax: `A[Signal] -->|activates| B[Target]`.
    Inhibition syntax: `B[Target] -.->|inhibits| C[Response]`.
    Use unique single-letter node IDs, plain-text labels, and valid paired brackets.
    Do not use HTML, subgraphs, styling directives, chained arrows, or unlabeled `-|` operators.
    Prefer a clean branching pathway that converges on a phenotypic outcome.
7. End with 1–2 testable predictions or follow-up experiments.
"""


# ── LLM dispatch ──────────────────────────────────────────────────────────────


def _call_openai_compatible(
    key: str,
    model: str,
    user_msg: str,
    max_tokens: Optional[int] = None,
    base_url: str = "https://api.openai.com/v1",
    extra_headers: Optional[dict] = None,
) -> str:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    client = OpenAI(api_key=key, base_url=base_url, default_headers=extra_headers or {})
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ── Public entry point ─────────────────────────────────────────────────────────


def generate_hypothesis(
    query: str,
    api_key: Optional[str] = None,
    model: str = "mimo-v2.5-free",
    provider: str = "OpenCode",
    max_tokens: Optional[int] = None,
) -> tuple[str, str]:
    """
    Generate an EMT assembly hypothesis for the given query.

    provider: "OpenCode" | "OpenRouter" | "OpenAI"
    Returns (hypothesis_text, context_used).
    Raises RuntimeError on missing key or unavailable package.
    """
    env_var = PROVIDER_ENV_KEY.get(provider, "OPENCODE_API_KEY")
    key = api_key or os.getenv(env_var)
    if not key:
        raise RuntimeError(
            f"No API key provided for {provider}. "
            f"Add {env_var} to the .env file or enter it in the UI."
        )

    context = build_context(query)
    user_message = (
        f"PlastInMap Knowledge Base Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Based on the above PlastInMap assembly context, generate a focused mechanistic "
        "hypothesis addressing this query. Reference specific assemblies and causal edges."
    )

    if provider == "OpenCode":
        hypothesis = _call_openai_compatible(
            key,
            model,
            user_message,
            max_tokens,
            base_url="https://opencode.ai/zen/v1",
            extra_headers={
                "HTTP-Referer": "https://emt-inmap.ohsu.edu",
                "X-Title": "PlastInMap Hypothesis Generator",
            },
        )

    elif provider == "OpenRouter":
        hypothesis = _call_openai_compatible(
            key,
            model,
            user_message,
            max_tokens,
            base_url="https://openrouter.ai/api/v1",
            extra_headers={
                "HTTP-Referer": "https://emt-inmap.ohsu.edu",
                "X-Title": "PlastInMap Hypothesis Generator",
            },
        )

    elif provider == "OpenAI":
        hypothesis = _call_openai_compatible(
            key,
            model,
            user_message,
            max_tokens,
            base_url="https://api.openai.com/v1",
        )

    else:
        choices = ", ".join(repr(name) for name in PROVIDER_MODELS)
        raise RuntimeError(f"Unknown provider: {provider!r}. Choose {choices}.")

    return hypothesis, context
