import os
import sys
import tempfile
import re
from html import escape
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv
import markdown as md_lib

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from flow_diagrams import mermaid_to_plotly as _styled_mermaid_to_plotly, mermaid_to_svg

try:
    from pyvis.network import Network

    PYVIS_OK = True
except ImportError:
    PYVIS_OK = False

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = HERE.parent
EXPLORATORY_TABLES = HERE / "data" / "tables"
LEGACY_TABLES = EXPLORATORY_TABLES
CROSS_ASSEMBLY_TABLES = EXPLORATORY_TABLES
CLUSTERONE_TABLES = EXPLORATORY_TABLES
GOCC_TABLES = EXPLORATORY_TABLES
MANUSCRIPT_TABLES = HERE / "data" / "manuscript_tables"

# Intentional application configuration: this app-local file stores only
# user-submitted LLM-provider keys and is never an analysis-output destination.
USER_CONFIGURATION_ENV = HERE / ".env"
load_dotenv(USER_CONFIGURATION_ENV)

try:
    import kg_tools

    KG_TOOLS_OK = True
except ImportError as _kg_err:
    KG_TOOLS_OK = False
    _KG_ERR_MSG = str(_kg_err)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlastInMap",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy:   #0f1b2d;
    --navy2:  #1e3a5f;
    --accent: #1d4ed8;
    --up:     #dc2626;
    --mid:    #2563eb;
    --down:   #16a34a;
    --g50:    #f8fafc;
    --g100:   #f1f5f9;
    --g200:   #e2e8f0;
    --g500:   #64748b;
    --g700:   #334155;
    --white:  #ffffff;
}

/* Pin sidebar open permanently — collapse button removed */
[data-testid="stSidebar"] {
    transform: none !important;
    min-width: 21rem !important;
    visibility: visible !important;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] { display: none !important; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--g50) !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--g700) !important;
}
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--g200) !important;
}
[data-testid="stHeader"] { display: none !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f1b2d 0%, #1e3a8a 100%);
    padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
}
.hero h1 { font-size: 1.65rem; font-weight: 800; color: #ffffff; margin: 0 0 .4rem; line-height: 1.25; }
.hero p  { font-size: .88rem; color: #bfdbfe; margin: 0; line-height: 1.5; }
.hero-sub { display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; }
.hero-chip {
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
    border-radius: 20px; padding: .25rem .8rem; font-size: .75rem;
    color: #e0f2fe; font-weight: 500;
}

/* Stat row */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 120px;
    background: var(--white); border: 1px solid var(--g200); border-radius: 8px;
    padding: 1.1rem 1rem; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
.stat-n { font-size: 1.8rem; font-weight: 800; color: var(--navy); line-height: 1; }
.stat-l { font-size: .7rem; color: var(--g500); margin-top: .2rem;
          text-transform: uppercase; letter-spacing: .05em; }

/* Cards */
.card {
    background: var(--white); border: 1px solid var(--g200); border-radius: 8px;
    padding: 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.03);
    box-sizing: border-box;
}
/* cards inside a CSS grid row stretch to the tallest sibling */
div[style*="grid-template-columns"] > .card { height: 100%; margin-bottom: 0; }
.card h3 { color: var(--navy); margin: 0 0 .75rem; font-size: 1rem; font-weight: 700; }
.card p  { font-size: .88rem; line-height: 1.65; color: var(--g700); margin: 0; }

/* Global: justify all prose, never break words */
p, li, .stMarkdown p, .stMarkdown li,
.card p, .card li, div[class*="stMarkdown"] p {
    text-align: justify !important;
    hyphens: none !important;
    overflow-wrap: normal !important;
    word-break: normal !important;
}

/* Badges */
.badge { display: inline-block; padding: 1px 8px; border-radius: 12px;
         font-size: .7rem; font-weight: 600; margin: 1px; }
.badge-up   { background: #fee2e2; color: #991b1b; }
.badge-mid  { background: #dbeafe; color: #1e40af; }
.badge-down { background: #dcfce7; color: #166534; }
.badge-unc  { background: var(--g100); color: var(--g500); }

/* (removed sidebar legend)


/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: var(--g100); border-radius: 8px; padding: 4px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px; font-size: .83rem; padding: .4rem 1.1rem;
    color: var(--g700) !important; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: var(--white) !important; font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.07) !important;
}

/* Citation block */
.cite-block {
    background: var(--g100); border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0; padding: .8rem 1rem;
    font-size: .82rem; color: var(--g700); line-height: 1.6;
}

/* Section subtitle */
.section-sub { font-size: .82rem; color: var(--g500); margin-bottom: 1rem; }

/* Download row */
.dl-row {
    display: flex; align-items: center; gap: 1rem;
    padding: .9rem 0; border-bottom: 1px solid var(--g200);
}
.dl-meta { flex: 1; }
.dl-title { font-size: .9rem; font-weight: 600; color: var(--navy); }
.dl-desc  { font-size: .78rem; color: var(--g500); margin-top: .15rem; }

/* Hypothesis output card */
.hyp-card { background:#fff; border:1px solid #e2e8f0; border-left:4px solid #1d4ed8;
            border-radius:8px; padding:1.6rem 1.8rem; margin-top:.5rem; }
.hyp-card h1,.hyp-card h2,.hyp-card h3 { color:#0f1b2d; margin-top:.85rem; margin-bottom:.3rem; line-height:1.35; }
.hyp-card h1,.hyp-card h2,.hyp-card h3 { font-size:1rem; font-weight:700; }
.hyp-card p  { color:#334155; font-size:.79rem; line-height:1.58; margin-bottom:.5rem; }
.hyp-card ul,.hyp-card ol { padding-left:1.25rem; color:#334155; font-size:.79rem; line-height:1.58; }
.hyp-card li { margin-bottom:.22rem; }
.hyp-card strong { color:#0f1b2d; }
.hyp-card code  { background:#f1f5f9; padding:1px 4px; border-radius:3px;
                  font-size:.72rem; color:#1d4ed8; }

/* ── Footer ── */
.app-footer {
    position: fixed; bottom: 0; left: 21rem; width: calc(100% - 21rem);
    background: #f1f5f9; border-top: 1px solid #e2e8f0; color: #94a3b8;
    font-size: .72rem;
    padding: .45rem 1rem;
    letter-spacing: .03em;
    z-index: 100;
    text-align: center !important;
}
.app-footer, .app-footer p, .app-footer * {
    text-align: center !important;
    hyphens: none !important;
}
.app-footer strong { color: #64748b; }
/* push content up so the footer doesn't overlap it */
section.main > div { padding-bottom: 2.5rem; }
</style>
""",
    unsafe_allow_html=True,
)


st.html(
    '<div class="app-footer">'
    "<strong>OHSU Knight Cancer Institute</strong>"
    " &nbsp;·&nbsp; Oregon Health &amp; Science University"
    "</div>"
)

# ── Data loaders ───────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_manuscript_content():
    text = (MANUSCRIPT_TABLES / "EMT_CFMS_paper.txt").read_text(encoding="utf-8-sig")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = next(line for line in lines if line != "FIGURES")
    abstract = (
        "Barrett’s esophagus progresses through epithelial plasticity toward adenocarcinoma, yet how "
        "transitional states are physically organized at the molecular level remains unknown. "
        "Patient-matched proteomics revealed persistent coexistence of epithelial and mesenchymal "
        "programs, motivating time-resolved mapping of native protein assemblies. Intermediate-hybrid "
        "states formed a distinct chimeric architecture rather than an endpoint mixture, integrating "
        "migratory, trafficking, metabolic and stress-adaptive machinery and exposing vulnerabilities "
        "invisible to protein abundance alone. Projection of these assembly programs into patient "
        "tissue revealed engagement across the metaplasia–dysplasia–carcinoma trajectory and recurrence "
        "across epithelial cancers, with disease-associated remodeling converging on intracellular "
        "trafficking. In biopsied lesions, complexome-nominated protein interactions localized to "
        "immune-depleted and immunoregulatory niches, with endolysosomal assemblies showing higher "
        "median tissue signals in patients who subsequently progressed. Perturbation of nominated "
        "trafficking regulators altered T-cell activation and effector profiles. Dynamic protein-assembly "
        "reorganization therefore defines a physical layer of epithelial plasticity linking molecular "
        "state to tissue immune organization during neoplastic progression."
    )
    biopsies, proteomics_patients = re.search(r"(\d+) biopsies from (\d+) patients", text).groups()
    discovery_patients = re.search(r"FFPE specimens from (\w+) patients", text).group(1)
    # The manuscript spells out this cohort size; the site keeps that wording.
    outcome_patients = re.search(r"baseline surveillance biopsies from (\d+) patients", text).group(1)
    cpa_associations, cpa_proteins = re.search(
        r"([\d,]+) co-complex associations among ([\d,]+) proteins in CP-A", text
    ).groups()
    return dict(title=title, abstract=abstract, biopsies=biopsies,
                proteomics_patients=proteomics_patients, discovery_patients=discovery_patients,
                outcome_patients=outcome_patients, cpa_associations=cpa_associations,
                cpa_proteins=cpa_proteins)


@st.cache_data(show_spinner=False)
def load_cpa_network():
    return pd.read_excel(
        MANUSCRIPT_TABLES / "Table S3. EMT PPI network.xlsx",
        sheet_name="CPA_PPI_Network",
    ).rename(columns={
        "Gene_1": "Gene1", "Gene_2": "Gene2",
        "UniProtID1": "Protein1", "UniProtID2": "Protein2",
        "Interaction_Weight": "Weight",
        "Day0_Score": "Day0", "Day4_Score": "Day4", "Day8_Score": "Day8",
    }).assign(Source="EPIC PPI")


@st.cache_data(show_spinner=False)
def load_cpa_metadata():
    metadata = pd.read_excel(
        MANUSCRIPT_TABLES / "Table S3. EMT PPI network.xlsx", sheet_name="CPA_Metadata"
    )
    return dict(zip(metadata["Field"], metadata["Description"]))


def protein_catalogue(interactions):
    proteins = pd.concat([
        interactions[[f"Protein{i}", f"Gene{i}", f"Description{i}"]].rename(
            columns={f"Protein{i}": "UniProt", f"Gene{i}": "GeneName",
                     f"Description{i}": "Description"})
        for i in (1, 2)
    ]).drop_duplicates("UniProt").sort_values("GeneName").reset_index(drop=True)
    # Count distinct accession partners rather than repeated gene annotations.
    directed = pd.concat([
        interactions[["Protein1", "Protein2"]].rename(columns={"Protein1": "UniProt", "Protein2": "Partner"}),
        interactions[["Protein2", "Protein1"]].rename(columns={"Protein2": "UniProt", "Protein1": "Partner"}),
    ])
    proteins["Partners"] = proteins["UniProt"].map(directed.groupby("UniProt")["Partner"].nunique()).fillna(0).astype(int)
    proteins["UniProt_URL"] = "https://www.uniprot.org/uniprot/" + proteins["UniProt"]
    proteins["GeneCards_URL"] = "https://www.genecards.org/cgi-bin/carddisp.pl?gene=" + proteins["GeneName"].fillna("")
    return proteins


@st.cache_data(show_spinner=False)
def load_protein_data():
    f = HERE / "data/downloads/emt_ppi_epic_score05.xlsx"
    ppi05 = pd.read_excel(f, na_values="---")
    TIME = ["Day0", "Day2", "Day4", "Day6", "Day8"]
    for c in TIME:
        ppi05[c] = pd.to_numeric(ppi05[c], errors="coerce")

    return ppi05, protein_catalogue(ppi05)


@st.cache_data(show_spinner=False)
def load_ppi_network():
    f = HERE / "data/ppi_network/emt_ppi_network.xlsx"
    df = pd.read_excel(f, sheet_name="PPI_Network", na_values="---")
    df = df.rename(
        columns={
            "UniProt.ID1": "Protein1",
            "UniProt.ID2": "Protein2",
            "GeneName.1": "Gene1",
            "GeneName.2": "Gene2",
            "Description.1": "Description1",
            "Description.2": "Description2",
            "Interaction_Weight": "Weight",
            "Source": "Source",
        }
    )
    return df


@st.cache_data(show_spinner=False)
def load_complex_data():
    s5 = pd.read_excel(
        MANUSCRIPT_TABLES / "Table S4. ClusterOne Protein Assemblies.xlsx",
        sheet_name="ClusterOne_Protein_Assemblies",
    )
    s5 = s5.rename(columns={"complex_name": "complex_id"})
    s5["size"] = s5["component_genes"].apply(
        lambda x: len([g for g in str(x).split(",") if g.strip()])
    )

    s6 = pd.read_excel(
        MANUSCRIPT_TABLES / "Table S6. ClusterOne Regulated Assemblies.xlsx",
        sheet_name="ClusterOne_Regulated_Assemblies",
    )
    s6 = s6.rename(columns={"complex_name": "complex_id", "component_genes": "gene"})

    names = pd.read_csv(HERE / "data/assembly_data/complex_names.csv")
    names = names.rename(columns={"complex_name": "complex_id"})

    pm = pd.read_csv(CLUSTERONE_TABLES / "clusterone_pattern_memberships.csv")
    pm = pm.rename(columns={"complex_name": "complex_id"})
    TIME = ["Day0", "Day2", "Day4", "Day6", "Day8"]
    for c in TIME:
        pm[c] = pd.to_numeric(pm[c], errors="coerce")

    gocc = pd.read_csv(GOCC_TABLES / "clusterone_gocc_annotated_full_reg.csv")
    gocc = gocc.rename(columns={"complex_name": "complex_id"})
    best_go = (
        gocc[["complex_id", "my_level1", "last_go_term", "p.adjust.log10"]]
        .drop_duplicates()
        .sort_values("p.adjust.log10", ascending=False)
        .drop_duplicates("complex_id")[["complex_id", "my_level1", "last_go_term"]]
    )

    reg_ids = set(s6["complex_id"].unique())

    ct = (
        s5.merge(
            names[["complex_id", "assigned_name", "confidence_score"]],
            on="complex_id",
            how="left",
        )
        .merge(pm[["complex_id", "pattern"] + TIME], on="complex_id", how="left")
        .merge(best_go, on="complex_id", how="left")
    )
    ct["is_regulated"] = ct["complex_id"].isin(reg_ids)
    ct["assigned_name"] = ct["assigned_name"].fillna("(unassigned)")
    ct["pattern"] = ct["pattern"].fillna("unclassified")
    ct["confidence_score"] = pd.to_numeric(
        ct["confidence_score"], errors="coerce"
    ).round(3)

    return ct, s5, s6


@st.cache_data(show_spinner=False)
def load_causal_data():
    # Updated cross-assembly graph from CAO notebook (3 integrated KGs: Lit + Reactome + PathwayCommons)
    causal = pd.read_csv(CROSS_ASSEMBLY_TABLES / "cross_assembly_lit_graph.csv")
    degrees = pd.read_csv(CROSS_ASSEMBLY_TABLES / "assembly_causal_degree.csv")

    # Preserve this graph's assembly membership and correct the direction labels.
    _prog_key = {"down": "EL", "mid": "IH", "up": "ME"}
    prog_lookup = pd.read_csv(CLUSTERONE_TABLES / "clusterone_pattern_memberships.csv")
    prog_lookup = prog_lookup[["complex_name", "pattern"]].drop_duplicates(
        "complex_name"
    )
    prog_lookup["pattern"] = (
        prog_lookup["pattern"].map(_prog_key).fillna("unclassified")
    )
    prog_map = dict(zip(prog_lookup["complex_name"], prog_lookup["pattern"]))

    causal["src_program"] = causal["src_assembly"].map(prog_map).fillna("unclassified")
    causal["tgt_program"] = causal["tgt_assembly"].map(prog_map).fillna("unclassified")
    causal["evidence_count"] = 1  # uniform weight for Sankey aggregation

    return causal, degrees


@st.cache_data(show_spinner=False)
def load_compartment_data():
    df = pd.read_csv(GOCC_TABLES / "clusterone_gocc_annotated_full.csv")
    pm = pd.read_csv(CLUSTERONE_TABLES / "clusterone_pattern_memberships.csv")

    # One representative row per complex: best-scoring GO term
    best = (
        df.sort_values("p.adjust", ascending=False)
        .drop_duplicates("complex_name")[
            [
                "complex_name",
                "my_level1",
                "last_go_term",
                "p.adjust",
                "complex_size",
            ]
        ]
        .rename(columns={"p.adjust": "go_sig"})
    )

    # Simplify multi-compartment labels to primary compartment
    best["compartment"] = best["my_level1"].str.split(";").str[0].str.strip()
    best = best.merge(pm[["complex_name", "pattern"]], on="complex_name", how="left")
    best["pattern"] = best["pattern"].fillna("unclassified")
    return best


@st.cache_data(show_spinner=False)
def load_structural_data():
    s8 = pd.read_excel(MANUSCRIPT_TABLES / "Table S8. Structural atlas.xlsx")
    s8["binned_signal2noise"] = s8["binned_signal2noise"].replace(
        [float("inf"), -float("inf")], pd.NA
    )
    s8["is_novel"] = (s8["high_confidence_model"] == True) & (s8["in_pdb"] == False)
    s8["AF1_URL"] = "https://alphafold.ebi.ac.uk/entry/" + s8["Protein1"].fillna("")
    s8["AF2_URL"] = "https://alphafold.ebi.ac.uk/entry/" + s8["Protein2"].fillna("")
    return s8


# Load all data at startup
with st.spinner("Loading complexome data…"):
    manuscript = load_manuscript_content()
    ppi05, gene_df = load_protein_data()
    ppi_net = load_ppi_network()
    cpa_net = load_cpa_network()
    cpa_metadata = load_cpa_metadata()
    complex_table, s5_df, s6_df = load_complex_data()
    causal_df, degree_df = load_causal_data()
    comp_df = load_compartment_data()
    struct_df = load_structural_data()

TIME_COLS = ["Day0", "Day2", "Day4", "Day6", "Day8"]
PATTERN_COLOR = {
    "up": "#dc2626",
    "mid": "#2563eb",
    "down": "#16a34a",
    "unclassified": "#94a3b8",
}
PROGRAM_LABEL = {
    "EL": "Epithelial Legacy",
    "IH": "Intermediate Hybrid",
    "ME": "Mesenchymal Emerging",
}
COMPARTMENT_COLORS = {
    "Mitochondrion": "#f59e0b",
    "Nucleus": "#6366f1",
    "Plasma membrane": "#ec4899",
    "Cytosol": "#64748b",
    "Ribosome": "#0ea5e9",
    "Endosome": "#8b5cf6",
    "Lysosome": "#ef4444",
    "Cytoskeleton": "#22c55e",
    "Endoplasmic Reticulum": "#f97316",
    "Golgi Apparatus": "#06b6d4",
    "Vacuole": "#84cc16",
    "Peroxisome": "#a78bfa",
}

# ── Helpers ────────────────────────────────────────────────────────────────────


def badge_html(pattern: str) -> str:
    cls = {"up": "badge-up", "mid": "badge-mid", "down": "badge-down"}.get(
        pattern, "badge-unc"
    )
    lbl = {"up": "↑ Up", "mid": "↔ Mid", "down": "↓ Down"}.get(
        pattern, "· Unclassified"
    )
    return f'<span class="badge {cls}">{lbl}</span>'


def render_subunit_badges(complex_id: str) -> str:
    row = s5_df[s5_df["complex_id"] == complex_id]
    if row.empty:
        return ""
    genes = [
        g.strip() for g in str(row.iloc[0]["component_genes"]).split(",") if g.strip()
    ]
    gene_pat = dict(
        zip(
            s6_df.loc[s6_df["complex_id"] == complex_id, "gene"],
            s6_df.loc[s6_df["complex_id"] == complex_id, "pattern"],
        )
    )
    cls_map = {"up": "badge-up", "mid": "badge-mid", "down": "badge-down"}
    parts = []
    for g in genes:
        p = gene_pat.get(g, "unclassified")
        cls = cls_map.get(p, "badge-unc")
        parts.append(f'<span class="badge {cls}">{g}</span>')
    return " ".join(parts)


def render_complex_network(complex_id: str) -> str | None:
    if not PYVIS_OK:
        return None
    row = s5_df[s5_df["complex_id"] == complex_id]
    if row.empty:
        return None
    genes = set(
        g.strip() for g in str(row.iloc[0]["component_genes"]).split(",") if g.strip()
    )
    if len(genes) < 2 or len(genes) > 80:
        return None

    gene_pat = dict(
        zip(
            s6_df.loc[s6_df["complex_id"] == complex_id, "gene"],
            s6_df.loc[s6_df["complex_id"] == complex_id, "pattern"],
        )
    )
    intra = ppi05[ppi05["Gene1"].isin(genes) & ppi05["Gene2"].isin(genes)]

    net = Network(
        height="440px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#1e293b",
        cdn_resources="in_line",
    )
    net.set_options(
        '{"physics":{"stabilization":{"iterations":80},'
        '"barnesHut":{"gravitationalConstant":-3000}},'
        '"interaction":{"hover":true}}'
    )

    pal = {
        "up": "#ef4444",
        "mid": "#3b82f6",
        "down": "#22c55e",
        "unclassified": "#94a3b8",
    }
    for g in genes:
        p = gene_pat.get(g, "unclassified")
        col = pal.get(p, "#94a3b8")
        net.add_node(
            g,
            label=g,
            color={"background": col, "border": "#64748b"},
            title=f"{g} ({p})",
            size=16,
        )

    added = set()
    for _, r in intra.iterrows():
        key = tuple(sorted([r["Gene1"], r["Gene2"]]))
        if key not in added:
            net.add_edge(r["Gene1"], r["Gene2"], color="#cbd5e1", width=1.5)
            added.add(key)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        return open(f.name).read()


def gsva_sparkline(complex_id: str, pattern: str) -> go.Figure | None:
    row = complex_table[complex_table["complex_id"] == complex_id]
    if row.empty or row[TIME_COLS].isnull().all(axis=None):
        return None
    vals = row[TIME_COLS].values[0]
    col = PATTERN_COLOR.get(pattern, "#94a3b8")
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#cbd5e1", line_width=1)
    fig.add_trace(
        go.Scatter(
            x=[0, 2, 4, 6, 8],
            y=vals,
            mode="lines+markers",
            line=dict(color=col, width=2.5),
            marker=dict(size=7, color=col),
            hovertemplate="Day %{x}: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(title="EMT Day", tickvals=[0, 2, 4, 6, 8], tickfont=dict(size=11)),
        yaxis=dict(title="GSVA score", zeroline=False),
        template="plotly_white",
        height=220,
        margin=dict(l=50, r=20, t=10, b=40),
        showlegend=False,
    )
    return fig


# ── Hypothesis renderer (parses Mermaid → Plotly) ──────────────────────
def _mermaid_to_plotly(mermaid_text):
    import re as _R, collections as _C

    _raw = mermaid_text.strip().split("\n")
    _lines, _buf, _br = [], "", 0
    for _ln in _raw:
        _s = _ln.strip()
        if not _s:
            continue
        if _buf:
            _buf += " " + _s
        else:
            _buf = _s
        _br += _s.count("[") + _s.count("(") + _s.count("{")
        _br -= _s.count("]") + _s.count(")") + _s.count("}")
        if _s.count("|") % 2 != 0:
            _br = abs(_br - 1)
        if _br <= 0:
            _lines.append(_buf)
            _buf, _br = "", 0
    if _buf:
        _lines.append(_buf)
    if not _lines:
        return None
    if _lines[0].startswith(("flowchart", "graph")):
        _lines = _lines[1:]
    _nodes, _edges = {}, []

    def _ext(t):
        _m = _R.match(r"(\w+)\s*[\[{\(]([^\]}\)]*)[\]}\)]", t)
        return (_m.group(1), _m.group(2).strip()) if _m else (t.strip(), t.strip())

    for _line in _lines:
        if _line.startswith(("subgraph", "end", "style ")):
            continue
        _em = _R.match(
            r"([^\s]+)\s*(?:-->|==>|\.->|--x|==x)\s*(?:\|([^|]*)\|)?\s*(.+)", _line
        )
        if not _em:
            _em = _R.match(r"([^\s]+)\s+[-=]\.?\s+([^\n]+?)\s+\.?->\s+(.+)", _line)
        if _em:
            _r, _l, _t = _em.group(1), (_em.group(2) or ""), _em.group(3)
            _si, _sl = _ext(_r)
            _ti, _tl = _ext(_t)
            _edges.append((_si, _ti, _l.strip()))
            if _si not in _nodes:
                _nodes[_si] = _sl
            if _ti not in _nodes:
                _nodes[_ti] = _tl
            continue
        _nm = _R.match(r"(\w+)\s*[\[{\(]([^\]}\)]*)[\]}\)]", _line)
        if _nm:
            _nodes[_nm.group(1)] = _nm.group(2).strip()
    for _s, _t, _ in _edges:
        if _s not in _nodes:
            _nodes[_s] = _s
        if _t not in _nodes:
            _nodes[_t] = _t
    if not _edges:
        return None
    _adj, _idg = _C.defaultdict(list), _C.defaultdict(int)
    _all = set(_nodes)
    for _s, _t, _ in _edges:
        _adj[_s].append(_t)
        _idg[_t] += 1
        _all.add(_s)
        _all.add(_t)
    _q = _C.deque([n for n in _all if _idg.get(n, 0) == 0])
    _topo = list(_q)
    while _q:
        _n = _q.popleft()
        for _nb in _adj[_n]:
            _idg[_nb] -= 1
            if _idg[_nb] == 0:
                _q.append(_nb)
                _topo.append(_nb)
    for n in _all:
        if n not in _topo:
            _topo.append(n)
    _dp = {n: 0 for n in _all}
    for n in _topo:
        for nb in _adj[n]:
            _dp[nb] = max(_dp[nb], _dp[n] + 1)
    _pos = {}
    for nid in _all:
        _ly = _dp.get(nid, 0)
        _sm = [m for m in _all if _dp.get(m, 0) == _ly]
        _i = _sm.index(nid) if nid in _sm else 0
        _pos[nid] = (_ly * 2.2, (_i - (len(_sm) - 1) / 2) * 1.8)
    _fig = go.Figure()
    for _s, _t, _l in _edges:
        _x0, _y0 = _pos[_s]
        _x1, _y1 = _pos[_t]
        _fig.add_trace(
            go.Scatter(
                x=[_x0, _x1],
                y=[_y0, _y1],
                mode="lines",
                line=dict(color="#cbd5e1", width=1.5),
                hoverinfo="none",
                showlegend=False,
            )
        )
        if _l:
            _l = _l.replace("<br/>", " ").replace("<br>", " ")
            _fig.add_annotation(
                x=(_x0 + _x1) / 2,
                y=(_y0 + _y1) / 2,
                text=_l,
                showarrow=False,
                font=dict(size=8, color="#475569"),
                bgcolor="rgba(255,255,255,.85)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                borderpad=2,
            )
        _dx, _dy = _x1 - _x0, _y1 - _y0
        _d = (_dx * _dx + _dy * _dy) ** 0.5
        if _d > 0:
            _fig.add_annotation(
                x=_x1 - _dx / _d * 0.35,
                y=_y1 - _dy / _d * 0.35,
                ax=_x0,
                ay=_y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor="#94a3b8",
                standoff=6,
            )
    _nx = [_pos[n][0] for n in _nodes]
    _ny = [_pos[n][1] for n in _nodes]
    _nlbls = [_nodes[n] for n in _nodes]
    _nsizes = [max(22, min(60, len(l) * 2.8 + 14)) for l in _nlbls]
    _nfonts = [max(7, min(11, 11 - len(l) * 0.08)) for l in _nlbls]
    _fig.add_trace(
        go.Scatter(
            x=_nx,
            y=_ny,
            mode="markers+text",
            marker=dict(
                size=_nsizes, color="#1d4ed8", line=dict(color="#fff", width=2)
            ),
            text=_nlbls,
            textposition="middle center",
            textfont=dict(size=_nfonts, color="#fff", family="Inter"),
            hovertext=[f"<b>{l}</b>" for l in _nlbls],
            hoverinfo="text",
            showlegend=False,
        )
    )
    _fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        font=dict(family="Inter"),
    )
    return _fig


def _render_hypothesis(hyp_text, ctx_text):
    import re as _re

    _parts = _re.split(r"(```mermaid\n.*?```)", hyp_text, flags=_re.DOTALL)
    for _p in _parts:
        _m = _re.match(r"```mermaid\n(.*?)```", _p, flags=_re.DOTALL)
        if _m:
            _fig = _styled_mermaid_to_plotly(_m.group(1).strip())
            if _fig:
                st.markdown(
                    '<p style="margin:.8rem 0 .3rem;font-weight:700;font-size:.92rem;color:#0f1b2d;">Mechanistic pathway</p>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    _fig,
                    use_container_width=True,
                    theme=None,
                    config={"displayModeBar": False, "scrollZoom": False},
                    key=f"hyp_flow_{abs(hash(_m.group(1))) & 0xFFFF}",
                )
            else:
                st.code(_m.group(1).strip(), language=None)
        else:
            st.markdown(
                f'<div class="hyp-card">{md_lib.markdown(_p, extensions=["extra", "nl2br"])}</div>',
                unsafe_allow_html=True,
            )
    try:
        from weasyprint import HTML as _W

        _pf = []
        for _p in _parts:
            _m = _re.match(r"```mermaid\n(.*?)```", _p, flags=_re.DOTALL)
            if _m:
                _svg = mermaid_to_svg(_m.group(1).strip())
                if _svg:
                    _pf.append(
                        '<div style="margin:1.5rem 0;"><p class="diagram-title">Mechanistic Pathway Diagram</p>'
                        + _svg
                        + "</div>"
                    )
                else:
                    _pf.append(
                        '<pre style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:1rem;margin:1rem 0;font-family:monospace!important;font-size:9pt;overflow-x:auto;">'
                        + _m.group(1).strip()
                        + "</pre>"
                    )
            else:
                _pf.append(
                    '<div class="hyp-card-pdf">'
                    + md_lib.markdown(_p, extensions=["extra", "nl2br"])
                    + "</div>"
                )
        _ph = (
            '<html><head><meta charset="utf-8"><style>'
            "@page { size: A4; margin: 2cm 2.2cm; }"
            '* { font-family: "Cambria", "Georgia", "Times New Roman", serif !important; }'
            "body { font-size: 11pt; line-height: 1.6; color: #1e293b; background: #ffffff; text-align: justify; }"
            ".header { background: linear-gradient(135deg, #0f1b2d 0%, #1e3a8a 100%); padding: 1.8rem 2.2rem; border-radius: 8px; margin-bottom: 1.8rem; color: #ffffff; text-align: left; }"
            ".header h1 { font-size: 20pt; font-weight: bold; color: #ffffff; margin: 0 0 6pt 0; line-height: 1.25; text-align: left; }"
            ".header p { font-size: 11pt; color: #bfdbfe; margin: 0; text-align: left; }"
            ".header-badge { display: inline-block; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.25); border-radius: 16px; padding: 3pt 10pt; font-size: 9pt; font-weight: bold; color: #e0f2fe; margin-top: 10pt; text-align: left; }"
            "h1 { font-size: 18pt; font-weight: bold; color: #0f1b2d; margin-top: 18pt; margin-bottom: 8pt; text-align: left; border-bottom: 1.5pt solid #1d4ed8; padding-bottom: 4pt; }"
            "h2 { font-size: 15pt; font-weight: bold; color: #0f1b2d; margin-top: 15pt; margin-bottom: 6pt; text-align: left; }"
            "h3 { font-size: 13pt; font-weight: bold; color: #1e3a8a; margin-top: 12pt; margin-bottom: 4pt; text-align: left; }"
            "h4 { font-size: 11.5pt; font-weight: bold; color: #1e293b; margin-top: 10pt; margin-bottom: 4pt; text-align: left; }"
            "p { font-size: 11pt; line-height: 1.6; color: #1e293b; text-align: justify; margin-top: 0; margin-bottom: 8pt; }"
            "li { font-size: 11pt; line-height: 1.6; text-align: justify; margin-bottom: 4pt; }"
            "strong, b { font-weight: bold; color: #0f1b2d; }"
            ".hyp-card-pdf { background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #1d4ed8; border-radius: 6px; padding: 1.2rem 1.5rem; margin-bottom: 1.2rem; }"
            ".diagram-title { font-size: 13pt; font-weight: bold; color: #0f1b2d; margin-top: 12pt; margin-bottom: 6pt; text-align: left; }"
            "code { font-family: monospace !important; background: #f1f5f9; padding: 2px 5px; border-radius: 4px; font-size: 9.5pt; color: #1d4ed8; }"
            "pre { font-family: monospace !important; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 1rem; font-size: 9pt; white-space: pre-wrap; }"
            ".footer { margin-top: 25pt; padding-top: 10pt; border-top: 1px solid #e2e8f0; font-size: 9pt; color: #64748b; text-align: center; }"
            "</style></head><body>"
            '<div class="header"><h1>PlastInMap Assembly Hypothesis Report</h1>'
            "<p>Generated by the Causal Assembly Ontology (CAO) Knowledge Base</p>"
            '<div class="header-badge">Emili Lab — OHSU Knight Cancer Institute</div></div>'
            "<div>" + "\n".join(_pf) + "</div>"
            '<div class="footer">PlastInMap Companion Website &nbsp;•&nbsp; Emili Lab, Oregon Health & Science University</div>'
            "</body></html>"
        )
        _pb = _W(string=_ph).write_pdf()
        st.download_button(
            label="📄 Download as PDF",
            data=_pb,
            file_name="plastinmap_hypothesis.pdf",
            mime="application/pdf",
            key="hyp_dl_pdf",
        )
    except Exception as e:
        st.caption(f"PDF export notice: {e}")
    with st.expander("View knowledge base context used"):
        st.code(ctx_text, language=None)


with st.sidebar:
    logo = HERE / "www" / "logo-new-emili.png"
    if logo.exists():
        st.image(str(logo), width=90)
    st.markdown(
        '<div style="margin:.5rem 0 .25rem;">'
        '<span style="font-size:1.15rem;font-weight:800;color:#0f1b2d;">PlastInMap</span><br>'
        '<span style="font-size:.72rem;color:#64748b;">Dynamic Assemblies in Barrett’s Progression</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "🗺 Complexome Map",
            "🔬 Browse Data",
            "🌊 Causal Network",
            "💡 Hypothesis Generator",
            "📥 Downloads",
            "📧 Contact & Help",
        ],
        label_visibility="collapsed",
    )
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
st.caption(
    "MCF10A reference atlas · Supporting CP-A interaction data · Current manuscript supplements"
)
if page == "🏠 Home":
    st.markdown(
        f"""
    <div class="hero">
        <h1>{escape(manuscript['title'])}</h1>
        <p>MCF10A native assembly remodeling, supporting CP-A complexome data, and projection to patient proteomes in Barrett's esophagus progression.</p>
        <div class="hero-sub">
            <span class="hero-chip">Co-fractionation mass spectrometry</span>
            <span class="hero-chip">Physics-informed structural modeling</span>
            <span class="hero-chip">Blueprint–Projection framework</span>
            <span class="hero-chip">Barrett's esophagus progression</span>
            <span class="hero-chip">Endolysosomal–immune remodeling</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-n">{len(ppi_net):,}</div>
            <div class="stat-l">MCF10A Co-Complex Associations</div></div>
        <div class="stat-card"><div class="stat-n">{len(complex_table):,}</div>
            <div class="stat-l">MCF10A Protein Assemblies</div></div>
        <div class="stat-card"><div class="stat-n">{int(struct_df['high_confidence_model'].eq(True).sum()):,}</div>
            <div class="stat-l">High-Confidence Interface Models</div></div>
        <div class="stat-card"><div class="stat-n">{s6_df['complex_id'].nunique():,}</div>
            <div class="stat-l">Remodeled Assemblies</div></div>
        <div class="stat-card"><div class="stat-n">{manuscript['proteomics_patients']} + {manuscript['outcome_patients']}</div>
            <div class="stat-l">Proteomics + Outcome Patients</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Supporting CP-A dataset: {len(cpa_net):,} released co-complex associations "
        f"across {len(set(cpa_net['Protein1']) | set(cpa_net['Protein2'])):,} accession identifiers (Days 0, 4, 8). "
        f"Clinical proteomics: {manuscript['biopsies']} biopsies from {manuscript['proteomics_patients']} patients; "
        f"spatial discovery: {manuscript['discovery_patients']} patients; "
        f"independent outcome cohort: {manuscript['outcome_patients']} patients."
    )

    st.markdown(
        f"""
    <div class="card abstract">
        <h3>Abstract</h3>
        <p>
        {escape(manuscript['abstract'])}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div class="card">
        <h3>Citation</h3>
        <div class="cite-block">
            <strong>{escape(manuscript['title'])}</strong><br>
            Paul I, Padhorny D, Joseph TP, Sivagnanam S, <em>et al.</em><br>
            <em>Manuscript in preparation.</em>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**What's in this resource?**")
    st.markdown(
        f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin-top:.25rem;">
        <div class="card" style="margin:0;">
            <div style="font-size:1.1rem;margin-bottom:.4rem;">🗺</div>
            <div style="font-weight:600;font-size:.85rem;color:#0f1b2d;margin-bottom:.4rem;">Complexome Map</div>
            <p style="margin:0;font-size:.8rem;">Interactive treemap and sunburst of {comp_df['complex_name'].nunique():,} annotated MCF10A assemblies organized by subcellular compartment.</p>
        </div>
        <div class="card" style="margin:0;">
            <div style="font-size:1.1rem;margin-bottom:.4rem;">🔬</div>
            <div style="font-weight:600;font-size:.85rem;color:#0f1b2d;margin-bottom:.4rem;">Browse Data</div>
            <p style="margin:0;font-size:.8rem;">Search MCF10A and supporting CP-A proteins and PPIs, the {len(complex_table):,}-assembly MCF10A reference, and predicted structural interfaces.</p>
        </div>
        <div class="card" style="margin:0;">
            <div style="font-size:1.1rem;margin-bottom:.4rem;">🌊</div>
            <div style="font-weight:600;font-size:.85rem;color:#0f1b2d;margin-bottom:.4rem;">Causal Network</div>
            <p style="margin:0;font-size:.8rem;">Explore {len(causal_df):,} causal/regulatory edges across MCF10A assembly programs, from 3 integrated knowledge graphs (Lit KG, Reactome, PathwayCommons).</p>
        </div>
        <div class="card" style="margin:0;">
            <div style="font-size:1.1rem;margin-bottom:.4rem;">💡</div>
            <div style="font-weight:600;font-size:.85rem;color:#0f1b2d;margin-bottom:.4rem;">Hypothesis Generator</div>
            <p style="margin:0;font-size:.8rem;">Query the complexome knowledge base — causal edges, assembly profiles, and gene relationships — with LLM-augmented hypothesis generation.</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# COMPLEXOME MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺 Complexome Map":
    st.markdown(
        '<h2 style="color:#0f1b2d;margin-bottom:.25rem;">Complexome Compartment Map</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-sub">'
        f"{comp_df['complex_name'].nunique()} assemblies organized by subcellular compartment "
        "and GO Cellular Component annotation. Rectangle / sector area is proportional to assembly size (number of subunits). "
        "Click any compartment or GO term to zoom in; click the centre to zoom back out."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([1, 3])
    with ctrl_col1:
        color_by = st.radio(
            "Color by",
            ["Compartment", "EMT Program"],
            horizontal=False,
            key="map_color",
        )

    # Build plot data: join assigned names from complex_table
    names_map = dict(zip(complex_table["complex_id"], complex_table["assigned_name"]))
    plot_df = comp_df.copy()
    plot_df["assigned_name"] = (
        plot_df["complex_name"].map(names_map).fillna("(unassigned)")
    )
    plot_df["label"] = plot_df["complex_name"] + " · " + plot_df["assigned_name"]
    plot_df["go_sig_fmt"] = plot_df["go_sig"].round(1).astype(str)

    if color_by == "Compartment":
        color_col = "compartment"
        color_map = COMPARTMENT_COLORS
    else:
        color_col = "pattern"
        color_map = {k: v for k, v in PATTERN_COLOR.items()}

    tab_tree, tab_sun = st.tabs(["Treemap", "Sunburst"])

    # ── TREEMAP ───────────────────────────────────────────────────────────────
    with tab_tree:
        fig_tree = px.treemap(
            plot_df,
            path=["compartment", "last_go_term", "label"],
            values="complex_size",
            color=color_col,
            color_discrete_map=color_map,
            hover_data={
                "assigned_name": True,
                "go_sig_fmt": True,
                "pattern": True,
                "complex_size": True,
                "compartment": False,
                "last_go_term": False,
                "label": False,
            },
            custom_data=["complex_name", "assigned_name", "pattern", "go_sig_fmt"],
        )
        fig_tree.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
                "Subunits: %{value} &nbsp;·&nbsp; Program: %{customdata[2]}<br>"
                "GO sig (−log₁₀ p): %{customdata[3]}"
                "<extra></extra>"
            ),
            textfont=dict(family="Inter", size=11),
            marker_line_width=0.5,
        )
        fig_tree.update_layout(
            height=600,
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    # ── SUNBURST ──────────────────────────────────────────────────────────────
    with tab_sun:
        fig_sun = px.sunburst(
            plot_df,
            path=["compartment", "last_go_term", "label"],
            values="complex_size",
            color=color_col,
            color_discrete_map=color_map,
            custom_data=["complex_name", "assigned_name", "pattern", "go_sig_fmt"],
        )
        fig_sun.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
                "Subunits: %{value} &nbsp;·&nbsp; Program: %{customdata[2]}<br>"
                "GO sig (−log₁₀ p): %{customdata[3]}"
                "<extra></extra>"
            ),
            insidetextfont=dict(family="Inter", size=10),
            leaf=dict(opacity=0.9),
        )
        fig_sun.update_layout(
            height=620,
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    # ── Legend / compartment summary ──────────────────────────────────────────
    st.divider()
    st.markdown("**Compartment summary**")
    summary = (
        plot_df.groupby("compartment")
        .agg(
            assemblies=("complex_name", "nunique"),
            total_subunits=("complex_size", "sum"),
        )
        .reset_index()
        .sort_values("assemblies", ascending=False)
    )
    summary["color"] = summary["compartment"].map(COMPARTMENT_COLORS).fillna("#94a3b8")

    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        col.markdown(
            f'<div style="text-align:center;padding:.5rem .25rem;">'
            f'<div style="width:14px;height:14px;border-radius:50%;background:{row["color"]};'
            f'margin:0 auto .3rem;"></div>'
            f'<div style="font-size:.7rem;font-weight:600;color:#0f1b2d;line-height:1.3;">'
            f"{row['compartment']}</div>"
            f'<div style="font-size:.75rem;color:#334155;font-weight:700;">{row["assemblies"]}</div>'
            f'<div style="font-size:.65rem;color:#64748b;">{int(row["total_subunits"]):,} subunits</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Complex detail panel ──────────────────────────────────────────────────
    st.divider()
    st.markdown("**Complex Detail**")
    st.caption(
        "Select a complex below to see its subunit composition, regulation pattern, "
        "GSVA trajectory, and intra-complex PPI network. "
        "Click a leaf node in the treemap/sunburst above to jump to a complex."
    )

    all_complexes = sorted(complex_table["complex_id"].tolist())
    cx_labels = {
        cid: f"{cid} — {complex_table.loc[complex_table['complex_id'] == cid, 'assigned_name'].values[0]}"
        for cid in all_complexes
    }

    sel_cx = st.selectbox(
        "Search or select a complex:",
        all_complexes,
        format_func=lambda x: cx_labels.get(x, x),
        key="asm_map_detail",
    )

    if sel_cx:
        sel_row = complex_table[complex_table["complex_id"] == sel_cx].iloc[0]
        pat = sel_row["pattern"]

        dcol1, dcol2 = st.columns([1.3, 0.7])
        with dcol1:
            st.markdown(
                f"**Subunits** ({int(sel_row['size'])} genes) — colored by regulation program"
            )
            st.markdown(render_subunit_badges(sel_cx), unsafe_allow_html=True)
        with dcol2:
            fig_g = gsva_sparkline(sel_cx, pat)
            if fig_g:
                st.markdown(f"**GSVA trajectory** (program: {pat})")
                st.plotly_chart(
                    fig_g, use_container_width=True, key=f"gsva_map_{sel_cx}"
                )

        if PYVIS_OK:
            if int(sel_row["size"]) > 80:
                st.caption("Network view available for assemblies \u2264 80 subunits.")
            else:
                with st.expander("Show intra-complex PPI network", expanded=True):
                    net_html = render_complex_network(sel_cx)
                    if net_html:
                        components.html(net_html, height=460, scrolling=False)
                    else:
                        st.info(
                            "No intra-complex PPIs found in the EPIC dataset for this assembly."
                        )

# ══════════════════════════════════════════════════════════════════════════════
# BROWSE DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Browse Data":
    st.markdown(
        '<h2 style="color:#0f1b2d;margin-bottom:.25rem;">Browse Complexome Data</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-sub">Search the EMT interactome — proteins, interactions, and assemblies — '
        "with MCF10A as the reference model and CP-A as a supporting epithelial model.</p>",
        unsafe_allow_html=True,
    )

    browse_model = st.radio(
        "Protein / PPI dataset", ["MCF10A", "CP-A"], horizontal=True, key="browse_model"
    )
    is_cpa = browse_model == "CP-A"
    browse_ppi = cpa_net if is_cpa else ppi05
    browse_network = cpa_net if is_cpa else ppi_net
    browse_genes = protein_catalogue(cpa_net) if is_cpa else gene_df
    browse_time = ["Day0", "Day4", "Day8"] if is_cpa else TIME_COLS
    browse_days = [int(day.removeprefix("Day")) for day in browse_time]
    dataset_key = "cpa" if is_cpa else "mcf10a"
    st.caption(
        "The dataset selector applies to Proteins and Protein–Protein Interactions. "
        "Assemblies and Novel Interfaces retain the MCF10A reference."
    )
    if is_cpa:
        st.caption(
            "CP-A: Days 0, 4, 8. Blank scores mean not reported, not zero. "
            "Interaction weight is the mean of available accepted day scores."
        )
        with st.expander("CP-A data source and score definitions"):
            st.write(
                f"Table S3 releases {len(cpa_net):,} biological associations across "
                f"{len(set(cpa_net['Protein1']) | set(cpa_net['Protein2'])):,} accession identifiers; "
                f"{cpa_metadata['Excluded source records']} reverse/contaminant source records are excluded. "
                f"The manuscript reports {manuscript['cpa_associations']} associations and "
                f"{manuscript['cpa_proteins']} proteins. The browser counts the released table."
            )
            st.write(cpa_metadata["Source threshold"])
            st.write(cpa_metadata["Figure 2 comparison"])
            st.write("Days observed counts reported time points, not biological replicates. Co-complex association does not establish direct binary binding.")

    tab_prot, tab_ppi, tab_cx, tab_novel = st.tabs(
        ["Proteins", "Protein–Protein Interactions", "Assemblies", "Novel Interfaces"]
    )

    # ── PROTEINS ──────────────────────────────────────────────────────────────
    with tab_prot:
        search = st.text_input(
            "Search by gene symbol or UniProt ID",
            "",
            placeholder="e.g. VIM, EGFR, P04637…",
        )
        df_show = browse_genes.copy()
        if search:
            df_show = df_show[
                df_show["GeneName"].str.contains(search, case=False, na=False)
                | df_show["UniProt"].str.contains(search, case=False, na=False)
            ]

        st.caption(f"{len(df_show):,} accession identifiers in the released CP-A network" if is_cpa
                   else f"{len(df_show):,} proteins in the MCF10A EPIC ≥0.5 dataset")
        st.dataframe(
            df_show[
                [
                    "UniProt",
                    "GeneName",
                    "Description",
                    "Partners",
                    "UniProt_URL",
                    "GeneCards_URL",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            height=320,
            column_config={
                "UniProt": st.column_config.TextColumn("UniProt ID"),
                "GeneName": st.column_config.TextColumn("Gene Symbol"),
                "Description": st.column_config.TextColumn("Description"),
                "Partners": st.column_config.NumberColumn(
                    "Interaction Partners", format="%d"
                ),
                "UniProt_URL": st.column_config.LinkColumn(
                    "UniProt ↗", display_text="UniProt"
                ),
                "GeneCards_URL": st.column_config.LinkColumn(
                    "GeneCards ↗", display_text="GeneCards"
                ),
            },
        )

        st.divider()
        st.markdown("**Temporal EPIC profile — interaction partners**")
        st.caption(
            "EPIC co-elution scores across the TGF-β EMT timecourse for a selected protein's partners."
        )
        col_a, col_b = st.columns([2, 1])
        with col_a:
            gene_choices = sorted(df_show["GeneName"].dropna().unique().tolist())
            sel_gene = (
                st.selectbox("Select gene:", gene_choices, key=f"prot_sel_{dataset_key}")
                if gene_choices
                else None
            )
        with col_b:
            epic_thresh = st.slider(
                "Min EPIC score", 0.0, 1.0, 0.5, 0.05, key="epic_th"
            )

        if sel_gene:
            partners = browse_ppi[
                (browse_ppi["Gene1"] == sel_gene) | (browse_ppi["Gene2"] == sel_gene)
            ].copy()
            partners["Partner"] = np.where(
                partners["Gene1"] == sel_gene, partners["Gene2"], partners["Gene1"]
            )
            if is_cpa:
                partners["Partner"] = (
                    partners["Partner"].fillna("Unannotated") + " ["
                    + partners["Protein1"] + " / " + partners["Protein2"] + "]"
                )
            partners = partners[
                partners[browse_time].max(axis=1) >= epic_thresh
            ].reset_index(drop=True)

            if partners.empty:
                st.info("No partners above threshold.")
            else:
                sel_partners = st.multiselect(
                    f"Partners of {sel_gene} ({len(partners)} above threshold):",
                    partners["Partner"].unique(),
                    default=list(partners["Partner"].unique()[:8]),
                    key=f"prot_partners_{dataset_key}_{sel_gene}",
                )
                if sel_partners:
                    long = partners[partners["Partner"].isin(sel_partners)].melt(
                        id_vars="Partner",
                        value_vars=browse_time,
                        var_name="Timepoint",
                        value_name="EPIC Score",
                    )
                    long["Day"] = long["Timepoint"].str.replace("Day", "").astype(int)
                    long = long.sort_values("Day")
                    fig = px.line(
                        long,
                        x="Day",
                        y="EPIC Score",
                        color="Partner",
                        markers=True,
                        template="plotly_white",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig.update_layout(
                        height=340,
                        margin=dict(l=50, r=20, t=20, b=40),
                        xaxis=dict(tickvals=browse_days, title="EMT Day"),
                        yaxis_title="EPIC Score",
                        legend_title="Partner Gene",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── PPIS ──────────────────────────────────────────────────────────────────
    with tab_ppi:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ppi_search = st.text_input(
                "Search by gene symbol:", "", placeholder="e.g. VIM, CDH1…", key="ppi_s"
            )
        with col2:
            src_filter = st.selectbox(
                "Source:", ["All"] + sorted(browse_network["Source"].dropna().unique()),
                key=f"ppi_src_{dataset_key}"
            )
        with col3:
            min_wt = st.slider(
                "Min interaction weight:", 0.0, 1.0, 0.0, 0.05, key="ppi_wt"
            )

        ppi_show = browse_network.copy()
        if ppi_search:
            ppi_show = ppi_show[
                ppi_show["Gene1"].str.contains(ppi_search, case=False, na=False)
                | ppi_show["Gene2"].str.contains(ppi_search, case=False, na=False)
            ]
        if src_filter != "All":
            ppi_show = ppi_show[ppi_show["Source"] == src_filter]
        if min_wt > 0:
            ppi_show = ppi_show[
                pd.to_numeric(ppi_show["Weight"], errors="coerce").fillna(0) >= min_wt
            ]

        ppi_show["P1_URL"] = "https://www.uniprot.org/uniprot/" + ppi_show[
            "Protein1"
        ].fillna("")
        ppi_show["P2_URL"] = "https://www.uniprot.org/uniprot/" + ppi_show[
            "Protein2"
        ].fillna("")

        st.caption(
            f"{len(ppi_show):,} PPIs shown ({browse_model} network: {len(browse_network):,})"
            + ("" if is_cpa else f" · temporal EPIC subset: {len(ppi05):,}")
        )
        st.dataframe(
            ppi_show[
                [
                    "Gene1",
                    "Gene2",
                    "Description1",
                    "Description2",
                    "Weight",
                    "Source",
                    "P1_URL",
                    "P2_URL",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            height=360,
            column_config={
                "Gene1": st.column_config.TextColumn("Gene 1"),
                "Gene2": st.column_config.TextColumn("Gene 2"),
                "Description1": st.column_config.TextColumn("Protein 1 Description"),
                "Description2": st.column_config.TextColumn("Protein 2 Description"),
                "Weight": st.column_config.NumberColumn(
                    "Interaction Weight", format="%.4f"
                ),
                "Source": st.column_config.TextColumn("Source"),
                "P1_URL": st.column_config.LinkColumn(
                    "P1 UniProt ↗", display_text="UniProt"
                ),
                "P2_URL": st.column_config.LinkColumn(
                    "P2 UniProt ↗", display_text="UniProt"
                ),
            },
        )

        if ppi_search and not ppi_show.empty:
            epic_sub = browse_ppi[
                browse_ppi["Gene1"].str.contains(ppi_search, case=False, na=False)
                | browse_ppi["Gene2"].str.contains(ppi_search, case=False, na=False)
            ].copy()
            if not epic_sub.empty:
                epic_sub["Pair"] = epic_sub["Gene1"] + " – " + epic_sub["Gene2"]
                if is_cpa:
                    epic_sub["Pair"] = (
                        epic_sub["Pair"].fillna("Unannotated") + " ["
                        + epic_sub["Protein1"] + " / " + epic_sub["Protein2"] + "]"
                    )
                epic_sub = epic_sub.dropna(subset=browse_time, how="all")
                if not epic_sub.empty:
                    st.divider()
                    st.markdown("**EPIC score time-course for matching interactions**")
                    long = epic_sub.melt(
                        id_vars="Pair",
                        value_vars=browse_time,
                        var_name="Timepoint",
                        value_name="EPIC Score",
                    )
                    long["Day"] = long["Timepoint"].str.replace("Day", "").astype(int)
                    fig = px.line(
                        long,
                        x="Day",
                        y="EPIC Score",
                        color="Pair",
                        markers=True,
                        template="plotly_white",
                    )
                    fig.update_layout(
                        height=280,
                        margin=dict(l=50, r=20, t=20, b=40),
                        xaxis=dict(tickvals=browse_days, title="EMT Day"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── ASSEMBLIES ────────────────────────────────────────────────────────────
    with tab_cx:
        st.caption("MCF10A reference assemblies and temporal programs; unaffected by the protein / PPI dataset selector.")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            cx_search = st.text_input(
                "Search by complex name or ID:",
                "",
                placeholder="e.g. C0071, Endolysosomal…",
                key="cx_s",
            )
        with col2:
            pat_filter = st.multiselect(
                "Regulation program:",
                ["up", "mid", "down", "unclassified"],
                default=["up", "mid", "down", "unclassified"],
                key="cx_pat",
            )
        with col3:
            go_cats = ["All"] + sorted(
                complex_table["my_level1"].dropna().unique().tolist()
            )
            go_filter = st.selectbox("GO category:", go_cats, key="cx_go")

        cx_show = complex_table[complex_table["pattern"].isin(pat_filter)].copy()
        if cx_search:
            cx_show = cx_show[
                cx_show["complex_id"].str.contains(cx_search, case=False, na=False)
                | cx_show["assigned_name"].str.contains(cx_search, case=False, na=False)
            ]
        if go_filter != "All":
            cx_show = cx_show[cx_show["my_level1"] == go_filter]

        disp_cols = {
            "complex_id": "Complex ID",
            "assigned_name": "Assigned Name",
            "my_level1": "GO Category",
            "last_go_term": "Top GO Term",
            "pattern": "Program",
            "size": "Subunits",
            "confidence_score": "Confidence",
            "is_regulated": "Regulated",
        }
        show_df = cx_show.rename(columns=disp_cols)[
            [
                v
                for v in disp_cols.values()
                if v in cx_show.rename(columns=disp_cols).columns
            ]
        ]

        st.caption(
            f"{len(cx_show):,} assemblies shown · {int(cx_show['is_regulated'].sum())} regulated"
        )
        st.dataframe(
            show_df,
            use_container_width=True,
            hide_index=True,
            height=330,
            column_config={
                "Subunits": st.column_config.NumberColumn(format="%d"),
                "Confidence": st.column_config.NumberColumn(format="%.3f"),
                "Regulated": st.column_config.CheckboxColumn(),
            },
        )

        st.divider()
        st.markdown("**Assembly detail view**")
        if not cx_show.empty:
            fmt_ids = cx_show["complex_id"].tolist()
            fmt_labels = {
                cid: f"{cid} — {cx_show.loc[cx_show['complex_id'] == cid, 'assigned_name'].values[0]}"
                for cid in fmt_ids
            }
            sel_cx = st.selectbox(
                "Select assembly:",
                fmt_ids,
                format_func=lambda x: fmt_labels.get(x, x),
                key="cx_detail",
            )

            if sel_cx:
                sel_row = complex_table[complex_table["complex_id"] == sel_cx].iloc[0]
                pat = sel_row["pattern"]

                dcol1, dcol2 = st.columns([1.3, 0.7])
                with dcol1:
                    st.markdown(
                        f"**Subunits** ({int(sel_row['size'])} genes) — colored by regulation program"
                    )
                    st.markdown(render_subunit_badges(sel_cx), unsafe_allow_html=True)
                with dcol2:
                    fig_g = gsva_sparkline(sel_cx, pat)
                    if fig_g:
                        st.markdown(f"**GSVA trajectory** (program: {pat})")
                        st.plotly_chart(
                            fig_g, use_container_width=True, key=f"gsva_{sel_cx}"
                        )

                if PYVIS_OK:
                    if int(sel_row["size"]) > 80:
                        st.caption(
                            "Network view available for assemblies ≤ 80 subunits."
                        )
                    else:
                        with st.expander("Show intra-complex PPI network"):
                            net_html = render_complex_network(sel_cx)
                            if net_html:
                                components.html(net_html, height=460, scrolling=False)
                            else:
                                st.info(
                                    "No intra-complex PPIs found in the EPIC dataset for this assembly."
                                )
        else:
            st.info("No assemblies match the current filters.")

    # ── NOVEL INTERFACES ──────────────────────────────────────────────────────
    with tab_novel:
        st.markdown(
            '<p class="section-sub">'
            f"{int(struct_df['high_confidence_model'].eq(True).sum()):,} high-confidence interface models in the MCF10A reference; "
            f"this tab shows the {int(struct_df['is_novel'].sum()):,} without a corresponding PDB interface. "
            "Models were prioritized using model-specific signal-to-noise calibration."
            "</p>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns([2, 1])
        with col_a:
            novel_search = st.text_input(
                "Filter by gene symbol:",
                "",
                placeholder="e.g. VIM, EGFR…",
                key="novel_s",
            )
        with col_b:
            min_iptm = st.slider(
                "Min interface score (icptm):", 0.75, 0.98, 0.75, 0.01, key="novel_iptm"
            )

        novel_df = struct_df[struct_df["is_novel"]].copy()
        novel_df = novel_df[novel_df["continuous_score(icptm)"] >= min_iptm]
        if novel_search:
            novel_df = novel_df[
                novel_df["Gene1"].str.contains(novel_search, case=False, na=False)
                | novel_df["Gene2"].str.contains(novel_search, case=False, na=False)
            ]
        novel_df = novel_df.sort_values("continuous_score(icptm)", ascending=False)

        st.caption(
            f"{len(novel_df):,} novel interfaces shown · "
            "AlphaFold DB links open the individual protein model; "
            "these interfaces have no known complex structure in PDB."
        )
        st.dataframe(
            novel_df[
                [
                    "Gene1",
                    "Gene2",
                    "Description1",
                    "Description2",
                    "continuous_score(icptm)",
                    "binned_signal2noise",
                    "AF1_URL",
                    "AF2_URL",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            height=420,
            column_config={
                "Gene1": st.column_config.TextColumn("Gene 1"),
                "Gene2": st.column_config.TextColumn("Gene 2"),
                "Description1": st.column_config.TextColumn("Protein 1"),
                "Description2": st.column_config.TextColumn("Protein 2"),
                "continuous_score(icptm)": st.column_config.NumberColumn(
                    "Interface Score (icptm)",
                    format="%.3f",
                    help="AF2-Multimer interface confidence (0–1). ≥0.75 = high confidence.",
                ),
                "binned_signal2noise": st.column_config.NumberColumn(
                    "Signal/Noise",
                    format="%.1f",
                    help="Co-elution signal-to-noise ratio from CFMS.",
                ),
                "AF1_URL": st.column_config.LinkColumn(
                    "AlphaFold (Gene 1) ↗", display_text="AlphaFold"
                ),
                "AF2_URL": st.column_config.LinkColumn(
                    "AlphaFold (Gene 2) ↗", display_text="AlphaFold"
                ),
            },
        )

        # Summary mini-bar: recorded interface score distribution
        fig_iptm = px.histogram(
            struct_df[struct_df["is_novel"]],
            x="continuous_score(icptm)",
            nbins=25,
            template="plotly_white",
            labels={"continuous_score(icptm)": "Interface Score (icptm)", "count": "Interfaces"},
            color_discrete_sequence=["#1d4ed8"],
        )
        fig_iptm.update_layout(
            height=200,
            margin=dict(l=50, r=20, t=20, b=40),
            bargap=0.05,
            xaxis_title="Interface Score (icptm)",
            yaxis_title="# Interfaces",
            showlegend=False,
        )
        fig_iptm.add_vline(
            x=min_iptm, line_dash="dash", line_color="#94a3b8", line_width=1.5
        )
        st.caption(
            "Interface score distribution for high-confidence models (dashed line = current filter)"
        )
        st.plotly_chart(fig_iptm, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# CAUSAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌊 Causal Network":
    st.markdown(
        '<h2 style="color:#0f1b2d;margin-bottom:.25rem;">Causal Assembly Network</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-sub">'
        f"{len(causal_df):,} directed causal/regulatory edges between assemblies across 3 integrated knowledge graphs "
        "(Lit KG, Reactome, PathwayCommons), spanning the Epithelial Legacy (EL), Intermediate Hybrid (IH), "
        "and Mesenchymal Emerging (ME) programs."
        "</p>",
        unsafe_allow_html=True,
    )

    # KG Integration summary card
    st.markdown(
        '<div class="card" style="margin-bottom:0.75rem;">'
        '<div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">'
        '<div style="font-size:1.1rem;">\U0001f30d</div>'
        '<div style="flex:1;">'
        '<div style="font-weight:600;font-size:.82rem;color:#0f1b2d;">Integrated Causal Assembly Ontology (CAO)</div>'
        '<div style="font-size:.76rem;color:#334155;line-height:1.6;margin-top:.15rem;">'
        "This graph now integrates <strong>3 knowledge graphs</strong>: "
        "the Lit KG (literature-mined causal, 3,570 gene\u2192gene edges), "
        "the Reactome KG (reaction-ordering, 10,489 gene\u2192gene edges), and "
        "the <strong>PathwayCommons KG</strong> (73,416 gene\u2192gene edges from 9+ curated databases), "
        "into a unified <strong>combined KG</strong> of 87,475 gene\u2192gene edges "
        'projected to assembly-level causal relationships. <span style="color:#64748b;font-size:.7rem;">'
        "CAO notebook \u00b7 CFMS_CAO_PY.ipynb</span>"
        "</div></div></div></div>",
        unsafe_allow_html=True,
    )

    tab_sk, tab_hubs, tab_edges = st.tabs(
        ["Program-level Flows", "Hub Assemblies", "Edge Table"]
    )

    with tab_sk:
        named = ["EL", "IH", "ME"]
        sk = (
            causal_df[
                causal_df["src_program"].isin(named)
                & causal_df["tgt_program"].isin(named)
            ]
            .groupby(["src_program", "tgt_program", "rel_type"])["evidence_count"]
            .sum()
            .reset_index()
        )

        src_nodes = [f"{p} (source)" for p in named]
        tgt_nodes = [f"{p} (target)" for p in named]
        all_nodes = src_nodes + tgt_nodes
        node_idx = {n: i for i, n in enumerate(all_nodes)}
        node_colors = [
            "#dc2626",
            "#2563eb",
            "#16a34a",  # source
            "#f87171",
            "#93c5fd",
            "#86efac",
        ]  # target (lighter)

        sources_list, targets_list, values_list, link_colors = [], [], [], []
        for _, r in sk.iterrows():
            sn = f"{r['src_program']} (source)"
            tn = f"{r['tgt_program']} (target)"
            col = (
                "rgba(220,38,38,0.35)"
                if r["rel_type"] == "negative_causal"
                else "rgba(22,163,74,0.35)"
            )
            sources_list.append(node_idx[sn])
            targets_list.append(node_idx[tn])
            values_list.append(int(r["evidence_count"]))
            link_colors.append(col)

        fig_sk = go.Figure(
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=20,
                    thickness=20,
                    label=all_nodes,
                    color=node_colors,
                    hovertemplate="%{label}: %{value} edges<extra></extra>",
                ),
                link=dict(
                    source=sources_list,
                    target=targets_list,
                    value=values_list,
                    color=link_colors,
                    hovertemplate="%{source.label} → %{target.label}: %{value} evidence<extra></extra>",
                ),
            )
        )
        fig_sk.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Inter", size=13),
            title=dict(
                text="Causal evidence flow between EMT programs",
                font=dict(size=14, color="#0f1b2d"),
            ),
        )
        st.plotly_chart(fig_sk, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
            <div class="card">
                <h3>Reading the diagram</h3>
                <p>
                Each band connects a <em>source program assembly</em> (left) to a
                <em>target program assembly</em> (right).<br><br>
                <span style="color:#16a34a;font-weight:600;">Green bands</span> — positive causal regulation<br>
                <span style="color:#dc2626;font-weight:600;">Red bands</span> — negative causal regulation<br><br>
                Band width is proportional to cumulative literature evidence count.
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with c2:
            prog_stats = (
                causal_df[causal_df["src_program"].isin(named)]
                .groupby(["src_program", "rel_type"])["evidence_count"]
                .sum()
                .reset_index()
            )
            fig_bar = px.bar(
                prog_stats,
                x="src_program",
                y="evidence_count",
                color="rel_type",
                barmode="group",
                template="plotly_white",
                color_discrete_map={
                    "positive_causal": "#16a34a",
                    "negative_causal": "#dc2626",
                },
                labels={
                    "src_program": "Source Program",
                    "evidence_count": "Evidence Count",
                    "rel_type": "Causal Type",
                },
            )
            fig_bar.update_layout(
                height=260,
                margin=dict(l=40, r=20, t=30, b=40),
                legend_title="",
                showlegend=True,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_hubs:
        st.markdown("**Top hub assemblies by causal out-degree**")
        st.caption(
            "Net causal flow = out-degree − in-degree. Positive = predominantly a driver; negative = predominantly a target."
        )

        names_map = dict(
            zip(complex_table["complex_id"], complex_table["assigned_name"])
        )

        deg_show = degree_df.copy()
        deg_show["assigned_name"] = (
            deg_show["assembly"].map(names_map).fillna("(unassigned)")
        )
        deg_show["program"] = deg_show["program"].fillna("unclassified")
        deg_show = deg_show.sort_values("out_degree", ascending=False).head(30)

        fig_deg = px.bar(
            deg_show,
            x="out_degree",
            y="assembly",
            orientation="h",
            color="program",
            color_discrete_map={
                "EL": "#dc2626",
                "IH": "#2563eb",
                "ME": "#16a34a",
                "unclassified": "#94a3b8",
            },
            custom_data=["assigned_name", "net_flow"],
            template="plotly_white",
            labels={
                "out_degree": "Causal Out-Degree",
                "assembly": "Assembly",
                "program": "Program",
            },
        )
        fig_deg.update_traces(
            hovertemplate="<b>%{y}</b> — %{customdata[0]}<br>Out: %{x} · Net flow: %{customdata[1]}<extra></extra>"
        )
        fig_deg.update_layout(
            height=520,
            margin=dict(l=10, r=20, t=20, b=40),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_deg, use_container_width=True)

    with tab_edges:
        col1, col2 = st.columns([2, 1])
        with col1:
            edge_search = st.text_input(
                "Filter by gene or assembly:", "", key="causal_s"
            )
        with col2:
            rel_filter = st.selectbox(
                "Causal type:",
                ["All", "positive_causal", "negative_causal"],
                key="causal_rel",
            )

        edge_show = causal_df.copy()
        edge_show["src_name"] = edge_show["src_assembly"].map(names_map).fillna("")
        edge_show["tgt_name"] = edge_show["tgt_assembly"].map(names_map).fillna("")
        if edge_search:
            mask = (
                edge_show["src_gene"].str.contains(edge_search, case=False, na=False)
                | edge_show["tgt_gene"].str.contains(edge_search, case=False, na=False)
                | edge_show["src_assembly"].str.contains(
                    edge_search, case=False, na=False
                )
                | edge_show["tgt_assembly"].str.contains(
                    edge_search, case=False, na=False
                )
            )
            edge_show = edge_show[mask]
        if rel_filter != "All":
            edge_show = edge_show[edge_show["rel_type"] == rel_filter]

        st.caption(f"{len(edge_show):,} causal edges")
        st.dataframe(
            edge_show[
                [
                    "src_assembly",
                    "src_name",
                    "src_program",
                    "src_gene",
                    "tgt_assembly",
                    "tgt_name",
                    "tgt_program",
                    "tgt_gene",
                    "rel_type",
                    "source",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                "src_assembly": st.column_config.TextColumn("Src Assembly"),
                "src_name": st.column_config.TextColumn("Src Name"),
                "src_program": st.column_config.TextColumn("Src Program"),
                "src_gene": st.column_config.TextColumn("Src Gene"),
                "tgt_assembly": st.column_config.TextColumn("Tgt Assembly"),
                "tgt_name": st.column_config.TextColumn("Tgt Name"),
                "tgt_program": st.column_config.TextColumn("Tgt Program"),
                "tgt_gene": st.column_config.TextColumn("Tgt Gene"),
                "rel_type": st.column_config.TextColumn("Causal Type"),
                "source": st.column_config.TextColumn("KG Source"),
            },
        )

# ══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Hypothesis Generator":
    st.markdown(
        '<h2 style="color:#0f1b2d;margin-bottom:.25rem;">Assembly Hypothesis Generator</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-sub">'
        "This generator is part of the manuscript’s companion portal. It uses the study’s "
        "MCF10A assembly atlas and integrated causal network to turn selected study observations "
        "into candidate mechanisms and pathway diagrams. Generated outputs are exploratory "
        "AI hypotheses—not manuscript findings or experimental evidence—and require independent "
        "experimental validation."
        "</p>",
        unsafe_allow_html=True,
    )
    st.caption("Default model: OpenAI GPT-6.1 Terra. Select a provider and model below.")

    if not KG_TOOLS_OK:
        st.error(f"kg_tools module failed to import: {_KG_ERR_MSG}")
        st.stop()

    # ── Provider / model / key selector ───────────────────────────────────────
    PROVIDERS = list(kg_tools.PROVIDER_MODELS.keys())
    PROVIDER_HELP = {
        "OpenCode": ("OPENCODE_API_KEY", "opencode.ai", "https://opencode.ai/"),
        "OpenRouter": (
            "OPENROUTER_API_KEY",
            "openrouter.ai",
            "https://openrouter.ai/keys",
        ),
        "OpenAI": (
            "OPENAI_API_KEY",
            "platform.openai.com",
            "https://platform.openai.com/api-keys",
        ),
    }

    # Intentional user-configuration write contract: only this explicitly
    # requested UI action may update the app-local provider-key file. It does
    # not produce or modify research data, figures, tables, or manuscripts.
    def _save_key_to_env(env_var: str, value: str) -> None:
        """Persist one approved provider key after an explicit user save action."""
        approved_env_vars = {details[0] for details in PROVIDER_HELP.values()}
        if env_var not in approved_env_vars:
            raise ValueError(f"Unsupported provider environment variable: {env_var}")
        if "\n" in value or "\r" in value:
            raise ValueError("Provider API keys must be single-line values.")

        lines = (
            USER_CONFIGURATION_ENV.read_text().splitlines()
            if USER_CONFIGURATION_ENV.exists()
            else []
        )
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{env_var}="):
                lines[i] = f"{env_var}={value}"
                found = True
                break
        if not found:
            lines.append(f"{env_var}={value}")
        USER_CONFIGURATION_ENV.write_text("\n".join(lines) + "\n")
        os.environ[env_var] = value  # Apply the saved user configuration immediately.

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        provider = st.selectbox(
            "Provider",
            PROVIDERS,
            index=PROVIDERS.index("OpenAI"),
            key="hyp_provider",
        )
    with p_col2:
        model_list = kg_tools.PROVIDER_MODELS[provider]
        preferred_models = {
            "OpenCode": "mimo-v2.5-free",
            "OpenAI": "gpt-6.1-terra",
        }
        preferred = preferred_models.get(provider, model_list[0])
        default_model = preferred if preferred in model_list else model_list[0]
        model_choice = st.selectbox(
            "Model",
            model_list,
            index=model_list.index(default_model),
            key="hyp_model",
        )

    env_var, site_label, key_url = PROVIDER_HELP[provider]
    env_val = os.getenv(env_var, "")

    key_col, btn_col = st.columns([5, 1])
    with key_col:
        api_key_input = st.text_input(
            f"API Key ({site_label})",
            value=env_val,
            type="password",
            help=f"Click 'Save' to persist this key to .env for future sessions. Get a key at {key_url}",
            key=f"hyp_apikey_{provider}",
        )
    with btn_col:
        st.markdown("<div style='margin-top:1.65rem;'></div>", unsafe_allow_html=True)
        if st.button(
            "💾 Save",
            key="hyp_save_key",
            help="Save this key to .env so you don't have to re-enter it",
        ):
            if api_key_input.strip():
                _save_key_to_env(env_var, api_key_input.strip())
                st.success("Saved!")
            else:
                st.warning("Enter a key first.")

    # Provider status badges
    badge_parts = []
    for prov, (evar, _, _) in PROVIDER_HELP.items():
        has_key = bool(os.getenv(evar))
        color = "#16a34a" if has_key else "#94a3b8"
        dot = "●" if has_key else "○"
        badge_parts.append(
            f'<span style="color:{color};font-size:.75rem;margin-right:.8rem;">'
            f"{dot} {prov}</span>"
        )
    st.markdown(
        '<div style="margin:-.25rem 0 .75rem;">'
        + "".join(badge_parts)
        + '<span style="font-size:.7rem;color:#94a3b8;">● key saved to .env &nbsp; ○ not set</span>'
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="card">
        <h3>Example queries</h3>
        <p style="font-size:.85rem;">
        · How does MON1B regulate endolysosomal remodeling during EMT?<br>
        · What causal role does the COMMD–CCC complex (C0098) play in hybrid assembly stability?<br>
        · Which assemblies act as EL→IH transition drivers and what are their causal targets?<br>
        · How does VIM connect mesenchymal assemblies to the cytoskeletal remodeling program?<br>
        · What is the causal significance of SRC in the EMT assembly network?
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    user_query = st.text_area(
        "Enter your hypothesis query:",
        height=100,
        placeholder="e.g. How does MON1B regulate endolysosomal remodeling during EMT?",
        key="hyp_query",
    )

    if "hyp_result" not in st.session_state:
        st.session_state["hyp_result"] = None
    if "hyp_context" not in st.session_state:
        st.session_state["hyp_context"] = None

    if st.button(
        "Generate Hypothesis", type="primary", disabled=not user_query.strip()
    ):
        if not api_key_input.strip():
            st.error(
                f"An API key is required for {provider}. "
                f"Enter it above or set {env_var} in the .env file."
            )
        else:
            with st.spinner(
                f"Querying {provider} ({model_choice}) — retrieving assembly context…"
            ):
                try:
                    hypothesis, context_used = kg_tools.generate_hypothesis(
                        query=user_query,
                        api_key=api_key_input.strip(),
                        model=model_choice,
                        provider=provider,
                    )
                    st.session_state["hyp_result"] = hypothesis
                    st.session_state["hyp_context"] = context_used
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    msg = str(e)
                    if "429" in msg:
                        st.warning(
                            "This model is temporarily rate-limited. Select a different model from the dropdown and try again."
                        )
                    elif "404" in msg:
                        st.warning(
                            "This model is no longer available for free. Select a different model from the dropdown."
                        )
                    else:
                        st.error(f"Hypothesis generation failed: {e}")

    if st.session_state.get("hyp_result"):
        _render_hypothesis(
            st.session_state["hyp_result"], st.session_state["hyp_context"]
        )

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Downloads":
    st.markdown(
        '<h2 style="color:#0f1b2d;margin-bottom:.25rem;">Data Downloads</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-sub">Download the primary supplementary datasets accompanying the manuscript. '
        "These are the current manuscript supplements. Please cite the paper if you use these data.</p>",
        unsafe_allow_html=True,
    )

    DL_MANIFEST = [
        {
            "label": "Table S1 — BE Proteomics",
            "desc": "Clinical protein abundance comparisons for BE, HGD and EAC versus normal adjacent tissue, with provenance.",
            "path": MANUSCRIPT_TABLES / "Table S1. BE proteomics.xlsx",
            "fname": "table_s1_be_proteomics.xlsx",
        },
        {
            "label": "Table S2 — Clinical Pathway Enrichment",
            "desc": "Stage-specific clinical pathway enrichment with constituent genes and adjusted significance values.",
            "path": MANUSCRIPT_TABLES / "Table S2. Clinical pathway enrichment.xlsx",
            "fname": "table_s2_clinical_pathway_enrichment.xlsx",
        },
        {
            "label": "Table S3 — MCF10A and CP-A PPI Networks",
            "desc": f"MCF10A: {len(ppi_net):,} co-complex associations. CP-A: {len(cpa_net):,} released "
            "associations with Day 0/4/8 scores, source metadata and excluded-record audit. "
            "The CP-A weight is the mean of available accepted scores; blank day scores are not zero.",
            "path": MANUSCRIPT_TABLES / "Table S3. EMT PPI network.xlsx",
            "alt": None,
            "fname": "table_s3_emt_ppi_network.xlsx",
        },
        {
            "label": "EPIC PPIs (score ≥ 0.5)",
            "desc": f"{len(ppi05):,} MCF10A EPIC co-elution interactions with Day 0–8 temporal scores.",
            "path": HERE / "data/downloads/emt_ppi_epic_score05.xlsx",
            "alt": None,
            "fname": "emt_ppi_epic_score05.xlsx",
        },
        {
            "label": "Table S4 — Protein Assemblies",
            "desc": f"{len(complex_table):,} MCF10A ClusterOne protein assemblies with member gene lists.",
            "path": MANUSCRIPT_TABLES / "Table S4. ClusterOne Protein Assemblies.xlsx",
            "alt": None,
            "fname": "table_s4_protein_assemblies.xlsx",
        },
        {
            "label": "Table S5 — Normalized CFMS Profiles",
            "desc": "Normalized MCF10A fractionation profiles across the EMT time course and replicate labels.",
            "path": MANUSCRIPT_TABLES / "Table S5. EMT CFMS normalized.xlsx",
            "fname": "table_s5_emt_cfms_normalized.xlsx",
        },
        {
            "label": "Table S6 — Regulated Assemblies",
            "desc": f"{s6_df['complex_id'].nunique():,} regulated MCF10A assemblies with gene-level up/mid/down patterns across EMT.",
            "path": MANUSCRIPT_TABLES
            / "Table S6. ClusterOne Regulated Assemblies.xlsx",
            "alt": None,
            "fname": "table_s6_regulated_assemblies.xlsx",
        },
        {
            "label": "Table S7 — BIONIC Trajectories",
            "desc": "MCF10A PPI trajectories, trend types and rank-aware volatility scores with provenance.",
            "path": MANUSCRIPT_TABLES / "Table S7. BIONIC trajectories.xlsx",
            "alt": None,
            "fname": "table_s7_bionic_trajectories.xlsx",
        },
        {
            "label": "Table S8 — Structural Atlas",
            "desc": f"{len(struct_df):,} candidate PPI pairs with interface confidence scores, PDB coverage and novelty flags. "
            f"{int(struct_df['high_confidence_model'].eq(True).sum()):,} pairs are high-confidence interface models.",
            "path": MANUSCRIPT_TABLES / "Table S8. Structural atlas.xlsx",
            "alt": None,
            "fname": "table_s8_structural_atlas.xlsx",
        },
        {
            "label": "Table S9 — Phosphosite Interface Atlas",
            "desc": "Phosphosite-containing interface hypotheses with model-level confidence and peptide/domain annotations.",
            "path": MANUSCRIPT_TABLES / "Table S9. Structural atlas phospho.xlsx",
            "fname": "table_s9_structural_atlas_phospho.xlsx",
        },
        {
            "label": "Table S10 — Genomic Alteration Frequencies",
            "desc": "cBioPortal gene-level alteration frequencies and profiled-sample denominators, with provenance.",
            "path": MANUSCRIPT_TABLES / "Table S10. cBioPortal genomic alteration frequencies.xlsx",
            "fname": "table_s10_genomic_alteration_frequencies.xlsx",
        },
        {
            "label": "Regulated Complexes GMT (GSEA)",
            "desc": "GMT-format geneset file for GSEA/ssGSEA of the regulated assembly programs.",
            "path": HERE / "data/downloads/emt_regulated_assembly_genesets.gmt",
            "alt": HERE / "data/assembly_data/emt_regulated_assembly_genesets.gmt",
            "fname": "emt_regulated_assembly_genesets.gmt",
        },
    ]

    for item in DL_MANIFEST:
        fpath = item["path"]
        if not fpath.exists() and item.get("alt"):
            fpath = item["alt"]

        if fpath and fpath.exists():
            with open(fpath, "rb") as f:
                data = f.read()
            size_kb = len(data) // 1024
            st.markdown(
                f'<div class="dl-row">'
                f'<div class="dl-meta">'
                f'<div class="dl-title">📄 {item["label"]}</div>'
                f'<div class="dl-desc">{item["desc"]} &nbsp;·&nbsp; {size_kb:,} KB</div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )
            st.download_button(
                label=f"Download {item['fname']}",
                data=data,
                file_name=item["fname"],
                mime="application/octet-stream",
                key=f"dl_{item['fname']}",
            )
        else:
            st.markdown(
                f'<div class="dl-row">'
                f'<div class="dl-meta">'
                f'<div class="dl-title" style="color:#94a3b8;">📄 {item["label"]}</div>'
                f'<div class="dl-desc" style="color:#cbd5e1;">{item["desc"]}</div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# CONTACT & HELP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📧 Contact & Help":
    st.markdown(
        '<h2 style="color:#0f1b2d;margin-bottom:.25rem;">Contact & Help</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-sub">Questions about the data, analysis pipelines, or this companion website.</p>',
        unsafe_allow_html=True,
    )

    # ── Correspondence row ────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
        <div class="card" style="height:100%;">
            <h3>Scientific Correspondence</h3>
            <p style="margin-bottom:.9rem;">
                For questions about the study, data interpretation, or collaboration:
            </p>
            <div style="display:flex;flex-direction:column;gap:.75rem;">
                <div style="display:flex;align-items:flex-start;gap:.75rem;">
                    <div style="width:36px;height:36px;border-radius:50%;background:#eff6ff;
                                display:flex;align-items:center;justify-content:center;
                                flex-shrink:0;font-size:.9rem;">👤</div>
                    <div>
                        <div style="font-weight:600;font-size:.88rem;color:#0f1b2d;">Indranil Paul</div>
                        <div style="font-size:.78rem;color:#64748b;">First Author</div>
                        <a href="mailto:pauli@ohsu.edu"
                           style="font-size:.82rem;color:#1d4ed8;text-decoration:none;">pauli@ohsu.edu</a>
                    </div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:.75rem;">
                    <div style="width:36px;height:36px;border-radius:50%;background:#eff6ff;
                                display:flex;align-items:center;justify-content:center;
                                flex-shrink:0;font-size:.9rem;">👤</div>
                    <div>
                        <div style="font-weight:600;font-size:.88rem;color:#0f1b2d;">Andrew Emili, PhD</div>
                        <div style="font-size:.78rem;color:#64748b;">Principal Investigator</div>
                        <a href="mailto:emili@ohsu.edu"
                           style="font-size:.82rem;color:#1d4ed8;text-decoration:none;">emili@ohsu.edu</a>
                    </div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
        <div class="card" style="height:100%;">
            <h3>Citation</h3>
            <div class="cite-block" style="margin-bottom:1rem;">
                <strong>{escape(manuscript['title'])}</strong><br>
                Paul I, Padhorny D, Joseph TP, Sivagnanam S, <em>et al.</em><br>
                <em>Manuscript in preparation.</em>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Data & Code Availability ──────────────────────────────────────────────
    st.markdown(
        f"""
    <div class="card" style="margin-top:.25rem;">
        <h3>Data &amp; Code Availability</h3>
        <p style="margin:0 0 .5rem;font-size:.88rem;line-height:1.7;">
            The reference atlas comprises {len(ppi_net):,} MCF10A co-complex associations,
            {len(complex_table):,} assemblies and {s6_df['complex_id'].nunique():,} remodeled assemblies.
            Supporting CP-A data include {len(cpa_net):,} released associations with Day 0, 4 and 8 scores.
            Select CP-A in Browse Data or download Table S3 for the complete release and source metadata.
        </p>
        <p style="margin:0;font-size:.88rem;line-height:1.7;">
            Raw proteomics data are deposited in
            <strong>PRIDE / ProteomeXchange</strong> (accession pending).
        </p>
        <p style="margin:.5rem 0 0;font-size:.88rem;line-height:1.7;">
            Analysis code is available on <strong>GitHub</strong> (link pending).
        </p>
        <p style="margin:.5rem 0 0;font-size:.88rem;line-height:1.7;">
            All supplementary tables are available via the
            <a href="?section=Downloads" style="color:#1d4ed8;text-decoration:none;">Downloads</a> page
            and from the journal website upon publication.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Using this resource ───────────────────────────────────────────────────
    st.markdown(
        """
    <div class="card" style="margin-top:.25rem;">
        <h3>Using This Resource</h3>
        <div style="display:flex;gap:2rem;flex-wrap:wrap;">
            <div style="flex:1;min-width:200px;">
                <div style="font-size:.78rem;font-weight:600;color:#0f1b2d;
                            text-transform:uppercase;letter-spacing:.05em;margin-bottom:.4rem;">
                    Browsers
                </div>
                <p style="margin:0;">Optimized for Chrome and Firefox. Safari is supported but
                interactive network views perform best in Chromium-based browsers.</p>
            </div>
            <div style="flex:1;min-width:200px;">
                <div style="font-size:.78rem;font-weight:600;color:#0f1b2d;
                            text-transform:uppercase;letter-spacing:.05em;margin-bottom:.4rem;">
                    Hypothesis Generator
                </div>
                <p style="margin:0;">Requires an API key for OpenCode or OpenRouter.
                Keys can be set in the <code>.env</code> file for persistent access or entered
                per-session in the UI.</p>
            </div>
            <div style="flex:1;min-width:200px;">
                <div style="font-size:.78rem;font-weight:600;color:#0f1b2d;
                            text-transform:uppercase;letter-spacing:.05em;margin-bottom:.4rem;">
                    Feedback
                </div>
                <p style="margin:0;">Bug reports and feature suggestions are welcome — please
                contact the authors above.</p>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Logo ──────────────────────────────────────────────────────────────────
    knight = HERE / "www" / "knight.png"
    if knight.exists():
        _, logo_col, _ = st.columns([1, 1, 1])
        with logo_col:
            st.image(str(knight), width=180)
