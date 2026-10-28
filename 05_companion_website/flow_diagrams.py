"""Polished Plotly rendering for LLM-generated Mermaid pathway flowcharts."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from html import escape
import math
import re
import textwrap

import plotly.graph_objects as go


@dataclass(frozen=True)
class FlowEdge:
    source: str
    target: str
    label: str = ""
    kind: str = "activation"  # activation | inhibition | association


_NODE_RE = re.compile(
    r'^([A-Za-z_][\w.-]*)\s*(?:\["?([^\]"]+)"?\]|\("?([^\)"]+)"?\)|\{"?([^\}"]+)"?\})?'
)
_EDGE_RE = re.compile(
    r"^(.*?)\s*(-->|==>|-\.->|\.->|--x|==x|--\||-\|)\s*"
    r"(?:\|([^|]*)\|\s*)?(.*?)\s*$"
)


def _join_mermaid_lines(text: str) -> list[str]:
    """Join wrapped Mermaid statements without confusing edge-label pipes."""
    raw = text.replace("```mermaid", "").replace("```", "").splitlines()
    lines: list[str] = []
    buffer: list[str] = []
    balance = 0
    open_edge_label = False

    for raw_line in raw:
        line = raw_line.strip().rstrip(";")
        if not line or line.startswith("%%"):
            continue
        if re.match(r"^(flowchart|graph)\s+", line, re.I):
            continue
        if line.startswith(
            ("subgraph", "end", "style ", "classDef ", "class ", "linkStyle ")
        ):
            continue

        buffer.append(line)
        balance += sum(line.count(char) for char in "[({")
        balance -= sum(line.count(char) for char in "])}")
        # Inhibitory operators (``-|``/``--|``) contain a pipe but do not
        # open an edge label; remove them before testing label-pipe parity.
        label_text = line.replace("--|", "").replace("-|", "")
        if label_text.count("|") % 2:
            open_edge_label = not open_edge_label

        if balance <= 0 and not open_edge_label:
            statement = " ".join(buffer)
            lines.extend(part.strip() for part in statement.split(";") if part.strip())
            buffer = []
            balance = 0

    if buffer:
        lines.append(" ".join(buffer))
    return lines


def _node(token: str) -> tuple[str, str]:
    token = token.strip()
    match = _NODE_RE.match(token)
    if not match:
        clean = re.sub(r"[^A-Za-z0-9_.-]", "", token)
        return clean, clean
    node_id = match.group(1)
    label = next((value for value in match.groups()[1:] if value is not None), node_id)
    return node_id, re.sub(r"\s+", " ", label).strip()


def parse_mermaid_flowchart(text: str) -> tuple[dict[str, str], list[FlowEdge]]:
    """Parse the conservative Mermaid subset emitted by the hypothesis generator.

    The parser also repairs conventional biological inhibition syntax (``A -| B``),
    which is not valid Mermaid but is commonly produced by scientific LLM prompts.
    """
    nodes: dict[str, str] = {}
    edges: list[FlowEdge] = []

    for statement in _join_mermaid_lines(text):
        edge_match = _EDGE_RE.match(statement)
        if edge_match:
            left, operator, edge_label, right = edge_match.groups()
            source, source_label = _node(left)
            target, target_label = _node(right)
            if not source or not target:
                continue
            kind = (
                "inhibition"
                if operator in {"--x", "==x", "--|", "-|"}
                or re.search(r"inhibit|suppress|block|restrain", edge_label or "", re.I)
                else "association"
                if operator in {"-.->", ".->"}
                else "activation"
            )
            nodes.setdefault(source, source_label)
            nodes.setdefault(target, target_label)
            edges.append(FlowEdge(source, target, (edge_label or "").strip(), kind))
            continue

        node_id, label = _node(statement)
        if node_id and label:
            nodes.setdefault(node_id, label)

    return nodes, edges


def _layered_layout(
    nodes: dict[str, str], edges: list[FlowEdge]
) -> dict[str, tuple[float, float]]:
    """Deterministic top-down Sugiyama-style layout with barycentric ordering."""
    order = list(nodes)
    rank = {node: index for index, node in enumerate(order)}
    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    indegree = {node: 0 for node in nodes}

    for edge in edges:
        if edge.target not in outgoing[edge.source]:
            outgoing[edge.source].append(edge.target)
            incoming[edge.target].append(edge.source)
            indegree[edge.target] = indegree.get(edge.target, 0) + 1

    queue = deque(node for node in order if indegree.get(node, 0) == 0)
    topo: list[str] = []
    remaining_indegree = indegree.copy()
    while queue:
        node = queue.popleft()
        topo.append(node)
        for target in outgoing[node]:
            remaining_indegree[target] -= 1
            if remaining_indegree[target] == 0:
                queue.append(target)
    topo.extend(node for node in order if node not in topo)
    topo_index = {node: index for index, node in enumerate(topo)}

    layer = {node: 0 for node in nodes}
    for node in topo:
        for target in outgoing[node]:
            # Ignore only true cycle-closing edges when assigning layers.
            if topo_index[target] <= topo_index[node]:
                continue
            layer[target] = max(layer[target], layer[node] + 1)

    layers: dict[int, list[str]] = defaultdict(list)
    for node in order:
        layers[layer[node]].append(node)

    x_position: dict[str, float] = {}
    for depth in sorted(layers):
        members = layers[depth]
        if depth:
            members.sort(
                key=lambda node: (
                    sum(x_position.get(parent, 0.0) for parent in incoming[node])
                    / max(1, len(incoming[node])),
                    rank[node],
                )
            )
        spacing = 3.7
        for index, node in enumerate(members):
            x_position[node] = (index - (len(members) - 1) / 2) * spacing

    # A reverse sweep aligns parents over their children without changing order.
    for depth in sorted(layers, reverse=True):
        members = layers[depth]
        for node in members:
            children = outgoing[node]
            if children and len(members) == 1:
                x_position[node] = sum(x_position[child] for child in children) / len(
                    children
                )

    return {node: (x_position[node], -layer[node] * 2.15) for node in nodes}


def _wrap_label(label: str) -> str:
    assembly = re.match(r"^(C\d{4})\s+(.+)$", label, re.I)
    if assembly:
        identifier, name = assembly.groups()
        wrapped = "<br>".join(escape(line) for line in textwrap.wrap(name, width=20))
        return f'<span style="font-size:10px">{escape(identifier.upper())}</span><br><b>{wrapped}</b>'
    wrapped = "<br>".join(escape(line) for line in textwrap.wrap(label, width=22))
    return f"<b>{wrapped}</b>"


def _node_style(label: str, indegree: int, outdegree: int) -> dict[str, str]:
    lower = label.lower()
    if indegree == 0:
        return {
            "fill": "#172554",
            "line": "#1e3a8a",
            "text": "#ffffff",
            "accent": "#60a5fa",
        }
    if re.search(r"\bC\d{4}\b", label, re.I):
        return {
            "fill": "#eef2ff",
            "line": "#6366f1",
            "text": "#312e81",
            "accent": "#818cf8",
        }
    if any(
        word in lower
        for word in ("suppress", "checkpoint", "stress", "damp", "hypoxia")
    ):
        return {
            "fill": "#fff1f2",
            "line": "#f43f5e",
            "text": "#881337",
            "accent": "#fb7185",
        }
    if outdegree == 0 or any(
        word in lower for word in ("priming", "activation", "response")
    ):
        return {
            "fill": "#ecfdf5",
            "line": "#10b981",
            "text": "#064e3b",
            "accent": "#34d399",
        }
    return {
        "fill": "#f8fafc",
        "line": "#94a3b8",
        "text": "#1e293b",
        "accent": "#38bdf8",
    }


def _rounded_rectangle_path(
    cx: float, cy: float, width: float, height: float, radius: float
) -> str:
    left, right = cx - width / 2, cx + width / 2
    bottom, top = cy - height / 2, cy + height / 2
    return (
        f"M {left + radius},{bottom} L {right - radius},{bottom} "
        f"Q {right},{bottom} {right},{bottom + radius} L {right},{top - radius} "
        f"Q {right},{top} {right - radius},{top} L {left + radius},{top} "
        f"Q {left},{top} {left},{top - radius} L {left},{bottom + radius} "
        f"Q {left},{bottom} {left + radius},{bottom} Z"
    )


def mermaid_to_plotly(text: str) -> go.Figure | None:
    """Render a Mermaid flowchart as a publication-styled interactive Plotly figure."""
    nodes, edges = parse_mermaid_flowchart(text)
    if not edges:
        return None

    positions = _layered_layout(nodes, edges)
    incoming_count = defaultdict(int)
    outgoing_count = defaultdict(int)
    for edge in edges:
        incoming_count[edge.target] += 1
        outgoing_count[edge.source] += 1

    width = 2.75
    height = 0.88
    figure = go.Figure()

    # Edges are drawn first so node cards mask their endpoints.
    target_edge_index = defaultdict(int)
    target_edge_total = defaultdict(int)
    for edge in edges:
        target_edge_total[edge.target] += 1

    for edge in edges:
        x0, y0 = positions[edge.source]
        x1, y1 = positions[edge.target]
        source_y = y0 - height / 2
        target_y = y1 + height / 2
        target_edge_index[edge.target] += 1
        count = target_edge_total[edge.target]
        offset = (target_edge_index[edge.target] - (count + 1) / 2) * 0.20
        target_x = x1 + offset
        color = "#e11d48" if edge.kind == "inhibition" else "#64748b"
        dash = "dot" if edge.kind in {"inhibition", "association"} else "solid"

        # Soft route halo followed by the actual connector.
        figure.add_shape(
            type="line",
            x0=x0,
            y0=source_y,
            x1=target_x,
            y1=target_y,
            line={"color": "rgba(255,255,255,0.95)", "width": 6},
            layer="below",
        )
        figure.add_shape(
            type="line",
            x0=x0,
            y0=source_y,
            x1=target_x,
            y1=target_y,
            line={"color": color, "width": 2.2, "dash": dash},
            layer="below",
        )

        dx, dy = target_x - x0, target_y - source_y
        distance = math.hypot(dx, dy) or 1.0
        ux, uy = dx / distance, dy / distance
        if edge.kind == "inhibition":
            bar_half = 0.15
            figure.add_shape(
                type="line",
                x0=target_x - uy * bar_half,
                y0=target_y + ux * bar_half,
                x1=target_x + uy * bar_half,
                y1=target_y - ux * bar_half,
                line={"color": color, "width": 3.2},
            )
        else:
            arrow_length, arrow_half = 0.22, 0.11
            base_x, base_y = target_x - ux * arrow_length, target_y - uy * arrow_length
            path = (
                f"M {target_x},{target_y} "
                f"L {base_x - uy * arrow_half},{base_y + ux * arrow_half} "
                f"L {base_x + uy * arrow_half},{base_y - ux * arrow_half} Z"
            )
            figure.add_shape(
                type="path",
                path=path,
                fillcolor=color,
                line={"color": color, "width": 1},
            )

        if edge.label:
            mid_x = (x0 + target_x) / 2
            mid_y = (source_y + target_y) / 2
            figure.add_annotation(
                x=mid_x,
                y=mid_y + 0.55,
                text=f"<b>{escape(edge.label)}</b>",
                showarrow=False,
                font={"family": "Inter, Arial, sans-serif", "size": 10, "color": color},
                bgcolor="rgba(255,255,255,0.96)",
                bordercolor=color,
                borderwidth=1,
                borderpad=4,
                opacity=0.98,
            )

    for node_id, label in nodes.items():
        x, y = positions[node_id]
        style = _node_style(label, incoming_count[node_id], outgoing_count[node_id])
        # Delicate shadow, card, and top accent rule.
        figure.add_shape(
            type="path",
            path=_rounded_rectangle_path(x + 0.06, y - 0.07, width, height, 0.15),
            fillcolor="rgba(15,23,42,0.12)",
            line={"width": 0},
            layer="below",
        )
        figure.add_shape(
            type="path",
            path=_rounded_rectangle_path(x, y, width, height, 0.15),
            fillcolor=style["fill"],
            line={"color": style["line"], "width": 1.6},
        )
        figure.add_shape(
            type="line",
            x0=x - width / 2 + 0.16,
            y0=y + height / 2 - 0.08,
            x1=x + width / 2 - 0.16,
            y1=y + height / 2 - 0.08,
            line={"color": style["accent"], "width": 3.2},
        )
        figure.add_annotation(
            x=x,
            y=y - 0.02,
            text=_wrap_label(label),
            showarrow=False,
            align="center",
            font={
                "family": "Inter, Arial, sans-serif",
                "size": 12,
                "color": style["text"],
            },
            hovertext=escape(label),
        )

    xs = [position[0] for position in positions.values()]
    ys = [position[1] for position in positions.values()]
    layer_count = len({round(y, 3) for y in ys})
    figure.update_layout(
        template="none",
        height=max(430, min(760, 155 + layer_count * 105)),
        margin={"l": 24, "r": 24, "t": 24, "b": 24},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Inter, Arial, sans-serif"},
        xaxis={
            "visible": False,
            "fixedrange": False,
            "range": [min(xs) - width / 2 - 0.5, max(xs) + width / 2 + 0.5],
        },
        yaxis={
            "visible": False,
            "fixedrange": False,
            "range": [min(ys) - height / 2 - 0.55, max(ys) + height / 2 + 0.55],
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        hovermode=False,
        showlegend=False,
    )
    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.995,
        y=0.005,
        text="Solid arrow: activation&nbsp;&nbsp; • &nbsp;&nbsp;Dashed bar: inhibition/constraint",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font={"size": 9, "color": "#94a3b8"},
    )
    return figure


def mermaid_to_svg(text: str) -> str | None:
    """Render a Mermaid flowchart as a vector SVG string suitable for WeasyPrint PDF embedding."""
    try:
        nodes, edges = parse_mermaid_flowchart(text)
        if not edges:
            return None
        positions = _layered_layout(nodes, edges)
        incoming_count = defaultdict(int)
        outgoing_count = defaultdict(int)
        for edge in edges:
            incoming_count[edge.target] += 1
            outgoing_count[edge.source] += 1

        xs = [pos[0] for pos in positions.values()]
        ys = [pos[1] for pos in positions.values()]
        min_x, max_x = min(xs), max(xs)

        scale_x = 140.0
        scale_y = 100.0
        margin_x, margin_y = 120.0, 50.0

        span_x = (max_x - min_x) if max_x > min_x else 1.0
        span_y = (max(ys) - min(ys)) if max(ys) > min(ys) else 1.0

        svg_width = max(680.0, span_x * scale_x + margin_x * 2)
        svg_height = max(220.0, span_y * scale_y + margin_y * 2)

        def tx(x):
            return margin_x + (x - min_x) * scale_x

        def ty(y):
            return margin_y + (-y) * scale_y

        node_w, node_h = 180.0, 50.0
        rx = 8.0

        svg_parts = [
            f'<svg width="100%" viewBox="0 0 {svg_width} {svg_height}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin:1.2rem 0;">'
        ]
        svg_parts.append("""<defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
          </marker>
          <marker id="arrow-inhibit" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#e11d48" />
          </marker>
        </defs>""")

        for edge in edges:
            x0, y0 = positions[edge.source]
            x1, y1 = positions[edge.target]
            sx, sy = tx(x0), ty(y0) + node_h / 2.2
            ex, ey = tx(x1), ty(y1) - node_h / 2.2

            is_inhibit = edge.kind == "inhibition"
            color = "#e11d48" if is_inhibit else "#64748b"
            dash = (
                'stroke-dasharray="4,4"'
                if is_inhibit or edge.kind == "association"
                else ""
            )
            marker = "url(#arrow-inhibit)" if is_inhibit else "url(#arrow)"

            svg_parts.append(
                f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{color}" stroke-width="2" {dash} marker-end="{marker}" />'
            )
            if edge.label:
                mx, my = (sx + ex) / 2, (sy + ey) / 2 - 40
                lbl = escape(edge.label)
                svg_parts.append(
                    f'<g><rect x="{mx - 35:.1f}" y="{my - 10:.1f}" width="70" height="20" rx="4" fill="#ffffff" stroke="{color}" stroke-width="0.8"/><text x="{mx:.1f}" y="{my + 4:.1f}" font-family="sans-serif" font-size="10" fill="{color}" text-anchor="middle">{lbl}</text></g>'
                )

        for node_id, label in nodes.items():
            x, y = positions[node_id]
            cx, cy = tx(x), ty(y)
            nx, ny = cx - node_w / 2, cy - node_h / 2
            style = _node_style(label, incoming_count[node_id], outgoing_count[node_id])
            fill = style["fill"]
            line = style["line"]
            text_color = style["text"]
            accent = style["accent"]

            svg_parts.append(
                f'<rect x="{nx:.1f}" y="{ny:.1f}" width="{node_w}" height="{node_h}" rx="{rx}" fill="{fill}" stroke="{line}" stroke-width="1.8" />'
            )
            svg_parts.append(
                f'<line x1="{nx + 10:.1f}" y1="{ny + 3:.1f}" x2="{nx + node_w - 10:.1f}" y2="{ny + 3:.1f}" stroke="{accent}" stroke-width="3" stroke-linecap="round" />'
            )

            clean_lbl = escape(label)
            lines = textwrap.wrap(clean_lbl, width=22)
            start_y = cy - (len(lines) - 1) * 7.0
            for i, l in enumerate(lines):
                svg_parts.append(
                    f'<text x="{cx:.1f}" y="{start_y + i * 14:.1f}" font-family="Inter, Arial, sans-serif" font-size="11" font-weight="600" fill="{text_color}" text-anchor="middle" dominant-baseline="middle">{l}</text>'
                )

        svg_parts.append("</svg>")
        return "".join(svg_parts)
    except Exception:
        return None
