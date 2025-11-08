"""OCR, RAG query, and citation graph routes."""
from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from collections import deque
import math
import sys
import networkx as nx

from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import current_user, jwt_required

from ..extensions import db
from ..models import Document

bp = Blueprint("ocr", __name__, url_prefix="/api/ocr")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_document_file_path(document: Document) -> Path | None:
    """Resolve the file system path for a document upload."""
    if not document.file_url:
        return None
    if document.file_url.startswith("/uploads/"):
        filename = document.file_url.replace("/uploads/", "")
        upload_folder = Path(current_app.config["UPLOAD_FOLDER"])  # configured in app factory
        return upload_folder / filename
    return Path(document.file_url)


def _ensure_project_root():
    """Add project root to sys.path for top-level imports (lexai/ etc.)."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _compute_internal_analysis(document: Document, file_path: Path) -> dict | None:
    """Run the multi-model internal coherence agent and cache result on the document.

    Returns the analysis dict on success, or None on failure.
    """
    try:
        _ensure_project_root()
        # Import here to avoid import overhead unless needed
        from lexai.agents.multi_model_internal_coherence_agent_light_pdf import (
            run_internal_coherence_agent,
        )

        analysis = run_internal_coherence_agent(str(file_path))
        # Cache to DB
        document.internal_analysis = analysis
        db.session.commit()
        return analysis
    except Exception as e:  # noqa: BLE001
        current_app.logger.exception(f"Internal analysis failed: {e}")
        return None


def _get_dummy_analysis(title: str) -> dict:
    """Generate dummy internal analysis data for demonstration purposes."""
    return {
        "File Name": f"{title}.pdf",
        "Title": title,
        "Citations": [
            "Article 21 of the Constitution",
            "Article 19(1)(a)",
            "K.S. Puttaswamy v. Union of India",
            "Maneka Gandhi v. Union of India",
            "Gobind v. State of Madhya Pradesh",
            "Section 66A of IT Act, 2000"
        ],
        "Claims": [
            "Right to privacy is a fundamental right under Article 21 of the Constitution.",
            "Privacy includes autonomy over fundamental personal choices.",
            "State surveillance must meet the test of proportionality.",
            "Data protection is essential for informational privacy.",
            "Privacy is not absolute and can be restricted by reasonable state action."
        ],
        "Contradictions": [
            "Claim that privacy is fundamental vs. earlier precedent treating it as non-fundamental",
            "Absolute protection claim vs. qualified protection in practice"
        ],
        "Final Report": {
            "Key Argument Flows": [
                "Privacy is intrinsic to dignity and autonomy",
                "Previous cases have recognized privacy in limited contexts",
                "Modern challenges require comprehensive privacy protection",
                "Balancing individual rights with legitimate state interests"
            ],
            "Detected Contradictions": [
                "Tension between absolute right rhetoric and qualified implementation",
                "Conflict between informational privacy and national security imperatives"
            ],
            "Logical Gaps": [
                "Insufficient framework for balancing privacy against competing interests",
                "Lack of clear standards for proportionality review",
                "Ambiguity in scope of reasonable restrictions"
            ],
            "Coherence Score": 0.78,
            "Brief Commentary": "The judgment establishes a strong foundation for privacy rights while acknowledging practical limitations. The analysis reveals good internal consistency in recognizing privacy as fundamental, though implementation details show some logical gaps in balancing tests. The coherence score of 0.78 reflects solid argumentation with room for refinement in proportionality standards."
        }
    }


# ---------------------------------------------------------------------------
# OCR Processing
# ---------------------------------------------------------------------------
@bp.post("/process/<document_id>")
@jwt_required()
def process_document(document_id: str):
    """Run OCR & build citation graph for a document, caching results."""
    document = Document.query.filter_by(id=document_id, user_id=current_user.id).one_or_none()
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    file_path = get_document_file_path(document)
    if file_path is None or not file_path.exists():
        return jsonify({"error": "File not found"}), HTTPStatus.NOT_FOUND

    try:
        _ensure_project_root()
        from lexai.agents.ocr_agent import process_pdf  # performs OCR
        from app.services.citation_graph_service import generate_citation_graph

        # If already processed, return cached
        if document.ocr_text and document.citation_graph:
            return jsonify({
                "success": True,
                "already_processed": True,
                "document_id": document.id,
            })

        ocr_result = process_pdf(str(file_path))
        ocr_text = ocr_result.get("full_text", "")
        ocr_meta = {
            "pages": ocr_result.get("pages", []),
            "stats": {
                "num_pages": len(ocr_result.get("pages", [])),
                "num_citations": len(ocr_result.get("citations", [])),
            },
        }

        citation_graph = generate_citation_graph(ocr_text, document.title)

        document.ocr_text = ocr_text
        document.ocr_metadata = ocr_meta
        document.citation_graph = citation_graph
        db.session.commit()

        # Optionally trigger internal analysis automatically if not present
        analysis_info = None
        try:
            if document.internal_analysis is None:
                analysis = _compute_internal_analysis(document, file_path)
                if analysis is not None:
                    analysis_info = {
                        "analysis_cached": True,
                        "coherence_score": analysis.get("Final Report", {}).get("Coherence Score"),
                    }
        except Exception:
            # Non-fatal; return primary OCR + graph result
            pass

        resp = {
            "success": True,
            "document_id": document.id,
            "num_nodes": len(citation_graph.get("nodes", [])),
            "num_edges": len(citation_graph.get("edges", [])),
        }
        if analysis_info:
            resp["internal_analysis"] = analysis_info
        return jsonify(resp)
    except Exception as e:  # noqa: BLE001
        current_app.logger.exception("OCR processing failed")
        return jsonify({"error": f"OCR processing failed: {e}"}), HTTPStatus.INTERNAL_SERVER_ERROR


# ---------------------------------------------------------------------------
# RAG Query
# ---------------------------------------------------------------------------
@bp.post("/query/<document_id>")
@jwt_required()
def query_document(document_id: str):
    """Answer a query using cached OCR text (fallback to live OCR)."""
    document = Document.query.filter_by(id=document_id, user_id=current_user.id).one_or_none()
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query text is required"}), HTTPStatus.BAD_REQUEST

    try:
        _ensure_project_root()
        from lexai.agents.query_pdf_rag_ocr import (
            query_rag_from_text,
            query_pdf_with_ocr,
        )

        if document.ocr_text:
            result = query_rag_from_text(document.ocr_text, query, document.title)
        else:
            file_path = get_document_file_path(document)
            if file_path is None or not file_path.exists():
                return jsonify({"error": "Document file not found"}), HTTPStatus.NOT_FOUND
            result = query_pdf_with_ocr(str(file_path), query)

        return jsonify({
            "success": True,
            "query": query,
            "answer": result.get("Answer", ""),
            "title": result.get("Title", document.title),
        })
    except Exception as e:  # noqa: BLE001
        current_app.logger.exception("Query processing failed")
        return jsonify({"error": f"Query processing failed: {e}"}), HTTPStatus.INTERNAL_SERVER_ERROR


# ---------------------------------------------------------------------------
# Citation Graph Raw
# ---------------------------------------------------------------------------
@bp.get("/citation-graph/<document_id>")
@jwt_required()
def get_citation_graph(document_id: str):
    """Return raw citation graph JSON (nodes & edges). Generate if missing."""
    document = Document.query.filter_by(id=document_id, user_id=current_user.id).one_or_none()
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    if document.citation_graph:
        return jsonify({"success": True, "graph": document.citation_graph})

    if not document.ocr_text:
        return jsonify({"error": "No OCR data available. Process the document first."}), HTTPStatus.BAD_REQUEST

    try:
        _ensure_project_root()
        from app.services.citation_graph_service import generate_citation_graph
        citation_graph = generate_citation_graph(document.ocr_text, document.title)
        document.citation_graph = citation_graph
        db.session.commit()
        return jsonify({"success": True, "graph": citation_graph})
    except Exception as e:  # noqa: BLE001
        current_app.logger.exception("Failed to generate citation graph")
        return jsonify({"error": f"Failed to generate citation graph: {e}"}), HTTPStatus.INTERNAL_SERVER_ERROR


# ---------------------------------------------------------------------------
# Citation Nodes (positioned + filtered)
# ---------------------------------------------------------------------------
@bp.get("/citation-nodes/<document_id>")
@jwt_required()
def get_citation_nodes(document_id: str):
    """Return positioned citation nodes with filtering and layout stats."""
    document = Document.query.filter_by(id=document_id, user_id=current_user.id).one_or_none()
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    # Ensure we have a graph
    if not document.citation_graph:
        if not document.ocr_text:
            return jsonify({"nodes": [], "edges": [], "total_nodes": 0}), HTTPStatus.OK
        try:
            _ensure_project_root()
            from app.services.citation_graph_service import generate_citation_graph
            document.citation_graph = generate_citation_graph(document.ocr_text, document.title)
            db.session.commit()
        except Exception as e:  # noqa: BLE001
            current_app.logger.error(f"Citation graph generation failed: {e}")
            return jsonify({"nodes": [], "edges": [], "total_nodes": 0}), HTTPStatus.OK

    graph = document.citation_graph or {}
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    total_nodes = len(nodes)
    if total_nodes == 0:
        return jsonify({
            "nodes": [],
            "edges": [],
            "total_nodes": 0,
            "filtered_nodes": 0,
            "showing_top": 0,
            "has_more": False,
        })

    # --- Filtering parameters ---
    limit = min(max(request.args.get("limit", type=int) or 50, 1), 200)
    min_citations = request.args.get("min_citations", default=0, type=int) or 0
    year_filter = request.args.get("year", type=int)

    # Build citation counts & adjacency
    citation_counts: dict[str, int] = {}
    adjacency: dict[str, list[str]] = {}
    reverse_cited: set[str] = set()

    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if not src or not tgt:
            continue
        citation_counts[tgt] = citation_counts.get(tgt, 0) + 1
        adjacency.setdefault(src, []).append(tgt)
        reverse_cited.add(tgt)

    # Enrich nodes
    for n in nodes:
        n_id = n.get("id")
        n["citation_count"] = citation_counts.get(n_id, 0)

    # Apply filters
    filtered = [n for n in nodes if n.get("citation_count", 0) >= min_citations]
    if year_filter is not None:
        filtered = [n for n in filtered if n.get("year") == year_filter]

    # Sort & slice
    filtered.sort(key=lambda n: n.get("citation_count", 0), reverse=True)
    visible = filtered[:limit]
    visible_ids = {n.get("id") for n in visible}

    # Choose layout: default to force-directed ("scattered"); tree available via layout=tree
    layout_kind = (request.args.get("layout") or "force").lower()
    positions: dict[str, dict[str, float]] = {}

    if layout_kind == "tree":
        # Hierarchical tree layout
        root_ids = [n.get("id") for n in visible if n.get("id") not in reverse_cited or n.get("id") not in visible_ids]
        if not root_ids:
            root_ids = [n.get("id") for n in visible[:3]]

        levels: dict[str, int] = {}
        q: deque[str] = deque()
        for rid in root_ids:
            levels[rid] = 0
            q.append(rid)
        for n in visible:
            nid = n.get("id")
            if nid not in levels:
                levels[nid] = 0
                q.append(nid)

        while q:
            current = q.popleft()
            cur_level = levels[current]
            for child in adjacency.get(current, []):
                if child in visible_ids and child not in levels:
                    levels[child] = cur_level + 1
                    q.append(child)

        max_level = max(levels.values()) if levels else 0
        level_groups: dict[int, list[str]] = {i: [] for i in range(max_level + 1)}
        for nid, lvl in levels.items():
            level_groups[lvl].append(nid)

        y_gap = 80 / (max_level + 1) if max_level > 0 else 40
        for lvl in range(max_level + 1):
            group = level_groups[lvl]
            count = len(group)
            x_gap = 80 / (count + 1) if count else 40
            for idx, nid in enumerate(group):
                positions[nid] = {"x": 10 + (idx + 1) * x_gap, "y": 10 + lvl * y_gap}

        # Slight anti-overlap pass
        repulsion = 120
        for _ in range(15):
            forces = {nid: {"x": 0.0, "y": 0.0} for nid in positions}
            ids = list(positions.keys())
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    ax, ay = positions[a]["x"], positions[a]["y"]
                    bx, by = positions[b]["x"], positions[b]["y"]
                    dx = ax - bx
                    dy = ay - by
                    dist = math.sqrt(dx * dx + dy * dy) + 0.1
                    force = repulsion / (dist * dist)
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    forces[a]["x"] += fx
                    forces[a]["y"] += fy * 0.25
                    forces[b]["x"] -= fx
                    forces[b]["y"] -= fy * 0.25
            for nid in positions:
                positions[nid]["x"] = max(5, min(95, positions[nid]["x"] + forces[nid]["x"] * 0.2))
                positions[nid]["y"] = max(5, min(95, positions[nid]["y"] + forces[nid]["y"] * 0.1))
    else:
        # Force-directed using NetworkX spring_layout (Fruchterman–Reingold)
        G = nx.DiGraph()
        for n in visible:
            nid = n.get("id")
            G.add_node(nid)
        for e in edges:
            src = e.get("source")
            tgt = e.get("target")
            if src in visible_ids and tgt in visible_ids:
                G.add_edge(src, tgt)

        n_count = max(len(G), 1)
        # k controls ideal distance; scale with node count for readability
        k = 1.5 / math.sqrt(n_count)
        pos = nx.spring_layout(G, k=k, iterations=200, seed=42, threshold=1e-4)

        # Normalize to 10..90 range
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = (max_x - min_x) or 1.0
            span_y = (max_y - min_y) or 1.0
            for nid, (px, py) in pos.items():
                nxp = 10 + 80 * (px - min_x) / span_x
                nyp = 10 + 80 * (py - min_y) / span_y
                positions[nid] = {"x": float(max(5, min(95, nxp))), "y": float(max(5, min(95, nyp)))}

    # Format nodes
    formatted_nodes = []
    for n in visible:
        nid = n.get("id")
        pos = positions.get(nid, {"x": 50, "y": 50})
        formatted_nodes.append({
            "id": nid,
            "title": n.get("title", "Unknown"),
            "x": pos["x"],
            "y": pos["y"],
            "citations": n.get("citation_count", 0),
            "year": n.get("year") or 0,
        })

    # Visible edges
    visible_edges = [
        e for e in edges
        if e.get("source") in visible_ids and e.get("target") in visible_ids
    ]

    return jsonify({
        "nodes": formatted_nodes,
        "edges": visible_edges,
        "total_nodes": total_nodes,
        "filtered_nodes": len(filtered),
        "showing_top": len(visible),
        "has_more": len(filtered) > len(visible),
    })


# ---------------------------------------------------------------------------
# Internal Coherence Analysis (compute on demand or return cached)
# ---------------------------------------------------------------------------
@bp.get("/internal-analysis/<document_id>")
@jwt_required()
def get_internal_analysis(document_id: str):
    """Return internal coherence analysis for a document.

    If not cached, compute using the multi-model internal agent, cache, and return.
    Falls back to dummy data if analysis is unavailable.
    """
    document = Document.query.filter_by(id=document_id, user_id=current_user.id).one_or_none()
    if document is None:
        return jsonify({"error": "Document not found"}), HTTPStatus.NOT_FOUND

    # Force recompute if requested
    force = request.args.get("force") in {"1", "true", "True", "yes", "on"}

    # If cached and not forcing, return directly
    if document.internal_analysis and not force:
        return jsonify({"success": True, "analysis": document.internal_analysis})

    # Need the file to compute
    file_path = get_document_file_path(document)
    if file_path is None or not file_path.exists():
        # Return dummy data if file not found
        return jsonify({"success": True, "analysis": _get_dummy_analysis(document.title), "is_dummy": True})

    analysis = _compute_internal_analysis(document, file_path)
    if analysis is None:
        # Return dummy data if analysis fails
        return jsonify({"success": True, "analysis": _get_dummy_analysis(document.title), "is_dummy": True})

    return jsonify({"success": True, "analysis": analysis})

