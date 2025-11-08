"""
CitationReplacementAgent

Given the output of ExternalInferenceAgent.infer(...), predict which citations
would be better replacements for currently used citations in the target document.

Usage:
    agent = CitationReplacementAgent()
    suggestions = agent.predict_replacements(inference_result, current_citations=["CASE_002", "CASE_004"])
"""
from typing import List, Dict, Any, Optional
import numpy as np
import difflib
import json


def _extract_trs(case: Dict[str, Any]) -> float:
    v = case.get("trs", 0.0)
    if isinstance(v, dict):
        return float(v.get("score", 0.0))
    try:
        return float(v)
    except Exception:
        return 0.0


def _safe_float(d: Dict[str, Any], k: str) -> float:
    try:
        return float(d.get(k, 0.0))
    except Exception:
        return 0.0


class CitationReplacementAgent:
    """
    Suggest replacements for OCR-extracted citations using only the
    ExternalInferenceAgent.infer(...) output (no online APIs).

    For each OCR citation string, the agent:
      - best-effort matches it to a retrieved case from inference_result
      - computes candidate ranking among retrieved_cases (TRS + title-similarity + jurisdiction)
      - returns replacements that improve TRS over the matched baseline by min_trs_improvement

    Output format:
      {
        "<ocr_text>": {
           "matched": { "case_id":..., "title":..., "baseline_trs":... } | None,
           "suggestions": [ { "case_id","title","score","trs","similarity","jurisdiction","justification" }, ... ]
        },
        ...
      }
    """

    def __init__(self, title_match_threshold: float = 0.45, weights: Optional[Dict[str, float]] = None):
        # Title-match threshold for considering an OCR string matched to a retrieved case
        self.title_match_threshold = title_match_threshold
        # Weights for local scoring function (higher => more important)
        self.weights = weights or {
            "trs": 0.7,
            "title_sim": 0.25,
            "jur": 0.05,
            "uncertainty_penalty": 0.05
        }

    def _title_similarity(self, a: str, b: str) -> float:
        """Simple fuzzy ratio normalized to [0,1] using difflib."""
        if not a or not b:
            return 0.0
        a_norm = " ".join(a.lower().split())
        b_norm = " ".join(b.lower().split())
        return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()

    def _jurisdiction_bonus(self, target_jur: Optional[str], cand_jur: Optional[str]) -> float:
        if not target_jur or not cand_jur:
            return 0.0
        return 1.0 if target_jur.lower().strip() == cand_jur.lower().strip() else 0.0

    def predict_replacements_from_ocr(
        self,
        inference_result: Dict[str, Any],
        current_citations_ocr: List[str],
        top_n: int = 3,
        min_trs_improvement: float = 0.03
    ) -> Dict[str, Any]:
        """
        Main function. Uses only inference_result['retrieved_cases'] and the OCR strings.

        Args:
          inference_result: output from ExternalInferenceAgent.infer(...)
          current_citations_ocr: list of OCR-extracted citation strings (raw)
          top_n: number of replacement suggestions per OCR citation
          min_trs_improvement: minimum TRS improvement over baseline to recommend

        Returns mapping described above.
        """
        if not isinstance(inference_result, dict) or "retrieved_cases" not in inference_result:
            raise ValueError("inference_result must be the output of ExternalInferenceAgent.infer(...)")

        retrieved: List[Dict[str, Any]] = inference_result.get("retrieved_cases", [])
        target = inference_result.get("target", {})
        target_jur = target.get("jurisdiction", None)

        # Precompute quick lookup and normalized titles
        candidates = []
        for c in retrieved:
            candidates.append({
                "case": c,
                "case_id": c.get("case_id"),
                "title": (c.get("title") or "").strip(),
                "trs": _extract_trs(c),
                "similarity_score": _safe_float(c, "similarity_score"),
                "jurisdiction": c.get("jurisdiction", None),
                "uncertainty": _safe_float(c, "uncertainty")
            })

        out: Dict[str, Any] = {}

        for ocr_text in current_citations_ocr:
            ocr_norm = (ocr_text or "").strip()
            # find best match among retrieved by title similarity or exact case_id match
            best_match = None
            best_match_score = 0.0
            for cand in candidates:
                # check case_id exact match first
                if ocr_norm and cand["case_id"] and ocr_norm.lower() == str(cand["case_id"]).lower():
                    best_match = cand
                    best_match_score = 1.0
                    break
                # title fuzzy match
                title_sim = self._title_similarity(ocr_norm, cand["title"])
                if title_sim > best_match_score:
                    best_match = cand
                    best_match_score = title_sim

            matched_info = None
            baseline_trs = 0.0
            if best_match and best_match_score >= self.title_match_threshold:
                matched_info = {
                    "case_id": best_match["case_id"],
                    "title": best_match["title"],
                    "match_score": round(best_match_score, 3),
                    "baseline_trs": round(best_match["trs"], 3)
                }
                baseline_trs = best_match["trs"]
            else:
                matched_info = None
                baseline_trs = 0.0

            # Score other candidates as potential replacements
            scored = []
            for cand in candidates:
                # skip if same as matched baseline case
                if matched_info and cand["case_id"] == matched_info["case_id"]:
                    continue
                # compute local combined score
                trs = float(cand["trs"])
                title_sim = self._title_similarity(ocr_norm, cand["title"])
                jur_bonus = self._jurisdiction_bonus(target_jur, cand["jurisdiction"])
                unc = float(cand["uncertainty"])
                score = (
                    self.weights["trs"] * trs +
                    self.weights["title_sim"] * title_sim +
                    self.weights["jur"] * jur_bonus -
                    self.weights["uncertainty_penalty"] * unc
                )
                scored.append({
                    "case_id": cand["case_id"],
                    "title": cand["title"],
                    "trs": round(trs, 3),
                    "title_similarity": round(title_sim, 3),
                    "jurisdiction": cand["jurisdiction"],
                    "uncertainty": round(unc, 3),
                    "score": round(score, 4)
                })

            # select candidates that improve TRS over baseline by min_trs_improvement
            scored.sort(key=lambda x: x["score"], reverse=True)
            suggestions = []
            for s in scored:
                if s["trs"] >= baseline_trs + min_trs_improvement:
                    # build short justification
                    parts = []
                    parts.append(f"TRS {s['trs']:.3f} vs baseline {baseline_trs:.3f}")
                    if s["title_similarity"] >= 0.5:
                        parts.append(f"title matches OCR (sim {s['title_similarity']:.2f})")
                    if s["jurisdiction"] and target_jur and s["jurisdiction"].lower().strip() == target_jur.lower().strip():
                        parts.append("same jurisdiction")
                    if s["uncertainty"] > 0.2:
                        parts.append("higher uncertainty in candidate flagged")
                    justification = "; ".join(parts)
                    s_out = s.copy()
                    s_out["justification"] = justification
                    suggestions.append(s_out)
                if len(suggestions) >= top_n:
                    break

            out[ocr_text] = {
                "matched": matched_info,
                "baseline_trs": round(float(baseline_trs), 3),
                "suggestions": suggestions
            }

        return out

    def _score_candidate(self, candidate: Dict[str, Any]) -> float:
        """Compute a lightweight ranking score for a candidate case (higher is better)."""
        trs = float(candidate.get("trs", 0.0) if not isinstance(candidate.get("trs", 0.0), dict) else candidate["trs"].get("score", 0.0))
        sim = float(candidate.get("similarity_score", 0.0))
        jur = float(candidate.get("jurisdiction_score", 0.0))
        unc = float(candidate.get("uncertainty", 0.0))
        score = (
            self.weights["trs"] * trs +
            self.weights["similarity"] * sim +
            self.weights["jurisdiction"] * jur +
            self.weights["uncertainty"] * unc
        )
        return float(score)

    def _index_by_case_id(self, retrieved_cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {c["case_id"]: c for c in retrieved_cases}

    def predict_replacements(
        self,
        inference_result: Dict[str, Any],
        current_citations: List[str],
        top_n: int = 3,
        min_trs_improvement: float = 0.03
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        For each citation in current_citations, return a ranked list of suggested replacements.

        Args:
            inference_result: output from ExternalInferenceAgent.infer(...)
            current_citations: list of case_id strings that are currently cited in the document
            top_n: number of replacement suggestions per citation
            min_trs_improvement: only suggest replacements whose TRS is at least this much higher
                                 than the current citation's TRS (helps avoid noisy swaps)

        Returns:
            dict keyed by original case_id -> list of suggestion dicts:
            {
                "orig_case_id": [
                    {"case_id": "...", "title": "...", "score": 0.92, "trs":0.9, "similarity_score":0.87, "justification": "..."},
                    ...
                ],
                ...
            }
        """
        if not isinstance(inference_result, dict) or "retrieved_cases" not in inference_result:
            raise ValueError("inference_result must be a dict containing 'retrieved_cases'")

        retrieved = inference_result["retrieved_cases"]
        idx = self._index_by_case_id(retrieved)

        # Precompute scores for all candidates
        scored_candidates = []
        for c in retrieved:
            c_score = self._score_candidate(c)
            scored_candidates.append((c_score, c))
        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        suggestions: Dict[str, List[Dict[str, Any]]] = {}
        # Build a fast lookup for candidate by id
        for orig in current_citations:
            orig_entry = idx.get(orig)
            if orig_entry is None:
                # If original citation not in retrieved set, treat its TRS as 0 and continue
                orig_trs = 0.0
                orig_jur = None
            else:
                orig_trs = float(orig_entry.get("trs", 0.0) if not isinstance(orig_entry.get("trs", 0.0), dict) else orig_entry["trs"].get("score", 0.0))
                orig_jur = orig_entry.get("jurisdiction")

            # collect top candidates that are not the original and that improve TRS by min_trs_improvement
            cand_list = []
            for score_val, cand in scored_candidates:
                if cand["case_id"] == orig:
                    continue
                cand_trs = float(cand.get("trs", 0.0) if not isinstance(cand.get("trs", 0.0), dict) else cand["trs"].get("score", 0.0))
                # require at least slight improvement (configurable)
                if cand_trs + 1e-9 < orig_trs + min_trs_improvement:
                    continue
                justification = self._build_justification(orig, orig_entry, cand)
                cand_list.append({
                    "case_id": cand["case_id"],
                    "title": cand.get("title", ""),
                    "score": float(score_val),
                    "trs": float(np.clip(cand_trs, 0.0, 1.0)),
                    "similarity_score": float(np.clip(cand.get("similarity_score", 0.0), 0.0, 1.0)),
                    "jurisdiction": cand.get("jurisdiction", "Unknown"),
                    "alignment_type": cand.get("alignment_type", "neutral"),
                    "justification": justification
                })
                if len(cand_list) >= top_n:
                    break

            suggestions[orig] = cand_list

        return suggestions

    def _build_justification(self, orig_id: str, orig_entry: Optional[Dict[str, Any]], candidate: Dict[str, Any]) -> str:
        """
        Small human-readable justification why candidate is suggested over original.
        Keeps to 1-2 sentences.
        """
        cand_trs = float(candidate.get("trs", 0.0) if not isinstance(candidate.get("trs", 0.0), dict) else candidate["trs"].get("score", 0.0))
        parts = []
        parts.append(f"Suggested because it has higher TRS ({cand_trs:.2f}).")
        # prefer same jurisdiction
        if orig_entry is not None:
            orig_jur = orig_entry.get("jurisdiction", "Unknown")
            cand_jur = candidate.get("jurisdiction", "Unknown")
            if orig_jur.lower() == cand_jur.lower():
                parts.append("Shares the same jurisdiction.")
        # alignment-based hint
        alignment = candidate.get("alignment_type", "")
        if alignment:
            parts.append(f"Alignment: {alignment}.")
        return " ".join(parts)


if __name__ == "__main__":
    # Quick local demo when running the file directly.
    import json, sys
    print("CitationReplacementAgent demo")
    # Expect path to a JSON file with inference_result or use example from ExternalInferenceAgent demo
    # Minimal fallback: build a tiny fake inference_result if none provided.
    fake_inference_result = {
        "target": {"case_id":"T1","title":"Target","year":2020,"jurisdiction":"Supreme Court of India"},
        "retrieved_cases": [
            {"case_id":"CASE_001","title":"A","trs":0.9,"similarity_score":0.92,"jurisdiction":"Supreme Court of India","alignment_type":"supports","uncertainty":0.02},
            {"case_id":"CASE_002","title":"B","trs":0.6,"similarity_score":0.65,"jurisdiction":"Supreme Court of India","alignment_type":"supports","uncertainty":0.1},
            {"case_id":"CASE_003","title":"C","trs":0.7,"similarity_score":0.72,"jurisdiction":"Other Court","alignment_type":"neutral","uncertainty":0.05},
            {"case_id":"CASE_004","title":"D","trs":0.4,"similarity_score":0.45,"jurisdiction":"Supreme Court of India","alignment_type":"contradicts","uncertainty":0.12}
        ],
        "overall_external_coherence_score": 0.65,
        "short_summary": "Fake"
    }
    agent = CitationReplacementAgent()
    current = ["CASE_002", "CASE_004", "NON_EXISTENT"]
    suggestions = agent.predict_replacements(fake_inference_result, current_citations=current, top_n=2)
    print(json.dumps(suggestions, indent=2))