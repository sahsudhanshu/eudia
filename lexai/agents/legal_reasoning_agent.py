
















"""Legal Reasoning Agent for generating structured multi-aspect analysis.

This agent receives a target case and retrieved precedent cases from the
ExternalInferenceAgent and generates comprehensive legal reasoning output.
"""

from typing import List, Dict, Any, Optional
import json
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


class LegalReasoningAgent:
    """
    Agent for generating structured legal reasoning analysis.
    
    Takes target case and retrieved precedents, generates multi-aspect
    reasoning report with evidence tables, uncertainty analysis, and
    recommendations.
    
    Can use:
    1. Local LLM (via transformers)
    2. Rule-based fallback (deterministic)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_llm: bool = False,
        device: Optional[str] = None,
        max_length: int = 2048
    ):
        """
        Initialize the LegalReasoningAgent.
        
        Args:
            model_name: Hugging Face model name for LLM reasoning.
                       If None and use_llm=True, uses "gpt2" as fallback.
            use_llm: Whether to use LLM or rule-based reasoning.
            device: Device to run model on ('cuda', 'cpu', or None for auto).
            max_length: Maximum token length for generation.
        """
        self.use_llm = use_llm
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        if use_llm:
            self._initialize_llm(model_name)
    
    def _initialize_llm(self, model_name: Optional[str] = None):
        """Initialize the LLM for reasoning generation."""
        if model_name is None:
            model_name = "gpt2"  # Fallback to GPT-2
        
        print(f"Loading LLM: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1,
                max_length=self.max_length
            )
            print(f"LLM loaded successfully on {self.device}")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            print("Falling back to rule-based reasoning")
            self.use_llm = False
    
    def generate_reasoning(
        self,
        target_case: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        external_coherence_score: float
    ) -> Dict[str, Any]:
        """
        Generate comprehensive legal reasoning analysis.
        
        Args:
            target_case: Target case dict with case_id, title, text, etc.
            retrieved_cases: List of retrieved precedent cases from ExternalInferenceAgent.
            external_coherence_score: Overall coherence score from external inference.
        
        Returns:
            Dictionary with multi-aspect reasoning analysis.
        """
        if self.use_llm and self.generator is not None:
            return self._generate_llm_reasoning(
                target_case, retrieved_cases, external_coherence_score
            )
        else:
            return self._generate_rule_based_reasoning(
                target_case, retrieved_cases, external_coherence_score
            )
    
    def _generate_llm_reasoning(
        self,
        target_case: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        external_coherence_score: float
    ) -> Dict[str, Any]:
        """Generate reasoning using LLM."""
        # Build prompt
        prompt = self._build_reasoning_prompt(target_case, retrieved_cases)
        
        try:
            # Generate with LLM
            outputs = self.generator(
                prompt,
                max_length=self.max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text'][len(prompt):]
            
            # Parse LLM output and structure
            reasoning = self._parse_llm_output(
                generated_text,
                target_case,
                retrieved_cases,
                external_coherence_score
            )
            reasoning['reasoning_method'] = 'local_llm'
            
        except Exception as e:
            print(f"LLM generation failed: {e}")
            print("Falling back to rule-based reasoning")
            reasoning = self._generate_rule_based_reasoning(
                target_case, retrieved_cases, external_coherence_score
            )
        
        return reasoning
    
    def _generate_rule_based_reasoning(
        self,
        target_case: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        external_coherence_score: float
    ) -> Dict[str, Any]:
        """Generate reasoning using deterministic rules."""
        
        # Count alignment types
        supporting = [c for c in retrieved_cases if c.get('alignment_type') == 'supports']
        contradicting = [c for c in retrieved_cases if c.get('alignment_type') == 'contradicts']
        neutral = [c for c in retrieved_cases if c.get('alignment_type') == 'neutral']
        
        # Generate summary
        summary_long = self._generate_summary(
            target_case, retrieved_cases, external_coherence_score,
            len(supporting), len(contradicting), len(neutral)
        )
        
        # Generate aspect analysis
        aspect_analysis = self._generate_aspect_analysis(
            target_case, supporting, contradicting, neutral
        )
        
        # Generate evidence table
        evidence_table = self._generate_evidence_table(retrieved_cases)
        
        # Generate uncertainty analysis
        uncertainty = self._generate_uncertainty_analysis(
            retrieved_cases, external_coherence_score
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            target_case, retrieved_cases, external_coherence_score,
            len(supporting), len(contradicting)
        )
        
        # Compute overall coherence
        overall_coherence = self._compute_overall_coherence(
            external_coherence_score, retrieved_cases
        )
        
        return {
            "summary_long": summary_long,
            "aspect_analysis": aspect_analysis,
            "detailed_evidence_table": evidence_table,
            "uncertainty_and_limits": uncertainty,
            "recommended_next_steps": recommendations,
            "reasoning_method": "rule_based",
            "overall_coherence_score": float(overall_coherence)
        }
    
    def _build_reasoning_prompt(
        self,
        target_case: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for LLM reasoning."""
        prompt = f"""Analyze the following legal case and provide comprehensive reasoning.

TARGET CASE:
Title: {target_case.get('title', 'N/A')}
Year: {target_case.get('year', 'N/A')}
Jurisdiction: {target_case.get('jurisdiction', 'N/A')}

RETRIEVED PRECEDENT CASES:
"""
        for i, case in enumerate(retrieved_cases[:5], 1):
            prompt += f"\n{i}. {case.get('title', 'N/A')} ({case.get('year', 'N/A')})"
            prompt += f"\n   TRS: {case.get('trs', 0):.3f}, Alignment: {case.get('alignment_type', 'N/A')}"
        
        prompt += "\n\nProvide analysis covering:\n"
        prompt += "1. Summary of key legal principles\n"
        prompt += "2. How precedents support or contradict the target\n"
        prompt += "3. Jurisdictional considerations\n"
        prompt += "4. Uncertainty factors\n"
        prompt += "5. Recommendations\n\n"
        
        return prompt
    
    def _parse_llm_output(
        self,
        generated_text: str,
        target_case: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        external_coherence_score: float
    ) -> Dict[str, Any]:
        """Parse LLM output into structured format."""
        # This is a simplified parser - in production, use more robust parsing
        
        # Extract sections from generated text
        summary = generated_text[:500] if len(generated_text) > 500 else generated_text
        
        # Fall back to rule-based for structured components
        supporting = [c for c in retrieved_cases if c.get('alignment_type') == 'supports']
        contradicting = [c for c in retrieved_cases if c.get('alignment_type') == 'contradicts']
        neutral = [c for c in retrieved_cases if c.get('alignment_type') == 'neutral']
        
        aspect_analysis = self._generate_aspect_analysis(
            target_case, supporting, contradicting, neutral
        )
        evidence_table = self._generate_evidence_table(retrieved_cases)
        uncertainty = self._generate_uncertainty_analysis(
            retrieved_cases, external_coherence_score
        )
        recommendations = self._generate_recommendations(
            target_case, retrieved_cases, external_coherence_score,
            len(supporting), len(contradicting)
        )
        
        return {
            "summary_long": summary,
            "aspect_analysis": aspect_analysis,
            "detailed_evidence_table": evidence_table,
            "uncertainty_and_limits": uncertainty,
            "recommended_next_steps": recommendations,
            "reasoning_method": "local_llm",
            "overall_coherence_score": float(external_coherence_score)
        }
    
    def _generate_summary(
        self,
        target_case: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        coherence_score: float,
        num_supporting: int,
        num_contradicting: int,
        num_neutral: int
    ) -> str:
        """Generate comprehensive summary."""
        title = target_case.get('title', 'Unknown Case')
        year = target_case.get('year', 'N/A')
        jurisdiction = target_case.get('jurisdiction', 'N/A')
        
        summary = f"Legal Analysis: {title}\n\n"
        summary += f"This analysis examines the case '{title}' ({year}, {jurisdiction}) "
        summary += f"in the context of {len(retrieved_cases)} retrieved precedent cases. "
        
        summary += f"The overall external coherence score is {coherence_score:.3f}, "
        if coherence_score >= 0.7:
            summary += "indicating strong alignment with established precedents. "
        elif coherence_score >= 0.5:
            summary += "suggesting moderate alignment with precedent case law. "
        else:
            summary += "revealing limited alignment with retrieved precedents. "
        
        summary += f"\n\nOf the retrieved cases, {num_supporting} support the target reasoning, "
        summary += f"{num_contradicting} present contradicting perspectives, "
        summary += f"and {num_neutral} are contextually neutral. "
        
        if num_supporting > num_contradicting:
            summary += "The weight of precedent favors the target case's legal position. "
        elif num_contradicting > num_supporting:
            summary += "The precedent analysis reveals significant challenges to the target position. "
        else:
            summary += "The precedent landscape presents a balanced perspective. "
        
        # Analyze top cases
        if retrieved_cases:
            top_case = retrieved_cases[0]
            summary += f"\n\nThe most relevant precedent is '{top_case.get('title', 'N/A')}' "
            summary += f"(TRS: {top_case.get('trs', 0):.3f}), which {top_case.get('alignment_type', 'relates to')} "
            summary += "the target case's reasoning. "
        
        summary += "\n\nThis analysis considers semantic similarity, contextual fit, "
        summary += "jurisdictional alignment, and temporal relevance in evaluating precedent applicability."
        
        return summary
    
    def _generate_aspect_analysis(
        self,
        target_case: Dict[str, Any],
        supporting: List[Dict[str, Any]],
        contradicting: List[Dict[str, Any]],
        neutral: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Generate aspect-by-aspect analysis."""
        aspects = []
        
        # Aspect 1: Precedent Support
        aspects.append({
            "aspect": "Precedent Support",
            "analysis": f"The target case has {len(supporting)} supporting precedents. " +
                       (f"Key supporting case: '{supporting[0].get('title', 'N/A')}' "
                        f"(TRS: {supporting[0].get('trs', 0):.3f}). " if supporting else "") +
                       "This indicates " +
                       ("strong" if len(supporting) >= 3 else "moderate" if len(supporting) > 0 else "limited") +
                       " precedential backing.",
            "strength": "high" if len(supporting) >= 3 else "medium" if len(supporting) > 0 else "low"
        })
        
        # Aspect 2: Contradicting Authority
        aspects.append({
            "aspect": "Contradicting Authority",
            "analysis": f"There are {len(contradicting)} cases that present contradicting views. " +
                       (f"Notable contradicting case: '{contradicting[0].get('title', 'N/A')}'. "
                        if contradicting else "No significant contradicting precedents found. ") +
                       "This represents " +
                       ("substantial" if len(contradicting) >= 2 else "limited") +
                       " counter-authority.",
            "strength": "high" if len(contradicting) >= 2 else "low"
        })
        
        # Aspect 3: Jurisdictional Consistency
        jurisdictions = set()
        for case in supporting + contradicting + neutral:
            jur = case.get('jurisdiction', 'Unknown')
            if jur != 'Unknown':
                jurisdictions.add(jur)
        
        target_jur = target_case.get('jurisdiction', 'Unknown')
        same_jur_count = sum(1 for c in supporting + contradicting + neutral 
                            if c.get('jurisdiction') == target_jur)
        
        aspects.append({
            "aspect": "Jurisdictional Consistency",
            "analysis": f"Cases span {len(jurisdictions)} jurisdiction(s). " +
                       f"{same_jur_count} case(s) are from the same jurisdiction as the target ({target_jur}). " +
                       "This " +
                       ("ensures strong" if same_jur_count >= 2 else "provides limited") +
                       " jurisdictional precedent.",
            "strength": "high" if same_jur_count >= 2 else "medium" if same_jur_count > 0 else "low"
        })
        
        # Aspect 4: Temporal Relevance
        target_year = target_case.get('year')
        if target_year and isinstance(target_year, int):
            recent_cases = sum(1 for c in supporting + contradicting + neutral
                             if isinstance(c.get('year'), int) and 
                             abs(c.get('year') - target_year) <= 10)
            
            aspects.append({
                "aspect": "Temporal Relevance",
                "analysis": f"{recent_cases} case(s) are within 10 years of the target ({target_year}). " +
                           "This indicates " +
                           ("strong" if recent_cases >= 2 else "moderate" if recent_cases > 0 else "limited") +
                           " temporal alignment with contemporary legal standards.",
                "strength": "high" if recent_cases >= 2 else "medium" if recent_cases > 0 else "low"
            })
        
        # Aspect 5: Semantic Coherence
        if supporting + contradicting + neutral:
            avg_similarity = np.mean([c.get('similarity_score', 0) 
                                     for c in supporting + contradicting + neutral])
            
            aspects.append({
                "aspect": "Semantic Coherence",
                "analysis": f"Average semantic similarity is {avg_similarity:.3f}. " +
                           "This indicates " +
                           ("high" if avg_similarity >= 0.7 else "moderate" if avg_similarity >= 0.5 else "low") +
                           " conceptual overlap between the target and precedent cases.",
                "strength": "high" if avg_similarity >= 0.7 else "medium" if avg_similarity >= 0.5 else "low"
            })
        
        return aspects
    
    def _generate_evidence_table(
        self,
        retrieved_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate detailed evidence table."""
        evidence_table = []
        
        for i, case in enumerate(retrieved_cases, 1):
            evidence_entry = {
                "rank": i,
                "case_id": case.get('case_id', 'N/A'),
                "case_title": case.get('title', 'N/A'),
                "year": case.get('year', 'N/A'),
                "jurisdiction": case.get('jurisdiction', 'N/A'),
                "trs": round(case.get('trs', 0), 4),
                "similarity_score": round(case.get('similarity_score', 0), 4),
                "context_fit": round(case.get('context_fit', 0), 4),
                "jurisdiction_score": round(case.get('jurisdiction_score', 0), 4),
                "alignment_type": case.get('alignment_type', 'neutral'),
                "internal_confidence": round(case.get('internal_confidence', 0), 4),
                "uncertainty": round(case.get('uncertainty', 0), 4),
                "key_reasoning": case.get('justification', 'No justification provided.'),
                "target_span": case.get('spans', {}).get('target_span', 'N/A')[:100],
                "candidate_span": case.get('spans', {}).get('candidate_span', 'N/A')[:100]
            }
            evidence_table.append(evidence_entry)
        
        return evidence_table
    
    def _generate_uncertainty_analysis(
        self,
        retrieved_cases: List[Dict[str, Any]],
        coherence_score: float
    ) -> str:
        """Generate uncertainty and limitations analysis."""
        if not retrieved_cases:
            return "No retrieved cases available for uncertainty analysis."
        
        # Calculate uncertainty metrics
        avg_uncertainty = np.mean([c.get('uncertainty', 0) for c in retrieved_cases])
        max_uncertainty = max([c.get('uncertainty', 0) for c in retrieved_cases])
        
        # Check for conflicting signals
        high_sim_low_context = sum(1 for c in retrieved_cases
                                   if c.get('similarity_score', 0) > 0.7 and 
                                   c.get('context_fit', 0) < 0.5)
        
        uncertainty = f"Uncertainty Analysis:\n\n"
        uncertainty += f"Average Uncertainty Score: {avg_uncertainty:.3f}\n"
        uncertainty += f"Maximum Uncertainty: {max_uncertainty:.3f}\n\n"
        
        if avg_uncertainty > 0.3:
            uncertainty += "⚠ High average uncertainty detected. "
            uncertainty += "This suggests significant variance between different relevance signals. "
        else:
            uncertainty += "✓ Low average uncertainty. Relevance signals are relatively consistent. "
        
        if high_sim_low_context > 0:
            uncertainty += f"\n\n⚠ {high_sim_low_context} case(s) show high semantic similarity "
            uncertainty += "but low contextual fit, indicating potential false positives. "
        
        uncertainty += f"\n\nOverall Coherence: {coherence_score:.3f}\n"
        if coherence_score < 0.6:
            uncertainty += "⚠ Limited coherence suggests the precedent corpus may not fully "
            uncertainty += "address the target case's specific legal questions. "
        
        uncertainty += "\n\nLimitations:\n"
        uncertainty += "1. Retrieved cases represent a subset of potentially relevant precedents.\n"
        uncertainty += "2. Semantic similarity may not capture nuanced legal distinctions.\n"
        uncertainty += "3. Jurisdictional differences may affect precedent applicability.\n"
        uncertainty += "4. Temporal gaps may indicate evolving legal standards.\n"
        uncertainty += "5. Missing metadata may affect jurisdiction/temporal scoring accuracy."
        
        return uncertainty
    
    def _generate_recommendations(
        self,
        target_case: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        coherence_score: float,
        num_supporting: int,
        num_contradicting: int
    ) -> List[str]:
        """Generate recommended next steps."""
        recommendations = []
        
        # Recommendation based on support ratio
        if num_supporting > num_contradicting * 2:
            recommendations.append(
                "Strong precedent support: Consider proceeding with confidence, "
                "citing the supporting cases identified in the evidence table."
            )
        elif num_contradicting > num_supporting:
            recommendations.append(
                "Significant counter-authority exists: Develop strong distinguishing "
                "arguments or consider alternative legal theories."
            )
        else:
            recommendations.append(
                "Mixed precedent signals: Conduct deeper case-by-case analysis to "
                "identify distinguishing factors and strengthen the legal position."
            )
        
        # Recommendation based on coherence
        if coherence_score < 0.5:
            recommendations.append(
                "Low coherence score: Expand the search to include additional cases, "
                "potentially from related jurisdictions or legal domains."
            )
        
        # Recommendation based on top case
        if retrieved_cases:
            top_case = retrieved_cases[0]
            if top_case.get('trs', 0) >= 0.75:
                recommendations.append(
                    f"Prioritize detailed analysis of '{top_case.get('title', 'N/A')}' "
                    f"(TRS: {top_case.get('trs', 0):.3f}) as the most relevant precedent."
                )
        
        # Jurisdictional recommendation
        target_jur = target_case.get('jurisdiction', 'Unknown')
        same_jur = [c for c in retrieved_cases if c.get('jurisdiction') == target_jur]
        if len(same_jur) < 2:
            recommendations.append(
                f"Limited cases from {target_jur}: Seek additional precedents from "
                "the target jurisdiction for stronger authority."
            )
        
        # Temporal recommendation
        target_year = target_case.get('year')
        if target_year and isinstance(target_year, int):
            old_cases = [c for c in retrieved_cases 
                        if isinstance(c.get('year'), int) and 
                        c.get('year') < target_year - 20]
            if len(old_cases) > len(retrieved_cases) // 2:
                recommendations.append(
                    "Many precedents are dated: Verify whether more recent cases have "
                    "modified or superseded these older authorities."
                )
        
        # General recommendations
        recommendations.append(
            "Review the extracted spans to identify key legal principles and "
            "terminology that can strengthen legal arguments."
        )
        
        recommendations.append(
            "Consider citation graph analysis to identify additional influential "
            "cases cited by the retrieved precedents."
        )
        
        return recommendations
    
    def _compute_overall_coherence(
        self,
        external_coherence: float,
        retrieved_cases: List[Dict[str, Any]]
    ) -> float:
        """Compute overall coherence score."""
        if not retrieved_cases:
            return external_coherence
        
        # Factor in internal metrics
        avg_confidence = np.mean([c.get('internal_confidence', 0) 
                                 for c in retrieved_cases])
        avg_uncertainty = np.mean([c.get('uncertainty', 0) 
                                  for c in retrieved_cases])
        
        # Combined coherence
        coherence = (
            0.6 * external_coherence +
            0.3 * avg_confidence -
            0.1 * avg_uncertainty
        )
        
        return float(np.clip(coherence, 0, 1))


# Demo and testing functions
def demo_legal_reasoning_agent():
    """Demonstrate the LegalReasoningAgent."""
    from lexai.agents.external_inference_agent import create_sample_data, ExternalInferenceAgent
    
    print("\n" + "="*70)
    print("LEGAL REASONING AGENT DEMO")
    print("="*70)
    
    # Step 1: Get data from ExternalInferenceAgent
    print("\n1. Running ExternalInferenceAgent...")
    candidates, target = create_sample_data()
    external_agent = ExternalInferenceAgent()
    external_agent.build_index(candidates)
    external_result = external_agent.infer(target, top_k=3, internal_confidence=0.8)
    
    print(f"   Retrieved {len(external_result['retrieved_cases'])} cases")
    print(f"   External coherence: {external_result['overall_external_coherence_score']:.3f}")
    
    # Step 2: Initialize LegalReasoningAgent
    print("\n2. Initializing LegalReasoningAgent (rule-based)...")
    reasoning_agent = LegalReasoningAgent(use_llm=False)
    
    # Step 3: Generate reasoning
    print("\n3. Generating legal reasoning analysis...")
    reasoning_result = reasoning_agent.generate_reasoning(
        target_case=external_result['target'],
        retrieved_cases=external_result['retrieved_cases'],
        external_coherence_score=external_result['overall_external_coherence_score']
    )
    
    # Step 4: Display results
    print("\n" + "="*70)
    print("REASONING ANALYSIS RESULTS")
    print("="*70)
    print(json.dumps(reasoning_result, indent=2))
    
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return reasoning_result


if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        print("Running Legal Reasoning Agent tests...")
        result = demo_legal_reasoning_agent()
        
        # Validation
        print("\n" + "="*70)
        print("VALIDATION")
        print("="*70)
        
        required_keys = [
            'summary_long', 'aspect_analysis', 'detailed_evidence_table',
            'uncertainty_and_limits', 'recommended_next_steps',
            'reasoning_method', 'overall_coherence_score'
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
            print(f"✓ {key} present")
        
        assert 0 <= result['overall_coherence_score'] <= 1, "Coherence score out of bounds"
        print(f"✓ Overall coherence score: {result['overall_coherence_score']:.3f}")
        
        assert len(result['aspect_analysis']) > 0, "No aspect analysis"
        print(f"✓ Generated {len(result['aspect_analysis'])} aspects")
        
        assert len(result['detailed_evidence_table']) > 0, "No evidence table"
        print(f"✓ Generated evidence table with {len(result['detailed_evidence_table'])} entries")
        
        assert len(result['recommended_next_steps']) > 0, "No recommendations"
        print(f"✓ Generated {len(result['recommended_next_steps'])} recommendations")
        
        print("\n" + "="*70)
        print("ALL VALIDATIONS PASSED ✓")
        print("="*70)
    else:
        demo_legal_reasoning_agent()
