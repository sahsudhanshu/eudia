"use client";

import React, { useEffect, useState } from 'react';
import { apiFetch } from '@/lib/api';

interface InternalAnalysisPanelProps {
	documentId: string | null;
}

interface AnalysisResult {
	[key: string]: any;
}

export default function InternalAnalysisPanel({ documentId }: InternalAnalysisPanelProps) {
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);

	useEffect(() => {
		if (!documentId) return;
		setLoading(true);
		setError(null);
		apiFetch<{ success?: boolean; analysis?: AnalysisResult; error?: string }>(`/api/ocr/internal-analysis/${documentId}`)
			.then(res => {
				if (res.error) {
					setError(res.error);
				} else {
					setAnalysis(res.analysis || null);
				}
			})
			.catch(err => {
				setError(err.message || 'Failed to load analysis');
			})
			.finally(() => setLoading(false));
	}, [documentId]);

	if (!documentId) {
		return (
			<div className="glass-card rounded-lg p-4 h-full flex items-center justify-center text-sm text-neutral-600">
				Select a document to view internal analysis.
			</div>
		);
	}

	if (loading) {
		return (
			<div className="glass-card rounded-lg p-4 h-full flex flex-col items-center justify-center text-center">
				<div className="animate-spin rounded-full h-12 w-12 border-4 border-neutral-200 border-t-green-500 mb-4" />
				<p className="text-neutral-600 text-sm">Computing internal coherence analysis...</p>
				<p className="text-[11px] text-neutral-500 mt-2">This may take a minute on first run (models loading).</p>
			</div>
		);
	}

	if (error) {
		return (
			<div className="glass-card rounded-lg p-4 h-full flex flex-col">
				<h3 className="text-sm font-semibold mb-2 text-neutral-900">Internal Analysis</h3>
				<p className="text-xs text-red-600 mb-2">{error}</p>
				<p className="text-[11px] text-neutral-500">Try re-uploading or ensure the backend models are available.</p>
			</div>
		);
	}

	if (!analysis) {
		return (
			<div className="glass-card rounded-lg p-4 h-full flex items-center justify-center text-sm text-neutral-600">
				No analysis available.
			</div>
		);
	}

	const coherenceScore = analysis?.['Final Report']?.['Coherence Score'] ?? null;
	const claims: string[] = analysis?.Claims || [];
	const contradictions: string[] = analysis?.Contradictions || analysis?.['Detected Contradictions'] || [];
	const briefCommentary = analysis?.['Final Report']?.['Brief Commentary'] || analysis?.['Brief Commentary'] || '';
	const keyFlows = analysis?.['Final Report']?.['Key Argument Flows'] || analysis?.['Key Argument Flows'] || [];

	return (
		<div className="glass-card rounded-lg p-4 h-full flex flex-col overflow-hidden">
			<div className="flex items-center justify-between mb-3">
				<h3 className="text-sm font-semibold text-neutral-900">Internal Coherence Analysis</h3>
				{coherenceScore !== null && (
					<span className="text-xs px-2 py-1 rounded-md bg-green-50 text-green-700 border border-green-200">
						Score: {coherenceScore.toFixed(2)}
					</span>
				)}
			</div>
			<div className="flex-1 overflow-y-auto custom-scrollbar space-y-4">
				<section>
					<h4 className="text-xs font-semibold text-neutral-700 mb-1">Summary Commentary</h4>
					<p className="text-[11px] leading-relaxed text-neutral-600 whitespace-pre-line">
						{briefCommentary || 'No commentary available.'}
					</p>
				</section>
				<section>
					<h4 className="text-xs font-semibold text-neutral-700 mb-1">Key Argument Flows</h4>
					{keyFlows && keyFlows.length > 0 ? (
						<ul className="space-y-1">
							{keyFlows.slice(0, 8).map((f: string, i: number) => (
								<li key={i} className="text-[11px] text-neutral-600">• {f}</li>
							))}
							{keyFlows.length > 8 && (
								<li className="text-[10px] text-neutral-500">+ {keyFlows.length - 8} more</li>
							)}
						</ul>
					) : (
						<p className="text-[11px] text-neutral-500">None detected.</p>
					)}
				</section>
				<section>
					<h4 className="text-xs font-semibold text-neutral-700 mb-1">Claims ({claims.length})</h4>
					{claims.length > 0 ? (
						<ul className="space-y-1">
							{claims.slice(0, 10).map((c, i) => (
								<li key={i} className="text-[11px] text-neutral-600">{c.slice(0, 160)}{c.length > 160 ? '…' : ''}</li>
							))}
							{claims.length > 10 && (
								<li className="text-[10px] text-neutral-500">+ {claims.length - 10} more</li>
							)}
						</ul>
					) : (
						<p className="text-[11px] text-neutral-500">No claims extracted.</p>
					)}
				</section>
				<section>
					<h4 className="text-xs font-semibold text-neutral-700 mb-1">Contradictions ({contradictions.length})</h4>
					{contradictions.length > 0 ? (
						<ul className="space-y-1">
							{contradictions.slice(0, 6).map((c, i) => (
								<li key={i} className="text-[11px] text-red-600">{c.slice(0, 180)}{c.length > 180 ? '…' : ''}</li>
							))}
							{contradictions.length > 6 && (
								<li className="text-[10px] text-neutral-500">+ {contradictions.length - 6} more</li>
							)}
						</ul>
					) : (
						<p className="text-[11px] text-neutral-500">No contradictions detected.</p>
					)}
				</section>
			</div>
		</div>
	);
}
