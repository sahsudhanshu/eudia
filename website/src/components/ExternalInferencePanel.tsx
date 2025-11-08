"use client";

import React, { useEffect, useState, useRef } from 'react';
import { fetchExternalInference, ExternalInferenceResponse, ExternalInferenceResultCase } from '@/lib/api';
import { useToast } from './ToastProvider';
import { BarChart2, RefreshCcw, X, Info } from 'lucide-react';

interface Props {
  documentId: string | null;
}

export default function ExternalInferencePanel({ documentId }: Props) {
  const { showToast } = useToast();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ExternalInferenceResponse | null>(null);
  const [expandedCaseId, setExpandedCaseId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [topK, setTopK] = useState(5);
  const [showFactors, setShowFactors] = useState(false);

  useEffect(() => {
    if (!documentId) {
      setData(null); setError(null); setLoading(false); return;
    }
    runInference();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentId]);

  const runInference = () => {
    if (!documentId) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError(null);
    showToast({ message: 'Running external inference…', variant: 'info' });
    fetchExternalInference(documentId, { topK, factors: showFactors, signal: controller.signal })
      .then(resp => {
        setData(resp);
        showToast({ message: 'External inference complete', variant: 'success' });
      })
      .catch(err => {
        if (controller.signal.aborted) return;
        console.error('External inference failed', err);
        setError(err.message || 'Failed to compute external inference');
        showToast({ message: 'External inference failed', variant: 'error' });
      })
      .finally(() => setLoading(false));
  };

  const overall = data?.overall_external_coherence_score ?? 0;
  const scoreBadgeColor = overall >= 0.7 ? 'bg-green-600' : overall >= 0.5 ? 'bg-amber-500' : 'bg-red-500';

  return (
    <div className="h-full glass-card rounded-lg p-4 flex flex-col">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h2 className="text-base font-semibold text-neutral-900 flex items-center gap-2">
            <BarChart2 size={18} className="text-neutral-700" />
            External Inference (TRS)
          </h2>
          <p className="text-xs text-neutral-600 mt-1">Similarity & contextual alignment across your corpus.</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => runInference()}
            disabled={loading || !documentId}
            className="text-xs px-2 py-1 rounded-md bg-neutral-800 text-white hover:bg-neutral-700 inline-flex items-center gap-1 disabled:opacity-50"
            title="Refresh external inference"
          >
            {loading ? <span className="inline-block h-3 w-3 border border-white border-t-transparent rounded-full animate-spin" /> : <RefreshCcw size={14} />}
            Refresh
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="mb-4 flex flex-wrap gap-2 items-center">
        <div className="flex items-center gap-1">
          <label className="text-[11px] text-neutral-600">Top K</label>
          <input
            type="number"
            min={1}
            max={15}
            value={topK}
            onChange={e => setTopK(Math.min(15, Math.max(1, parseInt(e.target.value || '5', 10))))}
            className="w-14 text-xs px-2 py-1 border rounded-md bg-white"
          />
          <button
            onClick={() => runInference()}
            disabled={loading}
            className="text-[11px] px-2 py-1 border rounded-md bg-white hover:bg-neutral-50 disabled:opacity-50"
          >Apply</button>
        </div>
        <label className="flex items-center gap-1 text-[11px] text-neutral-700">
          <input
            type="checkbox"
            checked={showFactors}
            onChange={e => setShowFactors(e.target.checked)}
          /> Factors
        </label>
        {data && (
          <div className={`text-xs text-white px-2 py-1 rounded-md font-medium ${scoreBadgeColor}`}>Overall: {(overall * 100).toFixed(1)}%</div>
        )}
        {loading && (
          <div className="flex items-center gap-1 text-neutral-500" title="Computing…">
            <span className="inline-block h-3 w-3 border-2 border-neutral-300 border-t-green-500 rounded-full animate-spin" />
            <span className="text-[10px]">Running</span>
          </div>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto pr-1 scrollbar-thin">
        {!documentId && (
          <div className="text-center text-neutral-500 text-xs py-8">Select a document to run external inference.</div>
        )}
        {error && (
          <div className="mb-2 bg-red-50 border border-red-200 text-red-700 text-xs p-2 rounded flex items-start gap-2">
            <X size={12} className="mt-0.5" />
            <span>{error}</span>
          </div>
        )}
        {data && data.retrieved_cases.length === 0 && !error && (
          <div className="text-center text-neutral-500 text-xs py-6">No candidate documents available to compare.</div>
        )}
        {data && data.retrieved_cases.length > 0 && (
          <ul className="space-y-2">
            {data.retrieved_cases.map(c => {
              const trsValue = typeof c.trs === 'number' ? c.trs : c.trs.score;
              const barColor = trsValue >= 0.7 ? 'bg-green-500' : trsValue >= 0.5 ? 'bg-amber-500' : 'bg-red-500';
              return (
                <li key={c.case_id} className="bg-white border border-neutral-200 rounded-lg p-2.5 text-xs">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="font-medium text-neutral-900 truncate" title={c.title}>{c.title}</div>
                      <div className="mt-1 flex flex-wrap gap-2 items-center">
                        <div className="flex items-center gap-1"><span className="text-neutral-500">TRS:</span>
                          <span className={`px-1.5 py-0.5 rounded text-white ${barColor}`}>{(trsValue*100).toFixed(1)}%</span>
                        </div>
                        <span className="text-neutral-500">Sim { (c.similarity_score*100).toFixed(0)}%</span>
                        <span className="text-neutral-500">Ctx { (c.context_fit*100).toFixed(0)}%</span>
                        <span className="text-neutral-500">J { (c.jurisdiction_score*100).toFixed(0)}%</span>
                        <span className="text-neutral-500">I { (c.internal_confidence*100).toFixed(0)}%</span>
                        <span className="text-neutral-500">U { (c.uncertainty*100).toFixed(0)}%</span>
                        <button
                          onClick={() => setExpandedCaseId(expandedCaseId === c.case_id ? null : c.case_id)}
                          className="text-neutral-600 hover:text-neutral-900 ml-auto"
                        >{expandedCaseId === c.case_id ? 'Hide' : 'Details'}</button>
                      </div>
                    </div>
                  </div>
                  {expandedCaseId === c.case_id && (
                    <div className="mt-2 space-y-1">
                      <p className="text-neutral-700 leading-relaxed">{c.justification}</p>
                      <div className="bg-neutral-50 rounded p-2">
                        <p className="font-medium text-[11px] text-neutral-700 mb-1 flex items-center gap-1"><Info size={12}/>Support Spans</p>
                        <p className="text-[11px] text-neutral-600"><strong className="text-neutral-800">Target:</strong> {c.spans.target_span}</p>
                        <p className="text-[11px] text-neutral-600 mt-1"><strong className="text-neutral-800">Candidate:</strong> {c.spans.candidate_span}</p>
                      </div>
                      {typeof c.trs !== 'number' && showFactors && (
                        <div className="bg-white border border-neutral-200 rounded p-2">
                          <p className="text-[11px] font-medium text-neutral-700 mb-1">Factor Breakdown</p>
                          <div className="grid grid-cols-2 gap-1">
                            {Object.entries(c.trs.factors).map(([k,v]) => (
                              <div key={k} className="flex justify-between text-[11px] text-neutral-600">
                                <span>{k}</span><span className="font-medium text-neutral-800">{(v*100).toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Summary */}
      {data && (
        <div className="mt-3 border-t pt-2 text-[11px] text-neutral-700">
          <p className="font-medium mb-1">Summary</p>
          <p className="text-neutral-600 leading-relaxed">{data.short_summary}</p>
        </div>
      )}
    </div>
  );
}
