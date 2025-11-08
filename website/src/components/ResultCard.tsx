"use client"

import React, { useState } from 'react';
import { ThumbsUp, ThumbsDown, ChevronDown, ChevronUp } from 'lucide-react';

interface Result {
  id: string;
  sourceTitle: string;
  metadata: {
    year: number;
    court: string;
    citations: number;
  };
  risk: 'low' | 'medium' | 'high';
  sentiment: 'positive' | 'neutral' | 'negative';
  trs: number;
  explanation: string;
  irac: {
    issue: string;
    rule: string;
    application: string;
    conclusion: string;
  };
}

interface ResultCardProps {
  result: Result;
  onFeedback?: (resultId: string, feedback: 'up' | 'down') => void;
}

export default function ResultCard({ result, onFeedback }: ResultCardProps) {
  const [showFullReasoning, setShowFullReasoning] = useState(false);
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null);

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'bg-green-50 text-green-700 border-green-200';
      case 'medium': return 'bg-yellow-50 text-yellow-700 border-yellow-200';
      case 'high': return 'bg-red-50 text-red-700 border-red-200';
      default: return 'bg-neutral-50 text-neutral-700 border-neutral-200';
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'bg-blue-50 text-blue-700 border-blue-200';
      case 'neutral': return 'bg-neutral-50 text-neutral-700 border-neutral-200';
      case 'negative': return 'bg-orange-50 text-orange-700 border-orange-200';
      default: return 'bg-neutral-50 text-neutral-700 border-neutral-200';
    }
  };

  const handleFeedback = (type: 'up' | 'down') => {
    setFeedback(type);
    onFeedback?.(result.id, type);
  };

  return (
    <div className="bg-white border border-neutral-200 rounded-lg p-4 hover:shadow-soft-lg transition-all">
      {/* Header */}
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-neutral-900 mb-2 leading-tight">{result.sourceTitle}</h3>
        <div className="flex items-center gap-3 text-xs text-neutral-600 flex-wrap">
          <span>{result.metadata.year}</span>
          <span>•</span>
          <span>{result.metadata.court}</span>
          <span>•</span>
          <span>{result.metadata.citations} citations</span>
        </div>
      </div>

      {/* Badges */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span className={`px-2.5 py-1 rounded-full text-xs font-medium border ${getRiskColor(result.risk)}`}>
          {result.risk.toUpperCase()} RISK
        </span>
        <span className={`px-2.5 py-1 rounded-full text-xs font-medium border ${getSentimentColor(result.sentiment)}`}>
          {result.sentiment.toUpperCase()}
        </span>
      </div>

      {/* TRS Score */}
      <div className="mb-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-neutral-700 font-medium text-xs">Relevance Score</span>
          <span className="text-neutral-900 font-semibold text-sm">{result.trs}%</span>
        </div>
        <div className="w-full h-1.5 bg-neutral-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-green-500 rounded-full transition-all duration-500"
            style={{ width: `${result.trs}%` }}
          ></div>
        </div>
      </div>

      {/* Explanation */}
      <div className="mb-3">
        <p className="text-neutral-700 text-xs leading-relaxed">{result.explanation}</p>
      </div>

      {/* Full Reasoning (I-R-A-C) */}
      <button
        onClick={() => setShowFullReasoning(!showFullReasoning)}
        className="w-full bg-neutral-50 hover:bg-neutral-100 text-neutral-900 font-medium py-2 px-3 rounded-lg transition-all flex items-center justify-between mb-3 text-xs border border-neutral-200"
      >
        <span>View Full Reasoning</span>
        {showFullReasoning ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {showFullReasoning && (
        <div className="bg-neutral-50 rounded-lg p-3 mb-3 space-y-3 border border-neutral-200">
          <div>
            <h4 className="text-blue-600 font-semibold text-xs mb-1.5">ISSUE</h4>
            <p className="text-neutral-700 text-xs leading-relaxed">{result.irac.issue}</p>
          </div>
          <div>
            <h4 className="text-purple-600 font-semibold text-xs mb-1.5">RULE</h4>
            <p className="text-neutral-700 text-xs leading-relaxed">{result.irac.rule}</p>
          </div>
          <div>
            <h4 className="text-orange-600 font-semibold text-xs mb-1.5">APPLICATION</h4>
            <p className="text-neutral-700 text-xs leading-relaxed">{result.irac.application}</p>
          </div>
          <div>
            <h4 className="text-green-600 font-semibold text-xs mb-1.5">CONCLUSION</h4>
            <p className="text-neutral-700 text-xs leading-relaxed">{result.irac.conclusion}</p>
          </div>
        </div>
      )}

      {/* Feedback Buttons */}
      <div className="flex items-center justify-between pt-3 border-t border-neutral-200">
        <span className="text-neutral-600 text-xs">Was this helpful?</span>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => handleFeedback('up')}
            className={`p-1.5 rounded-lg transition-all ${
              feedback === 'up'
                ? 'bg-green-100 text-green-600 border border-green-300'
                : 'bg-neutral-100 hover:bg-neutral-200 text-neutral-600 border border-neutral-200'
            }`}
          >
            <ThumbsUp size={14} />
          </button>
          <button
            onClick={() => handleFeedback('down')}
            className={`p-1.5 rounded-lg transition-all ${
              feedback === 'down'
                ? 'bg-red-100 text-red-600 border border-red-300'
                : 'bg-neutral-100 hover:bg-neutral-200 text-neutral-600 border border-neutral-200'
            }`}
          >
            <ThumbsDown size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}