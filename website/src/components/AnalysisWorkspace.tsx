"use client"

import React, { useState, useEffect } from 'react';
import { X, Sparkles } from 'lucide-react';
import ResultCard from './ResultCard';

interface SelectedItem {
  id: string;
  type: 'chunk' | 'node';
  title: string;
}

interface AnalysisWorkspaceProps {
  selectedItems: SelectedItem[];
  onItemDeselect: (id: string) => void;
}

export default function AnalysisWorkspace({ selectedItems, onItemDeselect }: AnalysisWorkspaceProps) {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);

  // Automatically trigger analysis when selectedItems change
  useEffect(() => {
    if (selectedItems.length > 0) {
      handleAnalyze();
    } else {
      setResults([]);
      setLoading(false);
    }
  }, [selectedItems]);

  const handleAnalyze = () => {
    setLoading(true);
    
    // Mock API call with timeout
    setTimeout(() => {
      const mockResults = [
        {
          id: '1',
          sourceTitle: 'Smith v. Jones (2019)',
          metadata: {
            year: 2019,
            court: '9th Circuit',
            citations: 234,
          },
          risk: 'low' as const,
          sentiment: 'positive' as const,
          trs: 92,
          explanation: 'This case provides strong precedential support for your argument. The court\'s reasoning aligns closely with the semantic content of your selected chunks.',
          irac: {
            issue: 'Whether the contract clause permits unilateral modification under commercial impracticability doctrine.',
            rule: 'Under UCC § 2-615, a seller\'s performance is excused when an unforeseen event fundamentally alters the nature of performance.',
            application: 'The pandemic-related supply chain disruption constitutes a supervening event that neither party anticipated.',
            conclusion: 'The court held that the seller was excused from performance.',
          },
        },
        {
          id: '2',
          sourceTitle: 'Anderson v. State (2021)',
          metadata: {
            year: 2021,
            court: 'Supreme Court',
            citations: 187,
          },
          risk: 'medium' as const,
          sentiment: 'neutral' as const,
          trs: 78,
          explanation: 'This case presents mixed precedent. While the outcome was favorable, the reasoning diverges from your fact pattern in key areas.',
          irac: {
            issue: 'Whether the lower court erred in applying the strict liability standard.',
            rule: 'Strict liability applies only when parties explicitly contract for such a standard.',
            application: 'The contract contained ambiguous language that could support either interpretation.',
            conclusion: 'The court remanded for further factual development.',
          },
        },
      ];
      
      setResults(mockResults);
      setLoading(false);
    }, 2000);
  };

  const handleFeedback = (resultId: string, feedback: 'up' | 'down') => {
    console.log(`Feedback for result ${resultId}: ${feedback}`);
  };

  return (
    <div className="h-full glass-card rounded-lg p-4 overflow-hidden flex flex-col">
      <div className="mb-4">
        <h2 className="text-base font-semibold text-neutral-900 flex items-center gap-2">
          <Sparkles size={18} className="text-neutral-700" />
          Analysis
        </h2>
        <p className="text-neutral-600 text-xs mt-1">
          {loading ? 'Analyzing...' : results.length > 0 ? `${results.length} results` : 'Select items'}
        </p>
      </div>

      {selectedItems.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center text-neutral-500 px-4">
            <Sparkles size={32} className="mx-auto mb-3 opacity-30" />
            <p className="text-xs">No items selected</p>
            <p className="text-xs mt-1 opacity-70">Click items to add them</p>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-hidden flex flex-col">
          {/* Selected Items */}
          <div className="mb-4">
            <h3 className="text-neutral-700 text-xs font-medium mb-2">
              Selected ({selectedItems.length})
            </h3>
            <div className="space-y-2 max-h-[150px] overflow-y-auto scrollbar-thin">
              {selectedItems.map((item) => (
                <div
                  key={item.id}
                  className="bg-white border border-neutral-200 rounded-lg p-2.5 flex items-center justify-between group hover:border-neutral-300"
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className={`text-xs font-medium px-2 py-0.5 rounded uppercase flex-shrink-0 ${
                      item.type === 'chunk' 
                        ? 'text-blue-600 bg-blue-50' 
                        : 'text-purple-600 bg-purple-50'
                    }`} style={{ fontSize: '10px' }}>
                      {item.type}
                    </span>
                    <span className="text-neutral-700 text-xs truncate">{item.title}</span>
                  </div>
                  <button
                    onClick={() => onItemDeselect(item.id)}
                    className="text-neutral-400 hover:text-red-500 transition-colors ml-2 flex-shrink-0"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Loading State */}
          {loading && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <div className="relative h-10 w-10 mx-auto mb-4">
                  <div className="absolute inset-0 rounded-full border-2 border-neutral-200"></div>
                  <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-blue-500 animate-spin"></div>
                </div>
                <p className="text-neutral-700 text-xs font-medium mb-2">Processing...</p>
                <div className="space-y-1.5 text-left inline-block">
                  <div className="flex items-center gap-2">
                    <div className="animate-pulse w-1.5 h-1.5 bg-blue-500 rounded-full"></div>
                    <span className="text-xs text-neutral-600">Retrieving cases</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="animate-pulse w-1.5 h-1.5 bg-green-500 rounded-full" style={{ animationDelay: '0.2s' }}></div>
                    <span className="text-xs text-neutral-600">Scoring relevance</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="animate-pulse w-1.5 h-1.5 bg-purple-500 rounded-full" style={{ animationDelay: '0.4s' }}></div>
                    <span className="text-xs text-neutral-600">Generating analysis</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Results */}
          {!loading && results.length > 0 && (
            <div className="flex-1 overflow-y-auto space-y-3 pr-1 scrollbar-thin">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-neutral-700 text-xs font-medium">
                  Results
                </h3>
                <span className="text-green-600 text-xs font-medium px-2 py-0.5 bg-green-50 rounded">
                  Complete
                </span>
              </div>
              {results.map((result) => (
                <ResultCard key={result.id} result={result} onFeedback={handleFeedback} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}