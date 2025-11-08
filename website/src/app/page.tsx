"use client"

import React, { useState, useEffect, useRef } from 'react';
import { AuthProvider } from '@/components/contexts/AuthContext';
import Navbar from '@/components/Navbar';
import PrivacyBanner from '@/components/PrivacyBanner';
import UploadPanel from '@/components/UploadPanel';
import DocumentExplorer from '@/components/DocumentExplorer';
import CitationGraph from '@/components/CitationGraph';
import AnalysisWorkspace from '@/components/AnalysisWorkspace';
import AuthModal from '@/components/AuthModal';
import { Upload, ArrowRight, Network, MessageSquare } from 'lucide-react';
import { useAuth } from '@/components/contexts/AuthContext';
import { apiFetch } from '@/lib/api';
import { useDebounce } from '@/hooks/useDebounce';
import ChatInterface from '@/components/ChatInterface';
import InternalAnalysisPanel from '@/components/InternalAnalysisPanel';
import { ToastProvider, useToast } from '@/components/ToastProvider';

type AppState = 'dashboard';

interface CitationNode {
  id: string;
  title: string;
  x: number;
  y: number;
  citations: number;
  year: number;
}

interface SelectedItem {
  id: string;
  type: 'node';
  title: string;
}

function HomePage() {
  const { showToast } = useToast();
  const { user } = useAuth();
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [nodes, setNodes] = useState<CitationNode[]>([]);
  const [selectedItems, setSelectedItems] = useState<SelectedItem[]>([]);
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);
  const [loadingCitations, setLoadingCitations] = useState(false);
  const [refreshDocuments, setRefreshDocuments] = useState(0);
  const [viewMode, setViewMode] = useState<'citation' | 'chat' | 'internal'>('citation');
  const [refreshingGraph, setRefreshingGraph] = useState(false);
  const [refreshingInternal, setRefreshingInternal] = useState(false);
  const [graphStats, setGraphStats] = useState<{
    totalNodes: number;
    filteredNodes: number;
    showingTop: number;
    hasMore: boolean;
  } | null>(null);

  const [graphQuery, setGraphQuery] = useState({
    limit: 50,
    layout: 'force' as 'force' | 'tree',
    minCitations: 0,
    year: ''
  });

  // Dynamic debounce: shorter while typing on small graphs, longer for heavy graphs
  const isHeavyGraph = (graphStats?.showingTop ?? 0) >= 120 || graphQuery.limit >= 150;
  const dynamicDebounceMs = isHeavyGraph ? 800 : 300;
  const debouncedMinCitations = useDebounce(graphQuery.minCitations, dynamicDebounceMs);
  const debouncedYear = useDebounce(graphQuery.year, dynamicDebounceMs);
  // Inline pending indicator when user is typing and a debounced refresh is queued
  const isDebouncePending = (
    graphQuery.minCitations !== debouncedMinCitations || graphQuery.year !== debouncedYear
  );
  const lastFetchKeyRef = useRef<string | null>(null);

  const handleDocumentSelect = async (documentId: string, opts?: Partial<typeof graphQuery>, preserveSelection = false) => {
    setSelectedDocumentId(documentId);
    const nextQuery = { ...graphQuery, ...opts };
    setGraphQuery(nextQuery);
    if (!preserveSelection) {
      setSelectedItems([]);
      setSelectedNodeIds([]);
    }

    // Build a key and mark it as last-fetch to avoid duplicate fetches from debounce effect
    const fetchKey = `${documentId}|${nextQuery.limit}|${nextQuery.layout}|${nextQuery.minCitations}|${nextQuery.year}`;
    lastFetchKeyRef.current = fetchKey;
    
    if (!user) return;
    
    // Fetch citation graph for the selected document
    try {
      setLoadingCitations(true);
      const params = new URLSearchParams();
      params.set('limit', String(nextQuery.limit));
      if (nextQuery.layout) params.set('layout', nextQuery.layout);
      if (nextQuery.minCitations > 0) params.set('min_citations', String(nextQuery.minCitations));
      if (nextQuery.year) params.set('year', nextQuery.year);

      const response = await apiFetch<{
        nodes: CitationNode[], 
        edges: any[],
        total_nodes: number,
        filtered_nodes: number,
        showing_top: number,
        has_more: boolean
      }>(`/api/ocr/citation-nodes/${documentId}?${params.toString()}`);
      
      setNodes(
        response.nodes.map((citation) => ({
          id: citation.id,
          title: citation.title,
          x: citation.x,
          y: citation.y,
          citations: citation.citations,
          year: citation.year,
        }))
      );
      
      setGraphStats({
        totalNodes: response.total_nodes || 0,
        filteredNodes: response.filtered_nodes || 0,
        showingTop: response.showing_top || 0,
        hasMore: response.has_more || false
      });
    } catch (err) {
      console.error('Error fetching citation graph:', err);
      setNodes([]);
      setGraphStats(null);
    } finally {
      setLoadingCitations(false);
    }
  };

    // When debounced filters change, fetch the graph but avoid duplicating a call
    useEffect(() => {
      if (!selectedDocumentId) return;

      const key = `${selectedDocumentId}|${graphQuery.limit}|${graphQuery.layout}|${debouncedMinCitations}|${debouncedYear}`;
      if (lastFetchKeyRef.current === key) {
        // This fetch was already triggered (manual or identical params)
        return;
      }

      // Trigger fetch preserving selection
      handleDocumentSelect(selectedDocumentId, { minCitations: debouncedMinCitations, year: debouncedYear }, true);
      // Mark last fetch key so effect doesn't re-trigger immediately
      lastFetchKeyRef.current = key;
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedDocumentId, debouncedMinCitations, debouncedYear]);

  const handleNodeSelect = (node: CitationNode) => {
    const isSelected = selectedNodeIds.includes(node.id);
    
    if (isSelected) {
      // Deselect node
      setSelectedNodeIds(selectedNodeIds.filter(id => id !== node.id));
      setSelectedItems(selectedItems.filter(item => item.id !== node.id));
    } else {
      // Select node
      setSelectedNodeIds([...selectedNodeIds, node.id]);
      setSelectedItems([
        ...selectedItems,
        {
          id: node.id,
          type: 'node',
          title: node.title,
        },
      ]);
    }
  };

  const handleNodePositionUpdate = async (nodeId: string, x: number, y: number) => {
    // Only persist positions for real DB citations (UUIDs). Graph nodes from OCR are not persisted.
    if (!user || !selectedDocumentId) return;
    const uuidV4Regex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    if (!uuidV4Regex.test(nodeId)) return; // skip backend call for non-DB nodes

    try {
      await apiFetch(`/api/documents/${selectedDocumentId}/citations/${nodeId}`, {
        method: 'PUT',
        body: JSON.stringify({ x, y }),
      });
    } catch (err) {
      console.error('Error updating node position:', err);
    }
  };

  const handleItemDeselect = (id: string) => {
    setSelectedItems(selectedItems.filter(item => item.id !== id));
    setSelectedNodeIds(selectedNodeIds.filter(nodeId => nodeId !== id));
  };

  const handleUploadComplete = (documentId: string) => {
    showToast({ message: 'Processing started… OCR + citation graph + internal analysis', variant: 'info' });
    // After upload, kick off processing (OCR -> graph -> internal analysis) automatically
    setRefreshDocuments(prev => prev + 1); // Trigger document list refresh
    setSelectedDocumentId(documentId);
    // Show loading while processing
    setLoadingCitations(true);
    apiFetch(`/api/ocr/process/${documentId}`, { method: 'POST' })
      .then(() => {
        showToast({ message: 'Processing complete!', variant: 'success' });
      })
      .catch(err => {
        console.error('Error processing document after upload:', err);
        showToast({ message: 'Processing failed', variant: 'error' });
      })
      .finally(() => {
        // Once processed (or even if it fails), fetch the citation graph to render
        handleDocumentSelect(documentId);
      });
  };

  const handleBackToUpload = () => {
    setSelectedDocumentId(null);
    setNodes([]);
    setSelectedItems([]);
    setSelectedNodeIds([]);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
     <div className="min-h-screen bg-neutral-50">
      <Navbar />
      <PrivacyBanner />
      <AuthModal />

      <div className="w-full px-3 sm:px-4 md:px-6 py-3 sm:py-4 md:py-6">
        <div className="max-w-screen-2xl mx-auto">
          {selectedDocumentId ? (
            <>
              {/* Header with Upload Button and View Toggle */}
              <div className="mb-4 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
                {/* View Toggle */}
                <div className="glass-card rounded-lg p-1 flex gap-1">
                  <button
                    onClick={() => setViewMode('citation')}
                    className={`px-4 py-2 rounded-md transition-all inline-flex items-center gap-2 text-sm font-medium ${
                      viewMode === 'citation'
                        ? 'bg-green-500 text-white shadow-soft'
                        : 'text-neutral-600 hover:text-neutral-900'
                    }`}
                  >
                    <Network size={16} />
                    Citation Network
                    {viewMode === 'citation' && (
                      <span
                        onClick={(e) => {
                          e.stopPropagation();
                          if (selectedDocumentId) {
                            setRefreshingGraph(true);
                            handleDocumentSelect(selectedDocumentId, {}, true)
                              .finally(() => {
                                setRefreshingGraph(false);
                                showToast({ message: 'Citation graph refreshed', variant: 'success' });
                              });
                          }
                        }}
                        className="ml-2 text-[10px] px-2 py-0.5 rounded bg-white/30 hover:bg-white/50 cursor-pointer inline-flex items-center gap-1"
                        title="Refresh citation graph"
                      >
                        {refreshingGraph && <span className="inline-block h-2 w-2 border border-white border-t-transparent rounded-full animate-spin" />}
                        Refresh
                      </span>
                    )}
                  </button>
                  <button
                    onClick={() => setViewMode('chat')}
                    className={`px-4 py-2 rounded-md transition-all inline-flex items-center gap-2 text-sm font-medium ${
                      viewMode === 'chat'
                        ? 'bg-green-500 text-white shadow-soft'
                        : 'text-neutral-600 hover:text-neutral-900'
                    }`}
                  >
                    <MessageSquare size={16} />
                    Chat
                  </button>
                  <button
                    onClick={() => setViewMode('internal')}
                    className={`px-4 py-2 rounded-md transition-all inline-flex items-center gap-2 text-sm font-medium ${
                      viewMode === 'internal'
                        ? 'bg-green-500 text-white shadow-soft'
                        : 'text-neutral-600 hover:text-neutral-900'
                    }`}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" /></svg>
                    Internal Analysis
                    {viewMode === 'internal' && (
                      <span
                        onClick={(e) => {
                          e.stopPropagation();
                          if (selectedDocumentId) {
                            setRefreshingInternal(true);
                            showToast({ message: 'Recomputing internal analysis…', variant: 'info' });
                            apiFetch(`/api/ocr/internal-analysis/${selectedDocumentId}?force=1`)
                              .then(() => {
                                setSelectedDocumentId(prev => prev);
                                showToast({ message: 'Internal analysis refreshed', variant: 'success' });
                              })
                              .catch(err => {
                                console.error('Failed to refresh internal analysis', err);
                                showToast({ message: 'Failed to refresh analysis', variant: 'error' });
                              })
                              .finally(() => setRefreshingInternal(false));
                          }
                        }}
                        className="ml-2 text-[10px] px-2 py-0.5 rounded bg-white/30 hover:bg-white/50 cursor-pointer inline-flex items-center gap-1"
                        title="Recompute internal analysis"
                      >
                        {refreshingInternal && <span className="inline-block h-2 w-2 border border-white border-t-transparent rounded-full animate-spin" />}
                        Refresh
                      </span>
                    )}
                  </button>
                </div>

                {/* Upload Button */}
                <button
                  onClick={handleBackToUpload}
                  className="bg-green-500 hover:bg-green-600 text-white font-medium px-4 py-2 rounded-lg transition-all inline-flex items-center gap-2 text-sm shadow-soft"
                >
                  <Upload size={16} />
                  Upload Document
                </button>
              </div>

              {/* Mobile: Stacked layout, Desktop: 3-column grid */}
              <div className="flex flex-col lg:grid lg:grid-cols-4 gap-3 sm:gap-4 md:gap-4">
                {/* Panel 1: Document Explorer - Always visible */}
                <div className="w-full lg:col-span-1 h-[400px] sm:h-[500px] lg:h-[calc(100vh-220px)]">
                  <DocumentExplorer 
                    onDocumentSelect={handleDocumentSelect}
                    selectedDocumentId={selectedDocumentId}
                    key={refreshDocuments}
                  />
                </div>

                {/* Panel 2 & 3: Conditional based on viewMode */}
                {viewMode === 'citation' ? (
                  <>
                    {/* Citation Graph */}
                    <div className="w-full lg:col-span-2 h-[400px] sm:h-[500px] lg:h-[calc(100vh-220px)]">
                      {loadingCitations ? (
                        <div className="h-full glass-card rounded-lg p-4 flex items-center justify-center">
                          <div className="text-center">
                            <div className="animate-spin rounded-full h-12 w-12 border-4 border-neutral-200 border-t-green-500 mx-auto mb-4"></div>
                            <p className="text-neutral-600 text-sm">Loading citations...</p>
                          </div>
                        </div>
                      ) : (
                          <div className="h-full flex flex-col">
                            {/* Graph Stats Bar */}
                            {graphStats && graphStats.totalNodes > 0 && (
                              <div className="glass-card rounded-t-lg px-4 py-2 mb-1 flex items-center justify-between text-xs">
                                <div className="flex gap-4 text-neutral-600">
                                  <span>Total: <strong className="text-neutral-900">{graphStats.totalNodes}</strong> citations</span>
                                  <span>Showing: <strong className="text-green-600">{graphStats.showingTop}</strong></span>
                                  {graphStats.hasMore && (
                                    <span className="text-amber-600">⚠ {graphStats.totalNodes - graphStats.showingTop} more available</span>
                                  )}
                                </div>
                                {graphStats.hasMore && (
                                  <button
                                    onClick={() => {
                                      setGraphQuery(q => {
                                        const nextLimit = Math.min(q.limit + 50, 200);
                                        const updated = { ...q, limit: nextLimit };
                                        handleDocumentSelect(selectedDocumentId!, { limit: nextLimit });
                                        return updated;
                                      });
                                    }}
                                    className="text-green-600 hover:text-green-700 font-medium"
                                  >
                                    Load More
                                  </button>
                                )}
                                <div className="flex items-center gap-2">
                                  <select
                                    value={graphQuery.layout}
                                    onChange={e => {
                                      const layout = e.target.value as 'force' | 'tree';
                                      setGraphQuery(q => ({ ...q, layout }));
                                      if (selectedDocumentId) {
                                        handleDocumentSelect(selectedDocumentId, { layout }, true);
                                      }
                                    }}
                                    className="text-xs px-2 py-1 border rounded-md bg-white"
                                  >
                                    <option value="force">Force</option>
                                    <option value="tree">Tree</option>
                                  </select>
                                  <input
                                    type="number"
                                    min={0}
                                    placeholder="Min citations"
                                    value={graphQuery.minCitations || ''}
                                    onChange={e => {
                                      const val = e.target.value ? parseInt(e.target.value, 10) : 0;
                                      setGraphQuery(q => ({ ...q, minCitations: val }));
                                    }}
                                    className="w-20 text-xs px-2 py-1 border rounded-md bg-white"
                                  />
                                  <input
                                    type="number"
                                    placeholder="Year"
                                    value={graphQuery.year}
                                    onChange={e => {
                                      setGraphQuery(q => ({ ...q, year: e.target.value }));
                                    }}
                                    className="w-20 text-xs px-2 py-1 border rounded-md bg-white"
                                  />
                                  {/* Tiny inline spinner shows when a debounced refresh is queued */}
                                  {isDebouncePending && !loadingCitations && (
                                    <div className="flex items-center gap-1 text-neutral-500" title="Applying filters…">
                                      <span className="inline-block h-3 w-3 border-2 border-neutral-300 border-t-green-500 rounded-full animate-spin" />
                                      <span className="text-[10px]">Updating…</span>
                                    </div>
                                  )}
                                  <button
                                    onClick={() => handleDocumentSelect(selectedDocumentId!, { limit: 50 })}
                                    className="text-xs px-2 py-1 border rounded-md bg-white hover:bg-neutral-50"
                                  >
                                    Reset
                                  </button>
                                </div>
                              </div>
                            )}
                          
                        <CitationGraph 
                          nodes={nodes} 
                          onNodeSelect={handleNodeSelect}
                          selectedNodeIds={selectedNodeIds}
                          onNodePositionUpdate={handleNodePositionUpdate}
                        />
                          </div>
                      )}
                    </div>

                    {/* Analysis Workspace */}
                    <div className="w-full lg:col-span-1 h-[500px] sm:h-[600px] lg:h-[calc(100vh-220px)]">
                      <AnalysisWorkspace
                        selectedItems={selectedItems}
                        onItemDeselect={handleItemDeselect}
                      />
                    </div>
                  </>
                ) : viewMode === 'chat' ? (
                  <>
                    {/* Chat Interface - Takes full remaining width */}
                    <div className="w-full lg:col-span-3 h-[600px] sm:h-[700px] lg:h-[calc(100vh-220px)]">
                      <ChatInterface documentId={selectedDocumentId} />
                    </div>
                  </>
                ) : (
                  <>
                    {/* Internal Analysis Panel */}
                    <div className="w-full lg:col-span-3 h-[600px] sm:h-[700px] lg:h-[calc(100vh-220px)]">
                      <InternalAnalysisPanel documentId={selectedDocumentId} />
                    </div>
                  </>
                )}
              </div>
            </>
          ) : (
            <>
              {/* Hero Section - Upload Centered */}
              <div className="max-w-4xl mx-auto">
                <div className="text-center mb-8">
                  <h1 className="text-3xl sm:text-4xl font-bold text-neutral-900 mb-3">
                    Legal Document Analysis
                  </h1>
                  <p className="text-neutral-600 text-base sm:text-lg max-w-2xl mx-auto">
                    Upload your legal documents to analyze citations, explore connections, and gain insights with AI-powered analysis
                  </p>
                </div>

                {/* Upload Panel */}
                <UploadPanel 
                  onUploadComplete={handleUploadComplete}
                />

                {/* Features Grid */}
                <div className="mt-12 grid grid-cols-1 sm:grid-cols-3 gap-6">
                  <div className="glass-card rounded-lg p-6 text-center">
                    <div className="w-12 h-12 bg-blue-50 rounded-lg flex items-center justify-center mx-auto mb-4">
                      <Upload className="text-blue-600" size={24} />
                    </div>
                    <h3 className="text-sm font-semibold text-neutral-900 mb-2">
                      Upload Documents
                    </h3>
                    <p className="text-xs text-neutral-600">
                      Drag and drop or browse PDF files to analyze
                    </p>
                  </div>

                  <div className="glass-card rounded-lg p-6 text-center">
                    <div className="w-12 h-12 bg-purple-50 rounded-lg flex items-center justify-center mx-auto mb-4">
                      <svg className="text-purple-600" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="3" />
                        <circle cx="19" cy="5" r="2" />
                        <circle cx="5" cy="19" r="2" />
                        <line x1="10.4" y1="13.4" x2="6.6" y2="17.6" />
                        <line x1="13.4" y1="10.4" x2="17.6" y2="6.6" />
                      </svg>
                    </div>
                    <h3 className="text-sm font-semibold text-neutral-900 mb-2">
                      Explore Citations
                    </h3>
                    <p className="text-xs text-neutral-600">
                      Interactive graph visualization of legal citations
                    </p>
                  </div>

                  <div className="glass-card rounded-lg p-6 text-center">
                    <div className="w-12 h-12 bg-green-50 rounded-lg flex items-center justify-center mx-auto mb-4">
                      <svg className="text-green-600" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 20h9" />
                        <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
                      </svg>
                    </div>
                    <h3 className="text-sm font-semibold text-neutral-900 mb-2">
                      AI Analysis
                    </h3>
                    <p className="text-xs text-neutral-600">
                      Get insights and analysis powered by AI
                    </p>
                  </div>
                </div>

                {/* Quick Access to Documents */}
                {user && (
                  <div className="mt-8 glass-card rounded-lg p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-base font-semibold text-neutral-900">
                        Your Documents
                      </h3>
                      <ArrowRight className="text-neutral-400" size={20} />
                    </div>
                    <DocumentExplorer 
                      onDocumentSelect={handleDocumentSelect}
                      selectedDocumentId={selectedDocumentId}
                      key={refreshDocuments}
                    />
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <AuthProvider>
      <ToastProvider>
        <HomePage />
      </ToastProvider>
    </AuthProvider>
  );
}