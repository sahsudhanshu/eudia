"use client"

import React, { useState, useEffect, useRef } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Move, MousePointer } from 'lucide-react';

interface CitationNode {
  id: string;
  title: string;
  x: number;
  y: number;
  citations: number;
  year: number;
}

interface CitationGraphProps {
  nodes: CitationNode[];
  onNodeSelect: (node: CitationNode) => void;
  selectedNodeIds?: string[];
  onNodePositionUpdate?: (nodeId: string, x: number, y: number) => void;
}

export default function CitationGraph({ nodes: initialNodes, onNodeSelect, selectedNodeIds = [], onNodePositionUpdate }: CitationGraphProps) {
  const [zoom, setZoom] = useState(1);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [nodes, setNodes] = useState(initialNodes);
  const [draggedNode, setDraggedNode] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [interactionMode, setInteractionMode] = useState<'pointer' | 'pan'>('pointer');
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragStartPos = useRef({ x: 0, y: 0 });
  const nodeStartPos = useRef({ x: 0, y: 0 });
  const panStartPos = useRef({ x: 0, y: 0 });

  useEffect(() => {
    setNodes(initialNodes);
  }, [initialNodes]);

  useEffect(() => {
    if (isDragging || isPanning) {
      document.body.style.userSelect = 'none';
      document.body.style.cursor = isPanning ? 'grabbing' : isDragging ? 'grabbing' : '';
    } else {
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    }
    return () => {
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
  }, [isDragging, isPanning]);

  const handleZoomIn = () => setZoom(Math.min(zoom + 0.2, 2));
  const handleZoomOut = () => setZoom(Math.max(zoom - 0.2, 0.5));
  const handleRecenter = () => {
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
    setNodes(initialNodes);
  };

  const toggleInteractionMode = () => {
    setInteractionMode(prev => prev === 'pointer' ? 'pan' : 'pointer');
  };

  const handlePointerDown = (e: React.PointerEvent, nodeId?: string) => {
    e.stopPropagation();
    e.preventDefault();
    
    if (interactionMode === 'pan' || !nodeId) {
      // Pan mode
      panStartPos.current = { x: e.clientX - panOffset.x, y: e.clientY - panOffset.y };
      setIsPanning(true);
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    } else {
      // Pointer mode - drag node
      const node = nodes.find(n => n.id === nodeId);
      if (!node || !containerRef.current) return;

      dragStartPos.current = { x: e.clientX, y: e.clientY };
      nodeStartPos.current = { x: node.x, y: node.y };
      
      setDraggedNode(nodeId);
      setIsDragging(true);
      setHoveredNode(null);

      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (isPanning) {
      setPanOffset({
        x: e.clientX - panStartPos.current.x,
        y: e.clientY - panStartPos.current.y
      });
    } else if (isDragging && draggedNode && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      
      const deltaX = e.clientX - dragStartPos.current.x;
      const deltaY = e.clientY - dragStartPos.current.y;
      
      const deltaXPercent = (deltaX / rect.width) * 100;
      const deltaYPercent = (deltaY / rect.height) * 100;
      
      let newX = nodeStartPos.current.x + deltaXPercent;
      let newY = nodeStartPos.current.y + deltaYPercent;
      
      newX = Math.max(8, Math.min(92, newX));
      newY = Math.max(8, Math.min(92, newY));
      
      setNodes(prevNodes =>
        prevNodes.map(node =>
          node.id === draggedNode
            ? { ...node, x: newX, y: newY }
            : node
        )
      );
    }
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (isDragging && draggedNode && onNodePositionUpdate) {
      const node = nodes.find(n => n.id === draggedNode);
      if (node) {
        onNodePositionUpdate(draggedNode, node.x, node.y);
      }
    }
    
    if (isDragging || isPanning) {
      e.stopPropagation();
      (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    }
    setDraggedNode(null);
    setIsDragging(false);
    setIsPanning(false);
  };

  const handleNodeClick = (e: React.MouseEvent, node: CitationNode) => {
    if (!isDragging && !isPanning && interactionMode === 'pointer') {
      e.stopPropagation();
      onNodeSelect(node);
    }
  };

  const getNodeSize = (citations: number) => {
    const minSize = 14;
    const maxSize = 28;
    const minCitations = Math.min(...nodes.map(n => n.citations));
    const maxCitations = Math.max(...nodes.map(n => n.citations));
    const range = maxCitations - minCitations || 1;
    return minSize + ((citations - minCitations) / range) * (maxSize - minSize);
  };

  return (
    <div className="h-full glass-card rounded-lg p-4 overflow-hidden flex flex-col">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-3">
        <div className="flex-1 min-w-0">
          <h2 className="text-base font-semibold text-neutral-900 truncate">
            Citation Network
          </h2>
          <p className="text-neutral-600 text-xs mt-1 hidden sm:block">
            {interactionMode === 'pointer' ? 'Drag nodes • Click to select' : 'Pan mode active'}
          </p>
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <button
            onClick={toggleInteractionMode}
            className={`p-1.5 border rounded-md transition-all ${
              interactionMode === 'pointer'
                ? 'bg-green-500 text-white border-green-500'
                : 'bg-white text-neutral-700 border-neutral-200 hover:bg-neutral-100'
            }`}
            title={interactionMode === 'pointer' ? 'Switch to Pan' : 'Switch to Pointer'}
          >
            {interactionMode === 'pointer' ? <MousePointer size={16} /> : <Move size={16} />}
          </button>
          <button
            onClick={handleZoomOut}
            className="p-1.5 bg-white hover:bg-neutral-100 border border-neutral-200 rounded-md transition-all"
            title="Zoom Out"
          >
            <ZoomOut className="text-neutral-700" size={16} />
          </button>
          <button
            onClick={handleZoomIn}
            className="p-1.5 bg-white hover:bg-neutral-100 border border-neutral-200 rounded-md transition-all"
            title="Zoom In"
          >
            <ZoomIn className="text-neutral-700" size={16} />
          </button>
          <button
            onClick={handleRecenter}
            className="p-1.5 bg-white hover:bg-neutral-100 border border-neutral-200 rounded-md transition-all"
            title="Recenter"
          >
            <RotateCcw className="text-neutral-700" size={16} />
          </button>
        </div>
      </div>

      <div 
        ref={containerRef}
        className={`flex-1 relative bg-neutral-50 rounded-lg overflow-hidden touch-none border border-neutral-200 ${
          interactionMode === 'pan' ? 'cursor-grab' : ''
        } ${isPanning ? 'cursor-grabbing' : ''}`}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        onPointerDown={(e) => interactionMode === 'pan' && handlePointerDown(e)}
      >
        {/* Axes Labels */}
        <div className="absolute top-2 left-1/2 -translate-x-1/2 text-neutral-600 text-xs font-medium bg-white px-3 py-1 rounded-full z-10 pointer-events-none border border-neutral-200">
          <span className="hidden sm:inline">Recent publications →</span>
          <span className="sm:hidden">Recent →</span>
        </div>
        <div className="absolute left-2 top-1/2 -translate-y-1/2 -rotate-90 text-neutral-600 text-xs font-medium bg-white px-3 py-1 rounded-full z-10 pointer-events-none whitespace-nowrap border border-neutral-200">
          <span className="hidden sm:inline">More citations →</span>
          <span className="sm:hidden">Citations →</span>
        </div>

        {/* Graph Content */}
        <div 
          className="absolute inset-0 flex items-center justify-center transition-transform duration-200"
          style={{ 
            transform: `scale(${zoom}) translate(${panOffset.x / zoom}px, ${panOffset.y / zoom}px)`
          }}
        >
          <svg className="w-full h-full">
            {/* Grid lines */}
            <line x1="0" y1="50%" x2="100%" y2="50%" stroke="#e5e5e5" strokeWidth="1" />
            <line x1="50%" y1="0" x2="50%" y2="100%" stroke="#e5e5e5" strokeWidth="1" />

            {/* Connection lines */}
            {nodes.slice(0, -1).map((node, i) => {
              const nextNode = nodes[i + 1];
              return (
                <line
                  key={`line-${node.id}`}
                  x1={`${node.x}%`}
                  y1={`${node.y}%`}
                  x2={`${nextNode.x}%`}
                  y2={`${nextNode.y}%`}
                  stroke="#d4d4d4"
                  strokeWidth="1.5"
                  strokeDasharray="4 4"
                  opacity="0.5"
                />
              );
            })}

            {/* Nodes */}
            {nodes.map((node) => {
              const nodeSize = getNodeSize(node.citations);
              const isSelected = selectedNodeIds.includes(node.id);
              const isActive = hoveredNode === node.id || draggedNode === node.id;
              
              return (
                <g key={node.id}>
                  {/* Outer ring for selected nodes */}
                  {isSelected && (
                    <circle
                      cx={`${node.x}%`}
                      cy={`${node.y}%`}
                      r={nodeSize + 6}
                      fill="none"
                      stroke="#10b981"
                      strokeWidth="2"
                      className="transition-all duration-300"
                    />
                  )}
                  
                  {/* Main node circle */}
                  <circle
                    cx={`${node.x}%`}
                    cy={`${node.y}%`}
                    r={nodeSize}
                    fill={isSelected ? "#10b981" : isActive ? "#3b82f6" : "#ffffff"}
                    stroke={isSelected ? "#10b981" : "#d4d4d4"}
                    strokeWidth="2"
                    className="transition-all duration-200"
                    style={{ 
                      cursor: interactionMode === 'pointer' 
                        ? (draggedNode === node.id ? 'grabbing' : 'grab')
                        : 'default',
                    }}
                    onPointerEnter={() => !isDragging && !isPanning && setHoveredNode(node.id)}
                    onPointerLeave={() => !isDragging && !isPanning && setHoveredNode(null)}
                    onPointerDown={(e) => interactionMode === 'pointer' && handlePointerDown(e as any, node.id)}
                    onClick={(e) => handleNodeClick(e as any, node)}
                  />
                  
                  {/* Citation count */}
                  <text
                    x={`${node.x}%`}
                    y={`${node.y}%`}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="text-[10px] font-semibold pointer-events-none select-none"
                    fill={isSelected ? "#ffffff" : "#525252"}
                    style={{ transform: `translateY(${nodeSize + 14}px)` }}
                  >
                    {node.citations}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Tooltips */}
          {nodes.map((node) => (
            hoveredNode === node.id && !isDragging && !isPanning && (
              <div
                key={`tooltip-${node.id}`}
                className="absolute bg-white border border-neutral-200 rounded-lg p-2.5 pointer-events-none z-20 shadow-soft-lg"
                style={{
                  left: `${node.x}%`,
                  top: `${node.y}%`,
                  transform: 'translate(-50%, calc(-100% - 15px))',
                  maxWidth: '200px',
                }}
              >
                <p className="text-neutral-900 font-medium text-xs mb-1 leading-tight">{node.title}</p>
                <div className="flex items-center gap-3 text-xs text-neutral-600">
                  <span>{node.citations} citations</span>
                  <span>{node.year}</span>
                </div>
              </div>
            )
          ))}
        </div>
      </div>
    </div>
  );
}