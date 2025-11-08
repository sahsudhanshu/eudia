"use client"

import React, { useState, useEffect } from 'react';
import { FileText, Loader2 } from 'lucide-react';
import { useAuth } from '@/components/contexts/AuthContext';
import { apiFetch } from '@/lib/api';

interface Document {
  id: string;
  title: string;
  fileUrl: string;
  fileSize: number;
  uploadDate: string;
  status: string;
}

interface DocumentExplorerProps {
  onDocumentSelect: (documentId: string) => void;
  selectedDocumentId: string | null;
}

export default function DocumentExplorer({ onDocumentSelect, selectedDocumentId }: DocumentExplorerProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { user, isLoading: authLoading, showAuthModal } = useAuth();

  useEffect(() => {
    if (!authLoading) {
      if (user) {
        fetchDocuments();
      } else {
        setLoading(false);
        setDocuments([]);
      }
    }
  }, [user, authLoading]);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      try {
        const data = await apiFetch<Document[]>(`/api/documents`);
        setDocuments(data);
        setError(null);
      } catch (err) {
        setError('Please sign in to view your documents');
        setDocuments([]);
        console.error(err);
      }
    } finally {
      setLoading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  if (authLoading) {
    return (
      <div className="h-full glass-card rounded-lg p-4 flex items-center justify-center">
        <Loader2 className="animate-spin text-neutral-400" size={32} />
      </div>
    );
  }

  if (!user) {
    return (
      <div className="h-full glass-card rounded-lg p-4 flex flex-col items-center justify-center">
        <FileText size={48} className="text-neutral-300 mb-4" />
        <h3 className="text-neutral-900 font-semibold mb-2">Sign In Required</h3>
        <p className="text-neutral-600 text-sm text-center mb-4">
          Please sign in to view and manage your documents
        </p>
        <button
          onClick={() => showAuthModal('signIn')}
          className="bg-green-500 hover:bg-green-600 text-white font-medium px-4 py-2 rounded-lg transition-all text-sm"
        >
          Sign In
        </button>
      </div>
    );
  }

  return (
    <div className="h-full glass-card rounded-lg p-4 overflow-hidden flex flex-col">
      <div className="mb-4">
        <h2 className="text-base font-semibold text-neutral-900 flex items-center gap-2">
          <FileText size={18} className="text-neutral-700" />
          Documents
        </h2>
        <p className="text-neutral-600 text-xs mt-1">
          {loading ? 'Loading...' : `${documents.length} document${documents.length !== 1 ? 's' : ''}`}
        </p>
      </div>

      {loading ? (
        <div className="flex-1 flex items-center justify-center">
          <Loader2 className="animate-spin text-neutral-400" size={32} />
        </div>
      ) : error ? (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-neutral-500 text-sm">{error}</p>
        </div>
      ) : documents.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-neutral-500 text-sm">No documents uploaded yet</p>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto space-y-2 pr-1 scrollbar-thin">
          {documents.map((doc) => (
            <div
              key={doc.id}
              onClick={() => onDocumentSelect(doc.id)}
              className={`bg-white border rounded-lg p-3 cursor-pointer transition-all group ${
                selectedDocumentId === doc.id
                  ? 'border-green-500 bg-green-50 shadow-soft'
                  : 'border-neutral-200 hover:border-neutral-300 hover:shadow-soft'
              }`}
            >
              <div className="flex items-start justify-between mb-2 gap-2">
                <h3 className="text-neutral-900 text-sm font-medium line-clamp-2 flex-1">
                  {doc.title}
                </h3>
                {selectedDocumentId === doc.id && (
                  <div className="w-2 h-2 bg-green-500 rounded-full shrink-0 mt-1"></div>
                )}
              </div>
              
              <div className="flex items-center justify-between text-xs text-neutral-600">
                <span>{formatFileSize(doc.fileSize)}</span>
                <span>{formatDate(doc.uploadDate)}</span>
              </div>
              
              <div className="mt-2">
                <span className={`text-xs px-2 py-0.5 rounded ${
                  doc.status === 'completed' 
                    ? 'bg-green-100 text-green-700' 
                    : doc.status === 'processing'
                    ? 'bg-blue-100 text-blue-700'
                    : 'bg-red-100 text-red-700'
                }`}>
                  {doc.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}