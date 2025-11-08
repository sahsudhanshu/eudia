"use client"

import React, { useState, useRef } from 'react';
import { Upload, FileText, Check, AlertCircle, X, Brain, Search, FileCheck } from 'lucide-react';
import { useAuth } from '@/components/contexts/AuthContext';
import { apiFetch } from '@/lib/api';

interface UploadPanelProps {
  onUploadComplete?: (documentId: string) => void;
  onClose?: () => void;
}

export default function UploadPanel({ onUploadComplete, onClose }: UploadPanelProps) {
  const { user, isLoading: authLoading, showAuthModal } = useAuth();
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState<'extracting' | 'analyzing' | 'validating' | 'complete'>('extracting');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Show sign in prompt if not authenticated
  if (authLoading) {
    return (
      <div className="w-full max-w-2xl mx-auto">
        <div className="glass-card rounded-xl p-8 sm:p-12 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-neutral-200 border-t-green-500 mx-auto mb-4"></div>
          <p className="text-neutral-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="w-full max-w-2xl mx-auto">
        <div className="glass-card rounded-xl p-8 sm:p-12 text-center">
          <Upload className="mx-auto text-neutral-300 mb-4" size={64} />
          <h2 className="text-2xl font-semibold text-neutral-900 mb-3">
            Sign In Required
          </h2>
          <p className="text-neutral-600 mb-6">
            Please sign in to upload and analyze legal documents
          </p>
          <button
            onClick={() => showAuthModal('signIn')}
            className="bg-green-500 hover:bg-green-600 text-white font-medium px-6 py-2.5 rounded-lg transition-all"
          >
            Sign In
          </button>
        </div>
      </div>
    );
  }

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Only set dragging false if leaving the drop zone completely
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX;
    const y = e.clientY;
    
    if (x <= rect.left || x >= rect.right || y <= rect.top || y >= rect.bottom) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

  const handleFile = (file: File) => {
    setError(null);
    setSuccess(false);

    // Validate file type
    if (file.type !== 'application/pdf') {
      setError('Please select a PDF file');
      return;
    }

    // Validate file size (50MB max)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File size exceeds 50MB limit');
      return;
    }

    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile || !user) return;

    try {
      setIsUploading(true);
      setError(null);
      setUploadProgress(0);

      const formData = new FormData();
      formData.append('file', selectedFile);

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const response = await apiFetch<{
        document: { id: string };
        message: string;
      }>(`/api/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);
      
      // Start document analysis
      setIsUploading(false);
      setIsAnalyzing(true);
      
      // Simulate analysis steps
      setAnalysisStep('extracting');
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setAnalysisStep('analyzing');
      await new Promise(resolve => setTimeout(resolve, 2500));
      
      setAnalysisStep('validating');
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setAnalysisStep('complete');
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Update document status to completed
      if (response.document?.id) {
        await apiFetch(`/api/documents/${response.document.id}`, {
          method: 'PUT',
          body: JSON.stringify({ status: 'completed' }),
        });
      }
      
      setIsAnalyzing(false);
      setSuccess(true);

      // Call completion callback after a brief delay
      setTimeout(() => {
        if (onUploadComplete && response.document) {
          onUploadComplete(response.document.id);
        }
        if (onClose) {
          onClose();
        }
      }, 1500);
    } catch (err) {
      console.error('Upload error:', err);
      setError(err instanceof Error ? err.message : 'Failed to upload file');
      setUploadProgress(0);
      setIsAnalyzing(false);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setError(null);
    setSuccess(false);
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const getAnalysisStepInfo = () => {
    switch (analysisStep) {
      case 'extracting':
        return { text: 'Extracting text from document...', icon: Search };
      case 'analyzing':
        return { text: 'Analyzing document structure...', icon: Brain };
      case 'validating':
        return { text: 'Validating citations and references...', icon: FileCheck };
      case 'complete':
        return { text: 'Analysis complete!', icon: Check };
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="glass-card rounded-xl p-8 sm:p-12 relative">
        {onClose && (
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-2 text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100 rounded-lg transition-colors"
          >
            <X size={20} />
          </button>
        )}

        {!selectedFile ? (
          <div
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-xl p-8 sm:p-12 transition-all ${
              isDragging 
                ? 'border-green-500 bg-green-50' 
                : 'border-neutral-300 hover:border-neutral-400'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileInput}
              className="hidden"
            />

            <div className="text-center">
              <div className="w-16 h-16 sm:w-20 sm:h-20 bg-neutral-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <Upload className="text-neutral-600" size={28} />
              </div>

              <h2 className="text-2xl sm:text-3xl font-semibold text-neutral-900 mb-3">
                Upload your document
              </h2>
              
              <p className="text-neutral-600 text-sm sm:text-base mb-8 max-w-md mx-auto">
                Drag and drop your PDF here, or click the button below to browse files
              </p>

              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-neutral-900 hover:bg-neutral-800 text-white font-medium px-6 py-2.5 rounded-lg transition-all inline-flex items-center gap-2 text-sm"
              >
                <FileText size={16} />
                Choose File
              </button>

              <p className="text-neutral-500 text-xs mt-6">
                Supported format: PDF • Maximum size: 50MB
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-neutral-900 mb-2">
                {success ? 'Upload Complete!' : isAnalyzing ? 'Analyzing Document...' : isUploading ? 'Uploading...' : 'Ready to Upload'}
              </h2>
              <p className="text-neutral-600 text-sm">
                {success 
                  ? 'Your document has been uploaded and analyzed successfully' 
                  : isAnalyzing 
                  ? 'AI model is analyzing your document' 
                  : isUploading 
                  ? 'Please wait while we process your file' 
                  : 'Review your file before uploading'}
              </p>
            </div>

            {/* File Info Card */}
            <div className="bg-white border border-neutral-200 rounded-lg p-4">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-red-50 rounded-lg flex items-center justify-center shrink-0">
                  <FileText className="text-red-600" size={24} />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-neutral-900 font-medium text-sm mb-1 truncate">
                    {selectedFile.name}
                  </h3>
                  <p className="text-neutral-600 text-xs">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
                {!isUploading && !success && !isAnalyzing && (
                  <button
                    onClick={handleRemoveFile}
                    className="p-1 text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100 rounded transition-colors"
                  >
                    <X size={18} />
                  </button>
                )}
              </div>

              {/* Upload Progress Bar */}
              {isUploading && (
                <div className="mt-4">
                  <div className="w-full bg-neutral-200 rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-green-500 h-full transition-all duration-300 ease-out"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <p className="text-xs text-neutral-600 mt-2 text-center">
                    {uploadProgress}% uploaded
                  </p>
                </div>
              )}

              {/* Analysis Progress */}
              {isAnalyzing && (
                <div className="mt-4 space-y-3">
                  {/* Analysis Steps */}
                  <div className="space-y-2">
                    {(['extracting', 'analyzing', 'validating', 'complete'] as const).map((step, index) => {
                      const isCurrentStep = step === analysisStep;
                      const isPastStep = (['extracting', 'analyzing', 'validating', 'complete'] as const).indexOf(step) < 
                                         (['extracting', 'analyzing', 'validating', 'complete'] as const).indexOf(analysisStep);
                      const isCompleted = analysisStep === 'complete' || isPastStep;
                      
                      return (
                        <div 
                          key={step}
                          className={`flex items-center gap-3 p-2 rounded-lg transition-all ${
                            isCurrentStep ? 'bg-green-50' : ''
                          }`}
                        >
                          <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${
                            isCompleted 
                              ? 'bg-green-500' 
                              : isCurrentStep 
                              ? 'bg-green-100' 
                              : 'bg-neutral-200'
                          }`}>
                            {isCompleted ? (
                              <Check size={14} className="text-white" />
                            ) : isCurrentStep ? (
                              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                            ) : (
                              <div className="w-2 h-2 bg-neutral-400 rounded-full" />
                            )}
                          </div>
                          <span className={`text-xs ${
                            isCurrentStep ? 'text-neutral-900 font-medium' : 'text-neutral-600'
                          }`}>
                            {step === 'extracting' && 'Extracting text from document'}
                            {step === 'analyzing' && 'Analyzing document structure'}
                            {step === 'validating' && 'Validating citations and references'}
                            {step === 'complete' && 'Analysis complete!'}
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Animated Progress Bar */}
                  <div className="w-full bg-neutral-200 rounded-full h-1.5 overflow-hidden">
                    <div className="bg-green-500 h-full animate-pulse" style={{ 
                      width: analysisStep === 'extracting' ? '25%' 
                            : analysisStep === 'analyzing' ? '50%' 
                            : analysisStep === 'validating' ? '75%' 
                            : '100%',
                      transition: 'width 0.5s ease-out'
                    }} />
                  </div>
                </div>
              )}

              {/* Success Message */}
              {success && (
                <div className="mt-4 flex items-center gap-2 text-green-600 bg-green-50 px-3 py-2 rounded-lg">
                  <Check size={16} />
                  <span className="text-sm font-medium">Document uploaded and analyzed successfully</span>
                </div>
              )}
            </div>

            {/* Error Message */}
            {error && (
              <div className="flex items-start gap-2 text-red-600 bg-red-50 px-4 py-3 rounded-lg">
                <AlertCircle size={18} className="shrink-0 mt-0.5" />
                <span className="text-sm">{error}</span>
              </div>
            )}

            {/* Action Buttons */}
            {!isUploading && !success && !isAnalyzing && (
              <div className="flex gap-3">
                <button
                  onClick={handleRemoveFile}
                  className="flex-1 bg-white hover:bg-neutral-50 text-neutral-700 font-medium px-6 py-2.5 rounded-lg transition-all border border-neutral-200 text-sm"
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpload}
                  className="flex-1 bg-green-500 hover:bg-green-600 text-white font-medium px-6 py-2.5 rounded-lg transition-all text-sm"
                >
                  Upload Document
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}