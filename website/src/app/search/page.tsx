"use client"

import React, { useState, useEffect } from 'react';
import { AuthProvider } from '@/components/contexts/AuthContext';
import Navbar from '@/components/Navbar';
import PrivacyBanner from '@/components/PrivacyBanner';
import AuthModal from '@/components/AuthModal';
import { Search, Loader2, FileText, ExternalLink, ChevronLeft, ChevronRight, Filter, Download, X, Calendar, Building2, FileCode, Tag, Users, Scale } from 'lucide-react';
import { useDebounce } from '@/hooks/useDebounce';

interface SearchResult {
  tid: number;
  title: string;
  headline?: string;
  docsource?: string;
  publishdate?: string;
  author?: string;
  numcites?: number;
  numcitedby?: number;
  citation?: string;
  docsize?: number;
}

interface CategoryItem {
  value: string;
  formInput: string;
  selected?: boolean;
}

interface Category {
  name: string;
  items: CategoryItem[];
}

export default function SearchPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [categories, setCategories] = useState<Category[]>([]);
  const [foundText, setFoundText] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [hasSearched, setHasSearched] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<SearchResult | null>(null);
  const [documentDetails, setDocumentDetails] = useState<any>(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  
  const debouncedSearchQuery = useDebounce(searchQuery, 500);

  useEffect(() => {
    if (debouncedSearchQuery.trim().length >= 3) {
      performSearch(debouncedSearchQuery, currentPage);
    } else if (debouncedSearchQuery.trim().length === 0) {
      setResults([]);
      setCategories([]);
      setFoundText('');
      setHasSearched(false);
      setError(null);
    }
  }, [debouncedSearchQuery, currentPage]);

  const performSearch = async (query: string, page: number) => {
    try {
      setLoading(true);
      setError(null);
      setHasSearched(true);

      const response = await fetch('/api/indian-kanoon/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          formInput: query,
          pagenum: page,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch search results');
      }

      const data = await response.json();
      
      // Parse categories
      if (data.categories && Array.isArray(data.categories)) {
        const parsedCategories: Category[] = data.categories.map((cat: any) => ({
          name: cat[0],
          items: cat[1] || [],
        }));
        setCategories(parsedCategories);
      }

      // Set results
      if (data.docs && Array.isArray(data.docs)) {
        setResults(data.docs);
      } else {
        setResults([]);
      }

      // Set found text
      if (data.found) {
        setFoundText(data.found);
      }
    } catch (err) {
      console.error('Search error:', err);
      setError('Failed to search. Please try again.');
      setResults([]);
      setCategories([]);
    } finally {
      setLoading(false);
    }
  };

  const handleCategoryClick = (categoryName: string, item: CategoryItem) => {
    setSearchQuery(item.value);
    setCurrentPage(0);
    performSearch(item.formInput, 0);
  };

  const handlePageChange = (newPage: number) => {
    if (newPage >= 0) {
      setCurrentPage(newPage);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handleDocumentClick = async (result: SearchResult) => {
    setSelectedDocument(result);
    setLoadingDetails(true);
    
    try {
      const response = await fetch(`/api/indian-kanoon/document/${result.tid}`);
      if (response.ok) {
        const data = await response.json();
        setDocumentDetails(data);
      }
    } catch (err) {
      console.error('Error fetching document details:', err);
    } finally {
      setLoadingDetails(false);
    }
  };

  const handleDownloadDocument = (docId: number, format: 'pdf' | 'txt') => {
    if (format === 'pdf') {
      window.open(`https://indiankanoon.org/origdoc/${docId}/`, '_blank', 'noopener,noreferrer');
    } else {
      window.open(`https://indiankanoon.org/doc/${docId}/`, '_blank', 'noopener,noreferrer');
    }
  };

  const stripHtmlTags = (html: string) => {
    return html?.replace(/<[^>]*>/g, '').replace(/&nbsp;/g, ' ').trim() || '';
  };

  const getCategoryIcon = (categoryName: string) => {
    if (categoryName.includes('AI Tags')) return <Tag size={16} />;
    if (categoryName.includes('Court') || categoryName.includes('Law')) return <Building2 size={16} />;
    if (categoryName.includes('Author') || categoryName.includes('Bench')) return <Users size={16} />;
    if (categoryName.includes('Year')) return <Calendar size={16} />;
    if (categoryName.includes('Document')) return <FileText size={16} />;
    return <Scale size={16} />;
  };

  return (
    <AuthProvider>
      <div className="min-h-screen bg-neutral-50">
        <Navbar />
        <PrivacyBanner />
        <AuthModal />

        <div className="w-full px-3 sm:px-4 md:px-6 py-3 sm:py-4 md:py-6">
          <div className="max-w-7xl mx-auto">
            {/* Search Header */}
            <div className="glass-card rounded-lg p-6 mb-6">
              <h1 className="text-2xl font-semibold text-neutral-900 mb-2">
                Indian Kanoon Legal Search
              </h1>
              <p className="text-neutral-600 text-sm mb-6">
                Search through the comprehensive Indian legal database for cases, judgments, and legal documents
              </p>

              {/* Search Input with Filter Button */}
              <div className="flex gap-3">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-400" size={20} />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Enter your search query (minimum 3 characters)..."
                    className="w-full pl-10 pr-4 py-3 border border-neutral-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent text-neutral-900 placeholder-neutral-400"
                  />
                </div>
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className={`px-4 py-3 border rounded-lg transition-all flex items-center gap-2 text-sm font-medium ${
                    showFilters 
                      ? 'bg-green-50 border-green-500 text-green-700' 
                      : 'bg-white border-neutral-200 text-neutral-700 hover:bg-neutral-50'
                  }`}
                >
                  <Filter size={18} />
                  <span className="hidden sm:inline">Filters</span>
                </button>
              </div>

              {/* Search Info */}
              <div className="mt-3 flex items-center justify-between text-xs text-neutral-600">
                <span>
                  {loading ? 'Searching...' : foundText ? foundText : hasSearched ? `${results.length} result${results.length !== 1 ? 's' : ''} found` : 'Enter a query to search'}
                </span>
                {searchQuery.length > 0 && searchQuery.length < 3 && (
                  <span className="text-orange-600">Minimum 3 characters required</span>
                )}
              </div>
            </div>

            {/* 2-Column Layout: Categories Sidebar + Results */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              {/* Categories Sidebar */}
              {showFilters && categories.length > 0 && (
                <div className="lg:col-span-1 space-y-4">
                  {categories.map((category, idx) => (
                    <div key={idx} className="glass-card rounded-lg p-4">
                      <h3 className="text-sm font-semibold text-neutral-900 mb-3 flex items-center gap-2">
                        {getCategoryIcon(category.name)}
                        {category.name}
                      </h3>
                      <div className="space-y-1.5">
                        {category.items.slice(0, 10).map((item, itemIdx) => (
                          <button
                            key={itemIdx}
                            onClick={() => handleCategoryClick(category.name, item)}
                            className={`w-full text-left px-3 py-2 rounded-md text-xs transition-all ${
                              item.selected
                                ? 'bg-green-50 text-green-700 font-medium'
                                : 'text-neutral-700 hover:bg-neutral-50'
                            }`}
                          >
                            {stripHtmlTags(item.value)}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Results Column */}
              <div className={showFilters && categories.length > 0 ? 'lg:col-span-3' : 'lg:col-span-4'}>
                {/* Loading State */}
                {loading && (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="animate-spin text-green-500" size={40} />
                  </div>
                )}

                {/* Error State */}
                {error && !loading && (
                  <div className="glass-card rounded-lg p-6 text-center">
                    <p className="text-red-600">{error}</p>
                  </div>
                )}

                {/* No Results */}
                {!loading && !error && hasSearched && results.length === 0 && (
                  <div className="glass-card rounded-lg p-12 text-center">
                    <FileText className="mx-auto text-neutral-400 mb-4" size={48} />
                    <h3 className="text-lg font-medium text-neutral-900 mb-2">No results found</h3>
                    <p className="text-neutral-600 text-sm">Try adjusting your search query or filters</p>
                  </div>
                )}

                {/* Results List */}
                {!loading && results.length > 0 && (
                  <>
                    <div className="space-y-4 mb-6">
                      {results.map((result, index) => (
                        <div
                          key={result.tid || index}
                          className="glass-card rounded-lg p-5 hover:shadow-soft-lg transition-all group"
                        >
                          <div className="flex items-start justify-between gap-4 mb-3">
                            <h3 
                              className="text-base font-medium text-neutral-900 group-hover:text-green-600 transition-colors flex-1 line-clamp-2 cursor-pointer"
                              onClick={() => handleDocumentClick(result)}
                            >
                              {stripHtmlTags(result.title || 'Untitled Document')}
                            </h3>
                            <div className="flex items-center gap-2 flex-shrink-0">
                              <button
                                onClick={() => handleDocumentClick(result)}
                                className="p-2 text-neutral-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all"
                                title="View Details"
                              >
                                <FileCode size={18} />
                              </button>
                              <button
                                onClick={() => handleDownloadDocument(result.tid, 'pdf')}
                                className="p-2 text-neutral-400 hover:text-green-600 hover:bg-green-50 rounded-lg transition-all"
                                title="Download PDF"
                              >
                                <Download size={18} />
                              </button>
                              <button
                                onClick={() => window.open(`https://indiankanoon.org/doc/${result.tid}/`, '_blank', 'noopener,noreferrer')}
                                className="p-2 text-neutral-400 hover:text-green-600 hover:bg-green-50 rounded-lg transition-all"
                                title="Open in Indian Kanoon"
                              >
                                <ExternalLink size={18} />
                              </button>
                            </div>
                          </div>

                          {result.citation && (
                            <p className="text-sm text-neutral-700 mb-2 font-medium">
                              {result.citation}
                            </p>
                          )}

                          {result.headline && (
                            <div 
                              className="text-sm text-neutral-600 line-clamp-3 mb-3"
                              dangerouslySetInnerHTML={{ __html: result.headline }}
                            />
                          )}

                          <div className="flex flex-wrap items-center gap-3 text-xs text-neutral-500">
                            {result.docsource && (
                              <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded">
                                {result.docsource}
                              </span>
                            )}
                            {result.publishdate && (
                              <span className="bg-neutral-100 text-neutral-700 px-2 py-1 rounded">
                                {result.publishdate}
                              </span>
                            )}
                            {result.author && (
                              <span className="bg-purple-50 text-purple-700 px-2 py-1 rounded">
                                {result.author}
                              </span>
                            )}
                            {result.numcites !== undefined && (
                              <span className="bg-green-50 text-green-700 px-2 py-1 rounded">
                                {result.numcites} citations
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Pagination */}
                    <div className="glass-card rounded-lg p-4 flex items-center justify-between">
                      <button
                        onClick={() => handlePageChange(currentPage - 1)}
                        disabled={currentPage === 0}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-neutral-700 bg-white border border-neutral-200 rounded-lg hover:bg-neutral-50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                      >
                        <ChevronLeft size={16} />
                        Previous
                      </button>

                      <span className="text-sm text-neutral-600">
                        Page {currentPage + 1}
                      </span>

                      <button
                        onClick={() => handlePageChange(currentPage + 1)}
                        disabled={results.length < 10}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-neutral-700 bg-white border border-neutral-200 rounded-lg hover:bg-neutral-50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                      >
                        Next
                        <ChevronRight size={16} />
                      </button>
                    </div>
                  </>
                )}

                {/* Initial State */}
                {!hasSearched && !loading && (
                  <div className="glass-card rounded-lg p-12 text-center">
                    <Search className="mx-auto text-neutral-300 mb-4" size={64} />
                    <h3 className="text-lg font-medium text-neutral-900 mb-2">
                      Start Your Legal Research
                    </h3>
                    <p className="text-neutral-600 text-sm max-w-md mx-auto mb-6">
                      Search through thousands of Indian legal cases, judgments, and documents. Use advanced filters to refine your search.
                    </p>
                    <div className="flex flex-wrap justify-center gap-4 text-xs text-neutral-500">
                      <div className="flex items-center gap-2">
                        <Building2 size={14} />
                        <span>Filter by Court</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Calendar size={14} />
                        <span>Date Range</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Download size={14} />
                        <span>Download Documents</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Document Details Modal */}
        {selectedDocument && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden shadow-soft-lg">
              {/* Modal Header */}
              <div className="bg-neutral-50 border-b border-neutral-200 p-6 flex items-start justify-between">
                <div className="flex-1 pr-4">
                  <h2 className="text-xl font-semibold text-neutral-900 mb-2">
                    {stripHtmlTags(selectedDocument.title)}
                  </h2>
                  {selectedDocument.citation && (
                    <p className="text-sm text-neutral-600">{selectedDocument.citation}</p>
                  )}
                </div>
                <button
                  onClick={() => {
                    setSelectedDocument(null);
                    setDocumentDetails(null);
                  }}
                  className="p-2 text-neutral-500 hover:text-neutral-700 hover:bg-neutral-200 rounded-lg transition-colors"
                >
                  <X size={20} />
                </button>
              </div>

              {/* Modal Body */}
              <div className="p-6 overflow-y-auto max-h-[calc(90vh-180px)]">
                {loadingDetails ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="animate-spin text-green-500" size={32} />
                  </div>
                ) : documentDetails ? (
                  <div className="space-y-6">
                    {/* Metadata */}
                    <div className="flex flex-wrap gap-3">
                      {selectedDocument.docsource && (
                        <span className="bg-blue-50 text-blue-700 px-3 py-1.5 rounded text-sm">
                          {selectedDocument.docsource}
                        </span>
                      )}
                      {selectedDocument.publishdate && (
                        <span className="bg-neutral-100 text-neutral-700 px-3 py-1.5 rounded text-sm">
                          {selectedDocument.publishdate}
                        </span>
                      )}
                    </div>

                    {/* Document Content */}
                    <div className="prose prose-sm max-w-none text-neutral-700">
                      {documentDetails.doc && (
                        <div dangerouslySetInnerHTML={{ __html: documentDetails.doc }} />
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <FileText className="mx-auto text-neutral-400 mb-4" size={48} />
                    <p className="text-neutral-600">No document details available</p>
                  </div>
                )}
              </div>

              {/* Modal Footer */}
              <div className="bg-neutral-50 border-t border-neutral-200 p-6 flex gap-3">
                <button
                  onClick={() => handleDownloadDocument(selectedDocument.tid, 'pdf')}
                  className="flex-1 bg-green-500 hover:bg-green-600 text-white font-medium px-4 py-2.5 rounded-lg transition-all flex items-center justify-center gap-2 text-sm"
                >
                  <Download size={16} />
                  Download PDF
                </button>
                <button
                  onClick={() => handleDownloadDocument(selectedDocument.tid, 'txt')}
                  className="flex-1 bg-white hover:bg-neutral-50 text-neutral-700 font-medium px-4 py-2.5 rounded-lg transition-all border border-neutral-200 flex items-center justify-center gap-2 text-sm"
                >
                  <FileText size={16} />
                  View Full Text
                </button>
                <button
                  onClick={() => window.open(`https://indiankanoon.org/doc/${selectedDocument.tid}/`, '_blank', 'noopener,noreferrer')}
                  className="flex-1 bg-white hover:bg-neutral-50 text-neutral-700 font-medium px-4 py-2.5 rounded-lg transition-all border border-neutral-200 flex items-center justify-center gap-2 text-sm"
                >
                  <ExternalLink size={16} />
                  Open in Browser
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AuthProvider>
  );
}