"use client"

import React from 'react';
import { Lock, Check } from 'lucide-react';

export default function PrivacyBanner() {
  return (
    <div className="bg-green-50 border-b border-green-100 py-3 sm:py-4">
      <div className="max-w-screen-2xl mx-auto px-4 sm:px-6">
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-8">
          {/* Main Privacy Message */}
          <div className="flex items-center gap-2 text-green-900">
            <Lock size={16} className="text-green-600 flex-shrink-0" />
            <p className="text-xs font-semibold">
              100% Local Processing
            </p>
          </div>

          {/* Separator */}
          <div className="hidden sm:block w-px h-4 bg-green-200" />

          {/* 3 Checkmarks - Key Features */}
          <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-6 text-xs text-green-800">
            <div className="flex items-center gap-1.5">
              <Check size={14} className="text-green-600 flex-shrink-0" />
              <span>Data stays on your machine</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Check size={14} className="text-green-600 flex-shrink-0" />
              <span>Secure processing</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Check size={14} className="text-green-600 flex-shrink-0" />
              <span>Privacy guaranteed</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}