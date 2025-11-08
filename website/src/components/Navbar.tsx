"use client"

import React, { useState } from 'react';
import { useAuth } from '@/components/contexts/AuthContext';
import { Scale, ChevronDown, Menu, X } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const { user, logout, showAuthModal } = useAuth();
  const [showProfileDropdown, setShowProfileDropdown] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
  const pathname = usePathname();

  const isActive = (path: string) => pathname === path;

  return (
    <nav className="sticky top-0 z-40 bg-white border-b border-neutral-200 shadow-soft">
      <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 py-3">
        <div className="flex items-center justify-between">
          {/* Left: Title */}
          <Link href="/" className="flex items-center gap-2 sm:gap-3 hover:opacity-80 transition-opacity">
            <Scale className="text-neutral-900 flex-shrink-0" size={20} />
            <h1 className="text-base sm:text-lg font-semibold text-neutral-900 truncate">
              LegalAI Analyst
            </h1>
          </Link>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setShowMobileMenu(!showMobileMenu)}
            className="md:hidden p-2 text-neutral-700 hover:bg-neutral-100 rounded-lg transition-colors"
            aria-label="Toggle menu"
          >
            {showMobileMenu ? <X size={20} /> : <Menu size={20} />}
          </button>

          {/* Desktop: Center Navigation + Right Auth */}
          <div className="hidden md:flex items-center gap-6">
            {/* Center: Navigation */}
            <div className="flex items-center gap-1">
              <Link
                href="/"
                className={`transition-colors font-medium text-sm px-3 py-1.5 rounded-md ${
                  isActive('/') 
                    ? 'text-green-600 bg-green-50' 
                    : 'text-neutral-700 hover:text-neutral-900 hover:bg-neutral-100'
                }`}
              >
                Documents
              </Link>
              <Link
                href="/search"
                className={`transition-colors font-medium text-sm px-3 py-1.5 rounded-md ${
                  isActive('/search') 
                    ? 'text-green-600 bg-green-50' 
                    : 'text-neutral-700 hover:text-neutral-900 hover:bg-neutral-100'
                }`}
              >
                Search
              </Link>
            </div>

            {/* Right: Auth */}
            <div>
              {user ? (
                <div className="relative">
                  <button
                    onClick={() => setShowProfileDropdown(!showProfileDropdown)}
                    className="flex items-center gap-2 bg-white hover:bg-neutral-50 text-neutral-900 px-3 py-1.5 rounded-lg transition-all border border-neutral-200"
                  >
                    <div className="w-6 h-6 bg-neutral-900 rounded-full flex items-center justify-center text-white font-medium text-xs">
                      {user.name.charAt(0).toUpperCase()}
                    </div>
                    <span className="font-medium text-sm hidden lg:inline">{user.name}</span>
                    <ChevronDown size={14} />
                  </button>

                  {showProfileDropdown && (
                    <>
                      <div 
                        className="fixed inset-0 z-10" 
                        onClick={() => setShowProfileDropdown(false)}
                      />
                      <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-soft-lg overflow-hidden z-20 border border-neutral-200">
                        <button
                          onClick={() => {
                            setShowProfileDropdown(false);
                            showAuthModal('editProfile');
                          }}
                          className="w-full text-left px-4 py-2.5 text-neutral-700 hover:bg-neutral-50 transition-colors text-sm"
                        >
                          Edit Profile
                        </button>
                        <button
                          onClick={() => {
                            setShowProfileDropdown(false);
                            logout();
                          }}
                          className="w-full text-left px-4 py-2.5 text-neutral-700 hover:bg-neutral-50 transition-colors border-t border-neutral-200 text-sm"
                        >
                          Sign Out
                        </button>
                      </div>
                    </>
                  )}
                </div>
              ) : (
                <button
                  onClick={() => showAuthModal('signIn')}
                  className="bg-green-500 hover:bg-green-600 text-white font-medium px-4 py-1.5 rounded-lg transition-all text-sm"
                >
                  Sign In
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        {showMobileMenu && (
          <div className="md:hidden mt-4 pt-4 border-t border-neutral-200 space-y-2">
            <Link
              href="/"
              onClick={() => setShowMobileMenu(false)}
              className={`block w-full text-left transition-colors font-medium py-2 px-3 rounded-md text-sm ${
                isActive('/') 
                  ? 'text-green-600 bg-green-50' 
                  : 'text-neutral-700 hover:text-neutral-900 hover:bg-neutral-100'
              }`}
            >
              Documents
            </Link>
            <Link
              href="/search"
              onClick={() => setShowMobileMenu(false)}
              className={`block w-full text-left transition-colors font-medium py-2 px-3 rounded-md text-sm ${
                isActive('/search') 
                  ? 'text-green-600 bg-green-50' 
                  : 'text-neutral-700 hover:text-neutral-900 hover:bg-neutral-100'
              }`}
            >
              Search
            </Link>
            
            <div className="pt-3 border-t border-neutral-200">
              {user ? (
                <div className="space-y-2">
                  <div className="flex items-center gap-3 py-2 px-3">
                    <div className="w-8 h-8 bg-neutral-900 rounded-full flex items-center justify-center text-white font-medium text-sm">
                      {user.name.charAt(0).toUpperCase()}
                    </div>
                    <span className="text-neutral-900 font-medium text-sm">{user.name}</span>
                  </div>
                  <button
                    onClick={() => {
                      setShowMobileMenu(false);
                      showAuthModal('editProfile');
                    }}
                    className="w-full text-left bg-white hover:bg-neutral-50 text-neutral-700 px-3 py-2 rounded-lg transition-colors border border-neutral-200 text-sm"
                  >
                    Edit Profile
                  </button>
                  <button
                    onClick={() => {
                      setShowMobileMenu(false);
                      logout();
                    }}
                    className="w-full text-left bg-white hover:bg-neutral-50 text-neutral-700 px-3 py-2 rounded-lg transition-colors border border-neutral-200 text-sm"
                  >
                    Sign Out
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => {
                    setShowMobileMenu(false);
                    showAuthModal('signIn');
                  }}
                  className="w-full bg-green-500 hover:bg-green-600 text-white font-medium px-4 py-2 rounded-lg transition-all text-sm"
                >
                  Sign In
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}