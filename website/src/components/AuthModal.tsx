"use client"

import React, { useState } from 'react';
import { useAuth } from '@/components/contexts/AuthContext';
import { X, Loader2 } from 'lucide-react';

export default function AuthModal() {
  const { authModalView, hideAuthModal, login, register, updateProfile, user, showAuthModal } = useAuth();
  const [formData, setFormData] = useState({
    name: user?.name || '',
    email: '',
    password: '',
    confirmPassword: '',
    rememberMe: false,
  });
  const [showEmailVerification, setShowEmailVerification] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  if (!authModalView) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      if (authModalView === 'signIn') {
        await login(formData.email, formData.password, formData.rememberMe);
      } else if (authModalView === 'register') {
        if (formData.password !== formData.confirmPassword) {
          alert('Passwords do not match');
          setIsLoading(false);
          return;
        }
        const success = await register(formData.name, formData.email, formData.password);
        if (success) {
          setShowEmailVerification(true);
        }
      } else if (authModalView === 'editProfile') {
        await updateProfile(formData.name);
      }
    } catch (error) {
      // Error already handled in AuthContext
    } finally {
      setIsLoading(false);
    }
  };

  const switchToRegister = () => {
    setShowEmailVerification(false);
    setFormData({ name: '', email: '', password: '', confirmPassword: '', rememberMe: false });
    showAuthModal('register');
  };

  const switchToSignIn = () => {
    setShowEmailVerification(false);
    setFormData({ name: '', email: '', password: '', confirmPassword: '', rememberMe: false });
    showAuthModal('signIn');
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm">
      <div className="relative w-full max-w-md bg-white rounded-xl shadow-soft-lg p-8 border border-neutral-200">
        <button
          onClick={hideAuthModal}
          className="absolute top-4 right-4 text-neutral-400 hover:text-neutral-900 transition-colors"
          disabled={isLoading}
        >
          <X size={20} />
        </button>

        {/* Sign In View */}
        {authModalView === 'signIn' && (
          <div>
            <h2 className="text-2xl font-semibold text-neutral-900 mb-6">
              Sign In
            </h2>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-neutral-700 text-sm font-medium mb-2">Email</label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="w-full bg-white border border-neutral-200 text-neutral-900 placeholder-neutral-400 rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500"
                  placeholder="your@email.com"
                  required
                  disabled={isLoading}
                  autoComplete="off"
                />
              </div>
              <div>
                <label className="block text-neutral-700 text-sm font-medium mb-2">Password</label>
                <input
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className="w-full bg-white border border-neutral-200 text-neutral-900 placeholder-neutral-400 rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500"
                  placeholder="••••••••"
                  required
                  disabled={isLoading}
                  autoComplete="off"
                />
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="rememberMe"
                  checked={formData.rememberMe}
                  onChange={(e) => setFormData({ ...formData, rememberMe: e.target.checked })}
                  className="w-4 h-4 text-green-600 bg-white border-neutral-300 rounded focus:ring-green-500"
                  disabled={isLoading}
                />
                <label htmlFor="rememberMe" className="ml-2 text-sm text-neutral-700">
                  Remember me
                </label>
              </div>
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-neutral-900 hover:bg-neutral-800 text-white font-medium py-2.5 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin" size={18} />
                    Signing in...
                  </>
                ) : (
                  'Sign In'
                )}
              </button>
            </form>

            <p className="text-neutral-600 text-sm text-center mt-6">
              Don't have an account?{' '}
              <button
                onClick={switchToRegister}
                className="text-green-600 hover:text-green-700 font-medium"
                disabled={isLoading}
              >
                Register
              </button>
            </p>
          </div>
        )}

        {/* Register View */}
        {authModalView === 'register' && (
          <div>
            {showEmailVerification ? (
              <div className="text-center">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h2 className="text-2xl font-semibold text-neutral-900 mb-3">Account Created!</h2>
                <p className="text-neutral-600 mb-6 text-sm">
                  Your account has been created successfully. You can now sign in.
                </p>
                <button
                  onClick={switchToSignIn}
                  className="w-full bg-neutral-900 hover:bg-neutral-800 text-white font-medium py-2.5 rounded-lg transition-all"
                >
                  Go to Sign In
                </button>
              </div>
            ) : (
              <>
                <h2 className="text-2xl font-semibold text-neutral-900 mb-6">
                  Register
                </h2>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label className="block text-neutral-700 text-sm font-medium mb-2">Full Name</label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      className="w-full bg-white border border-neutral-200 text-neutral-900 placeholder-neutral-400 rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500"
                      placeholder="John Doe"
                      required
                      disabled={isLoading}
                      autoComplete="off"
                    />
                  </div>
                  <div>
                    <label className="block text-neutral-700 text-sm font-medium mb-2">Email</label>
                    <input
                      type="email"
                      value={formData.email}
                      onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                      className="w-full bg-white border border-neutral-200 text-neutral-900 placeholder-neutral-400 rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500"
                      placeholder="your@email.com"
                      required
                      disabled={isLoading}
                      autoComplete="off"
                    />
                  </div>
                  <div>
                    <label className="block text-neutral-700 text-sm font-medium mb-2">Password</label>
                    <input
                      type="password"
                      value={formData.password}
                      onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                      className="w-full bg-white border border-neutral-200 text-neutral-900 placeholder-neutral-400 rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500"
                      placeholder="••••••••"
                      required
                      disabled={isLoading}
                      autoComplete="off"
                    />
                  </div>
                  <div>
                    <label className="block text-neutral-700 text-sm font-medium mb-2">Confirm Password</label>
                    <input
                      type="password"
                      value={formData.confirmPassword}
                      onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                      className="w-full bg-white border border-neutral-200 text-neutral-900 placeholder-neutral-400 rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500"
                      placeholder="••••••••"
                      required
                      disabled={isLoading}
                      autoComplete="off"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full bg-neutral-900 hover:bg-neutral-800 text-white font-medium py-2.5 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="animate-spin" size={18} />
                        Creating account...
                      </>
                    ) : (
                      'Register'
                    )}
                  </button>
                </form>

                <p className="text-neutral-600 text-sm text-center mt-6">
                  Already have an account?{' '}
                  <button
                    onClick={switchToSignIn}
                    className="text-green-600 hover:text-green-700 font-medium"
                    disabled={isLoading}
                  >
                    Sign In
                  </button>
                </p>
              </>
            )}
          </div>
        )}

        {/* Edit Profile View */}
        {authModalView === 'editProfile' && (
          <div>
            <h2 className="text-2xl font-semibold text-neutral-900 mb-6">
              Edit Profile
            </h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-neutral-700 text-sm font-medium mb-2">Full Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full bg-white border border-neutral-200 text-neutral-900 placeholder-neutral-400 rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500"
                  placeholder="John Doe"
                  required
                  disabled={isLoading}
                />
              </div>
              <div>
                <label className="block text-neutral-700 text-sm font-medium mb-2">Email</label>
                <input
                  type="email"
                  value={user?.email || ''}
                  className="w-full bg-neutral-100 border border-neutral-200 text-neutral-500 rounded-lg px-4 py-2.5 cursor-not-allowed"
                  disabled
                />
              </div>
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-neutral-900 hover:bg-neutral-800 text-white font-medium py-2.5 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin" size={18} />
                    Saving...
                  </>
                ) : (
                  'Save Changes'
                )}
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}