"use client"

import React, { createContext, useContext, ReactNode, useCallback, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import {
  login as loginRequest,
  register as registerRequest,
  logout as logoutRequest,
  fetchCurrentUser,
  refreshAccessToken,
} from '@/lib/auth';
import { moveTokensToSession, getRefreshTokenValue } from '@/lib/api';

interface User {
  id: string;
  name: string;
  email: string;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  login: (email: string, password: string, rememberMe: boolean) => Promise<void>;
  logout: () => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<boolean>;
  updateProfile: (name: string) => Promise<void>;
  showAuthModal: (view?: 'signIn' | 'register' | 'editProfile') => void;
  hideAuthModal: () => void;
  authModalView: 'signIn' | 'register' | 'editProfile' | null;
  refetchSession: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const router = useRouter();
  const [authModalView, setAuthModalView] = useState<'signIn' | 'register' | 'editProfile' | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const loadSession = useCallback(async () => {
    if (typeof window !== 'undefined') {
      const accessToken = localStorage.getItem('access_token') ?? sessionStorage.getItem('access_token');
      const refreshToken = getRefreshTokenValue();
      if (!accessToken && !refreshToken) {
        setUser(null);
        setIsLoading(false);
        return;
      }
    }

    setIsLoading(true);
    try {
      const response = await fetchCurrentUser();
      setUser(response.user ?? null);
    } catch (error) {
      const refreshedToken = await refreshAccessToken();
      if (refreshedToken) {
        try {
          const response = await fetchCurrentUser();
          setUser(response.user ?? null);
        } catch (err) {
          console.error('Session fetch failed after refresh', err);
          setUser(null);
        }
      } else {
        setUser(null);
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSession();
  }, [loadSession]);

  const login = async (email: string, password: string, rememberMe: boolean) => {
    try {
      await loginRequest(email, password);
      if (!rememberMe) {
        moveTokensToSession();
      }
      toast.success("Signed in successfully!");
      setAuthModalView(null);
      await loadSession();
      router.push("/");
    } catch (error) {
      console.error('Login failed', error);
      toast.error("Invalid email or password. Please make sure you have already registered an account and try again.");
      throw error;
    }
  };

  const logout = async () => {
    try {
      await logoutRequest();
    } catch (error) {
      console.error('Logout error', error);
    } finally {
      setUser(null);
      toast.success("Signed out successfully!");
      router.push("/");
    }
  };

  const register = async (name: string, email: string, password: string): Promise<boolean> => {
    try {
      await registerRequest(name, email, password);
      toast.success("Account created! Please sign in.");
      return true;
    } catch (error) {
      console.error('Registration failed', error);
      toast.error(error instanceof Error ? error.message : "Registration failed");
      return false;
    }
  };

  const updateProfile = async (name: string) => {
    // Update profile logic here if needed
    toast.success("Profile updated successfully!");
    setAuthModalView(null);
    await loadSession();
  };

  const showAuthModal = (view: 'signIn' | 'register' | 'editProfile' = 'signIn') => {
    setAuthModalView(view);
  };

  const hideAuthModal = () => {
    setAuthModalView(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        login,
        logout,
        register,
        updateProfile,
        showAuthModal,
        hideAuthModal,
        authModalView,
        refetchSession: loadSession,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};