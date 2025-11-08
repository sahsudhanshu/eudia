"use client";

import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';

export type ToastVariant = 'info' | 'success' | 'error' | 'warning';

export interface ToastOptions {
  message: string;
  variant?: ToastVariant;
  durationMs?: number;
}

interface ToastItem extends Required<ToastOptions> {
  id: string;
}

interface ToastContextValue {
  showToast: (opts: ToastOptions) => void;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}

function ToastStack({ toasts, dismiss }: { toasts: ToastItem[]; dismiss: (id: string) => void }) {
  return (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={
            'min-w-[220px] max-w-[360px] px-3 py-2 rounded-md shadow-lg text-sm flex items-start gap-2 border ' +
            (t.variant === 'success'
              ? 'bg-green-50 text-green-800 border-green-200'
              : t.variant === 'error'
              ? 'bg-red-50 text-red-800 border-red-200'
              : t.variant === 'warning'
              ? 'bg-amber-50 text-amber-800 border-amber-200'
              : 'bg-neutral-50 text-neutral-800 border-neutral-200')
          }
        >
          <span className="mt-0.5">
            {t.variant === 'success' && '✅'}
            {t.variant === 'error' && '⛔'}
            {t.variant === 'warning' && '⚠️'}
            {t.variant === 'info' && 'ℹ️'}
          </span>
          <div className="flex-1">{t.message}</div>
          <button
            onClick={() => dismiss(t.id)}
            className="text-xs text-neutral-500 hover:text-neutral-700"
            aria-label="Dismiss"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const showToast = useCallback((opts: ToastOptions) => {
    const id = Math.random().toString(36).slice(2);
    const toast: ToastItem = {
      id,
      message: opts.message,
      variant: opts.variant ?? 'info',
      durationMs: opts.durationMs ?? 2500,
    };
    setToasts((prev) => [...prev, toast]);
    window.setTimeout(() => dismiss(id), toast.durationMs);
  }, [dismiss]);

  const value = useMemo(() => ({ showToast }), [showToast]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastStack toasts={toasts} dismiss={dismiss} />
    </ToastContext.Provider>
  );
}
