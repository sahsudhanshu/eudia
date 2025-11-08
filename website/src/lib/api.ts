const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_API_BASE;

type FetchOptions = RequestInit & {
  skipAuth?: boolean;
};

function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return (
    localStorage.getItem("access_token") ?? sessionStorage.getItem("access_token")
  );
}

function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return (
    localStorage.getItem("refresh_token") ?? sessionStorage.getItem("refresh_token")
  );
}

export async function apiFetch<T>(path: string, options: FetchOptions = {}): Promise<T> {
  const { skipAuth, headers, ...rest } = options;
  const resolvedHeaders = new Headers(headers ?? {});

  if (!resolvedHeaders.has("Content-Type") && !(rest.body instanceof FormData)) {
    resolvedHeaders.set("Content-Type", "application/json");
  }

  if (!skipAuth) {
    const token = getAccessToken();
    if (token && !resolvedHeaders.has("Authorization")) {
      resolvedHeaders.set("Authorization", `Bearer ${token}`);
    }
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...rest,
    headers: resolvedHeaders,
  });

  if (response.status === 204) {
    return null as T;
  }

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const message = typeof payload === "string" ? payload : payload?.error || "Request failed";
    throw new Error(message);
  }

  return payload as T;
}

export function persistTokens(accessToken: string, refreshToken?: string | null, opts?: { remember?: boolean }) {
  if (typeof window === "undefined") return;
  const storage = opts?.remember === false ? sessionStorage : localStorage;
  storage.setItem("access_token", accessToken);
  if (refreshToken) {
    storage.setItem("refresh_token", refreshToken);
  }
}

export function clearTokens() {
  if (typeof window === "undefined") return;
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
  sessionStorage.removeItem("access_token");
  sessionStorage.removeItem("refresh_token");
}

export function moveTokensToSession() {
  if (typeof window === "undefined") return;
  const accessToken = localStorage.getItem("access_token");
  const refreshToken = localStorage.getItem("refresh_token");
  if (accessToken) {
    sessionStorage.setItem("access_token", accessToken);
    localStorage.removeItem("access_token");
  }
  if (refreshToken) {
    sessionStorage.setItem("refresh_token", refreshToken);
    localStorage.removeItem("refresh_token");
  }
}

export function getRefreshTokenValue(): string | null {
  return getRefreshToken();
}

export function areTokensInSession(): boolean {
  if (typeof window === "undefined") return false;
  return (
    sessionStorage.getItem("refresh_token") !== null ||
    sessionStorage.getItem("access_token") !== null
  );
}
