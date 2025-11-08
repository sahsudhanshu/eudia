import { apiFetch, persistTokens, clearTokens, getRefreshTokenValue, areTokensInSession } from "./api";

interface AuthResponse {
	user: {
		id: string;
		name: string;
		email: string;
		createdAt?: string | null;
		updatedAt?: string | null;
	};
	accessToken?: string;
	refreshToken?: string;
}

export async function login(email: string, password: string): Promise<AuthResponse> {
	const response = await apiFetch<AuthResponse>("/api/auth/login", {
		method: "POST",
		body: JSON.stringify({ email, password }),
		skipAuth: true,
	});

	if (!response.accessToken) {
		throw new Error("Login response missing access token");
	}

		persistTokens(response.accessToken, response.refreshToken ?? null);
	return response;
}

export async function register(name: string, email: string, password: string): Promise<AuthResponse> {
	return apiFetch<AuthResponse>("/api/auth/register", {
		method: "POST",
		body: JSON.stringify({ name, email, password }),
		skipAuth: true,
	});
}

export async function fetchCurrentUser(): Promise<AuthResponse> {
	return apiFetch<AuthResponse>("/api/auth/me");
}

export async function refreshAccessToken(): Promise<string | null> {
		const refreshToken = getRefreshTokenValue();
	if (!refreshToken) {
		return null;
	}

	try {
		const response = await apiFetch<{ accessToken: string }>("/api/auth/refresh", {
			method: "POST",
			headers: {
				Authorization: `Bearer ${refreshToken}`,
			},
			skipAuth: true,
		});

		persistTokens(response.accessToken, refreshToken, { remember: !areTokensInSession() });
		return response.accessToken;
	} catch (error) {
		clearTokens();
		return null;
	}
}

export async function logout(): Promise<void> {
	try {
		await apiFetch("/api/auth/logout", { method: "POST" });
	} finally {
		clearTokens();
	}
}