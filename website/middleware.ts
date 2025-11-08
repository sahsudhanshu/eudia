import { NextRequest, NextResponse } from "next/server";

export async function middleware(request: NextRequest) {
  // Middleware for JWT-based auth
  // Token validation is handled by the frontend via AuthContext
  // Protected routes should check auth state in components
  return NextResponse.next();
}

export const config = {
  matcher: ["/api/:path*"],
};