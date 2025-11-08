import { NextResponse } from 'next/server';

export function POST() {
  return NextResponse.json(
    {
      error: 'This route has moved to the Flask backend. Use NEXT_PUBLIC_API_BASE_URL instead.',
    },
    { status: 410 }
  );
}