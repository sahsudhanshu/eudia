import { NextResponse } from 'next/server';

function deprecated() {
  return NextResponse.json(
    {
      error: 'This route has moved to the Flask backend. Call NEXT_PUBLIC_API_BASE_URL instead.',
    },
    { status: 410 }
  );
}

export const GET = deprecated;
export const POST = deprecated;