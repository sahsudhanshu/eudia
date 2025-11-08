import { NextResponse } from 'next/server';

export function GET() {
  return NextResponse.json(
    {
      error: 'This route has moved to the Flask backend API.',
    },
    { status: 410 }
  );
}

export const PUT = GET;
export const DELETE = GET;