import { NextResponse } from 'next/server';

export function PUT() {
  return NextResponse.json(
    {
      error: 'This route has moved to the Flask backend API.',
    },
    { status: 410 }
  );
}

export const DELETE = PUT;