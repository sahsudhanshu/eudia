import { NextResponse } from 'next/server';

export function GET() {
	return NextResponse.json(
		{
			error: 'Authentication routes now live on the Flask backend.',
		},
		{ status: 410 }
	);
}

export const POST = GET;