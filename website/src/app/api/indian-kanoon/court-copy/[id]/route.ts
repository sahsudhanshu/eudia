import { NextRequest, NextResponse } from 'next/server';

const INDIAN_KANOON_API_TOKEN = 'd393d383ef7d0004e37c866904d87f8273026a16';

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const docId = params.id;

    if (!docId) {
      return NextResponse.json(
        { error: 'Document ID is required' },
        { status: 400 }
      );
    }

    // Fetch court copy (original document) from Indian Kanoon API
    const indianKanoonUrl = `https://api.indiankanoon.org/origdoc/${docId}/`;

    const response = await fetch(indianKanoonUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Token ${INDIAN_KANOON_API_TOKEN}`,
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Indian Kanoon API returned ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Indian Kanoon court copy API error:', error);
    return NextResponse.json(
      {
        error: 'Failed to fetch court copy from Indian Kanoon',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
