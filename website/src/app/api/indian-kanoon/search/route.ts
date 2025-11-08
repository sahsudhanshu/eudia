import { NextRequest, NextResponse } from 'next/server';

const INDIAN_KANOON_API_TOKEN = 'd393d383ef7d0004e37c866904d87f8273026a16';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { formInput, pagenum = 0 } = body;

    if (!formInput) {
      return NextResponse.json(
        { error: 'Search query (formInput) is required' },
        { status: 400 }
      );
    }

    // Make POST request to Indian Kanoon API
    const searchUrl = `https://api.indiankanoon.org/search/?formInput=${encodeURIComponent(formInput)}&pagenum=${pagenum}`;
    
    const response = await fetch(searchUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Token ${INDIAN_KANOON_API_TOKEN}`,
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      console.error('Indian Kanoon API error:', response.status, response.statusText);
      return NextResponse.json(
        { error: `Indian Kanoon API returned ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Search API error:', error);
    return NextResponse.json(
      { error: 'Failed to perform search' },
      { status: 500 }
    );
  }
}