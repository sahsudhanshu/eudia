"use client"

import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Bot, User } from 'lucide-react';
import { apiFetch } from '@/lib/api';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

interface ChatInterfaceProps {
    documentId: string;
}

export default function ChatInterface({ documentId }: ChatInterfaceProps) {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '1',
            role: 'assistant',
            content: 'Hello! I can help you analyze this document. Ask me anything about its content, citations, or legal implications.',
            timestamp: new Date(),
        }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input.trim(),
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
            // Call the OCR query endpoint
            const response = await apiFetch<{
                success: boolean;
                query: string;
                answer: string;
                title: string;
            }>(`/api/ocr/query/${documentId}`, {
                method: 'POST',
                body: JSON.stringify({ query: userMessage.content }),
            });

            const aiMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: response.answer || 'I apologize, but I could not generate a response. Please try rephrasing your question.',
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, aiMessage]);
        } catch (error) {
            console.error('Error querying document:', error);
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: 'I encountered an error while processing your question. Please try again.',
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    const formatTime = (date: Date) => {
        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    };

    return (
        <div className="h-full glass-card rounded-lg flex flex-col overflow-hidden">
            {/* Chat Header */}
            <div className="p-4 border-b border-neutral-200 bg-white/50">
                <h2 className="text-base font-semibold text-neutral-900 flex items-center gap-2">
                    <Bot size={18} className="text-green-600" />
                    AI Legal Assistant
                </h2>
                <p className="text-neutral-600 text-xs mt-1">
                    Ask questions about this document
                </p>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin">
                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        {message.role === 'assistant' && (
                            <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center shrink-0">
                                <Bot size={16} className="text-green-600" />
                            </div>
                        )}

                        <div className={`flex flex-col gap-1 max-w-[80%] ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
                            <div
                                className={`rounded-lg px-4 py-2 ${message.role === 'user'
                                        ? 'bg-green-500 text-white'
                                        : 'bg-white border border-neutral-200 text-neutral-900'
                                    }`}
                            >
                                <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                            </div>
                            <span className="text-xs text-neutral-500">{formatTime(message.timestamp)}</span>
                        </div>

                        {message.role === 'user' && (
                            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                                <User size={16} className="text-blue-600" />
                            </div>
                        )}
                    </div>
                ))}

                {loading && (
                    <div className="flex gap-3 justify-start">
                        <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center shrink-0">
                            <Bot size={16} className="text-green-600" />
                        </div>
                        <div className="bg-white border border-neutral-200 rounded-lg px-4 py-2">
                            <Loader2 className="animate-spin text-green-600" size={16} />
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Form */}
            <div className="p-4 border-t border-neutral-200 bg-white/50">
                <form onSubmit={handleSubmit} className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask about this document..."
                        disabled={loading}
                        className="flex-1 px-4 py-2 border border-neutral-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent text-sm disabled:bg-neutral-100 disabled:cursor-not-allowed"
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || loading}
                        className="bg-green-500 hover:bg-green-600 disabled:bg-neutral-300 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg transition-all flex items-center gap-2 text-sm font-medium"
                    >
                        <Send size={16} />
                        Send
                    </button>
                </form>
            </div>
        </div>
    );
}