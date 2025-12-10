import { Copy, Download, Send } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { realApi } from '../services/api';
import './ChatPage.css';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  chart?: any;
  hasChart?: boolean;
  confidence?: string;
}

const ChatPage: React.FC = () => {
  // Start with empty messages array - no mock initial message
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId] = useState<string>(`conv_${Date.now()}`);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load initial welcome message from real API
  useEffect(() => {
    const loadWelcomeMessage = async () => {
      try {
        const welcomeMessage: Message = {
          id: 'welcome',
          type: 'assistant',
          content: 'Hello! I\'m your AI Research Assistant for Ottawa\'s economic development data. I can help you analyze business trends, employment statistics, and economic indicators for the Ottawa region.\n\nHow can I assist you today?',
          timestamp: new Date()
        };
        setMessages([welcomeMessage]);
      } catch (error) {
        console.error('Failed to load welcome message:', error);
      }
    };

    loadWelcomeMessage();
  }, []);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputValue;
    setInputValue('');
    setIsLoading(true);

    try {
      // Use REAL API ONLY - no fallback
      const apiResponse = await realApi.sendMessage(currentInput, conversationId);
      
      const assistantMessage: Message = {
        id: apiResponse.id || (Date.now() + 1).toString(),
        type: 'assistant',
        content: apiResponse.content,
        timestamp: new Date(apiResponse.timestamp || Date.now()),
        hasChart: false, // API response might include chart data
        chart: undefined,
        confidence: apiResponse.confidence
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('API request failed:', error);
      
      // Show error message to user instead of falling back to mock
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I\'m having trouble connecting to the server right now. Please check your connection and try again.',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = e.target;
    setInputValue(textarea.value);
    
    // Reset height to auto to get the correct scrollHeight
    textarea.style.height = 'auto';
    // Set the height to the scrollHeight, but not more than max-height
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const downloadChart = () => {
    // In a real app, this would generate and download the chart
    alert('Chart download functionality would be implemented here');
  };

  return (
    <div className="chat-page">
      <div className="chat-container">
        <div className="chat-messages" role="log" aria-live="polite" aria-label="Chat messages">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.type}`}>
              <div className="message-content">
                <div className="message-text">
                  {message.type === 'assistant' ? (
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                  ) : (
                    <p>{message.content}</p>
                  )}
                </div>
                {message.type === 'assistant' && message.confidence && (
                  <div className="confidence-indicator">
                    Confidence: <strong>{message.confidence}</strong>
                  </div>
                )}
                {message.hasChart && message.chart && (
                  <div className="message-chart">
                    <div className="chart-header">
                      <h4>Business Growth Trends</h4>
                      <button 
                        onClick={downloadChart}
                        className="chart-download-btn"
                        aria-label="Download chart"
                      >
                        <Download size={16} />
                      </button>
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={message.chart}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip />
                        <Line 
                          type="monotone" 
                          dataKey="businesses" 
                          stroke="#667eea" 
                          strokeWidth={2}
                          name="New Businesses"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="growth" 
                          stroke="#10b981" 
                          strokeWidth={2}
                          name="Growth Rate (%)"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {message.type === 'assistant' && (
                  <div className="message-actions">
                    <button 
                      onClick={() => copyToClipboard(message.content)}
                      className="action-btn"
                      aria-label="Copy message"
                    >
                      <Copy size={14} />
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-container">
          <div className="chat-input">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about economic development data..."
              rows={1}
              disabled={isLoading}
              aria-label="Message input"
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="send-button"
              aria-label="Send message"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatPage; 