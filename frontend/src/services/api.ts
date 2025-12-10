// Real API Service Layer - Production Implementation
import { Message, Report, StatData, Translations, UploadedFile } from '../mock/types';
import { authService } from './authService';

// Base API configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api/v1';

// Backend response types
interface BackendDocumentInfo {
  id: string;
  filename: string;
  size: number;
  upload_date: string;
  processed: boolean;
  page_count?: number;
  language?: string;
}

interface BackendDocumentList {
  documents: BackendDocumentInfo[];
  total: number;
}

interface BackendUploadResponse {
  id: string;
  filename: string;
  size: number;
  message: string;
  processing_status: string;
}

// HTTP client utility
class ApiClient {
  private baseURL: string;
  private headers: Record<string, string>;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
    this.headers = {
      'Content-Type': 'application/json',
    };
  }

  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      headers: this.headers,
      ...options,
    };
    
    // Retrieve login token from authService or localStorage
    const token =
      authService.getToken() || localStorage.getItem('auth_token');

    // Add Authorization to the request header.
    if (token) {
      const existingHeaders =
        (config.headers as Record<string, string> | undefined) || {};
      existingHeaders['Authorization'] = `Bearer ${token}`;
      config.headers = existingHeaders;
    }

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async upload<T>(endpoint: string, file: File): Promise<T> {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.request<T>(endpoint, {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set Content-Type for FormData
    });
  }

  // NEW: generic DELETE method with auth header
  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'DELETE',
    });
  }
}

// Initialize API client
const apiClient = new ApiClient(API_BASE_URL);

// Real API implementation
export const realApi = {
  // Reports API
  getReports: async (): Promise<Report[]> => {
    return apiClient.get<Report[]>('/reports');
  },

  getReport: async (id: string): Promise<Report> => {
    return apiClient.get<Report>(`/reports/${id}`);
  },

  generateReport: async (query: string): Promise<Report> => {
    return apiClient.post<Report>('/reports/generate', { query });
  },

  // Chat API
  sendMessage: async (message: string, conversationId?: string): Promise<Message> => {
    const response = await apiClient.post<any>('/chat/message', { 
      message,
      language: 'en',
      context: conversationId 
    });
    
    // Transform backend response to frontend Message format
    return {
      id: response.id,
      type: 'assistant',
      content: response.response,
      timestamp: new Date(response.timestamp),
      confidence: response.confidence
    };
  },

  getConversationHistory: async (conversationId: string): Promise<Message[]> => {
    return apiClient.get<Message[]>(`/chat/conversations/${conversationId}`);
  },

  // Document Upload API
  uploadFile: async (file: File): Promise<UploadedFile> => {
    const response = await apiClient.upload<BackendUploadResponse>('/documents/upload', file);
    
    // Transform backend format to frontend format
    return {
      id: response.id,
      name: response.filename,
      size: response.size,
      type: file.type,
      status: response.processing_status === 'completed' ? 'completed' as const : 'uploading' as const,
      progress: response.processing_status === 'completed' ? 100 : 50,
      uploadedAt: new Date()
    };
  },

  getUploadedFiles: async (): Promise<UploadedFile[]> => {
    const response = await apiClient.get<BackendDocumentList>('/documents/list');
    
    // Transform backend format to frontend format
    return response.documents.map(doc => ({
      id: doc.id,
      name: doc.filename,
      size: doc.size,
      type: 'application/pdf',
      status: doc.processed ? 'completed' as const : 'uploading' as const,
      progress: doc.processed ? 100 : 0,
      uploadedAt: new Date(doc.upload_date)
    }));
  },

  deleteFile: async (fileId: string): Promise<void> => {
    await apiClient.delete<void>(`/documents/${fileId}`);
  },

  // Statistics API
  getStats: async (): Promise<StatData[]> => {
    return apiClient.get<StatData[]>('/stats');
  },

  // Translations API
  getTranslations: async (language: string): Promise<Translations> => {
    return apiClient.get<Translations>(`/translations/${language}`);
  },

  // Health Check
  healthCheck: async (): Promise<{ status: string; version: string }> => {
    return apiClient.get('/health');
  },
};

export default realApi; 