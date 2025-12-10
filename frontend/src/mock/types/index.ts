// Shared types for mock data
export interface Report {
  id: string;
  title: string;
  generatedAt: Date;
  type: 'summary' | 'analysis' | 'trend';
  status: 'completed' | 'processing' | 'error';
}

export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  chart?: any;
  hasChart?: boolean;
  confidence?: string;
}

export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  status: 'uploading' | 'completed' | 'error';
  progress: number;
  uploadedAt: Date;
}

export interface ChartData {
  [key: string]: any;
}

export interface BusinessData {
  sector: string;
  growth: number;
  businesses: number;
}

export interface EmploymentData {
  name: string;
  value: number;
  color: string;
}

export interface EconomicData {
  month: string;
  businesses: number;
  growth: number;
}

export interface StatData {
  number: string;
  label: string;
}

export interface Translations {
  [language: string]: {
    [key: string]: string;
  };
} 