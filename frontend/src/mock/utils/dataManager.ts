import { mockChartData } from '../data/charts';
import { mockUploadedFiles } from '../data/files';
import { mockResponsePatterns } from '../data/messages';
import { mockReports } from '../data/reports';
import { getMockStats } from '../data/stats';
import { mockTranslations } from '../data/translations';

export interface MockDataSet {
  name: string;
  description: string;
  reports: any[];
  files: any[];
  stats: any[];
  translations: any;
  charts: any;
  responsePatterns: any;
}

export const mockDataSets: Record<string, MockDataSet> = {
  demo: {
    name: 'Demo Data',
    description: '演示用数据集 - 用于展示和演讲',
    reports: mockReports,
    files: mockUploadedFiles,
    stats: getMockStats((key: string) => key), // Use default key as fallback
    translations: mockTranslations,
    charts: mockChartData,
    responsePatterns: mockResponsePatterns
  }
};

export class MockDataManager {
  private currentDataSet: string;
  private listeners: Array<(dataSet: string) => void> = [];

  constructor() {
    this.currentDataSet = this.getStoredDataSet();
  }

  getCurrentDataSet(): string {
    return this.currentDataSet;
  }

  getCurrentData(): MockDataSet {
    return mockDataSets[this.currentDataSet] || mockDataSets.demo;
  }

  switchDataSet(dataSetName: string): boolean {
    if (mockDataSets[dataSetName]) {
      this.currentDataSet = dataSetName;
      this.storeDataSet(dataSetName);
      this.notifyListeners();
      return true;
    }
    console.warn(`数据集 "${dataSetName}" 不存在`);
    return false;
  }

  addListener(callback: (dataSet: string) => void): void {
    this.listeners.push(callback);
  }

  removeListener(callback: (dataSet: string) => void): void {
    this.listeners = this.listeners.filter(listener => listener !== callback);
  }

  private notifyListeners(): void {
    this.listeners.forEach(listener => {
      try {
        listener(this.currentDataSet);
      } catch (error) {
        console.error('监听器执行错误:', error);
      }
    });
  }

  private getStoredDataSet(): string {
    const envDataSet = process.env.REACT_APP_MOCK_DATA_SET;
    if (envDataSet && mockDataSets[envDataSet]) {
      return envDataSet;
    }

    if (process.env.NODE_ENV === 'development') {
      const stored = localStorage.getItem('mock_data_set');
      if (stored && mockDataSets[stored]) {
        return stored;
      }
    }
    return 'demo';
  }

  private storeDataSet(dataSetName: string): void {
    if (process.env.NODE_ENV === 'development') {
      localStorage.setItem('mock_data_set', dataSetName);
    }
  }

  reset(): void {
    this.switchDataSet('demo');
  }

  validate(): Array<string> {
    const issues: string[] = [];
    const data = this.getCurrentData();

    if (!Array.isArray(data.reports)) {
      issues.push('Reports data is not an array');
    }

    if (!Array.isArray(data.files)) {
      issues.push('Files data is not an array');
    }

    if (!Array.isArray(data.stats)) {
      issues.push('Stats data is not an array');
    }

    if (!data.translations || typeof data.translations !== 'object') {
      issues.push('Translations data is invalid');
    }

    if (!data.charts || typeof data.charts !== 'object') {
      issues.push('Charts data is invalid');
    }

    if (!data.responsePatterns || typeof data.responsePatterns !== 'object') {
      issues.push('Response patterns data is invalid');
    }

    return issues;
  }

  exportCurrentDataSet(): any {
    return {
      dataSet: this.currentDataSet,
      data: this.getCurrentData(),
      exportedAt: new Date().toISOString()
    };
  }

  getDataSetStats(): any {
    const data = this.getCurrentData();
    return {
      dataSet: this.currentDataSet,
      reports: data.reports.length,
      files: data.files.length,
      stats: data.stats.length,
      translations: Object.keys(data.translations).length,
      charts: Object.keys(data.charts).length,
      responsePatterns: Object.keys(data.responsePatterns).length
    };
  }
}

export const mockDataManager = new MockDataManager();

if (process.env.NODE_ENV === 'development') {
  (window as any).mockDataManager = mockDataManager;
  const handleKeyPress = (event: KeyboardEvent) => {
    if ((event.ctrlKey || event.metaKey) && event.key >= '1' && event.key <= '4') {
      event.preventDefault();
      switch (event.key) {
        case '1':
          mockDataManager.switchDataSet('demo');
          break;
      }
    }
  };
  
  document.addEventListener('keydown', handleKeyPress);
  
  console.log(`
🛠️  Mock Data Manager Developer tool is enabled

shortcut:
- Ctrl/Cmd + 1: Demo Dataset

Global object:
- window.mockDataManager: Data Manager Instance

Current dataset: ${mockDataManager.getCurrentDataSet()}
  `);
}

export const mockDataUtils = {
  switchToDemo: () => mockDataManager.switchDataSet('demo'),
  
  getCurrentReports: () => mockDataManager.getCurrentData().reports,
  getCurrentFiles: () => mockDataManager.getCurrentData().files,
  getCurrentStats: () => mockDataManager.getCurrentData().stats,
  getCurrentTranslations: () => mockDataManager.getCurrentData().translations,
  getCurrentCharts: () => mockDataManager.getCurrentData().charts,
  getCurrentResponsePatterns: () => mockDataManager.getCurrentData().responsePatterns,
  
  validateCurrentData: () => mockDataManager.validate(),
  
  exportData: () => mockDataManager.exportCurrentDataSet(),
  
  getStats: () => mockDataManager.getDataSetStats()
};
