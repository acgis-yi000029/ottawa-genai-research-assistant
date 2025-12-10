// Mock Data Configuration

export const MOCK_CONFIG = {
  currentDataSet: 'demo', // 'demo'
  
  apiDelay: {
    short: 300,   
    medium: 800,  
    long: 1500    
  },
  
  dataRefreshInterval: 30000, 
  
  development: {
    enableConsoleLog: true,
    enablePerformanceTracking: true,
    showDataSourceInUI: true
  },
  
  validation: {
    enableDataValidation: true,
    strictModeEnabled: false
  }
};

export const getEnvironmentConfig = () => {
  const isDevelopment = process.env.NODE_ENV === 'development';
  
  return {
    ...MOCK_CONFIG,
    development: {
      ...MOCK_CONFIG.development,
      enableConsoleLog: isDevelopment,
      enablePerformanceTracking: isDevelopment,
      showDataSourceInUI: isDevelopment
    }
  };
};

export const DATA_SET_DESCRIPTIONS = {
  demo: {
    name: 'Demo Data',
    description: 'Demonstration dataset - for presentations and speeches',
    recommended: 'presentations',
    features: ['Full feature demonstration', 'Aesthetically pleasing visual effects', 'Real-world data scenarios']
  }
};

export const getCurrentConfig = () => getEnvironmentConfig();
export const isDevMode = () => process.env.NODE_ENV === 'development';
export const isMockEnabled = () => true; // Always true for this prototype 