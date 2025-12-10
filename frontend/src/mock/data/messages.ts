import { EconomicData, Message } from '../types';
import { mockChartData } from './charts';

export const mockInitialMessage: Message = {
  id: '1',
  type: 'assistant',
  content: 'Hello! I\'m your Ottawa Economic Development AI Assistant. I can help you analyze economic data, answer questions about city reports, and generate insights. What would you like to know?',
  timestamp: new Date()
};

export const mockEconomicData: EconomicData[] = [
  { month: 'Jan', businesses: 120, growth: 5.2 },
  { month: 'Feb', businesses: 125, growth: 6.1 },
  { month: 'Mar', businesses: 135, growth: 7.8 },
  { month: 'Apr', businesses: 142, growth: 8.3 },
  { month: 'May', businesses: 156, growth: 9.1 },
  { month: 'Jun', businesses: 168, growth: 10.2 }
];

export interface AIResponse {
  content: string;
  hasChart: boolean;
  chart?: any;
}

// Mock AI response patterns
export const mockResponsePatterns = {
  business: {
    content: `## Business Growth Analysis

Based on the latest Economic Development reports, here's what I found:

### Key Findings:
- **New business registrations** have increased by **15.2%** over the past 6 months
- **Small business loan approvals** are up by **8.3%** compared to last year
- **The technology sector** shows the strongest growth at **22.1%**

### Recommendations:
1. Continue supporting tech startup initiatives
2. Expand small business loan programs
3. Focus on downtown revitalization projects

The chart below shows the monthly progression of new business registrations and growth rates.`,
    hasChart: true,
    chart: mockChartData.businessGrowth
  },
  employment: {
    content: `## Employment Trends in Ottawa

### Current Employment Statistics:
- **Unemployment rate**: 4.2% (down from 5.1% last year)
- **Job openings**: 12,500 active postings
- **Average salary growth**: 3.8% annually

### Sector Breakdown:
- **Technology**: 35% of new jobs
- **Healthcare**: 18% of new jobs  
- **Professional Services**: 15% of new jobs
- **Manufacturing**: 12% of new jobs

### Key Initiatives:
- Skills training programs launched
- Partnership with local colleges expanded
- Remote work policies updated`,
    hasChart: false
  },
  default: {
    content: `I can help you with questions about:

📊 **Economic Data Analysis**
- Business growth trends
- Employment statistics
- Sector performance

📋 **Report Generation**
- Quarterly summaries
- Custom analysis reports
- Data visualizations

📁 **Document Processing**
- PDF report analysis
- Data extraction
- Content summarization

Try asking me about business growth, employment trends, or upload a document for analysis!`,
    hasChart: false
  }
};

export const simulateAIResponse = (userMessage: string): AIResponse => {
  const lowerMessage = userMessage.toLowerCase();
  
  if (lowerMessage.includes('business') || lowerMessage.includes('growth') || lowerMessage.includes('trend')) {
    return mockResponsePatterns.business;
  }
  
  if (lowerMessage.includes('unemploy') || lowerMessage.includes('employ') || lowerMessage.includes('job')) {
    return {
      content: `## Employment Statistics Update

Here's the current employment landscape in Ottawa:

### Current Status:
- **Unemployment rate**: 4.2% (down from 5.1% last quarter)
- **New job postings**: Up 12% this month
- **Labor force participation**: 68.5%
- **Youth employment** (15-24): Improved to 89.3%

### Sector Breakdown:
- **Technology**: +2,340 jobs (highest growth)
- **Healthcare**: +1,120 jobs  
- **Professional Services**: +890 jobs

### Skills in Demand:
1. Digital skills development
2. Healthcare support roles
3. Project management
4. Data analysis
5. Customer service

The employment outlook remains positive with continued growth expected.`,
      hasChart: true,
      chart: mockChartData.employmentDistribution
    };
  }
  
  if (lowerMessage.includes('revenue') || lowerMessage.includes('income') || lowerMessage.includes('budget')) {
    return {
      content: `## Revenue and Budget Analysis

### Municipal Revenue Breakdown:
- **Property taxes**: 45% of total revenue
- **Federal transfers**: 25% of budget
- **Provincial funding**: 15% of budget
- **User fees**: 10% of revenue
- **Economic Development**: 15% of budget

### Investment Priorities:
1. Infrastructure development
2. Technology innovation hubs
3. Small business support programs
4. Skills training initiatives

### Economic Impact:
- **ROI on ED programs**: 3.2x return
- **Jobs created**: 2,500+ annually
- **Business retention rate**: 87%

Budget allocation shows strong commitment to economic growth.`,
      hasChart: true,
      chart: mockChartData.revenueTrends
    };
  }
  
  if (lowerMessage.includes('hello') || lowerMessage.includes('hi') || lowerMessage.includes('help')) {
    return mockResponsePatterns.default;
  }
  
  // Default fallback response
  return {
    content: `I understand you're looking for information related to Ottawa's economic development. While I don't have specific data on this topic in my current knowledge base, I can help you with:

- **Business growth analysis**
- **Employment statistics and trends**
- **Revenue and budget information**
- **Development project updates**
- **Economic indicators and forecasts**

Could you please rephrase your question or ask about one of these specific areas?`,
    hasChart: false
  };
}; 