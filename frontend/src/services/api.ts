// API service for Time Series Forecasting
import axios from 'axios';
import { ModelConfig } from '@/contexts/WorkflowContext';

const BASE_URL = 'http://localhost:5000/api';

const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

export const api = {
  // File upload
  uploadCsvFile: async (formData: FormData) => {
    const response = await apiClient.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  // Get available tables for a database type
  getTables: async (databaseType: string) => {
    if (databaseType === 'local') {
      return ['uploaded_csv'];
    }
    const response = await apiClient.get(`/tables?type=${databaseType}`);
    return response.data.tables;
  },

  // Get columns for a selected table
  getColumns: async (table: string) => {
    const response = await apiClient.get(`/columns?table=${table}`);
    return response.data.columns;
  },

  // Detect date format from a column
  detectDateFormat: async (config: { timeColumn: string }) => {
    const response = await apiClient.post('/detect-date-format', config);
    return response.data;
  },
  // Process data with configuration
  processData: async (config: {
    timeColumn: string;
    targetVariable: string;
    frequency: string;
    features: string[];
    dateFormat?: string;
    aggregationMethod?: 'mean' | 'sum' | 'max' | 'min';
  }) => {
    try {
      const response = await apiClient.post('/process', config, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      return response.data;
    } catch (error: any) {
      if (error.response) {
        throw new Error(error.response.data.error || 'Failed to process data');
      }
      throw new Error('Network error - please check if the server is running');
    }
  },

  // Add a generic post method for API calls
  post: async (endpoint: string, data: any) => {
    try {
      const response = await apiClient.post(endpoint, data);
      return response.data;
    } catch (error: any) {
      if (error.response) {
        throw new Error(error.response.data.error || `Failed to ${endpoint}`);
      }
      throw new Error('Network error - please check if the server is running');
    }
  },

  // Train model with configuration
  trainModel: async (config: ModelConfig & { 
    timeSteps?: number;
    units?: number;
    epochs?: number;
    batchSize?: number;
    ensembleModels?: string[];
    ensembleMethod?: 'voting' | 'stacking';
    ensembleWeights?: number[] | null;
  }) => {
    try {
      const modelConfig = {
        modelType: config.modelType,
        hyperparameterTuning: config.hyperparameterTuning,
        timeSteps: config.timeSteps || 12,
        units: config.units || 50,
        epochs: config.epochs || 100,
        batchSize: config.batchSize || 32,
        ensembleLearning: config.ensembleLearning,
        transferLearning: config.transferLearning,
        // Ensemble specific configuration
        ...config.ensembleLearning ? {
          ensembleModels: config.ensembleModels,
          ensembleMethod: config.ensembleMethod || 'voting',
          ensembleWeights: config.ensembleWeights || null
        } : {},
        // Model specific parameters for hyperparameter tuning
        n_estimators: 100,
        max_depth: 10,
        learning_rate: 0.1,
        subsample: 0.8,
        colsample_bytree: 0.8
      };

      return await api.post('/train', modelConfig);
    } catch (error: any) {
      if (error.response) {
        throw new Error(error.response.data.error || 'Failed to train model');
      }
      throw new Error('Network error - please check if the server is running');
    }
  },

  // Get available trained models for transfer learning
  getModels: async () => {
    try {
      const response = await apiClient.get('/models');
      return response.data;
    } catch (error: any) {
      if (error.response) {
        throw new Error(error.response.data.error || 'Failed to fetch models');
      }
      throw new Error('Network error - please check if the server is running');
    }
  },

  // Export results
  exportResults: async (format: 'csv' | 'excel' | 'json', data: any, predictionDate?: Date) => {
    const response = await apiClient.post('/export', {
      format,
      data,
      predictionDate: predictionDate?.toISOString()
    }, {
      responseType: 'blob'
    });
    
    // Create and trigger download
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `forecast_results.${format}`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  }
};
