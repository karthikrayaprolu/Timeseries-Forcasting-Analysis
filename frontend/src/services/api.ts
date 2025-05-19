// API service for Time Series Forecasting
import axios from 'axios';
import {
  ProcessConfig,
  ModelConfig,
  ResultsData,
  TrainingMetrics,
  ModelInfo,
  DatabaseConfig
} from '@/shared/types';

// Base API configuration
const BASE_URL = 'http://localhost:5000/api';


const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Error handler
const handleApiError = (error: any, fallbackMessage: string) => {
  if (error.response) {
    throw new Error(error.response.data.error || fallbackMessage);
  }
  throw new Error('Network error - please check if the server is running');
};

export const api = {
  // File upload
  uploadCsvFile: async (formData: FormData) => {
    try {
      const response = await apiClient.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to upload CSV file');
    }
  },

  // Get available tables for a database type
  getTables: async (databaseType: string) => {
    try {
      if (databaseType === 'local') {
        return ['uploaded_csv'];
      }
      const response = await apiClient.get(`/tables?type=${databaseType}`);
      return response.data.tables;
    } catch (error) {
      handleApiError(error, 'Failed to get tables');
    }
  },

  // Get columns for a table
  getColumns: async () => {
    try {
      const response = await apiClient.get('/columns');
      return response.data.columns;
    } catch (error) {
      handleApiError(error, 'Failed to get columns');
    }
  },

  // Process data with selected configuration
  processData: async (config: ProcessConfig) => {
    try {
      const response = await apiClient.post('/process', config);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Data processing failed');
    }
  },

  // Train model with selected configuration
  trainModel: async (config: ModelConfig) => {
    try {
      const response = await apiClient.post('/train', config);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Model training failed');
    }
  },

  // Get available models for transfer learning
  getAvailableModels: async () => {
    try {
      const response = await apiClient.get('/models');
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to get available models');
    }
  },

  // Get model metrics and forecasts
  getResults: async (modelId: string): Promise<ResultsData> => {
    try {
      const response = await apiClient.get(`/results/${modelId}`);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to get results');
      throw error;
    }
  },
};
