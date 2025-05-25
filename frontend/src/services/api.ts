// API service for Time Series Forecasting
import axios from 'axios';
import { toast } from 'sonner';

import {
  ProcessConfig,
  ModelConfig,
  ResultsData,
  TrainingMetrics,
  ModelInfo,
  DatabaseConfig
} from '@/shared/types';

const BASE_URL = 'http://localhost:5000/api';

// Utility to fetch user ID from localStorage
const getUserId = () => localStorage.getItem("user_id") || "";

// Create axios client with dynamic user headers
const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Interceptor to inject user ID
apiClient.interceptors.request.use((config) => {
  const userId = getUserId();
  if (userId && config.headers) {
    config.headers['X-User-Id'] = userId;
  }
  return config;
});

// Centralized error handler
const handleApiError = (error: any, fallbackMessage: string) => {
  if (error.response) {
    throw new Error(error.response.data.error || fallbackMessage);
  }
  throw new Error('Network error - please check if the server is running');
};

export const api = {
  // Upload CSV file
  uploadCsvFile: async (formData: FormData) => {
    try {
      const response = await apiClient.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }, // overrides JSON
      });
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to upload CSV file');
    }
  },

  // Get available tables (for future DB support)
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

  // Get columns for uploaded or selected dataset
  getColumns: async () => {
    try {
      const response = await apiClient.get('/columns');
      return response.data.columns;
    } catch (error) {
      handleApiError(error, 'Failed to get columns');
    }
  },

  // Process uploaded data
  processData: async (config: ProcessConfig) => {
    try {
      const response = await apiClient.post('/process', config);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Data processing failed');
    }
  },

  // Train model with user context

trainModel: async (config: ModelConfig) => {
  try {
    console.log("🚀 Sending training config:", config);
    const response = await apiClient.post('/train', config);

    
    // Handle transfer learning specific response
    if (config.transferLearning && config.sourceModelId) {
      toast.success("Transfer learning completed successfully!", {
        description: `Model was initialized from ${config.sourceModelId} and fine-tuned on your data`
      });
    }
    
    return response.data;
  } catch (error) {
    if (config.transferLearning) {
      toast.error("Transfer learning failed", {
        description: error instanceof Error ? error.message : String(error)
      });
    } else {
      toast.error("Model training failed", {
        description: error instanceof Error ? error.message : String(error)
      });
    }
    throw error;
  }
},
  // Get pretrained models for transfer learning
  getAvailableModels: async () => {
    try {
      const response = await apiClient.get('/models');
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to get available models');
    }
  },

  // Get model training results
  getResults: async (modelId: string): Promise<ResultsData> => {
    try {
      const response = await apiClient.get(`/results/${modelId}`);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to get results');
      throw error;
    }
  }
};
