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

// Import Firebase Auth
import { getAuth } from "firebase/auth";

// Utility to fetch user ID from Firebase
const getAuthHeaders = async () => {
  const auth = getAuth();
  const user = auth.currentUser;
  if (!user) return {};
  const token = await user.getIdToken();
  return {
    "Authorization": `Bearer ${token}`,
    "X-User-Id": user.uid, // ✅ critical for user-specific models
  };
};

// Create axios client
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

// ✅ Single export block
export const api = {
  // Upload CSV
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

  // Get columns
  getColumns: async () => {
    try {
      const response = await apiClient.get('/columns');
      return response.data.columns;
    } catch (error) {
      handleApiError(error, 'Failed to get columns');
    }
  },

  // Process data
  processData: async (config: ProcessConfig) => {
    try {
      const response = await apiClient.post('/process', config);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Data processing failed');
    }
  },

  // ✅ Get pretrained models (with user ID)
  getAvailableModels: async () => {
    try {
      const headers = await getAuthHeaders();
      const response = await apiClient.get("/models", { headers });
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to get available models');
    }
  },

  // ✅ Train model (with user ID)
  trainModel: async (config: ModelConfig) => {
    try {
      const headers = await getAuthHeaders();
      const response = await apiClient.post("/train", config, { headers });

      if (config.transferLearning && config.sourceModelId) {
        toast.success("Transfer learning completed successfully!", {
          description: `Model initialized from ${config.sourceModelId} and fine-tuned`
        });
      }

      return response.data;
    } catch (error) {
      toast.error("Model training failed", {
        description: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  },

  // Get results
  getResults: async (modelId: string): Promise<ResultsData> => {
    try {
      const response = await apiClient.get(`/results/${modelId}`);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to get results');
    }
  }
};
