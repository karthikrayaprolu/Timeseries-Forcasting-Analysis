// Shared types between frontend and backend

export type ModelType = 'ARIMA' | 'Prophet' | 'LSTM' | 'RandomForest' | 'XGBoost';

export type TimeFrequency = 'daily' | 'weekly' | 'monthly';

export type AggregationMethod = 'mean' | 'sum' | 'max' | 'min';

export type EnsembleMethod = 'voting' | 'stacking';

export interface DatabaseConfig {
  databaseType: string;
  connectionString?: string;
  schema?: string;
  table?: string;
}

export interface ProcessConfig {
  timeColumn: string;
  targetVariable: string;
  frequency: TimeFrequency;
  features: string[];
  dateFormat?: string;
  aggregationMethod?: AggregationMethod;
}

export interface ModelConfig {
  modelType: ModelType;
  hyperparameterTuning: boolean;
  ensembleLearning: boolean;
  transferLearning: boolean;
  timeSteps?: number;
  units?: number;
  epochs?: number;
  batchSize?: number;
  ensembleModels?: string[];
  ensembleMethod?: EnsembleMethod;
  ensembleWeights?: number[] | null;
}

export interface TrainingMetrics {
  mse: number;
  rmse: number;
  mae: number;
  mape: number;
}

export interface ForecastResults {
  dates: string[];
  actual: number[];
  predicted: number[];
}

export interface ModelInfo {
  type: ModelType;
  parameters: Record<string, any>;
  features: {
    hyperparameterTuning: boolean;
    transferLearning: boolean;
    ensembleLearning: boolean;
  };
}

export interface ResultsData {
  metrics: TrainingMetrics;
  modelInfo: ModelInfo;
  dataInfo: {
    title: string;
    filename: string;
  };
  forecasts: ForecastResults;
}
