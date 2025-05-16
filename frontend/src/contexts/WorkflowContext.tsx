import React, { createContext, useContext, useState, ReactNode } from "react";

// Define types for our workflow
export type WorkflowStep = "home" | "database" | "process" | "train" | "results";

export type DatabaseConfig = {
  databaseType: string;
  connectionString?: string;
  schema?: string;
  table?: string;
};

export type ProcessConfig = {
  timeColumn: string;
  targetVariable: string;
  frequency: "daily" | "weekly" | "monthly";
  features: string[];
  aggregationMethod?: "mean" | "sum" | "max" | "min";
  dateFormat?: string;
};

export type ModelParameters = {
  // Common parameters
  timeSteps?: number;
  units?: number;
  epochs?: number;
  batchSize?: number;
  
  // LSTM specific
  dropout?: number;
  learningRate?: number;
  
  // Tree models specific
  n_estimators?: number;
  max_depth?: number;
  
  // XGBoost specific
  learning_rate?: number;
  subsample?: number;
  colsample_bytree?: number;
  
  // Ensemble specific
  ensembleModels?: string[];
  ensembleMethod?: 'voting' | 'stacking';
  ensembleWeights?: number[] | null;
};

export type ModelConfig = {
  modelType: "arima" | "prophet" | "lstm" | "random_forest" | "xgboost";
  hyperparameterTuning: boolean;
  ensembleLearning: boolean;
  transferLearning: boolean;
  sourceModelId?: string;
} & ModelParameters;

export type ResultsData = {
  metrics: {
    mse: number;
    rmse: number;
    mae: number;
    mape: number;
  };
  dates: string[];
  actual: number[];
  forecasts: number[];
  modelInfo: {
    type: string;
    parameters: Record<string, any>;
    features: {
      hyperparameterTuning: boolean;
      transferLearning: boolean;
      ensembleLearning: boolean;
    };
  };
  dataInfo: {
    title: string;
    filename: string;
  };
};

interface WorkflowContextType {
  currentStep: WorkflowStep | null;
  setCurrentStep: (step: WorkflowStep) => void;
  database: DatabaseConfig;
  setDatabase: (config: DatabaseConfig) => void;
  process: ProcessConfig;
  setProcess: (config: ProcessConfig) => void;
  model: ModelConfig;
  setModel: (config: ModelConfig) => void;
  results: ResultsData | null;
  setResults: (data: ResultsData | null) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  availableTables: string[];
  setAvailableTables: (tables: string[]) => void;
  availableColumns: string[];
  setAvailableColumns: (columns: string[]) => void;
}


const WorkflowContext = createContext<WorkflowContextType | undefined>(undefined);

export const WorkflowProvider = ({ children }: { children: ReactNode }) => {
  const [currentStep, setCurrentStep] = useState<WorkflowStep | null>("home");
  const [database, setDatabase] = useState<DatabaseConfig>({
    databaseType: "mongodb",
  });  const [process, setProcess] = useState<ProcessConfig>({
    timeColumn: "",
    targetVariable: "",
    frequency: "daily",
    features: [],
    aggregationMethod: "mean"
  });  const [model, setModel] = useState<ModelConfig>({
    modelType: "random_forest",
    hyperparameterTuning: false,
    ensembleLearning: false,
    transferLearning: false,
  });
  const [results, setResults] = useState<ResultsData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [availableTables, setAvailableTables] = useState<string[]>([]);
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);

  const value = {
    currentStep,
    setCurrentStep,
    database,
    setDatabase,
    process,
    setProcess,
    model,
    setModel,
    results,
    setResults,
    isLoading,
    setIsLoading,
    availableTables,
    setAvailableTables,
    availableColumns,
    setAvailableColumns,
  };

  return <WorkflowContext.Provider value={value}>{children}</WorkflowContext.Provider>;
};

export const useWorkflow = () => {
  const context = useContext(WorkflowContext);
  if (context === undefined) {
    throw new Error("useWorkflow must be used within a WorkflowProvider");
  }
  return context;
};
