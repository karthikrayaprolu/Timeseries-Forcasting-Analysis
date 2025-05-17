import React, { createContext, useContext, useState, ReactNode } from "react";
import { 
  DatabaseConfig, 
  ProcessConfig, 
  ModelConfig,
  ResultsData,
  ModelType,
  TimeFrequency
} from "@/shared/types";

// Define workflow step type
export type WorkflowStep = "home" | "database" | "process" | "train" | "results";

// Define context type using shared types
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
  });
  const [process, setProcess] = useState<ProcessConfig>({
    timeColumn: "",
    targetVariable: "",
    frequency: "daily",
    features: [],
  });
  const [model, setModel] = useState<ModelConfig>({
    modelType: "Prophet",
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
