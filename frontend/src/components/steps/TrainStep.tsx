import React, { useEffect, useRef, useState } from "react";
import { useWorkflow } from "@/contexts/WorkflowContext";
import { api } from "@/services/api";
import { gsap } from "gsap";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";

interface SourceModel {
  id: string;
  type: string;
  target: string;
  metrics: {
    rmse?: number;
    mae?: number;
  };
}

const fetchAvailableModels = async (
  setAvailableModels: (models: SourceModel[]) => void,
  currentModelType: string
) => {
  try {
    const response = await api.getAvailableModels();
    const models = response.data || response;

    if (!Array.isArray(models)) {
      throw new Error("Invalid response format from server");
    }

    const compatibleModels = models.filter(m =>
      m.type.toLowerCase() === currentModelType.toLowerCase()
    );

    setAvailableModels(compatibleModels);

    if (compatibleModels.length === 0) {
      toast.warning(`No compatible pre-trained ${currentModelType} models available`, {
        description: `Train a ${currentModelType} model first to use transfer learning`
      });
    }

    return compatibleModels;
  } catch (err) {
    console.error('Error fetching models:', err);
    toast.error("Failed to fetch available models", {
      description: err instanceof Error ? err.message : "Please check your connection"
    });
    return [];
  }
};

const Spinner = () => (
  <svg className="animate-spin h-5 w-5 text-white mr-2" viewBox="0 0 24 24">
    <circle
      className="opacity-25"
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth="4"
      fill="none"
    />
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
    />
  </svg>
);

const LoadingRobot = () => (
  <div className="flex flex-col items-center justify-center py-6">
    <svg width="80" height="80" viewBox="0 0 80 80" className="animate-bounce mb-2">
      <rect x="20" y="30" width="40" height="30" rx="8" fill="#60a5fa" />
      <rect x="30" y="20" width="20" height="20" rx="6" fill="#3b82f6" />
      <circle cx="35" cy="45" r="4" fill="#fff" />
      <circle cx="45" cy="45" r="4" fill="#fff" />
      <rect x="36" y="52" width="8" height="4" rx="2" fill="#fff" />
      <rect x="18" y="38" width="4" height="12" rx="2" fill="#a7f3d0" />
      <rect x="58" y="38" width="4" height="12" rx="2" fill="#a7f3d0" />
    </svg>
    <p className="text-blue-700 text-sm font-medium text-center">
      🤖 Training in progress...<br />
      Sit tight while our robot crunches the numbers!
    </p>
  </div>
);

const TrainStep = () => {
  const {
    process,
    model,
    setModel,
    setCurrentStep,
    setResults,
    isLoading,
    setIsLoading,
  } = useWorkflow();
const [forecastHorizon, setForecastHorizon] = useState<number>(30);

  const [availableModels, setAvailableModels] = useState<SourceModel[]>([]);
  const [selectedSourceModel, setSelectedSourceModel] = useState<string>("");
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [showManualParams, setShowManualParams] = useState<boolean>(false);

  const componentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (componentRef.current) {
      gsap.fromTo(
        componentRef.current,
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }
      );
    }
  }, []);

  useEffect(() => {
    const fetchModels = async () => {
      if (model.transferLearning) {
        setIsLoading(true);
        try {
          const models = await fetchAvailableModels(setAvailableModels, model.modelType);
          if (models.length > 0) {
            setSelectedSourceModel(models[0].id);
            setModel(prev => ({ ...prev, sourceModelId: models[0].id }));
          }
        } catch (error) {
          console.error("Model fetch error:", error);
        } finally {
          setIsLoading(false);
        }
      }
    };
    fetchModels();
  }, [model.transferLearning, model.modelType]);

  useEffect(() => {
    if (!model.transferLearning) {
      setSelectedSourceModel("");
    }
  }, [model.transferLearning]);

  const modelTypes = [
    {
      id: "arima",
      name: "ARIMA",
      description: "Classical time series forecasting with auto regression"
    },
    {
      id: "prophet",
      name: "Prophet",
      description: "Facebook's powerful forecasting tool with automatic seasonality"
    },
    { 
      id: "lstm", 
      name: "LSTM Neural Network",
      description: "Deep learning for complex temporal patterns"
    },
    { 
      id: "random_forest", 
      name: "Random Forest",
      description: "Powerful ensemble learning for time series forecasting"
    },
    { 
      id: "xgboost", 
      name: "XGBoost",
      description: "High performance gradient boosting for accurate predictions"
    },
    {
      id: "ets",
      name: "ETS (Exponential Smoothing)",
      description: "Statistical model for time series with trend and seasonality"
    },
    {
      id: "lightgbm",
      name: "LightGBM",
      description: "Efficient gradient boosting for fast and accurate predictions"
    }
  ];

  const modelLevels = [
    {
      id: "basic",
      name: "Basic",
      description: "Fast training with default parameters (less accurate)"
    },
    {
      id: "balanced",
      name: "Balanced",
      description: "Good trade-off between speed and accuracy"
    },
    {
      id: "high_accuracy",
      name: "High Accuracy",
      description: "Slower training with optimized parameters (best results)"
    },
    {
      id: "custom",
      name: "Custom",
      description: "Manually configure all parameters"
    }
  ];

  const handleTrain = async () => {
    setIsLoading(true);
    try {
      let modelParams: any = {};
      
      if (model.modelLevel !== 'custom') {
        switch (model.modelLevel) {
          case 'basic':
            modelParams = getBasicParams(model.modelType);
            break;
          case 'balanced':
            modelParams = getBalancedParams(model.modelType);
            break;
          case 'high_accuracy':
            modelParams = getHighAccuracyParams(model.modelType);
            break;
          case 'ets':
            modelParams = {
              trend: model.trend ?? 'add',
              seasonal: model.seasonal ?? 'add',
              seasonal_periods: model.seasonal_periods ?? 12
            };
            break;
          case 'lightgbm':
            modelParams = {
              num_leaves: model.num_leaves ?? 31,
              learning_rate: model.learning_rate ?? 0.1,
              n_estimators: model.n_estimators ?? 100
            };
            break;
          default:
            modelParams = getBasicParams(model.modelType);
        }
      } else {
        switch (model.modelType) {
          case 'lstm':
            modelParams = {
              units: model.units ?? 50,
              dropout: model.dropout ?? 0.2,
              epochs: model.epochs ?? 100,
              batch_size: model.batchSize ?? 32,
              sequence_length: model.sequence_length ?? 10
            };
            break;
          case 'random_forest':
            modelParams = {
              n_estimators: model.n_estimators ?? 100,
              max_depth: model.max_depth ?? 10
            };
            break;
          case 'xgboost':
            modelParams = {
              n_estimators: model.n_estimators ?? 100,
              max_depth: model.max_depth ?? 6,
              learning_rate: model.learning_rate ?? 0.1
            };
            break;
          case 'arima':
            modelParams = {
              order: model.order ?? [1, 1, 1]
            };
            break;
          case 'prophet':
            modelParams = {
              changepoint_prior_scale: model.changepoint_prior_scale ?? 0.05,
              seasonality_prior_scale: model.seasonality_prior_scale ?? 10,
              seasonality_mode: model.seasonality_mode ?? 'additive'
            };
            break;
          case 'ets':
            modelParams = {
              trend: model.trend ?? 'add',
              seasonal: model.seasonal ?? 'add',
              seasonal_periods: model.seasonal_periods ?? 12
            };
            break;
          case 'lightgbm':
            modelParams = {
              num_leaves: model.num_leaves ?? 31,
              learning_rate: model.learning_rate ?? 0.1,
              n_estimators: model.n_estimators ?? 100
            };
            break;
        }
      }

      let ensembleModels = model.ensembleModels || [];
      if (model.ensembleLearning && !ensembleModels.includes(model.modelType)) {
        ensembleModels = [...ensembleModels, model.modelType];
      }
      if (model.ensembleLearning && ensembleModels.length === 0) {
        toast.warning("Please select at least one additional model for ensemble.");
        setIsLoading(false);
        return;
      }

      const payload = {
        ...model,
        ...modelParams,
        sourceModelId: model.transferLearning ? selectedSourceModel : null,
        timeColumn: process.timeColumn,
        targetVariable: process.targetVariable,
        frequency: process.frequency,
        ensembleModels: model.ensembleLearning ? ensembleModels : undefined,
          forecast_horizon: process.forecast_horizon,
     

      };

      const response = await api.trainModel(payload);

      if (!response || response.error) {
        throw new Error(response?.error || 'Training failed');
      }

      setResults(response);
      setCurrentStep("results");
      toast.success("Training completed successfully!");
    } catch (error) {
      console.error('Training error:', error);
      toast.error("Training failed: " + (error instanceof Error ? error.message : String(error)));
    } finally {
      setIsLoading(false);
    }
  };

  const getBasicParams = (modelType: string) => {
    switch (modelType) {
      case 'lstm':
        return {
          units: 32,
          dropout: 0.1,
          epochs: 50,
          batch_size: 32,
          sequence_length: 5
        };
      case 'random_forest':
        return {
          n_estimators: 50,
          max_depth: 5
        };
      case 'xgboost':
        return {
          n_estimators: 50,
          max_depth: 3,
          learning_rate: 0.3
        };
      case 'ets':
        return {
          trend: 'add',
          seasonal: 'add',
          seasonal_periods: 7
        };
      case 'lightgbm':
        return {
          num_leaves: 15,
          learning_rate: 0.3,
          n_estimators: 50
        };
      default:
        return {};
    }
  };

  const getBalancedParams = (modelType: string) => {
    switch (modelType) {
      case 'lstm':
        return {
          units: 64,
          dropout: 0.2,
          epochs: 100,
          batch_size: 32,
          sequence_length: 10
        };
      case 'random_forest':
        return {
          n_estimators: 100,
          max_depth: 10
        };
      case 'xgboost':
        return {
          n_estimators: 100,
          max_depth: 6,
          learning_rate: 0.1
        };
      case 'ets':
        return {
          trend: 'add',
          seasonal: 'add',
          seasonal_periods: 12
        };
      case 'lightgbm':
        return {
          num_leaves: 31,
          learning_rate: 0.1,
          n_estimators: 100
        };
      default:
        return {};
    }
  };

  const getHighAccuracyParams = (modelType: string) => {
    switch (modelType) {
      case 'lstm':
        return {
          units: 128,
          dropout: 0.3,
          epochs: 200,
          batch_size: 16,
          sequence_length: 15
        };
      case 'random_forest':
        return {
          n_estimators: 200,
          max_depth: 15
        };
      case 'xgboost':
        return {
          n_estimators: 200,
          max_depth: 8,
          learning_rate: 0.05
        };
      case 'ets':
        return {
          trend: 'mul',
          seasonal: 'mul',
          seasonal_periods: 12
        };
      case 'lightgbm':
        return {
          num_leaves: 63,
          learning_rate: 0.05,
          n_estimators: 200
        };
      default:
        return {};
    }
  };

  const handleBack = () => {
    setCurrentStep("process");
  };

  const renderModelParams = () => {
    if (model.modelLevel !== 'custom') return null;

    switch (model.modelType) {
      case 'lstm':
        return (
          <div className="space-y-4 bg-blue-50 p-4 rounded-lg border border-blue-200 mt-4">
            <h4 className="font-medium text-blue-900 mb-3">LSTM Parameters</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Units</label>
                <input
                  type="number"
                  value={model.units || 50}
                  onChange={(e) => setModel({...model, units: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  min="1"
                  max="1000"
                />
                <p className="text-xs text-gray-500">Number of LSTM units (10-200 recommended)</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Dropout Rate</label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  value={model.dropout || 0.2}
                  onChange={(e) => setModel({...model, dropout: parseFloat(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <p className="text-xs text-gray-500">Dropout rate for regularization (0.0-0.5)</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Sequence Length</label>
                <input
                  type="number"
                  value={model.sequence_length || 10}
                  onChange={(e) => setModel({ ...model, sequence_length: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  min="1"
                  max="100"
                />
                <p className="text-xs text-gray-500">Number of time steps to look back</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Epochs</label>
                <input
                  type="number"
                  value={model.epochs || 100}
                  onChange={(e) => setModel({...model, epochs: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  min="1"
                  max="1000"
                />
                <p className="text-xs text-gray-500">Training epochs (50-200 recommended)</p>
              </div>
            </div>
          </div>
        );
      
      case 'random_forest':
        return (
          <div className="space-y-4 bg-green-50 p-4 rounded-lg border border-green-200 mt-4">
            <h4 className="font-medium text-green-900 mb-3">Random Forest Parameters</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Number of Estimators</label>
                <input
                  type="number"
                  value={model.n_estimators || 100}
                  onChange={(e) => setModel({...model, n_estimators: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  min="10"
                  max="1000"
                />
                <p className="text-xs text-gray-500">Number of trees in the forest (50-500)</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Max Depth</label>
                <input
                  type="number"
                  value={model.max_depth || 10}
                  onChange={(e) => setModel({...model, max_depth: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  min="1"
                  max="50"
                />
                <p className="text-xs text-gray-500">Maximum depth of trees (5-20 recommended)</p>
              </div>
            </div>
          </div>
        );

      case 'xgboost':
        return (
          <div className="space-y-4 bg-purple-50 p-4 rounded-lg border border-purple-200 mt-4">
            <h4 className="font-medium text-purple-900 mb-3">XGBoost Parameters</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Number of Estimators</label>
                <input
                  type="number"
                  value={model.n_estimators || 100}
                  onChange={(e) => setModel({...model, n_estimators: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                  min="10"
                  max="1000"
                />
                <p className="text-xs text-gray-500">Number of boosting rounds</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Max Depth</label>
                <input
                  type="number"
                  value={model.max_depth || 6}
                  onChange={(e) => setModel({...model, max_depth: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                  min="1"
                  max="20"
                />
                <p className="text-xs text-gray-500">Maximum tree depth</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Learning Rate</label>
                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  max="1"
                  value={model.learning_rate || 0.1}
                  onChange={(e) => setModel({...model, learning_rate: parseFloat(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
                <p className="text-xs text-gray-500">Step size shrinkage</p>
              </div>
            </div>
          </div>
        );
      case 'ets':
        return (
          <div className="space-y-4 bg-teal-50 p-4 rounded-lg border border-teal-200 mt-4">
            <h4 className="font-medium text-teal-900 mb-3">ETS Parameters</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Trend Type</label>
                <Select
                  value={model.trend || 'add'}
                  onValueChange={(value) => setModel({ ...model, trend: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="add">Additive</SelectItem>
                    <SelectItem value="mul">Multiplicative</SelectItem>
                    <SelectItem value="none">None</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-gray-500">Type of trend component</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Seasonal Type</label>
                <Select
                  value={model.seasonal || 'add'}
                  onValueChange={(value) => setModel({ ...model, seasonal: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="add">Additive</SelectItem>
                    <SelectItem value="mul">Multiplicative</SelectItem>
                    <SelectItem value="none">None</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-gray-500">Type of seasonal component</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Seasonal Periods</label>
                <input
                  type="number"
                  value={model.seasonal_periods || 12}
                  onChange={(e) => setModel({ ...model, seasonal_periods: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500"
                  min="1"
                  max="52"
                />
                <p className="text-xs text-gray-500">Number of periods in a season (e.g., 12 for monthly)</p>
              </div>
            </div>
          </div>
        );
      case 'lightgbm':
        return (
          <div className="space-y-4 bg-orange-50 p-4 rounded-lg border border-orange-200 mt-4">
            <h4 className="font-medium text-orange-900 mb-3">LightGBM Parameters</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Number of Leaves</label>
                <input
                  type="number"
                  value={model.num_leaves || 31}
                  onChange={(e) => setModel({ ...model, num_leaves: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                  min="2"
                  max="131072"
                />
                <p className="text-xs text-gray-500">Maximum number of leaves in trees</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Learning Rate</label>
                <input
                  type="number"
                  step="0.01"
                  min="0.001"
                  max="1"
                  value={model.learning_rate || 0.1}
                  onChange={(e) => setModel({ ...model, learning_rate: parseFloat(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                />
                <p className="text-xs text-gray-500">Step size shrinkage</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Number of Estimators</label>
                <input
                  type="number"
                  value={model.n_estimators || 100}
                  onChange={(e) => setModel({ ...model, n_estimators: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                  min="10"
                  max="1000"
                />
                <p className="text-xs text-gray-500">Number of boosting iterations</p>
              </div>
            </div>
          </div>
        );
      
      default:
        return null;
    }
  };

  const getSelectedModelName = () => {
    const selectedModel = modelTypes.find(type => type.id === model.modelType);
    return selectedModel ? selectedModel.name : "Select Model";
  };

  const requiresModelLevel = ['lstm', 'random_forest', 'xgboost','ets','lightgbm'].includes(model.modelType);

  return (
    <div
      ref={componentRef}
      className="workflow-step max-w-4xl mx-auto p-6 bg-white rounded-xl shadow-lg"
      aria-label="Train Model Step"
    >
      <h2 className="text-3xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-indigo-800 to-blue-600">
        Train Model
      </h2>

      {isLoading && <LoadingRobot />}

      <form
        className={`space-y-8 ${isLoading ? "pointer-events-none opacity-60" : ""}`}
        onSubmit={e => {
          e.preventDefault();
          handleTrain();
        }}
        aria-disabled={isLoading}
      >
        {/* Model Selection */}
        <div className="space-y-4">
          <label className="text-lg font-medium text-gray-700">Model Type</label>
          <Select
            value={model.modelType}
            onValueChange={(value) => setModel({ ...model, modelType: value as any, modelLevel: 'balanced' })}
            disabled={isLoading}
          >
            <SelectTrigger className="w-full h-16 text-left">
              <SelectValue>
                <div className="flex flex-col items-start">
                  <span className="font-medium text-gray-900">{getSelectedModelName()}</span>
                  {model.modelType && (
                    <span className="text-sm text-gray-500">
                      {modelTypes.find(t => t.id === model.modelType)?.description}
                    </span>
                  )}
                </div>
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {modelTypes.map((type) => (
                <SelectItem key={type.id} value={type.id} className="py-3">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">{type.name}</span>
                    <span className="text-sm text-gray-500">{type.description}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Model Level Selection */}
        {requiresModelLevel && (
          <div className="space-y-4">
            <label className="text-lg font-medium text-gray-700">Tuning Mode</label>
            <Select
              value={model.modelLevel || 'balanced'}
              onValueChange={(value) => {
                setModel({ ...model, modelLevel: value as any });
                if (value === 'custom') {
                  setShowManualParams(true);
                }
              }}
              disabled={isLoading}
            >
              <SelectTrigger className="w-full h-16 text-left">
                <SelectValue>
                  <div className="flex flex-col items-start">
                    <span className="font-medium text-gray-900">
                      {modelLevels.find(l => l.id === (model.modelLevel || 'balanced'))?.name}
                    </span>
                    <span className="text-sm text-gray-500">
                      {modelLevels.find(l => l.id === (model.modelLevel || 'balanced'))?.description}
                    </span>
                  </div>
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {modelLevels.map((level) => (
                  <SelectItem key={level.id} value={level.id} className="py-3">
                    <div className="flex flex-col items-start">
                      <span className="font-medium">{level.name}</span>
                      <span className="text-sm text-gray-500">{level.description}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Show parameters when custom is selected */}
            {renderModelParams()}
          </div>
        )}

        {/* Advanced Options */}
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border-2 border-gray-200">
            <div className="space-y-1">
              <label htmlFor="advanced" className="text-lg font-medium text-gray-700">
                Advanced Features
              </label>
              <p className="text-sm text-gray-500">
                Enable ensemble learning, transfer learning, and hyperparameter tuning
              </p>
            </div>
            <Switch
              id="advanced"
              checked={showAdvanced}
              onCheckedChange={setShowAdvanced}
              disabled={isLoading}
              className="data-[state=checked]:bg-indigo-600"
            />
          </div>

          {/* Advanced Options Content */}
          {showAdvanced && (
            <div className="space-y-6 bg-gray-50 p-6 rounded-lg border-2 border-gray-200">
              {/* Hyperparameter Tuning */}
              <div className="flex items-center justify-between p-4 bg-white rounded-lg border border-gray-200">
                <div className="space-y-1">
                  <label htmlFor="hyperparameter" className="text-base font-medium text-gray-700">
                    Hyperparameter Tuning
                  </label>
                  <p className="text-sm text-gray-500">
                    Use grid/random search to find optimal parameters
                  </p>
                </div>
                <Switch
                  id="hyperparameter"
                  checked={model.hyperparameterTuning}
                  onCheckedChange={(checked) =>
                    setModel({ ...model, hyperparameterTuning: checked })
                  }
                  disabled={isLoading}
                  className="data-[state=checked]:bg-indigo-600"
                />
              </div>

              {/* Ensemble Learning */}
              <div className="space-y-4 p-4 bg-white rounded-lg border border-gray-200">
                <div className="flex items-center justify-between"> 
                  <div className="space-y-1">
                    <label htmlFor="ensemble" className="text-base font-medium text-gray-700">
                      Ensemble Learning
                    </label>
                    <p className="text-sm text-gray-500">
                      Combine multiple models for better performance
                    </p>
                  </div>
                  <Switch
                    id="ensemble"
                    checked={model.ensembleLearning}
                    onCheckedChange={(checked) => {
                      setModel({ ...model, ensembleLearning: checked });
                    }}
                    disabled={isLoading}
                    className="data-[state=checked]:bg-indigo-600"
                  />
                </div>

                {model.ensembleLearning && (
                  <div className="mt-4 space-y-3 border-t pt-4">
                    <label className="text-sm font-medium text-gray-700">Select Additional Models</label>
                    <div className="grid grid-cols-2 gap-3">
                      {['arima', 'prophet', 'lstm', 'random_forest', 'xgboost', 'ets', 'lightgbm']
                        .filter((m) => m !== model.modelType)
                        .map((m) => (
                          <div key={m} className="flex items-center space-x-2 p-2 bg-gray-50 rounded">
                            <input
                              type="checkbox"
                              id={`ensemble-${m}`}
                              checked={model.ensembleModels?.includes(m)}
                              onChange={(e) => {
                                const checked = e.target.checked;
                                let updated = model.ensembleModels || [];
                                if (checked) {
                                  updated = [...updated, m];
                                } else {
                                  updated = updated.filter((item) => item !== m);
                                }
                                setModel({ ...model, ensembleModels: updated });
                              }}
                              className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                            />
                            <label htmlFor={`ensemble-${m}`} className="text-sm font-medium capitalize cursor-pointer">
                              {m.replace('_', ' ')}
                            </label>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Transfer Learning */}
              <div className="space-y-4 p-4 bg-white rounded-lg border border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <label htmlFor="transfer" className="text-base font-medium text-gray-700">
                      Transfer Learning
                    </label>
                    <p className="text-sm text-gray-500">
                      Use pre-trained models to improve performance
                    </p>
                  </div>
                  <Switch
                    id="transfer"
                    checked={model.transferLearning}
                    onCheckedChange={(checked) =>
                      setModel({ ...model, transferLearning: checked })
                    }
                    disabled={isLoading}
                    className="data-[state=checked]:bg-indigo-600"
                  />
                </div>

                {model.transferLearning && (
                  <div className="mt-4 border-t pt-4">
                    <Label htmlFor="sourceModel" className="mb-2 block font-medium">Source Model</Label>
                    {isLoading ? (
                      <div className="flex items-center justify-center py-4">
                        <Spinner />
                        <span className="ml-2">Loading models...</span>
                      </div>
                    ) : (
                      <>
                        <Select
                          value={selectedSourceModel}
                          onValueChange={(value) => {
                            setSelectedSourceModel(value);
                            setModel(prev => ({...prev, sourceModelId: value}));
                          }}
                          disabled={isLoading || availableModels.length === 0}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder={
                              availableModels.length === 0 
                                ? "No compatible models available" 
                                : "Select a source model"
                            } />
                          </SelectTrigger>
                          {availableModels.length > 0 && (
                            <SelectContent>
                              {availableModels.map((sourceModel) => (
                                <SelectItem key={sourceModel.id} value={sourceModel.id}>
                                  {`${sourceModel.type} (trained on ${sourceModel.target}) - RMSE: ${sourceModel.metrics?.rmse?.toFixed(2) ?? 'N/A'}`}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          )}
                        </Select>
                        {availableModels.length === 0 && !isLoading && (
                          <div className="mt-2 p-3 bg-yellow-50 rounded-md border border-yellow-200">
                            <p className="text-sm text-yellow-700">
                              No compatible pre-trained {model.modelType} models available. 
                              Train a model first to use transfer learning.
                            </p>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>


        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row justify-between gap-4 pt-6">
          <Button
            type="button"
            onClick={handleBack}
            variant="outline"
            disabled={isLoading}
            className="h-12 px-8 border-2 border-gray-200 hover:border-gray-300 text-gray-700 font-medium rounded-lg transition-all"
          >
            Back
          </Button>
          <Button
            type="submit"
            disabled={isLoading}
            className="h-12 px-8 bg-gradient-to-r from-indigo-600 to-blue-500 hover:from-indigo-700 hover:to-blue-600 text-white font-medium rounded-lg shadow-md hover:shadow-lg transition-all flex items-center justify-center"
          >
            {isLoading ? (
              <>
                <Spinner />
                Training...
              </>
            ) : (
              "Train Model"
            )}
          </Button>
        </div>
      </form>
    </div>
  );
};

export default TrainStep;