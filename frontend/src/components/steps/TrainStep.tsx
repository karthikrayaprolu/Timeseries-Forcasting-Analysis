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

const fetchAvailableModels = async (setAvailableModels: (models: SourceModel[]) => void) => {
  try {
    const models = await api.getModels();
    setAvailableModels(models);
    if (models.length === 0) {
      toast("No pre-trained models available yet", {
        description: "Train a model first to use transfer learning"
      });
    }
    return models;
  } catch (err) {
    console.error('Error fetching models:', err);
    toast.error("Failed to fetch available models");
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
    model,
    setModel,
    setCurrentStep,
    setResults,
    isLoading,
    setIsLoading,
  } = useWorkflow();

  const [availableModels, setAvailableModels] = useState<SourceModel[]>([]);
  const [selectedSourceModel, setSelectedSourceModel] = useState<string>("");

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

  // Fetch available models when transfer learning is enabled
  useEffect(() => {
    const fetchModels = async () => {
      if (model.transferLearning) {
        setIsLoading(true);
        await fetchAvailableModels(setAvailableModels);
        setIsLoading(false);
      }
    };
    fetchModels();
  }, [model.transferLearning]);

  // Reset selected model when transfer learning is disabled
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
    },    { 
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
    }
  ];

  const handleTrain = async () => {
    setIsLoading(true);
    try {
      // Add model-specific parameters based on the selected model type
      let modelParams: any = {};      switch (model.modelType) {
        case 'lstm':
          modelParams = {
            units: model.units || 50,
            dropout: model.dropout || 0.2,
            epochs: model.epochs || 100,
            batch_size: model.batchSize || 32,
            sequence_length: model.timeSteps || 10
          };
          break;
        case 'random_forest':
        case 'xgboost':
          modelParams = {
            n_estimators: model.n_estimators || 100,
            max_depth: model.max_depth || 6,
            learning_rate: model.learning_rate || 0.1 // for xgboost
          };
          break;
        case 'arima':
          modelParams = {
            order: [1, 1, 1] // Default ARIMA parameters
          };
          break;
        case 'prophet':
          modelParams = {
            changepoint_prior_scale: 0.05,
            seasonality_prior_scale: 10,
            seasonality_mode: 'additive'
          };
          break;
      }      const response = await api.trainModel({
        ...model,
        ...modelParams,
        sourceModelId: model.transferLearning ? selectedSourceModel : undefined
      });

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

  const handleBack = () => {
    setCurrentStep("process");
  };
  const renderModelParams = () => {
    switch (model.modelType) {
      case 'lstm':
        return (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label>Units</label>
              <input
                type="number"
                value={model.units || 50}
                onChange={(e) => setModel({...model, units: parseInt(e.target.value)})}
                className="w-24 px-2 py-1 border rounded"
              />
            </div>
            <div className="flex items-center justify-between">
              <label>Dropout</label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={model.dropout || 0.2}
                onChange={(e) => setModel({...model, dropout: parseFloat(e.target.value)})}
                className="w-24 px-2 py-1 border rounded"
              />
            </div>
            <div className="flex items-center justify-between">
              <label>Sequence Length</label>
              <input
                type="number"
                value={model.timeSteps || 10}
                onChange={(e) => setModel({...model, timeSteps: parseInt(e.target.value)})}
                className="w-24 px-2 py-1 border rounded"
              />
            </div>
          </div>
        );
      case 'random_forest':
      case 'xgboost':
        return (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label>Number of Estimators</label>
              <input
                type="number"
                value={model.n_estimators || 100}
                onChange={(e) => setModel({...model, n_estimators: parseInt(e.target.value)})}
                className="w-24 px-2 py-1 border rounded"
              />
            </div>
            <div className="flex items-center justify-between">
              <label>Max Depth</label>
              <input
                type="number"
                value={model.max_depth || 6}
                onChange={(e) => setModel({...model, max_depth: parseInt(e.target.value)})}
                className="w-24 px-2 py-1 border rounded"
              />
            </div>
            {model.modelType === 'xgboost' && (
              <div className="flex items-center justify-between">
                <label>Learning Rate</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={model.learning_rate || 0.1}
                  onChange={(e) => setModel({...model, learning_rate: parseFloat(e.target.value)})}
                  className="w-24 px-2 py-1 border rounded"
                />
              </div>
            )}
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div
      ref={componentRef}
      className="workflow-step max-w-3xl mx-auto p-6 bg-white rounded-xl shadow-lg"
      aria-label="Train Model Step"
    >
      <h2 className="text-3xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-indigo-800 to-blue-600">Train Model</h2>

      {isLoading && <LoadingRobot />}

      <form
        className={`space-y-8 ${isLoading ? "pointer-events-none opacity-60" : ""}`}
        onSubmit={e => {
          e.preventDefault();
          handleTrain();
        }}
        aria-disabled={isLoading}
      >
        <div className="space-y-4">
          <label className="text-lg font-medium text-gray-700">Model Selection</label>
          <div className="grid grid-cols-1 gap-4">
            {modelTypes.map((type) => (
              <div
                key={type.id}
                className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                  model.modelType === type.id
                    ? "border-indigo-500 bg-indigo-50"
                    : "border-gray-200 hover:border-indigo-200 hover:bg-gray-50"
                }`}
                onClick={() => setModel({ ...model, modelType: type.id as any })}
              >
                <div className="flex items-center space-x-3">
                  <div className={`w-4 h-4 rounded-full ${
                    model.modelType === type.id ? "bg-indigo-500" : "bg-gray-200"
                  }`} />
                  <div>
                    <h3 className="font-medium text-gray-900">{type.name}</h3>
                    <p className="text-sm text-gray-500">{type.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <label className="text-lg font-medium text-gray-700">Model Parameters</label>
          {renderModelParams()}
        </div>

        <div className="space-y-4">
          <label className="text-lg font-medium text-gray-700">Advanced Options</label>
          <div className="space-y-6 bg-gray-50 p-6 rounded-lg border-2 border-gray-200">
            <div className="flex items-center justify-between">
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
                onCheckedChange={(checked) =>
                  setModel({ ...model, ensembleLearning: checked })
                }
                disabled={isLoading}
                className="data-[state=checked]:bg-indigo-600"
              />
            </div>            <div className="space-y-4">
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
                  <Label htmlFor="sourceModel" className="mb-2 block">Source Model</Label>
                  <Select
                    value={selectedSourceModel}
                    onValueChange={setSelectedSourceModel}
                    disabled={isLoading || availableModels.length === 0}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select a source model" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableModels.map((sourceModel) => (
                        <SelectItem key={sourceModel.id} value={sourceModel.id}>
                          {`${sourceModel.type} - ${sourceModel.target} (RMSE: ${sourceModel.metrics.rmse?.toFixed(2) ?? 'N/A'})`}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {availableModels.length === 0 && (
                    <p className="text-sm text-yellow-600 mt-2">
                      No pre-trained models available. Train a model first to use transfer learning.
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

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