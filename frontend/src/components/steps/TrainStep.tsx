import React, { useEffect, useRef } from "react";
import { useWorkflow } from "@/contexts/WorkflowContext";
import { api } from "@/services/api";
import { gsap } from "gsap";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";

// Simple SVG Spinner
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

// Fun Animated Robot SVG (engaging during loading)
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

  const componentRef = useRef<HTMLDivElement>(null);

  // GSAP animation
  useEffect(() => {
    if (componentRef.current) {
      gsap.fromTo(
        componentRef.current,
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }
      );
    }
  }, []);

  const modelTypes = [
    { id: "ARIMA", name: "ARIMA" },
    { id: "Prophet", name: "Facebook Prophet" },
    { id: "LSTM", name: "LSTM (Deep Learning)" },
    { id: "RandomForest", name: "Random Forest" },
    { id: "XGBoost", name: "XGBoost" },
  ];

  const handleTrain = async () => {
    setIsLoading(true);
    try {
      const trainingResults = await api.trainModel({ ...model });
      // Structure the results data properly
      const results = {
        dataInfo: {
          title: trainingResults.dataInfo?.title || "Time Series Data",
          filename: trainingResults.dataInfo?.filename || "data.csv"
        },
        modelInfo: {
          type: model.modelType,
          parameters: {}, // Add empty parameters object to satisfy type
          features: {
            hyperparameterTuning: model.hyperparameterTuning,
            transferLearning: model.transferLearning,
            ensembleLearning: model.ensembleLearning
          }
        },
        metrics: trainingResults.metrics,
        forecasts: trainingResults.forecasts
      };
      setResults(results);
      toast.success("Model training completed successfully");
      setCurrentStep("results");
    } catch (error) {
      console.error("Error training model:", error);
      toast.error("Failed to train model");
    } finally {
      setIsLoading(false);
    }
  };

  // Back to previous step
  const handleBack = () => {
    setCurrentStep("process");
  };

  return (
    <div
      ref={componentRef}
      className="workflow-step max-w-3xl mx-auto p-6 bg-white rounded-xl shadow-lg"
      aria-label="Train Model Step"
    >
      <h2 className="text-3xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-indigo-800 to-blue-600">Train Model</h2>

      {/* Show engaging robot and message while loading */}
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
          <label className="text-lg font-medium text-gray-700" htmlFor="model-type">
            Model Type
          </label>
          <Select
            value={model.modelType}
            onValueChange={(value) =>
              setModel({ ...model, modelType: value as any })
            }
            disabled={isLoading}
          >
            <SelectTrigger className="w-full h-12 bg-gray-50 border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 rounded-lg transition-all" id="model-type">
              <SelectValue placeholder="Select model type" />
            </SelectTrigger>
            <SelectContent className="bg-white border-2 border-gray-200 rounded-lg shadow-lg">
              {modelTypes.map((type) => (
                <SelectItem key={type.id} value={type.id as any} className="hover:bg-indigo-50 cursor-pointer">
                  {type.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
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
            </div>

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
