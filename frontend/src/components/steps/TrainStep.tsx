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
      const results = await api.trainModel({ ...model });
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
      className="workflow-step w-full min-h-screen p-4 sm:p-8 bg-white rounded-none shadow-none"
      aria-label="Train Model Step"
    >
      <h2 className="text-2xl font-semibold mb-6 text-center">Train Model</h2>

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
        <div className="space-y-2">
          <label className="text-sm font-medium" htmlFor="model-type">
            Model Type
          </label>
          <Select
            value={model.modelType}
            onValueChange={(value) =>
              setModel({ ...model, modelType: value as any })
            }
            disabled={isLoading}
          >
            <SelectTrigger className="w-full" id="model-type">
              <SelectValue placeholder="Select model type" />
            </SelectTrigger>
            <SelectContent>
              {modelTypes.map((type) => (
                <SelectItem key={type.id} value={type.id as any}>
                  {type.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-4">
          <label className="text-sm font-medium">Advanced Options</label>
          <div className="space-y-4 bg-secondary/50 p-4 rounded-md">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <label htmlFor="hyperparameter" className="text-sm font-medium">
                  Hyperparameter Tuning
                </label>
                <p className="text-xs text-muted-foreground">
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
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <label htmlFor="ensemble" className="text-sm font-medium">
                  Ensemble Learning
                </label>
                <p className="text-xs text-muted-foreground">
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
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <label htmlFor="transfer" className="text-sm font-medium">
                  Transfer Learning
                </label>
                <p className="text-xs text-muted-foreground">
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
              />
            </div>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row justify-between gap-2 pt-4">
          <Button
            type="button"
            onClick={handleBack}
            variant="outline"
            disabled={isLoading}
            className="w-full sm:w-auto"
          >
            Back
          </Button>
          <Button
            type="submit"
            disabled={isLoading}
            className="w-full sm:w-auto flex items-center justify-center bg-blue-accent hover:bg-blue-accent/90"
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
