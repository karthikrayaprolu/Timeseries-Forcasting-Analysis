
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
      const results = await api.trainModel({
        ...model,
      });
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
    <div ref={componentRef} className="workflow-step">
      <h2 className="step-title">Train Model</h2>
      
      <div className="space-y-6">
        <div className="space-y-2">
          <label className="text-sm font-medium">Model Type</label>
          <Select
            value={model.modelType}
            onValueChange={(value: "ARIMA" | "Prophet" | "LSTM" | "RandomForest" | "XGBoost") => 
              setModel({ ...model, modelType: value })
            }
            disabled={isLoading}
          >
            <SelectTrigger className="w-full">
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

        <div className="flex justify-between pt-4">
          <Button onClick={handleBack} variant="outline" disabled={isLoading}>
            Back
          </Button>
          <Button 
            onClick={handleTrain} 
            disabled={isLoading}
            className="bg-blue-accent hover:bg-blue-accent/90"
          >
            {isLoading ? "Training..." : "Train Model"}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default TrainStep;
