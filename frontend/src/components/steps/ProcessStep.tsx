import React, { useEffect, useRef } from "react";
import { useWorkflow } from "@/contexts/WorkflowContext";
import { api } from "@/services/api";
import { gsap } from "gsap";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";

// --- Wave Bars Loader ---
const WaveBarsLoader = () => (
  <div className="flex flex-col items-center py-6">
    <div className="flex space-x-1 mb-2">
      {[...Array(5)].map((_, i) => (
        <div
          key={i}
          className="w-2 h-6 bg-blue-500 rounded animate-wave"
          style={{
            animationDelay: `${i * 0.15}s`,
          }}
        />
      ))}
    </div>
    <p className="text-blue-700 text-sm font-medium text-center">
      Please wait while we prepare your features...
    </p>
    <style>
      {`
        @keyframes wave {
          0%, 60%, 100% { transform: scaleY(1); }
          30% { transform: scaleY(1.8); }
        }
        .animate-wave {
          display: inline-block;
          animation: wave 1s infinite ease-in-out;
        }
      `}
    </style>
  </div>
);

const ProcessStep = () => {
  const {
    process,
    setProcess,
    setCurrentStep,
    database,
    availableColumns,
    setAvailableColumns,
    isLoading,
    setIsLoading,
  } = useWorkflow();

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
    const loadColumns = async () => {
      if (database.table) {
        setIsLoading(true);
        try {
          const columns = await api.getColumns(database.table);
          setAvailableColumns(columns);
        } catch (error) {
          console.error("Error loading columns:", error);
          toast.error("Failed to load columns from table");
        } finally {
          setIsLoading(false);
        }
      }
    };
    loadColumns();
  }, [database.table, setAvailableColumns, setIsLoading]);

  const handleProcess = async () => {
    if (!process.timeColumn || !process.targetVariable) {
      toast.warning("Please select time column and target variable");
      return;
    }
    setIsLoading(true);
    try {
      await api.processData({
        table: database.table,
        ...process,
      });
      toast.success("Data processed successfully");
      setCurrentStep("train");
    } catch (error) {
      console.error("Error processing data:", error);
      toast.error("Failed to process data");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeatureToggle = (column) => {
    const features = [...process.features];
    if (features.includes(column)) {
      setProcess({
        ...process,
        features: features.filter((f) => f !== column),
      });
    } else {
      setProcess({
        ...process,
        features: [...features, column],
      });
    }
  };

  const handleBack = () => {
    setCurrentStep("database");
  };

  return (
    <div ref={componentRef} className="workflow-step">
      <h2 className="step-title">Process Data</h2>

      {/* Engaging loader */}
      {isLoading && <WaveBarsLoader />}

      <form
        className={`space-y-6 ${isLoading ? "pointer-events-none opacity-60" : ""}`}
        onSubmit={e => {
          e.preventDefault();
          handleProcess();
        }}
        aria-disabled={isLoading}
      >
        <div className="space-y-2">
          <label className="text-sm font-medium">Time Column</label>
          <p className="text-xs text-muted-foreground mb-2">
            Select the column containing date/time values
          </p>
          <Select
            value={process.timeColumn}
            onValueChange={(value) => setProcess({ ...process, timeColumn: value })}
            disabled={isLoading}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select time column" />
            </SelectTrigger>
            <SelectContent>
              {availableColumns
                .filter(
                  (col) =>
                    col.toLowerCase().includes("month") ||
                    col.toLowerCase().includes("date") ||
                    col.toLowerCase().includes("time")
                )
                .map((column) => (
                  <SelectItem key={column} value={column}>
                    {column}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Target Variable</label>
          <Select
            value={process.targetVariable}
            onValueChange={(value) => setProcess({ ...process, targetVariable: value })}
            disabled={isLoading}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select target variable" />
            </SelectTrigger>
            <SelectContent>
              {availableColumns
                .filter(
                  (col) =>
                    !col.toLowerCase().includes("date") &&
                    !col.toLowerCase().includes("time") &&
                    col !== process.timeColumn
                )
                .map((column) => (
                  <SelectItem key={column} value={column}>
                    {column}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Time Frequency</label>
          <Select
            value={process.frequency}
            onValueChange={(value) =>
              setProcess({ ...process, frequency: value })
            }
            disabled={isLoading}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select frequency" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="daily">Daily</SelectItem>
              <SelectItem value="weekly">Weekly</SelectItem>
              <SelectItem value="monthly">Monthly</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Additional Features</label>
          <div className="bg-secondary/50 p-4 rounded-md max-h-40 overflow-y-auto space-y-2">
            {availableColumns
              .filter(
                (col) =>
                  col !== process.timeColumn && col !== process.targetVariable
              )
              .map((column) => (
                <div key={column} className="flex items-center space-x-2">
                  <Checkbox
                    id={`feature-${column}`}
                    checked={process.features.includes(column)}
                    onCheckedChange={() => handleFeatureToggle(column)}
                    disabled={isLoading}
                  />
                  <label
                    htmlFor={`feature-${column}`}
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    {column}
                  </label>
                </div>
              ))}
          </div>
        </div>

        <div className="flex justify-between pt-4">
          <Button onClick={handleBack} variant="outline" disabled={isLoading}>
            Back
          </Button>
          <Button type="submit" disabled={isLoading}>
            {isLoading ? "Processing..." : "Process & Continue"}
          </Button>
        </div>
      </form>
    </div>
  );
};

export default ProcessStep;
