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
          const columns = await api.getColumns();
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
    // Validate required fields
    if (!process.timeColumn || !process.targetVariable || !process.frequency) {
      toast.warning("Please select time column, target variable, and frequency");
      return;
    }

    setIsLoading(true);
    try {
      const processResult = await api.processData({
        timeColumn: process.timeColumn,
        targetVariable: process.targetVariable,
        frequency: process.frequency,
        features: process.features || [],
        aggregationMethod: process.aggregationMethod
      });

      if (processResult.preview) {
        toast.success(`Data processed successfully. ${processResult.preview.rows} rows processed.`);
        setTimeout(() => setCurrentStep("train"), 1000);
      } else {
        throw new Error("Invalid response from server");
      }
    } catch (error: any) {
      console.error("Error processing data:", error);
      toast.error(error.message || "Failed to process data");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeatureToggle = (column: string) => {
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
    <div ref={componentRef} className="workflow-step max-w-6xl mx-auto p-8 bg-white rounded-xl shadow-lg">
      <h2 className="text-3xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-indigo-800 to-blue-600">
        Process Data
      </h2>

      {isLoading && <WaveBarsLoader />}

      <form
        className={`${isLoading ? "pointer-events-none opacity-60" : ""}`}
        onSubmit={e => {
          e.preventDefault();
          handleProcess();
        }}
        aria-disabled={isLoading}
      >
        <div className="grid grid-cols-2 gap-8">
          {/* Left Column */}
          <div className="space-y-6">
            {/* Time Column Selection */}
            <div className="bg-gray-50 p-6 rounded-xl border-2 border-gray-200">
              <label className="text-lg font-medium text-gray-700 block mb-2">Time Column</label>
              <p className="text-sm text-gray-500 mb-3">
                Select the column containing date/time values
              </p>
              <Select
                value={process.timeColumn}
                onValueChange={(value) => setProcess({ ...process, timeColumn: value })}
                disabled={isLoading}
              >
                <SelectTrigger className="w-full h-12 bg-white border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 rounded-lg transition-all">
                  <SelectValue placeholder="Select time column" />
                </SelectTrigger>
                <SelectContent className="bg-white border-2 border-gray-200 rounded-lg shadow-lg">
                  {availableColumns
                    .filter(col => 
                      col.toLowerCase().includes("month") ||
                      col.toLowerCase().includes("date") ||
                      col.toLowerCase().includes("time")
                    )
                    .map((column) => (
                      <SelectItem key={column} value={column} className="hover:bg-indigo-50 cursor-pointer">
                        {column}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>

            {/* Target Variable Selection */}
            <div className="bg-gray-50 p-6 rounded-xl border-2 border-gray-200">
              <label className="text-lg font-medium text-gray-700 block mb-2">Target Variable</label>
              <Select
                value={process.targetVariable}
                onValueChange={(value) => setProcess({ ...process, targetVariable: value })}
                disabled={isLoading}
              >
                <SelectTrigger className="w-full h-12 bg-white border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 rounded-lg transition-all">
                  <SelectValue placeholder="Select target variable" />
                </SelectTrigger>
                <SelectContent className="bg-white border-2 border-gray-200 rounded-lg shadow-lg">
                  {availableColumns
                    .filter(col =>
                      !col.toLowerCase().includes("date") &&
                      !col.toLowerCase().includes("time") &&
                      col !== process.timeColumn
                    )
                    .map((column) => (
                      <SelectItem key={column} value={column} className="hover:bg-indigo-50 cursor-pointer">
                        {column}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>

            {/* Time Frequency Selection */}
            <div className="bg-gray-50 p-6 rounded-xl border-2 border-gray-200">
              <label className="text-lg font-medium text-gray-700 block mb-2">Time Frequency</label>
              <div className="space-y-3">
                <Select
                  value={process.frequency}
                  onValueChange={(value) =>
                    setProcess({ ...process, frequency: value as "daily" | "weekly" | "monthly" })
                  }
                  disabled={isLoading}
                >
                  <SelectTrigger className="w-full h-12 bg-white border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 rounded-lg transition-all">
                    <SelectValue placeholder="Select frequency" />
                  </SelectTrigger>
                  <SelectContent className="bg-white border-2 border-gray-200 rounded-lg shadow-lg">
                    <SelectItem value="daily" className="hover:bg-indigo-50 cursor-pointer">Daily</SelectItem>
                    <SelectItem value="weekly" className="hover:bg-indigo-50 cursor-pointer">Weekly</SelectItem>
                    <SelectItem value="monthly" className="hover:bg-indigo-50 cursor-pointer">Monthly</SelectItem>
                  </SelectContent>
                </Select>

                {(process.frequency === 'weekly' || process.frequency === 'monthly') && (
                  <Select
                    value={process.aggregationMethod}
                    onValueChange={(value) =>
                      setProcess({ ...process, aggregationMethod: value as "mean" | "sum" | "max" | "min" })
                    }
                    disabled={isLoading}
                  >
                    <SelectTrigger className="w-full h-12 bg-white border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 rounded-lg transition-all">
                      <SelectValue placeholder="Select aggregation method" />
                    </SelectTrigger>
                    <SelectContent className="bg-white border-2 border-gray-200 rounded-lg shadow-lg">
                      <SelectItem value="mean" className="hover:bg-indigo-50 cursor-pointer">Mean (Average)</SelectItem>
                      <SelectItem value="sum" className="hover:bg-indigo-50 cursor-pointer">Sum</SelectItem>
                      <SelectItem value="max" className="hover:bg-indigo-50 cursor-pointer">Maximum</SelectItem>
                      <SelectItem value="min" className="hover:bg-indigo-50 cursor-pointer">Minimum</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              </div>
            </div>
          </div>

          {/* Right Column - Additional Features */}
          <div className="bg-gray-50 p-6 rounded-xl border-2 border-gray-200 h-full">
            <label className="text-lg font-medium text-gray-700 block mb-4">Additional Features</label>
            <div className="grid grid-cols-2 gap-4">
              {availableColumns
                .filter(col => 
                  col !== process.timeColumn && 
                  col !== process.targetVariable
                )
                .map((column) => (
                  <div key={column} className="flex items-center space-x-3 bg-white p-3 rounded-lg border border-gray-200">
                    <Checkbox
                      id={`feature-${column}`}
                      checked={process.features.includes(column)}
                      onCheckedChange={() => handleFeatureToggle(column)}
                      disabled={isLoading}
                      className="h-5 w-5 border-2 border-gray-300 rounded-md checked:bg-indigo-600 checked:border-indigo-600 focus:ring-2 focus:ring-indigo-200"
                    />
                    <label
                      htmlFor={`feature-${column}`}
                      className="text-sm font-medium text-gray-700 leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      {column}
                    </label>
                  </div>
                ))}
            </div>
          </div>
        </div>

        {/* Navigation Buttons */}
        <div className="flex justify-between mt-8 pt-6 border-t border-gray-200">
          <Button 
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
            className="h-12 px-8 bg-gradient-to-r from-indigo-600 to-blue-500 hover:from-indigo-700 hover:to-blue-600 text-white font-medium rounded-lg shadow-md hover:shadow-lg transition-all"
          >
            {isLoading ? "Processing..." : "Process & Continue"}
          </Button>
        </div>
      </form>
    </div>
  );
};

export default ProcessStep;
