import React, { useEffect, useRef, useState } from "react";
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
import Plot from "react-plotly.js";
import { Info, TrendingUp, TrendingDown, Calendar, Target, Clock, Search, BarChart3, Plus, ArrowUp, ArrowDown, Sparkles } from "lucide-react";

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
      Please wait while we analyze your data...
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

type CorrelationMap = { [feature: string]: number };

// Helper function to get relationship strength description
const getRelationshipStrength = (correlation: number) => {
  const abs = Math.abs(correlation);
  if (abs >= 0.7) return { strength: "Very Strong", color: "text-green-600" };
  if (abs >= 0.5) return { strength: "Strong", color: "text-blue-600" };
  if (abs >= 0.3) return { strength: "Moderate", color: "text-yellow-600" };
  if (abs >= 0.1) return { strength: "Weak", color: "text-orange-600" };
  return { strength: "Very Weak", color: "text-red-600" };
};

// Helper function to get relationship direction
const getRelationshipDirection = (correlation: number) => {
  if (correlation > 0.1) return { direction: "Positive", icon: TrendingUp, color: "text-green-500" };
  if (correlation < -0.1) return { direction: "Negative", icon: TrendingDown, color: "text-red-500" };
  return { direction: "Neutral", icon: Info, color: "text-gray-500" };
};

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

  // New state to store correlations
  const [correlations, setCorrelations] = useState<CorrelationMap>({});
  const [isCorrLoading, setIsCorrLoading] = useState(false);

  useEffect(() => {
    if (componentRef.current) {
      gsap.fromTo(
        componentRef.current,
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }
      );
    }
  }, []);

  // Load columns when database.table changes
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

  // Load correlations whenever targetVariable or columns change
  useEffect(() => {
    const loadRelationships = async () => {
      if (!process.targetVariable) return;
      setIsCorrLoading(true);
      try {
        const corrData: CorrelationMap = await api.getCorrelations({
          targetVariable: process.targetVariable,
        });

        setCorrelations(corrData);
      } catch (err) {
        console.error("Failed to load relationships:", err);
        toast.error("Failed to analyze feature relationships");
        setCorrelations({});
      } finally {
        setIsCorrLoading(false);
      }
    };
    loadRelationships();
  }, [process.targetVariable, database.table]);

  const handleProcess = async () => {
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
        aggregationMethod: process.aggregationMethod,
      });

      if (processResult.preview) {
        toast.success(
          `Data processed successfully. ${processResult.preview.rows} rows processed.`
        );
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

  // Prepare heatmap data for Plotly with user-friendly labels
  const heatmapData = [
    {
      z: availableColumns
        .filter(
          (col) => col !== process.timeColumn && col !== process.targetVariable
        )
        .map((col) => [correlations[col] ?? 0]),
      x: [process.targetVariable || "What You Want to Predict"],
      y: availableColumns.filter(
        (col) => col !== process.timeColumn && col !== process.targetVariable
      ),
      type: "heatmap",
      colorscale: [
        [0, '#ff4444'],     // Red for negative
        [0.5, '#ffffff'],   // White for neutral
        [1, '#4444ff']      // Blue for positive
      ],
      colorbar: {
        title: "Relationship<br>Strength",
        titleside: "right",
        tickvals: [-1, -0.5, 0, 0.5, 1],
        ticktext: ["Strong Negative", "Weak Negative", "No Relationship", "Weak Positive", "Strong Positive"]
      },
      hovertemplate:
        "<b>%{y}</b><br>Relationship: %{z:.2f}<br><extra></extra>",
    },
  ];

  return (
    <div
      ref={componentRef}
      className="workflow-step max-w-6xl mx-auto p-8 bg-white rounded-xl shadow-lg"
    >
      <h2 className="text-3xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-indigo-800 to-blue-600">
        Configure Your Prediction Settings
      </h2>

      {(isLoading || isCorrLoading) && <WaveBarsLoader />}

      <form
        className={`${isLoading || isCorrLoading ? "pointer-events-none opacity-60" : ""}`}
        onSubmit={(e) => {
          e.preventDefault();
          handleProcess();
        }}
        aria-disabled={isLoading || isCorrLoading}
      >
        <div className="grid grid-cols-2 gap-8">
          {/* Left Column */}
          <div className="space-y-6">
            {/* Time Column Selection */}
            <div className="bg-gray-50 p-6 rounded-xl border-2 border-gray-200">
              <label className="text-lg font-medium text-gray-700 flex items-center gap-2 mb-2">
                <Calendar className="w-5 h-5 text-indigo-600" />
                Date/Time Column
              </label>
              <p className="text-sm text-gray-500 mb-3">
                Which column contains your dates or time information?
              </p>
              <Select
                value={process.timeColumn}
                onValueChange={(value) =>
                  setProcess({ ...process, timeColumn: value })
                }
                disabled={isLoading || isCorrLoading}
              >
                <SelectTrigger className="w-full h-12 bg-white border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 rounded-lg transition-all">
                  <SelectValue placeholder="Choose your date/time column" />
                </SelectTrigger>
                <SelectContent className="bg-white border-2 border-gray-200 rounded-lg shadow-lg">
                  {availableColumns
                    .filter(
                      (col) =>
                        col.toLowerCase().includes("month") ||
                        col.toLowerCase().includes("date") ||
                        col.toLowerCase().includes("time")
                    )
                    .map((column) => (
                      <SelectItem
                        key={column}
                        value={column}
                        className="hover:bg-indigo-50 cursor-pointer"
                      >
                        {column}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>

            {/* Target Variable Selection */}
            <div className="bg-gray-50 p-6 rounded-xl border-2 border-gray-200">
              <label className="text-lg font-medium text-gray-700 flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-indigo-600" />
                What Do You Want to Predict?
              </label>
              <p className="text-sm text-gray-500 mb-3">
                Choose the main thing you want to forecast (like sales, revenue, etc.)
              </p>
              <Select
                value={process.targetVariable}
                onValueChange={(value) =>
                  setProcess({ ...process, targetVariable: value })
                }
                disabled={isLoading || isCorrLoading}
              >
                <SelectTrigger className="w-full h-12 bg-white border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 rounded-lg transition-all">
                  <SelectValue placeholder="Select what you want to predict" />
                </SelectTrigger>
                <SelectContent className="bg-white border-2 border-gray-200 rounded-lg shadow-lg">
                  {availableColumns
                    .filter(
                      (col) =>
                        !col.toLowerCase().includes("date") &&
                        !col.toLowerCase().includes("time") &&
                        col !== process.timeColumn
                    )
                    .map((column) => (
                      <SelectItem
                        key={column}
                        value={column}
                        className="hover:bg-indigo-50 cursor-pointer"
                      >
                        {column}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>

            {/* Time Frequency Selection */}
            <div className="bg-gray-50 p-6 rounded-xl border-2 border-gray-200">
              <label className="text-lg font-medium text-gray-700 flex items-center gap-2 mb-2">
                <Clock className="w-5 h-5 text-indigo-600" />
                How Often Do You Want Predictions?
              </label>
              <p className="text-sm text-gray-500 mb-3">
                Choose how frequently you want to make predictions
              </p>
              <div className="space-y-3">
                <Select
                  value={process.frequency}
                  onValueChange={(value) =>
                    setProcess({
                      ...process,
                      frequency: value as "daily" | "weekly" | "monthly",
                    })
                  }
                  disabled={isLoading || isCorrLoading}
                >
                  <SelectTrigger className="w-full h-12 bg-white border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 rounded-lg transition-all">
                    <SelectValue placeholder="Choose prediction frequency" />
                  </SelectTrigger>
                  <SelectContent className="bg-white border-2 border-gray-200 rounded-lg shadow-lg">
                    <SelectItem
                      value="daily"
                      className="hover:bg-indigo-50 cursor-pointer"
                    >
                      <div className="flex items-center gap-2">
                        <BarChart3 className="w-4 h-4" />
                        Daily Predictions
                      </div>
                    </SelectItem>
                    <SelectItem
                      value="weekly"
                      className="hover:bg-indigo-50 cursor-pointer"
                    >
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-4 h-4" />
                        Weekly Predictions
                      </div>
                    </SelectItem>
                    <SelectItem
                      value="monthly"
                      className="hover:bg-indigo-50 cursor-pointer"
                    >
                      <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4" />
                        Monthly Predictions
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>

                {(process.frequency === "weekly" ||
                  process.frequency === "monthly") && (
                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                    <p className="text-sm text-blue-800 mb-2">
                      How should we combine your daily data into {process.frequency} summaries?
                    </p>
                    <Select
                      value={process.aggregationMethod}
                      onValueChange={(value) =>
                        setProcess({
                          ...process,
                          aggregationMethod: value as "mean" | "sum" | "max" | "min",
                        })
                      }
                      disabled={isLoading || isCorrLoading}
                    >
                      <SelectTrigger className="w-full h-12 bg-white border-2 border-blue-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 rounded-lg transition-all">
                        <SelectValue placeholder="Choose how to combine data" />
                      </SelectTrigger>
                      <SelectContent className="bg-white border-2 border-blue-200 rounded-lg shadow-lg">
                        <SelectItem
                          value="mean"
                          className="hover:bg-blue-50 cursor-pointer"
                        >
                          <div className="flex items-center gap-2">
                            <BarChart3 className="w-4 h-4" />
                            Average (Most Common)
                          </div>
                        </SelectItem>
                        <SelectItem
                          value="sum"
                          className="hover:bg-blue-50 cursor-pointer"
                        >
                          <div className="flex items-center gap-2">
                            <Plus className="w-4 h-4" />
                            Add Up All Values
                          </div>
                        </SelectItem>
                        <SelectItem
                          value="max"
                          className="hover:bg-blue-50 cursor-pointer"
                        >
                          <div className="flex items-center gap-2">
                            <ArrowUp className="w-4 h-4" />
                            Highest Value
                          </div>
                        </SelectItem>
                        <SelectItem
                          value="min"
                          className="hover:bg-blue-50 cursor-pointer"
                        >
                          <div className="flex items-center gap-2">
                            <ArrowDown className="w-4 h-4" />
                            Lowest Value
                          </div>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column - Additional Features */}
          <div className="bg-gray-50 p-6 rounded-xl border-2 border-gray-200 h-full flex flex-col">
            <div className="mb-4">
              <label className="text-lg font-medium text-gray-700 flex items-center gap-2 mb-2">
                <Search className="w-5 h-5 text-indigo-600" />
                Additional Information to Help Predictions
              </label>
              <div className="bg-blue-50 p-3 rounded-lg border border-blue-200 mb-4">
                <p className="text-sm text-blue-800">
                  <Info className="inline w-4 h-4 mr-1" />
                  Select other data that might help predict your target. We've sorted them by how much they relate to what you want to predict.
                </p>
              </div>
            </div>
            
            <div className="grid grid-cols-1 gap-3 max-h-[300px] overflow-y-auto mb-6">
              {availableColumns
                .filter(
                  (col) =>
                    col !== process.timeColumn && col !== process.targetVariable
                )
                .sort((a, b) => Math.abs((correlations[b] ?? 0)) - Math.abs((correlations[a] ?? 0)))
                .map((column) => {
                  const correlation = correlations[column] ?? 0;
                  const strength = getRelationshipStrength(correlation);
                  const direction = getRelationshipDirection(correlation);
                  const DirectionIcon = direction.icon;
                  
                  return (
                    <div
                      key={column}
                      className={`flex items-center justify-between bg-white p-4 rounded-lg border-2 transition-all hover:shadow-md ${
                        process.features.includes(column) 
                          ? 'border-indigo-300 bg-indigo-50' 
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex items-center space-x-3 flex-1">
                        <Checkbox
                          id={`feature-${column}`}
                          checked={process.features.includes(column)}
                          onCheckedChange={() => handleFeatureToggle(column)}
                          disabled={isLoading || isCorrLoading}
                          className="h-5 w-5 border-2 border-gray-300 rounded-md checked:bg-indigo-600 checked:border-indigo-600 focus:ring-2 focus:ring-indigo-200"
                        />
                        <div className="flex-1">
                          <label
                            htmlFor={`feature-${column}`}
                            className="text-sm font-medium text-gray-700 leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                          >
                            {column}
                          </label>
                          {correlations[column] !== undefined && (
                            <div className="flex items-center space-x-2 mt-1">
                              <DirectionIcon className={`w-3 h-3 ${direction.color}`} />
                              <span className={`text-xs ${strength.color} font-medium`}>
                                {strength.strength} {direction.direction === "Neutral" ? "No Clear" : direction.direction} Relationship
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
            </div>

            {/* Relationship Visualization */}
            {process.targetVariable && !isCorrLoading && Object.keys(correlations).length === 0 && (
              <div className="text-center p-6 bg-yellow-50 rounded-lg border border-yellow-200">
                <Info className="w-6 h-6 text-yellow-600 mx-auto mb-2" />
                <p className="text-sm text-yellow-700">
                  No relationship data available yet. Try choosing a different target variable to see how other factors relate to it.
                </p>
              </div>
            )}

            {process.targetVariable && Object.keys(correlations).length > 0 && (
              <>
                <h3 className="text-md font-semibold text-gray-700 flex items-center gap-2 mb-2">
                  <BarChart3 className="w-5 h-5 text-indigo-600" />
                  Data Relationships Visualization
                </h3>
                <p className="text-xs text-gray-600 mb-2">
                  This chart shows how strongly each factor relates to what you want to predict
                </p>
                <div className="w-full h-64">
                  <Plot
                    data={heatmapData}
                    layout={{
                      autosize: true,
                      margin: { t: 30, b: 50, l: 100, r: 50 },
                      yaxis: {
                        automargin: true,
                        tickfont: { size: 10 },
                      },
                      xaxis: {
                        automargin: true,
                        tickfont: { size: 12 },
                      },
                      paper_bgcolor: "#f9fafb",
                      plot_bgcolor: "#f9fafb",
                    }}
                    config={{ displayModeBar: false }}
                    style={{ width: "100%", height: "100%" }}
                  />
                </div>
              </>
            )}
          </div>
        </div>

        <div className="flex justify-between mt-8">
          <Button
            variant="outline"
            onClick={handleBack}
            disabled={isLoading || isCorrLoading}
            className="px-6 py-2"
          >
            ← Go Back
          </Button>
          <Button
            type="submit"
            disabled={isLoading || isCorrLoading}
            className="bg-indigo-700 hover:bg-indigo-800 text-white px-8 py-2 flex items-center gap-2"
          >
            Process My Data
          </Button>
        </div>
      </form>
    </div>
  );
};

export default ProcessStep;