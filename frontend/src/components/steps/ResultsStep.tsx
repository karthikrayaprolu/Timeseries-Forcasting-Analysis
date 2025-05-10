import React, { useEffect, useRef } from "react";
import { useWorkflow } from "@/contexts/WorkflowContext";
import { api } from "@/services/api";
import { gsap } from "gsap";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const ResultsStep = () => {
  const { results, setCurrentStep, isLoading, setIsLoading } = useWorkflow();
  const componentRef = useRef<HTMLDivElement>(null);

  // GSAP animation
  useEffect(() => {
    if (componentRef.current) {
      gsap.fromTo(
        componentRef.current,
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }
      );
      gsap.fromTo(
        ".metric-card, .export-card",
        { scale: 0.9, opacity: 0 },
        { scale: 1, opacity: 1, duration: 0.5, stagger: 0.1, ease: "back.out(1.7)" }
      );
    }
  }, []);

  if (!results) {
    return (
      <div className="workflow-step text-center">
        <h2 className="step-title">Results</h2>
        <p>No results available. Please train a model first.</p>
        <Button onClick={() => setCurrentStep("train")} className="mt-4">
          Go to Training
        </Button>
      </div>
    );
  }

  // Format data for the chart
  const chartData = results.forecasts.dates.map((date, index) => {
    const actualValue = results.forecasts.actual[index];
    const predictedValue = results.forecasts.predicted[index];
    return {
      date: date.slice(0, 7),
      Actual: actualValue !== null ? Number(actualValue) : null,
      Predicted: !isNaN(Number(predictedValue)) ? Number(predictedValue) : null,
    };
  }).filter(data => data.Actual !== null || data.Predicted !== null);

  const handleExport = async (format) => {
    setIsLoading(true);
    try {
      const exportResult = await api.exportResults(format, results);
      toast.success(exportResult.message);
    } catch (error) {
      console.error("Error exporting results:", error);
      toast.error("Failed to export results");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRestart = () => setCurrentStep("database");
  const handleBack = () => setCurrentStep("train");
  const { modelInfo } = results;

  // Metrics for cards
  const metrics = [
    { label: "MSE", value: Number(results.metrics.mse).toFixed(2) },
    { label: "RMSE", value: Number(results.metrics.rmse).toFixed(2) },
    { label: "MAE", value: Number(results.metrics.mae).toFixed(2) },
    { label: "MAPE", value: Number(results.metrics.mape).toFixed(2) + "%" },
  ];

  return (
    <div ref={componentRef} className="workflow-step w-full">
      <h2 className="step-title mb-8 text-center">Forecasting Results</h2>
      <div className="space-y-8">

        {/* Settings Cards: Basic & Advanced side by side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Basic Settings Card */}
          <div className="bg-white p-6 rounded-lg shadow-md border border-border flex flex-col">
            <h3 className="text-lg font-medium mb-4">Basic Settings</h3>
            <div className="space-y-2 text-sm">
              <p><span className="font-medium">Model Type:</span> {modelInfo?.type}</p>
              <p><span className="font-medium">Data Source:</span> {results.dataInfo.filename}</p>
              <p><span className="font-medium">Target Variable:</span> {results.dataInfo.title}</p>
            </div>
          </div>
          {/* Advanced Features Card */}
          <div className="bg-white p-6 rounded-lg shadow-md border border-border flex flex-col">
            <h3 className="text-lg font-medium mb-4">Advanced Features</h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${modelInfo?.features.hyperparameterTuning ? 'bg-green-500' : 'bg-gray-300'}`} />
                <span>Hyperparameter Tuning</span>
              </div>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${modelInfo?.features.transferLearning ? 'bg-green-500' : 'bg-gray-300'}`} />
                <span>Transfer Learning</span>
              </div>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${modelInfo?.features.ensembleLearning ? 'bg-green-500' : 'bg-gray-300'}`} />
                <span>Ensemble Learning</span>
              </div>
            </div>
          </div>
        </div>

        {/* Metrics Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {metrics.map((metric) => (
            <div key={metric.label} className="metric-card p-4 rounded-lg bg-gray-50 shadow border border-border flex flex-col items-center">
              <div className="text-sm text-muted-foreground">{metric.label}</div>
              <div className="text-2xl font-bold text-primary">{metric.value}</div>
            </div>
          ))}
        </div>

        {/* Forecast Chart */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-border flex flex-col mb-8">
          <h3 className="text-lg font-medium mb-4">{results.dataInfo.title} Forecast</h3>
          <p className="text-sm text-muted-foreground mb-4">Based on data from: {results.dataInfo.filename}</p>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 12 }}
                  interval={2}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis
                  label={{
                    value: results.dataInfo.title,
                    angle: -90,
                    position: "insideLeft",
                    style: { textAnchor: "middle" },
                  }}
                />
                <Tooltip
                  formatter={(value) => {
                    if (value === null || isNaN(value)) return ["N/A", ""];
                    return [Number(value).toFixed(2), ""];
                  }}
                  labelFormatter={(label) => `Date: ${label}`}
                  contentStyle={{ backgroundColor: "white", border: "1px solid #ccc" }}
                />
                <Legend verticalAlign="top" height={36} wrapperStyle={{ paddingBottom: "20px" }} />
                <Line
                  type="monotone"
                  dataKey="Actual"
                  stroke="#8884d8"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  activeDot={{ r: 4 }}
                  name="Historical Data"
                />
                <Line
                  type="monotone"
                  dataKey="Predicted"
                  stroke="#82ca9d"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  activeDot={{ r: 4 }}
                  name="Forecast"
                  strokeDasharray="5 5"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Export Buttons as Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
          <button
            className="export-card p-4 rounded-lg bg-white shadow-md border border-border flex flex-col items-center justify-center font-semibold hover:bg-blue-50 transition"
            onClick={() => handleExport("csv")}
            disabled={isLoading}
            type="button"
          >
            Export as CSV
          </button>
          <button
            className="export-card p-4 rounded-lg bg-white shadow-md border border-border flex flex-col items-center justify-center font-semibold hover:bg-blue-50 transition"
            onClick={() => handleExport("excel")}
            disabled={isLoading}
            type="button"
          >
            Export as Excel
          </button>
          <button
            className="export-card p-4 rounded-lg bg-white shadow-md border border-border flex flex-col items-center justify-center font-semibold hover:bg-blue-50 transition"
            onClick={() => handleExport("json")}
            disabled={isLoading}
            type="button"
          >
            Export as JSON
          </button>
        </div>

        {/* Action Buttons: Back (left) & Start New Forecast (right) */}
        <div className="flex flex-col sm:flex-row justify-between gap-4 pt-6">
          <Button
            onClick={handleBack}
            variant="outline"
            disabled={isLoading}
            className="w-full sm:w-1/2 max-w-xs"
          >
            Back
          </Button>
          <Button
            onClick={handleRestart}
            variant="secondary"
            disabled={isLoading}
            className="w-full sm:w-1/2 max-w-xs"
          >
            Start New Forecast
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ResultsStep;
