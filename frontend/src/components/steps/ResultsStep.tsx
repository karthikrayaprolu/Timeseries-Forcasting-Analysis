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
import { FileDown } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

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
      <div className="workflow-step max-w-5xl mx-auto p-6 bg-white rounded-xl shadow-lg">
        <h2 className="text-3xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-indigo-800 to-blue-600">Results</h2>
        <div className="text-center">
          <p className="text-gray-600 mb-4">No results available. Please train a model first.</p>
          <Button 
            onClick={() => setCurrentStep("train")} 
            className="h-12 px-8 bg-gradient-to-r from-indigo-600 to-blue-500 hover:from-indigo-700 hover:to-blue-600 text-white font-medium rounded-lg shadow-md hover:shadow-lg transition-all"
          >
            Go to Training
          </Button>
        </div>
      </div>
    );
  }

  // Format data for the chart
  const chartData = React.useMemo(() => {
    if (!results?.dates?.length) return [];
    return results.dates.map((date, index) => ({
      date: (date || '').slice(0, 7) || '',
      Actual: Number(results.actual?.[index] ?? 0),
      Predicted: Number(results.forecasts?.[index] ?? 0),
    }));
  }, [results?.dates, results?.actual, results?.forecasts]);

  const handleExport = async (format: 'csv' | 'excel' | 'json') => {
    if (isLoading || !results) return;
    
    setIsLoading(true);
    try {
      const exportData = {
        Results: {
          modelInfo: results.modelInfo,
          metrics: results.metrics,
          data: results.dates.map((date, i) => ({
            date,
            actual: results.actual[i],
            forecast: results.forecasts[i],
            error: results.actual[i] - results.forecasts[i]
          }))
        }
      };
      
      await api.exportResults(format, exportData);
      toast.success(`Results exported successfully as ${format.toUpperCase()}`);
    } catch (error) {
      console.error("Export error:", error);
      toast.error(`Failed to export results as ${format.toUpperCase()}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRestart = () => setCurrentStep("database");
  const handleBack = () => setCurrentStep("train");
  const { modelInfo } = results;

  // Metrics for cards
  const metrics = [
    { label: "MSE", value: Number(results.metrics?.mse || 0).toFixed(2) },
    { label: "RMSE", value: Number(results.metrics?.rmse || 0).toFixed(2) },
    { label: "MAE", value: Number(results.metrics?.mae || 0).toFixed(2) },
    { label: "MAPE", value: Number(results.metrics?.mape || 0).toFixed(2) + "%" },
  ];

  // Export card content
  const exportButtons = [
    { format: "csv", label: "CSV" },
    { format: "excel", label: "Excel" },
    { format: "json", label: "JSON" },
  ];

  return (
    <div ref={componentRef} className="workflow-step max-w-5xl mx-auto p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-3xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-indigo-800 to-blue-600">Forecasting Results</h2>
      <div className="space-y-8">
        {/* Settings Cards: Basic & Advanced side by side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Basic Settings Card */}
          <div className="bg-gray-50 p-6 rounded-lg border-2 border-gray-200 flex flex-col">
            <h3 className="text-xl font-semibold mb-4 text-gray-800">Basic Settings</h3>
            <div className="space-y-3 text-sm">
              <p className="flex items-center gap-2">
                <span className="font-medium text-gray-700">Model Type:</span>
                <span className="text-gray-600">{results.modelInfo?.type || 'N/A'}</span>
              </p>
              <p className="flex items-center gap-2">
                <span className="font-medium text-gray-700">Data Source:</span>
                <span className="text-gray-600">{results.dataInfo?.filename || 'N/A'}</span>
              </p>
              <p className="flex items-center gap-2">
                <span className="font-medium text-gray-700">Target Variable:</span>
                <span className="text-gray-600">{results.dataInfo?.title || 'N/A'}</span>
              </p>
            </div>
          </div>
          {/* Advanced Features Card */}
          <div className="bg-gray-50 p-6 rounded-lg border-2 border-gray-200">
            <h3 className="text-xl font-semibold mb-4 text-gray-800">Advanced Features</h3>
            <div className="space-y-3 text-sm">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${results.modelInfo?.features?.hyperparameterTuning ? 'bg-green-500' : 'bg-gray-300'}`} />
                <span className="text-gray-700">Hyperparameter Tuning</span>
              </div>
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${results.modelInfo?.features?.transferLearning ? 'bg-green-500' : 'bg-gray-300'}`} />
                <span className="text-gray-700">Transfer Learning</span>
              </div>
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${results.modelInfo?.features?.ensembleLearning ? 'bg-green-500' : 'bg-gray-300'}`} />
                <span className="text-gray-700">Ensemble Learning</span>
              </div>
            </div>
          </div>
        </div>

        {/* Metrics Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {metrics.map((metric) => (
            <div key={metric.label} className="metric-card p-6 rounded-lg bg-gray-50 border-2 border-gray-200 flex flex-col items-center shadow-sm hover:shadow-md transition-all">
              <div className="text-sm font-medium text-gray-500 mb-1">{metric.label}</div>
              <div className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-blue-500">{metric.value}</div>
            </div>
          ))}
        </div>
        
        {/* Forecast Chart
        <Card className="shadow-lg border-2 border-gray-200">
          <CardHeader className="space-y-2">
            <CardTitle className="text-xl font-semibold text-gray-800">Forecast Visualization</CardTitle>
            <CardDescription className="text-gray-500">
              Historical data vs. predicted values
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[400px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#6b7280"
                    tick={{ fill: '#6b7280' }}
                  />
                  <YAxis 
                    stroke="#6b7280"
                    tick={{ fill: '#6b7280' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'white',
                      border: '2px solid #e5e7eb',
                      borderRadius: '0.5rem',
                      boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="Actual"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={{ r: 3, fill: '#6366f1' }}
                    activeDot={{ r: 5, fill: '#6366f1' }}
                    name="Historical Data"
                  />
                  <Line
                    type="monotone"
                    dataKey="Predicted"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ r: 3, fill: '#3b82f6' }}
                    activeDot={{ r: 5, fill: '#3b82f6' }}
                    name="Forecast"
                    strokeDasharray="5 5"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card> */}

        {/* Export Card */}
        <Card className="shadow-lg border-2 border-gray-200">
          <CardHeader className="space-y-2">
            <CardTitle className="text-xl font-semibold text-gray-800">Export Results</CardTitle>
            <CardDescription className="text-gray-500">
              Download your results in different formats
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4">
              {exportButtons.map(({ format, label }) => (
                <Button
                  key={format}
                  variant="outline"
                  className="h-12 flex items-center justify-center gap-3 text-base font-medium border-2 border-gray-200 hover:border-gray-300 text-gray-700 rounded-lg transition-all"
                  disabled={isLoading}
                  onClick={() => handleExport(format as 'csv' | 'excel' | 'json')}
                >
                  <FileDown className="h-5 w-5" />
                  <span>Export as {label}</span>
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row justify-between gap-4 pt-6">
          <Button
            onClick={handleBack}
            variant="outline"
            disabled={isLoading}
            className="h-12 px-8 border-2 border-gray-200 hover:border-gray-300 text-gray-700 font-medium rounded-lg transition-all"
          >
            Back
          </Button>
          <Button
            onClick={handleRestart}
            className="h-12 px-8 bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700 text-white font-medium rounded-lg shadow-md hover:shadow-lg transition-all"
            disabled={isLoading}
          >
            Start New Forecast
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ResultsStep;
