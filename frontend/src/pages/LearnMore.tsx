import React from "react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { Database, FileInput, Brain, BarChart, ArrowRight } from "lucide-react";

const LearnMore = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Database,
      title: "Database Integration",
      description: "Seamlessly connect and manage your time series data from various sources. Our platform supports multiple database formats and provides easy data import capabilities."
    },
    {
      icon: FileInput,
      title: "Data Processing",
      description: "Powerful data preprocessing tools to clean, transform, and prepare your time series data for analysis. Handle missing values, outliers, and data normalization with ease."
    },
    {
      icon: Brain,
      title: "Advanced Forecasting",
      description: "State-of-the-art machine learning models for accurate time series forecasting. Choose from various algorithms including ARIMA, Prophet, and deep learning models."
    },
    {
      icon: BarChart,
      title: "Visualization & Analysis",
      description: "Interactive visualizations and comprehensive analysis tools to understand your data patterns and forecast results. Generate detailed reports and insights."
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-white">
        <div className="container mx-auto px-6 py-16">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl md:text-6xl">
              Time Series Forecasting
              <span className="text-indigo-600"> Made Simple</span>
            </h1>
            <p className="mt-4 text-xl text-gray-600 max-w-3xl mx-auto">
              Transform your time series data into accurate predictions with our powerful forecasting platform.
              Designed for both beginners and experts in data analysis.
            </p>
            <div className="mt-8 flex justify-center gap-4">
              <Button
                size="lg"
                onClick={() => navigate("/auth/signup")}
                className="bg-indigo-600 hover:bg-indigo-700"
              >
                Get Started
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={() => navigate("/")}
              >
                Back to Home
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="container mx-auto px-6 py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300"
            >
              <div className="flex items-center gap-4 mb-4">
                <div className="p-3 bg-indigo-100 rounded-lg">
                  <feature.icon className="h-6 w-6 text-indigo-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900">
                  {feature.title}
                </h3>
              </div>
              <p className="text-gray-600">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* How It Works Section */}
      <div className="bg-gray-900 text-white">
        <div className="container mx-auto px-6 py-16">
          <h2 className="text-3xl font-bold text-center mb-12">
            How It Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-indigo-600 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-xl font-bold">1</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Connect Your Data</h3>
              <p className="text-gray-400">
                Import your time series data from various sources or use our sample datasets
              </p>
            </div>
            <div className="text-center">
              <div className="bg-indigo-600 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-xl font-bold">2</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Train Models</h3>
              <p className="text-gray-400">
                Choose and configure forecasting models to analyze your data
              </p>
            </div>
            <div className="text-center">
              <div className="bg-indigo-600 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-xl font-bold">3</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Get Insights</h3>
              <p className="text-gray-400">
                View predictions, analyze results, and export your forecasts
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-indigo-600 text-white">
        <div className="container mx-auto px-6 py-16 text-center">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Start Forecasting?
          </h2>
          <p className="text-xl mb-8 text-indigo-100">
            Join thousands of users who trust our platform for their time series forecasting needs
          </p>
          <Button
            size="lg"
            variant="secondary"
            onClick={() => navigate("/auth/signup")}
            className="bg-white text-indigo-600 hover:bg-gray-100"
          >
            Create Free Account
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default LearnMore; 