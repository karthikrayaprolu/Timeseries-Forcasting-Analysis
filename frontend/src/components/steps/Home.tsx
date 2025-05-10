import React from "react";
import { Button } from "@/components/ui/button";
import { Rocket, Database, LineChart, Settings } from "lucide-react";

const Home = () => {
  return (
    <div className="bg-gray-900 min-h-screen text-gray-100">
      {/* Hero Section */}
      <section className="container mx-auto px-6 py-20 text-center">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            Time Series Forecasting Platform
          </h1>
          <p className="text-lg md:text-xl text-gray-300 mb-10">
            Powerful tools for analyzing and predicting time series data with machine learning
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Button className="bg-indigo-600 hover:bg-indigo-700 px-8 py-6 text-lg">
              <Rocket className="mr-2" />
              Get Started
            </Button>
            <Button variant="outline" className="border-gray-600 text-gray-200 hover:bg-gray-800 px-8 py-6 text-lg">
              <Settings className="mr-2" />
              Learn More
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-gray-800 py-16">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: <Database size={32} className="text-amber-200" />,
                title: "Data Management",
                description: "Easily register and organize your time series datasets"
              },
              {
                icon: <Settings size={32} className="text-amber-200" />,
                title: "Processing Tools",
                description: "Clean, transform and prepare your data for analysis"
              },
              {
                icon: <LineChart size={32} className="text-amber-200" />,
                title: "Advanced Forecasting",
                description: "Build and train models with state-of-the-art algorithms"
              }
            ].map((feature, index) => (
              <div key={index} className="bg-gray-700 p-6 rounded-lg hover:bg-gray-600 transition-all">
                <div className="flex justify-center mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-3 text-center">{feature.title}</h3>
                <p className="text-gray-300 text-center">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to get started?</h2>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Join thousands of analysts using our platform for accurate time series forecasting
          </p>
          <Button className="bg-amber-500 hover:bg-amber-600 text-gray-900 px-10 py-6 text-lg font-bold">
            Start Your Free Trial
          </Button>
        </div>
      </section>
    </div>
  );
};

export default Home;