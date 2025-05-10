import React, { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Rocket, Database, LineChart, Settings, ChevronRight } from "lucide-react";
import { useWorkflow } from "@/contexts/WorkflowContext";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { lazy, Suspense } from "react";

// Dynamic import for heavier animation components
const AnimatedChart = lazy(() => import('../AnimatedChart'));

const Home = () => {
  const navigate = useNavigate();
  const { setCurrentStep } = useWorkflow();

  // Animation variants
  const fadeIn = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 0.8 } }
  };

  const slideUp = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.6 } }
  };

  return (
    <div className="bg-white min-h-screen">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-indigo-50 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob"></div>
        <div className="absolute top-1/3 right-1/4 w-64 h-64 bg-amber-50 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-2000"></div>
        <div className="absolute bottom-1/4 left-1/2 w-64 h-64 bg-blue-50 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-4000"></div>
      </div>

      {/* Hero Section */}
      <section className="relative container mx-auto px-6 py-24 text-center">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={fadeIn}
          className="max-w-4xl mx-auto"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-indigo-800 to-blue-600">
            Time Series Forecasting Platform
          </h1>
          <p className="text-lg md:text-xl text-gray-600 mb-10">
            Powerful tools for analyzing and predicting time series data with machine learning
          </p>
          <motion.div variants={slideUp} className="flex flex-col sm:flex-row justify-center gap-4">
            <Button 
              className="bg-gradient-to-r from-indigo-600 to-blue-500 hover:from-indigo-700 hover:to-blue-600 px-8 py-6 text-lg text-white shadow-lg hover:shadow-xl transition-all"
              onClick={() => setCurrentStep("database")}
            >
              <Rocket className="mr-2" />
              Get Started
              <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
            <Button 
              variant="outline" 
              className="border-amber-400 text-amber-600 hover:bg-amber-50 px-8 py-6 text-lg hover:border-amber-500 transition-all"
              onClick={() => navigate('/learn-more')}
            >
              <Settings className="mr-2" />
              Learn More
            </Button>
          </motion.div>
        </motion.div>

        {/* Animated chart preview */}
        <motion.div 
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mt-16 mx-auto max-w-4xl"
        >
          <Suspense fallback={<div>Loading...</div>}>
            <AnimatedChart />
          </Suspense>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="relative bg-gradient-to-b from-white to-indigo-50 py-24">
        <div className="container mx-auto px-6">
          <motion.h2 
            initial="hidden"
            whileInView="visible"
            variants={slideUp}
            viewport={{ once: true }}
            className="text-3xl font-bold text-center mb-16 text-indigo-900"
          >
            Key Features
          </motion.h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: <Database size={32} className="text-amber-500" />,
                title: "Data Management",
                description: "Easily register and organize your time series datasets",
                color: "from-indigo-100 to-indigo-50"
              },
              {
                icon: <Settings size={32} className="text-blue-500" />,
                title: "Processing Tools",
                description: "Clean, transform and prepare your data for analysis",
                color: "from-amber-100 to-amber-50"
              },
              {
                icon: <LineChart size={32} className="text-indigo-500" />,
                title: "Advanced Forecasting",
                description: "Build and train models with state-of-the-art algorithms",
                color: "from-blue-100 to-blue-50"
              }
            ].map((feature, index) => (
              <motion.div 
                key={index}
                initial="hidden"
                whileInView="visible"
                variants={slideUp}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className={`bg-gradient-to-br ${feature.color} p-8 rounded-xl shadow-sm hover:shadow-md transition-all border border-gray-100`}
              >
                <div className="flex justify-center mb-6">
                  <div className="p-4 bg-white rounded-full shadow-sm">
                    {feature.icon}
                  </div>
                </div>
                <h3 className="text-xl font-semibold mb-3 text-center text-indigo-800">{feature.title}</h3>
                <p className="text-gray-600 text-center">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative py-24 bg-gradient-to-b from-indigo-50 to-white">
        <div className="container mx-auto px-6 text-center">
          <motion.h2 
            initial="hidden"
            whileInView="visible"
            variants={slideUp}
            viewport={{ once: true }}
            className="text-3xl font-bold mb-6 text-indigo-900"
          >
            Ready to get started?
          </motion.h2>
          <motion.p 
            initial="hidden"
            whileInView="visible"
            variants={slideUp}
            transition={{ delay: 0.1 }}
            viewport={{ once: true }}
            className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto"
          >
            Join thousands of analysts using our platform for accurate time series forecasting
          </motion.p>
          <motion.div 
            initial="hidden"
            whileInView="visible"
            variants={slideUp}
            transition={{ delay: 0.2 }}
            viewport={{ once: true }}
            className="flex flex-col sm:flex-row justify-center gap-4"
          >
            <Button 
              className="bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700 text-white px-10 py-6 text-lg font-bold shadow-lg hover:shadow-xl transition-all"
              onClick={() => navigate('/auth/signup')}
            >
              Start Your Free Trial
            </Button>
            <Button 
              variant="outline" 
              className="border-indigo-400 text-indigo-600 hover:bg-indigo-50 px-10 py-6 text-lg hover:border-indigo-500 transition-all"
              onClick={() => navigate('/auth/login')}
            >
              Login to Existing Account
            </Button>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;