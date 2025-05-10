import React, { useEffect } from "react";
import { useWorkflow } from "@/contexts/WorkflowContext";
import WorkflowHeader from "./WorkflowHeader";
import Home from "./steps/Home";
import DatabaseStep from "./steps/DatabaseStep";
import ProcessStep from "./steps/ProcessStep";
import TrainStep from "./steps/TrainStep";
import ResultsStep from "./steps/ResultsStep";
import { gsap } from "gsap";

const Dashboard = () => {
  const { currentStep } = useWorkflow();

  // GSAP timeline for container animations
  useEffect(() => {
    // Create a timeline for the step transitions
    const tl = gsap.timeline();
    
    // Initial animation of the dashboard
    tl.fromTo(
      ".dashboard-container",
      { opacity: 0, scale: 0.95 },
      { 
        opacity: 1, 
        scale: 1, 
        duration: 0.8, 
        ease: "power3.out",
      }
    );
    
    return () => {
      tl.kill();
    };
  }, []);

  // Render the current step
  const renderStep = () => {
    switch (currentStep) {
      case "home":
      case null:
        return <Home />;
      case "database":
        return <DatabaseStep />;
      case "process":
        return <ProcessStep />;
      case "train":
        return <TrainStep />;
      case "results":
        return <ResultsStep />;
      default:
        return <Home />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="dashboard-container container mx-auto py-8 px-4">
        <WorkflowHeader />
        <div className="gsap-container">
          {renderStep()}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
