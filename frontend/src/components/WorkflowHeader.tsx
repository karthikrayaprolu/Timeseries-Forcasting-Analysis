
import React, { useEffect, useRef } from "react";
import { useWorkflow, WorkflowStep } from "@/contexts/WorkflowContext";
import { Button } from "@/components/ui/button";
import { gsap } from "gsap";
import { Database, FileInput, Brain, BarChart } from "lucide-react";

const WorkflowHeader = () => {
  const { currentStep, setCurrentStep } = useWorkflow();
  const headerRef = useRef<HTMLDivElement>(null);

  // Steps configuration
  const steps: { id: WorkflowStep; title: string; icon: React.ElementType }[] = [
    { id: "database", title: "Register Database", icon: Database },
    { id: "process", title: "Process Data", icon: FileInput },
    { id: "train", title: "Train Model", icon: Brain },
    { id: "results", title: "Results", icon: BarChart },
  ];

  // GSAP animation for header
  useEffect(() => {
    if (headerRef.current) {
      gsap.fromTo(
        headerRef.current,
        { y: -50, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.8, ease: "power3.out" }
      );
    }
  }, []);

  // Animation for step transition
  const handleStepChange = (step: WorkflowStep) => {
    setCurrentStep(step);
  };

  return (
    <div ref={headerRef} className="bg-white shadow-md mb-8 rounded-lg">
      <div className="container mx-auto px-4 py-4">
        <h1 className="text-2xl font-bold mb-6 text-center text-primary">
          Time Series Forecasting Platform
        </h1>
        <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isActive = currentStep === step.id;
            const isPast = steps.findIndex((s) => s.id === currentStep) > index;

            return (
              <Button
                key={step.id}
                variant={isActive ? "default" : isPast ? "secondary" : "outline"}
                className={`flex-1 gap-2 ${isActive ? "shadow-md" : ""}`}
                onClick={() => handleStepChange(step.id)}
              >
                <Icon size={18} />
                <span className="hidden sm:inline">{step.title}</span>
                <span className="inline sm:hidden">{index + 1}</span>
              </Button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default WorkflowHeader;
