import React, { useEffect, useRef } from "react";
import { useWorkflow, WorkflowStep } from "@/contexts/WorkflowContext";
import { Button } from "@/components/ui/button";
import { gsap } from "gsap";
import { Home, Database, FileInput, Brain, BarChart } from "lucide-react";
import { useNavigate } from "react-router-dom";

const WorkflowHeader = () => {
  const { currentStep, setCurrentStep } = useWorkflow();
  const headerRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  // Steps configuration including Home
  const steps: { id: WorkflowStep | "home"; title: string; icon: React.ComponentType }[] = [
    { id: "home", title: "Home", icon: Home },
    { id: "database", title: "Register Database", icon: Database },
    { id: "process", title: "Process Data", icon: FileInput },
    { id: "train", title: "Train Model", icon: Brain },
    { id: "results", title: "Results", icon: BarChart },
  ];

  // Right side menu items
  const menuItems = [
    { id: "login", title: "Login" },
    { id: "signup", title: "Signup" },
    { id: "instructions", title: "Instructions" },
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

  const handleStepChange = (step: WorkflowStep ) => {
 
      setCurrentStep(step);
    
  };

  return (
    <div ref={headerRef} className="bg-gray-900 shadow-sm">
      <div className="container mx-auto px-6 py-4">
        <div className="flex flex-col sm:flex-row justify-between items-center gap-6">
          {/* Main Navigation */}
          <div className="flex items-center space-x-8 overflow-x-auto w-full py-2">
            {steps.map((step) => {
              const Icon = step.icon;
              const isActive = currentStep === step.id || (step.id === "home" && currentStep === null);

              return (
                <div key={step.id} className="relative group">
                  <Button
                    variant="ghost"
                    className={`p-0 h-auto text-gray-300 hover:bg-transparent group-hover:text-indigo-400 ${
                      isActive ? "text-amber-200" : ""
                    }`}
                    onClick={() => handleStepChange(step.id)}
                  >
                    <div className="flex items-center gap-2">
                      <span style={{ fontSize: 18, color: "currentColor" }}>
                        <Icon />
                      </span>
                      <span className="text-[15px] font-medium tracking-wide">{step.title}</span>
                    </div>
                  </Button>
                  <div className={`
                    absolute bottom-0 left-0 h-[2px] bg-indigo-400 
                    transition-all duration-300 ease-out 
                    ${isActive ? "w-full bg-amber-200" : "w-0 group-hover:w-full"}
                  `} />
                </div>
              );
            })}
          </div>
          
          {/* Right Side Menu */}
          <div className="flex items-center space-x-6">
            {menuItems.map((item) => (
              <div key={item.id} className="relative group">
                <Button 
                  variant="ghost" 
                  className="p-0 h-auto text-gray-300 hover:bg-transparent group-hover:text-indigo-400"
                >
                  <span className="text-[15px] font-medium tracking-wide">{item.title}</span>
                </Button>
                <div className={`
                  absolute bottom-0 left-0 h-[2px] bg-indigo-400 
                  transition-all duration-300 ease-out w-0 group-hover:w-full
                `} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WorkflowHeader;