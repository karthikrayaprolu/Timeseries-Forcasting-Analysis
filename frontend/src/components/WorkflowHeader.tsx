import React, { useEffect, useRef } from "react";
import { useWorkflow, WorkflowStep } from "@/contexts/WorkflowContext";
import { Button } from "@/components/ui/button";
import { gsap } from "gsap";
import { Home, Database, FileInput, Brain, BarChart, User, LogOut, Loader2, LineChart } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { toast } from "sonner";

const WorkflowHeader = () => {
  const { currentStep, setCurrentStep } = useWorkflow();
  const headerRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const { user, logout, loading } = useAuth();

  // Steps configuration including Home
  const steps = [
    { id: "home", title: "Home", icon: Home },
    { id: "database", title: "Register Database", icon: Database },
    { id: "process", title: "Process Data", icon: FileInput },
    { id: "train", title: "Train Model", icon: Brain },
    { id: "results", title: "Results", icon: BarChart },
  ];

  const handleLogout = async () => {
    try {
      await logout();
      toast.success("Logged out successfully");
      navigate('/');
    } catch (error) {
      console.error('Logout failed:', error);
      toast.error("Failed to logout. Please try again.");
    }
  };

  // GSAP animation
  useEffect(() => {
    if (headerRef.current) {
      gsap.fromTo(
        headerRef.current,
        { y: -50, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.8, ease: "power3.out" }
      );
    }
  }, []);

const handleStepChange = (step: WorkflowStep) => {
    if (!user && step !== "home") {
      toast.error("Please login to access this feature");
      navigate('/auth/login');
      return;
    }
    setCurrentStep(step);
  };

  // Profile Menu Component
  const ProfileMenu = () => (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button 
          variant="ghost" 
          className="relative h-10 w-10 rounded-full hover:bg-gray-800"
        >
          <Avatar>
            <AvatarFallback className="bg-indigo-600 text-white">
              {user?.email?.charAt(0).toUpperCase() || 'U'}
            </AvatarFallback>
          </Avatar>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <div className="flex items-center justify-start gap-2 p-2">
          <div className="flex flex-col space-y-1 leading-none">
            {user?.email && (
              <p className="font-medium text-sm">{user.email}</p>
            )}
          </div>
        </div>
        <DropdownMenuSeparator />
        <DropdownMenuItem 
          onClick={handleLogout}
          className="text-red-600 focus:text-red-600 cursor-pointer"
        >
          <LogOut className="mr-2 h-4 w-4" />
          <span>Log out</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );

  // Auth Buttons Component
  const AuthButtons = () => (
    <>
      <Button 
        variant="ghost" 
        className="text-gray-300 hover:text-indigo-400"
        onClick={() => navigate('/auth/login')}
      >
        Login
      </Button>
      <Button 
        variant="ghost" 
        className="text-gray-300 hover:text-indigo-400"
        onClick={() => navigate('/auth/signup')}
      >
        Sign Up
      </Button>
    </>
  );

  return (
    <div ref={headerRef} className="bg-gray-900 shadow-sm">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo Section */}
          <div className="flex items-center gap-2">
            <LineChart className="h-8 w-8 text-indigo-400" />
            <span className="text-xl font-bold text-white">TimeSeries</span>
          </div>

          {/* Main Navigation - Centered */}
          <div className="flex-1 flex justify-center">
            <div className="flex items-center space-x-8 overflow-x-auto py-2">
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
                      onClick={() => handleStepChange(step.id as WorkflowStep)}
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
          </div>
          
          {/* Right Side Menu */}
          <div className="flex items-center space-x-6">
            {loading ? (
              <div className="w-10 h-10 flex items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-gray-400" />
              </div>
            ) : (
              user ? <ProfileMenu /> : <AuthButtons />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WorkflowHeader;