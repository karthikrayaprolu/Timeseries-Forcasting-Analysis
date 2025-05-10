
import React from "react";
import { WorkflowProvider } from "@/contexts/WorkflowContext";
import Dashboard from "@/components/Dashboard";

const Index = () => {
  return (
    <WorkflowProvider>
      <Dashboard />
    </WorkflowProvider>
  );
};

export default Index;
