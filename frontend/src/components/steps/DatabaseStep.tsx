import React, { useState, useEffect, useRef } from "react";
import { useWorkflow } from "@/contexts/WorkflowContext";
import { api } from "@/services/api";
import { gsap } from "gsap";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { FileUploader } from "@/components/ui/file-uploader";

const DatabaseStep = () => {
  const {
    database,
    setDatabase,
    setCurrentStep,
    setAvailableTables,
    availableTables,
    isLoading,
    setIsLoading,
  } = useWorkflow();

  const [isConnected, setIsConnected] = useState(false);
  const componentRef = useRef<HTMLDivElement>(null);
  const [csvData, setCsvData] = useState<File | null>(null);

  // GSAP animation
  useEffect(() => {
    if (componentRef.current) {
      gsap.fromTo(
        componentRef.current,
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }
      );
    }
  }, []);

  // Database types
  const databaseTypes = [
    { id: "mongodb", name: "MongoDB" },
    { id: "postgres", name: "PostgreSQL" },
    { id: "local", name: "CSV Upload" },
  ];

  // Handle CSV file upload
  const handleFileUpload = async (file: File) => {
    if (file && file.type === "text/csv") {
      setCsvData(file);
      setDatabase({ 
        ...database, 
        connectionString: file.name,
        databaseType: "local" 
      });
    } else {
      toast.error("Please upload a valid CSV file");
    }
  };

  // Connect to database
  const handleConnect = async () => {
    setIsLoading(true);
    try {
      if (database.databaseType === "local" && csvData) {
        // Handle CSV file upload
        const formData = new FormData();
        formData.append("file", csvData);
        await api.uploadCsvFile(formData);
        const tables = await api.getTables("local");
        setAvailableTables(tables);
        setIsConnected(true);
        toast.success("CSV file uploaded successfully");
      } else {
        // Handle database connection
        await api.connectToDatabase(database.connectionString || "");
        const tables = await api.getTables(database.databaseType);
        setAvailableTables(tables);
        setIsConnected(true);
        toast.success("Connected to database successfully");
      }
    } catch (error) {
      console.error("Error connecting to database:", error);
      toast.error("Failed to connect to database");
    } finally {
      setIsLoading(false);
    }
  };

  // Update database config
  const handleDatabaseTypeChange = (value: string) => {
    setDatabase({ ...database, databaseType: value });
    setIsConnected(false); // Reset connection status
  };

  const handleTableChange = (value: string) => {
    setDatabase({ ...database, table: value });
  };

  // Next step
  const handleNext = () => {
    if (database.table) {
      setCurrentStep("process");
    } else {
      toast.warning("Please select a table before proceeding");
    }
  };

  return (
    <div ref={componentRef} className="workflow-step">
      <h2 className="step-title">Register Database</h2>
      
      <div className="space-y-6">
        <div className="space-y-2">
          <label className="text-sm font-medium">Database Type</label>
          <Select
            value={database.databaseType}
            onValueChange={handleDatabaseTypeChange}
            disabled={isLoading}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select database type" />
            </SelectTrigger>
            <SelectContent>
              {databaseTypes.map((type) => (
                <SelectItem key={type.id} value={type.id}>
                  {type.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {database.databaseType === "local" ? (
          <div className="space-y-2">
            <label className="text-sm font-medium">Upload CSV File</label>
            <FileUploader
              onFileSelect={handleFileUpload}
              accept=".csv"
              disabled={isLoading}
            />
            <p className="text-sm text-muted-foreground">
              Upload a CSV file containing time series data
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <label className="text-sm font-medium">Connection String</label>
            <Input
              placeholder="Connection string"
              value={database.connectionString || ""}
              onChange={(e) =>
                setDatabase({ ...database, connectionString: e.target.value })
              }
              disabled={isLoading}
            />
          </div>
        )}

        <Button
          onClick={handleConnect}
          disabled={isLoading || !database.databaseType || (database.databaseType === "local" && !csvData)}
          className="w-full"
        >
          {isLoading ? "Connecting..." : "Connect"}
        </Button>

        {isConnected && (
          <div className="space-y-4 mt-4 animate-fade-in">
            <div className="space-y-2">
              <label className="text-sm font-medium">Select Table</label>
              <Select
                value={database.table}
                onValueChange={handleTableChange}
                disabled={isLoading}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select table" />
                </SelectTrigger>
                <SelectContent>
                  {availableTables.map((table) => (
                    <SelectItem key={table} value={table}>
                      {table}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Button
              onClick={handleNext}
              disabled={!database.table}
              className="w-full"
              variant="default"
            >
              Next: Process Data
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DatabaseStep;
