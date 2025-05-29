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
  };  // Connect to database
  const handleConnect = async () => {
    setIsLoading(true);
    try {
      if (database.databaseType === "local" && csvData) {
        // Handle CSV file upload
        const formData = new FormData();
        formData.append("file", csvData, csvData.name); // Add filename as third parameter
        const result = await api.uploadCsvFile(formData);
        if (result.tables) {
          setAvailableTables(result.tables);
          setIsConnected(true);
          toast.success("CSV file uploaded successfully");
        }
      } else if (database.databaseType !== "local" && database.connectionString) {
        // For now we only support local CSV uploads
        toast.error("Only local CSV files are supported at this time");
        setIsLoading(false);
        return;
      }
    } catch (error) {
      console.error("Error processing file:", error);
      toast.error(error instanceof Error ? error.message : "Failed to process file");
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
    <div ref={componentRef} className="workflow-step max-w-3xl mx-auto p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-3xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-indigo-800 to-blue-600">Register Database</h2>
      
      <div className="space-y-8">
        <div className="space-y-4">
          <label className="text-lg font-medium text-gray-700">Database Type</label>
          <Select
            value={database.databaseType}
            onValueChange={handleDatabaseTypeChange}
            disabled={isLoading}
          >
            <SelectTrigger className="w-full h-12 bg-gray-50 border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 rounded-lg transition-all">
              <SelectValue placeholder="Select database type" />
            </SelectTrigger>
            <SelectContent className="bg-white border-2 border-gray-200 rounded-lg shadow-lg">
              {databaseTypes.map((type) => (
                <SelectItem key={type.id} value={type.id} className="hover:bg-indigo-50 cursor-pointer">
                  {type.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {database.databaseType === "local" ? (
          <div className="space-y-4">
            <label className="text-lg font-medium text-gray-700">Upload CSV File</label>
            <FileUploader
              onFileSelect={handleFileUpload}
              accept=".csv"              disabled={isLoading}
            />
            <p className="text-sm text-gray-500">
              Upload a CSV file containing time series data
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <label className="text-lg font-medium text-gray-700">Connection String</label>
            <Input
              placeholder="Connection string"
              value={database.connectionString || ""}
              onChange={(e) =>
                setDatabase({ ...database, connectionString: e.target.value })
              }
              disabled={isLoading}
              className="h-12 bg-gray-50 border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 rounded-lg transition-all"
            />
          </div>
        )}

        <Button
          onClick={handleConnect}
          disabled={isLoading || !database.databaseType || (database.databaseType === "local" && !csvData)}
          className="w-full h-12 bg-gradient-to-r from-indigo-600 to-blue-500 hover:from-indigo-700 hover:to-blue-600 text-white font-medium rounded-lg shadow-md hover:shadow-lg transition-all"
        >
          {isLoading ? "Connecting..." : "Connect"}
        </Button>

        {isConnected && (
          <div className="space-y-6 mt-6 animate-fade-in">
            <div className="space-y-4">
              <label className="text-lg font-medium text-gray-700">Select Table</label>
              <Select
                value={database.table}
                onValueChange={handleTableChange}
                disabled={isLoading}
              >
                <SelectTrigger className="w-full h-12 bg-gray-50 border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 rounded-lg transition-all">
                  <SelectValue placeholder="Select table" />
                </SelectTrigger>
                <SelectContent className="bg-white border-2 border-gray-200 rounded-lg shadow-lg">
                  {availableTables.map((table) => (
                    <SelectItem key={table} value={table} className="hover:bg-indigo-50 cursor-pointer">
                      {table}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Button
              onClick={handleNext}
              disabled={!database.table}
              className="w-full h-12 bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700 text-white font-medium rounded-lg shadow-md hover:shadow-lg transition-all"
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
