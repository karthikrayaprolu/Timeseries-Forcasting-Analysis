// Mock API service for Time Series Forecasting

// Mock data for available tables
const mockTables = [
  "sales_data",
  "stock_prices",
  "website_traffic",
  "temperature_records",
  "covid_cases",
  "uploaded_csv", // Add default table name for CSV uploads
];

// Mock data storage for CSV
let csvData: any = null;

// Mock data for available columns
const mockColumns = [
  "date",
  "timestamp",
  "sales",
  "revenue",
  "visitors",
  "temperature",
  "humidity",
  "stock_price",
  "volume",
  "cases",
  "deaths",
  "recovered",
  "region_id",
  "product_id",
  "customer_id",
];

// Mock forecast results
const mockResults = {
  metrics: {
    mse: 245.67,
    rmse: 15.67,
    mae: 12.34,
    mape: 8.76,
  },
  forecasts: {
    dates: Array.from({ length: 30 }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() + i);
      return date.toISOString().split("T")[0];
    }),
    actual: Array.from({ length: 15 }, () => Math.floor(Math.random() * 1000) + 500),
    predicted: Array.from({ length: 30 }, () => Math.floor(Math.random() * 1000) + 500),
  },
};

// Simulates API call delay
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Helper function to safely format numbers
const formatNumber = (num: number): number => {
  if (isNaN(num) || !isFinite(num)) return 0;
  return Number(num.toFixed(2));
};

// Helper functions
const calculateMetrics = (actual: number[], predicted: number[]) => {
  // Filter out invalid pairs and pairs where actual is too close to zero (for MAPE)
  const validPairs = actual.map((val, i) => [val, predicted[i]])
    .filter(([a, p]) => !isNaN(a) && !isNaN(p) && isFinite(a) && isFinite(p));
  
  if (validPairs.length === 0) {
    return {
      mse: 0,
      rmse: 0,
      mae: 0,
      mape: 0
    };
  }

  // Calculate errors
  const errors = validPairs.map(([a, p]) => a - p);
  const absErrors = errors.map(e => Math.abs(e));
  const squaredErrors = errors.map(e => e * e);
  
  // Calculate basic metrics
  const mse = squaredErrors.reduce((a, b) => a + b, 0) / validPairs.length;
  const rmse = Math.sqrt(mse);
  const mae = absErrors.reduce((a, b) => a + b, 0) / validPairs.length;
  
  // Calculate MAPE only for non-zero actual values to avoid division by zero
  const validMapePairs = validPairs.filter(([a]) => Math.abs(a) > 1e-10);
  const mape = validMapePairs.length > 0
    ? validMapePairs.reduce((sum, [a, p]) => sum + Math.abs((a - p) / a), 0) * 100 / validMapePairs.length
    : 0;

  return {
    mse: formatNumber(mse),
    rmse: formatNumber(rmse),
    mae: formatNumber(mae),
    mape: formatNumber(mape)
  };
};

// Time series model classes and utilities
class TimeSeriesModel {
  protected data: number[] = [];
  protected params: any = {};
  
  constructor(params = {}) {
    this.params = params;
  }
  
  fit(data: number[]): TimeSeriesModel {
    this.data = data;
    return this;
  }
  
  predict(horizon: number): number[] {
    return [];
  }
}

class ARIMAModel extends TimeSeriesModel {
  private p: number;
  private d: number;
  private q: number;
  private coefficients: number[] = [];
  
  constructor(p: number, d: number, q: number) {
    super({ p, d, q });
    this.p = p;
    this.d = d;
    this.q = q;
  }
  
  fit(data: number[]): ARIMAModel {
    super.fit(data);
    
    // Simple AR coefficient calculation
    this.coefficients = new Array(this.p).fill(0).map((_, i) => 
      0.8 / Math.pow(2, i)
    );
    
    return this;
  }
  
  predict(horizon: number): number[] {
    const predictions = [...this.data];
    
    for (let i = 0; i < horizon; i++) {
      let pred = 0;
      for (let j = 0; j < this.p; j++) {
        if (i - j >= 0) {
          pred += this.coefficients[j] * predictions[predictions.length - 1 - j];
        }
      }
      predictions.push(pred);
    }
    
    return predictions;
  }
}

class ProphetModel extends TimeSeriesModel {
  private trend: number[] = [];
  private seasonality: number[] = [];
  private changepoints: number[] = [];
  
  constructor(params: any) {
    super(params);
  }
  
  fit(data: number[]): ProphetModel {
    super.fit(data);
    
    // Calculate trend
    this.trend = this.calculateTrend(data);
    
    // Calculate seasonality
    this.seasonality = this.calculateSeasonality(data, this.trend);
    
    // Identify changepoints
    this.changepoints = this.findChangepoints(data);
    
    return this;
  }
  
  predict(horizon: number): number[] {
    const predictions = [...this.data];
    const period = 12;
    
    // Calculate growth rate
    const growthRate = this.trend.length > period ? 
      (this.trend[this.trend.length - 1] / this.trend[this.trend.length - period - 1] - 1) / period : 
      0.01;
    
    // Generate future predictions
    for (let i = 0; i < horizon; i++) {
      const trendValue = this.trend[this.trend.length - 1] * (1 + growthRate * (i + 1));
      const seasonalValue = this.seasonality[i % 12];
      predictions.push(trendValue * seasonalValue);
    }
    
    return predictions;
  }
  
  private calculateTrend(data: number[]): number[] {
    const window = 12;
    return data.map((_, i) => {
      const start = Math.max(0, i - window);
      const segment = data.slice(start, i + 1);
      return segment.reduce((sum, val) => sum + val, 0) / segment.length;
    });
  }
  
  private calculateSeasonality(data: number[], trend: number[]): number[] {
    const period = 12;
    const seasonalFactors = new Array(period).fill(0);
    const counts = new Array(period).fill(0);
    
    data.forEach((val, i) => {
      if (trend[i] !== 0) {
        seasonalFactors[i % period] += val / trend[i];
        counts[i % period]++;
      }
    });
    
    const rawFactors = seasonalFactors.map((factor, i) => 
      factor / (counts[i] || 1)
    );
    
    const mean = rawFactors.reduce((a, b) => a + b) / period;
    return rawFactors.map(factor => factor / mean);
  }
  
  private findChangepoints(data: number[]): number[] {
    const window = 5;
    const threshold = 2;
    const changes: number[] = [];
    
    for (let i = window; i < data.length - window; i++) {
      const before = data.slice(i - window, i);
      const after = data.slice(i, i + window);
      const beforeMean = before.reduce((a, b) => a + b) / window;
      const afterMean = after.reduce((a, b) => a + b) / window;
      
      if (Math.abs(afterMean - beforeMean) / beforeMean > threshold) {
        changes.push(i);
      }
    }
    
    return changes;
  }
}

class LSTMModel extends TimeSeriesModel {
  private weights: number[][] = [];
  private units: number;
  
  constructor(units: number) {
    super({ units });
    this.units = units;
    this.initializeWeights();
  }
  
  fit(data: number[], params: any = {}): LSTMModel {
    super.fit(data);
    
    const { epochs = 100, batch_size = 32 } = params;
    
    // Simplified training loop
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < data.length - batch_size; i += batch_size) {
        const batch = data.slice(i, i + batch_size);
        this.trainOnBatch(batch);
      }
    }
    
    return this;
  }
  
  predict(horizon: number): number[] {
    const predictions = [...this.data];
    const lookback = 12;
    
    for (let i = 0; i < horizon; i++) {
      const input = predictions.slice(-lookback);
      const pred = this.forward(input);
      predictions.push(pred);
    }
    
    return predictions;
  }
  
  loadWeights(weights: number[][]) {
    this.weights = weights;
  }
  
  private initializeWeights() {
    this.weights = Array(this.units).fill(0).map(() =>
      Array(this.units).fill(0).map(() => Math.random() - 0.5)
    );
  }
  
  private forward(input: number[]): number {
    const lookback = Math.min(input.length, this.units);
    let output = 0;
    
    for (let i = 0; i < lookback; i++) {
      for (let j = 0; j < this.units; j++) {
        output += input[input.length - 1 - i] * this.weights[i][j];
      }
    }
    
    return output / (lookback * this.units);
  }
  
  private trainOnBatch(batch: number[]) {
    // Simplified update rule
    const learningRate = 0.01;
    
    for (let i = 0; i < this.units; i++) {
      for (let j = 0; j < this.units; j++) {
        this.weights[i][j] += learningRate * (Math.random() - 0.5);
      }
    }
  }
}

class SimpleModel extends TimeSeriesModel {
  private window: number = 3;
  
  predict(horizon: number): number[] {
    const predictions = [...this.data];
    
    for (let i = 0; i < horizon; i++) {
      const start = predictions.length - this.window;
      const window = predictions.slice(start);
      const pred = window.reduce((a, b) => a + b) / this.window;
      predictions.push(pred);
    }
    
    return predictions;
  }
}

// Utility functions
const generateRandomWeights = (size: number): number[][] => {
  return Array(size).fill(0).map(() =>
    Array(size).fill(0).map(() => Math.random() - 0.5)
  );
};

// Time series model configuration
const modelConfig = {
  ARIMA: {
    defaultParams: { p: 1, d: 1, q: 1 },
    paramGrid: {
      p: [1, 2, 3],
      d: [0, 1, 2],
      q: [0, 1, 2]
    }
  },
  Prophet: {
    defaultParams: {
      changepoint_prior_scale: 0.05,
      seasonality_prior_scale: 10,
      seasonality_mode: 'multiplicative'
    },
    paramGrid: {
      changepoint_prior_scale: [0.001, 0.01, 0.05, 0.1],
      seasonality_prior_scale: [1, 5, 10, 15],
      seasonality_mode: ['multiplicative', 'additive']
    }
  },
  LSTM: {
    defaultParams: {
      units: 50,
      epochs: 100,
      batch_size: 32
    },
    paramGrid: {
      units: [32, 50, 64, 128],
      epochs: [50, 100, 150],
      batch_size: [16, 32, 64]
    }
  }
};

// Mock API service
export const api = {
  // Time series forecasting helper functions
  timeSeriesForecasting: {
    calculateTrend: (data: number[], window: number) => {
      return data.map((_, i) => {
        const start = Math.max(0, i - window);
        const segment = data.slice(start, i + 1);
        return segment.reduce((sum, val) => sum + val, 0) / segment.length;
      });
    },

    calculateSeasonality: (data: number[], trends: number[], period: number) => {
      const seasonalFactors = new Array(period).fill(0);
      const counts = new Array(period).fill(0);

      data.forEach((val, i) => {
        if (trends[i] !== 0) {
          const period_idx = i % period;
          seasonalFactors[period_idx] += val / trends[i];
          counts[period_idx]++;
        }
      });

      const rawFactors = seasonalFactors.map((factor, i) => factor / (counts[i] || 1));
      const mean = rawFactors.reduce((a, b) => a + b) / period;
      return rawFactors.map(factor => factor / mean);
    },

    generateFuturePredictions: (lastValue: number, growthRate: number, seasonality: number[], numPeriods: number) => {
      return new Array(numPeriods).fill(0).map((_, i) => {
        const trend = lastValue * (1 + growthRate * (i + 1));
        return trend * seasonality[i % seasonality.length];
      });
    }
  },

  // Connect to database
  connectToDatabase: async (connectionString: string) => {
    await delay(1500);
    return { success: true, message: "Connected successfully" };
  },

  // Get available tables
  getTables: async (databaseType: string) => {
    await delay(1000);
    if (databaseType === 'local' && csvData) {
      return ['uploaded_csv'];
    }
    return mockTables;
  },

  // Get columns for a selected table
  getColumns: async (table: string) => {
    await delay(800);
    if (table === 'uploaded_csv' && csvData) {
      return csvData.headers;
    }
    return mockColumns;
  },

  // Process data with selected configuration
  processData: async (config: any) => {
    await delay(2000);
    
    if (config.table === 'uploaded_csv' && csvData) {
      const processedData = {
        timeColumn: csvData.rows.map((row: any) => row[config.timeColumn]),
        targetVariable: csvData.rows.map((row: any) => row[config.targetVariable]),
        features: config.features.map((feature: string) => 
          csvData.rows.map((row: any) => row[feature])
        ),
      };
      
      csvData.processedData = processedData;
    }
    
    return { success: true, message: "Data processed successfully" };
  },

  // Train model with selected configuration
  trainModel: async (config: any) => {
    await delay(1000);
    
    if (csvData?.processedData) {
      // Clean and validate data
      const actualData = csvData.processedData.targetVariable
        .map(val => {
          const num = Number(val);
          return isNaN(num) || !isFinite(num) ? null : num;
        })
        .filter((val): val is number => val !== null);
      
      if (actualData.length === 0) {
        throw new Error("No valid numeric data found in target variable");
      }

      // Scale data to [0,1] range to prevent numerical issues
      const maxVal = Math.max(...actualData);
      const minVal = Math.min(...actualData);
      const range = maxVal - minVal;
      const scaledData = range > 0 
        ? actualData.map(val => (val - minVal) / range)
        : actualData;
      
      // Split data for training and testing
      const trainSize = Math.floor(scaledData.length * 0.8);
      const trainData = scaledData.slice(0, trainSize);
      const testData = scaledData.slice(trainSize);
      
      let predictions: number[] = [];
      
      try {
        switch (config.modelType) {
          case 'ARIMA': {
            const model = new ARIMAModel(1, 1, 1);
            model.fit(trainData);
            predictions = model.predict(scaledData.length + 12);
            break;
          }
          
          case 'Prophet': {
            const model = new ProphetModel({
              changepoint_prior_scale: 0.05,
              seasonality_prior_scale: 10
            });
            model.fit(trainData);
            predictions = model.predict(scaledData.length + 12);
            break;
          }
          
          case 'LSTM': {
            const model = new LSTMModel(32);
            model.fit(trainData, { epochs: 100, batch_size: 32 });
            predictions = model.predict(scaledData.length + 12);
            break;
          }
          
          default: {
            const model = new SimpleModel();
            model.fit(trainData);
            predictions = model.predict(scaledData.length + 12);
          }
        }
        
        // Safely unscale predictions
        if (range > 0) {
          predictions = predictions.map(val => {
            // Clamp scaled predictions to [0,1] to prevent out-of-range values
            const clampedVal = Math.max(0, Math.min(1, val));
            return formatNumber(clampedVal * range + minVal);
          });
        }
        
        // Ensure all predictions are valid numbers
        predictions = predictions.map(val => 
          isNaN(val) || !isFinite(val) ? actualData[actualData.length - 1] : val
        );
        
        // Calculate final metrics on test data
        const testPreds = predictions.slice(trainSize, trainSize + testData.length);
        const unscaledTestData = testData.map(val => range > 0 ? val * range + minVal : val);
        const metrics = calculateMetrics(unscaledTestData, testPreds);
        
        // Prepare dates for visualization
        const dates = [
          ...csvData.processedData.timeColumn,
          ...new Array(12).fill(0).map((_, i) => {
            const lastDate = new Date(csvData.processedData.timeColumn[csvData.processedData.timeColumn.length - 1]);
            lastDate.setMonth(lastDate.getMonth() + i + 1);
            return lastDate.toISOString().split('T')[0];
          })
        ];

        // Format data for visualization
        return {
          dataInfo: {
            title: csvData.dataType || 'Time Series Forecast',
            filename: csvData.filename || 'data'
          },
          metrics,
          modelInfo: {
            type: config.modelType,
            parameters: modelConfig[config.modelType]?.defaultParams || {},
            features: {
              hyperparameterTuning: config.hyperparameterTuning || false,
              transferLearning: config.transferLearning || false,
              ensembleLearning: config.ensembleLearning || false
            }
          },
          forecasts: {
            dates,
            actual: [...actualData, ...new Array(12).fill(null)],
            predicted: predictions.map(val => formatNumber(val))
          }
        };
      } catch (error) {
        console.error('Error during model training:', error);
        throw new Error('Failed to train model: ' + (error as Error).message);
      }
    }
    
    return mockResults;
  },

  // Export results to file
  exportResults: async (format: "csv" | "excel" | "json", results: any) => {
    let content = '';
    const timestamp = new Date().toISOString().split('T')[0];
    let filename = `forecast_results_${timestamp}`;
    
    switch (format) {
      case 'csv':
      case 'excel':
        content = 'Date,Actual,Predicted\n';
        results.forecasts.dates.forEach((date: string, i: number) => {
          const actual = results.forecasts.actual[i] === null ? '' : results.forecasts.actual[i].toFixed(2);
          const predicted = results.forecasts.predicted[i] === null ? '' : results.forecasts.predicted[i].toFixed(2);
          content += `${date},${actual},${predicted}\n`;
        });
        filename += format === 'csv' ? '.csv' : '.xlsx';
        break;
        
      case 'json':
        content = JSON.stringify({
          metrics: results.metrics,
          dataInfo: results.dataInfo,
          forecasts: {
            dates: results.forecasts.dates,
            actual: results.forecasts.actual,
            predicted: results.forecasts.predicted.map(v => Number(v.toFixed(2)))
          }
        }, null, 2);
        filename += '.json';
        break;
    }
    
    const blob = new Blob([content], { 
      type: format === 'json' 
        ? 'application/json' 
        : 'text/csv;charset=utf-8;' 
    });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    return {
      success: true,
      message: `Results exported as ${format.toUpperCase()} successfully`,
    };
  },

  // Upload and process CSV file
  uploadCsvFile: async (formData: FormData) => {
    await delay(1000);
    
    const file = formData.get('file') as File;
    if (!file) {
      throw new Error('No file provided');
    }

    const text = await file.text();
    const lines = text.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    csvData = {
      filename: file.name,
      dataType: headers[1] || 'Data',
      headers,
      rows: lines
        .filter(line => line.trim())
        .slice(1)
        .map(line => {
          const values = line.split(',').map(v => v.trim());
          return headers.reduce((obj: any, header, index) => {
            obj[header] = index === 1 ? parseFloat(values[index]) : values[index];
            return obj;
          }, {});
        }),
    };

    return { success: true, message: 'CSV file uploaded and processed successfully' };
  },

  // Helper functions for model training
  helpers: {
    trainSingleModel: (data: number[], type: string, params: any) => {
      switch (type) {
        case 'ARIMA': {
          const { p = 1, d = 1, q = 1 } = params;
          return new ARIMAModel(p, d, q).fit(data);
        }
        case 'Prophet': {
          const { changepoint_prior_scale, seasonality_prior_scale, seasonality_mode } = params;
          return new ProphetModel({
            changepoint_prior_scale,
            seasonality_prior_scale,
            seasonality_mode
          }).fit(data);
        }
        case 'LSTM': {
          const { units, epochs, batch_size, pretrained, weights } = params;
          const model = new LSTMModel(units);
          if (pretrained && weights) {
            model.loadWeights(weights);
          }
          return model.fit(data, { epochs, batch_size });
        }
        default:
          return new SimpleModel().fit(data);
      }
    },
    
    calculateRMSE: (actual: number[], predicted: number[]) => {
      const errors = actual.map((val, i) => val - predicted[i]);
      const mse = errors.reduce((sum, err) => sum + err * err, 0) / errors.length;
      return Math.sqrt(mse);
    },
    
    generateParamCombinations: (paramGrid: any) => {
      const keys = Object.keys(paramGrid);
      const combinations = [{}];
      
      keys.forEach(key => {
        const values = paramGrid[key];
        const temp: any[] = [];
        
        combinations.forEach(combo => {
          values.forEach(value => {
            temp.push({ ...combo, [key]: value });
          });
        });
        
        combinations.splice(0, combinations.length, ...temp);
      });
      
      return combinations;
    },
    
    getPretrainedWeights: (modelType: string) => {
      // Simulated pre-trained weights for different model types
      return {
        LSTM: generateRandomWeights(100),
        Prophet: generateRandomWeights(50)
      }[modelType];
    }
  }
};
