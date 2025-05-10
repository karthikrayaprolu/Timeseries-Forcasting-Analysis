// components/AnimatedChart.tsx
import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import * as d3 from 'd3';

interface DataPoint {
  date: Date;
  value: number;
  forecast?: boolean;
}

const AnimatedChart = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const width = 800;
  const height = 400;
  const margin = { top: 30, right: 30, bottom: 50, left: 50 };

  // Sample time series data with actual and forecast values
  const generateData = (): DataPoint[] => {
    const data: DataPoint[] = [];
    const today = new Date();
    
    // Generate historical data (past 30 days)
    for (let i = 30; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      data.push({
        date,
        value: 50 + Math.random() * 30 + Math.sin(i * 0.3) * 20,
        forecast: false
      });
    }
    
    // Generate forecast data (next 15 days)
    for (let i = 1; i <= 15; i++) {
      const date = new Date(today);
      date.setDate(date.getDate() + i);
      data.push({
        date,
        value: 80 + Math.random() * 15 + Math.sin(i * 0.3) * 10,
        forecast: true
      });
    }
    
    return data;
  };

  useEffect(() => {
    if (!svgRef.current) return;

    const today = new Date();

    const data = generateData();
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous renders

    // Set up scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(data, d => d.date) as [Date, Date])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.value)! * 1.1])
      .range([height - margin.bottom, margin.top]);

    // Create axes
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5);

    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(xAxis)
      .selectAll("text")
      .style("font-family", "sans-serif")
      .style("font-size", "12px");

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(yAxis)
      .selectAll("text")
      .style("font-family", "sans-serif")
      .style("font-size", "12px");

    // Create line generator
    const line = d3.line<DataPoint>()
      .x(d => xScale(d.date))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    // Draw actual data line
    const actualData = data.filter(d => !d.forecast);
    svg.append('path')
      .datum(actualData)
      .attr('fill', 'none')
      .attr('stroke', '#4f46e5') // Indigo-600
      .attr('stroke-width', 3)
      .attr('d', line)
      .attr('stroke-dasharray', function() { return this.getTotalLength() + ' ' + this.getTotalLength(); })
      .attr('stroke-dashoffset', function() { return this.getTotalLength(); })
      .transition()
      .duration(1500)
      .attr('stroke-dashoffset', 0);

    // Draw forecast line with animation
    const forecastData = data.filter(d => d.forecast);
    const forecastLine = svg.append('path')
      .datum(forecastData)
      .attr('fill', 'none')
      .attr('stroke', '#f59e0b') // Amber-500
      .attr('stroke-width', 3)
      .attr('stroke-dasharray', '5,5')
      .attr('d', line)
      .style('opacity', 0);

    forecastLine.transition()
      .delay(1500)
      .duration(1000)
      .style('opacity', 1);

    // Add confidence area for forecast
    const area = d3.area<DataPoint>()
      .x(d => xScale(d.date))
      .y0(d => yScale(d.value - 5))
      .y1(d => yScale(d.value + 5))
      .curve(d3.curveMonotoneX);

    const confidenceArea = svg.append('path')
      .datum(forecastData)
      .attr('fill', '#f59e0b')
      .attr('fill-opacity', 0.1)
      .attr('d', area)
      .style('opacity', 0);

    confidenceArea.transition()
      .delay(1500)
      .duration(1000)
      .style('opacity', 1);

    // Add today's line
    svg.append('line')
      .attr('x1', xScale(today))
      .attr('x2', xScale(today))
      .attr('y1', margin.top)
      .attr('y2', height - margin.bottom)
      .attr('stroke', '#ef4444') // Red-500
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .style('opacity', 0)
      .transition()
      .delay(2000)
      .duration(500)
      .style('opacity', 1);

    // Add legend
    const legend = svg.append('g')
      .attr('transform', `translate(${width - margin.right - 150},${margin.top})`);

    legend.append('line')
      .attr('x1', 0)
      .attr('x2', 30)
      .attr('y1', 0)
      .attr('y2', 0)
      .attr('stroke', '#4f46e5')
      .attr('stroke-width', 2);

    legend.append('text')
      .attr('x', 40)
      .attr('y', 0)
      .attr('dy', '0.35em')
      .text('Historical Data')
      .style('font-family', 'sans-serif')
      .style('font-size', '12px');

    legend.append('line')
      .attr('x1', 0)
      .attr('x2', 30)
      .attr('y1', 20)
      .attr('y2', 20)
      .attr('stroke', '#f59e0b')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5');

    legend.append('text')
      .attr('x', 40)
      .attr('y', 20)
      .attr('dy', '0.35em')
      .text('Forecast')
      .style('font-family', 'sans-serif')
      .style('font-size', '12px');

    legend.append('line')
      .attr('x1', 0)
      .attr('x2', 30)
      .attr('y1', 40)
      .attr('y2', 40)
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3');

    legend.append('text')
      .attr('x', 40)
      .attr('y', 40)
      .attr('dy', '0.35em')
      .text('Today')
      .style('font-family', 'sans-serif')
      .style('font-size', '12px');

  }, []);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="bg-white rounded-xl shadow-md p-6 border border-gray-100"
    >
      <svg 
        ref={svgRef} 
        width={width} 
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto"
      />
    </motion.div>
  );
};

export default AnimatedChart;