import { Component, OnInit, AfterViewInit, ElementRef, ViewChild, NgZone } from '@angular/core';
import * as d3 from 'd3';
import { CommonModule } from '@angular/common';

interface Point2D {
  x: number;
  y: number;
}

interface ProjectedPoint {
  pc1: number;
  pc2: number;
}

interface NamedValue {
  name: string;
  value: number;
}

@Component({
  selector: 'app-pca-simulation',
  templateUrl: './pca-simulation.component.html',
  styleUrls: ['./pca-simulation.component.scss'],
  standalone: true,
  imports: [CommonModule]
})
export class PcaSimulationComponent implements OnInit, AfterViewInit {
  @ViewChild('simulationContainer') simulationContainer!: ElementRef;
  @ViewChild('scatterPlot') scatterPlotContainer!: ElementRef;
  @ViewChild('screePlot') screePlotContainer!: ElementRef;
  @ViewChild('variancePlot') variancePlotContainer!: ElementRef;

  private width = 500;
  private height = 350;
  private margin = { top: 30, right: 45, bottom: 50, left: 50 };
  private innerWidth = this.width - this.margin.left - this.margin.right;
  private innerHeight = this.height - this.margin.top - this.margin.bottom;

  // Animation variables
  private autoPlayInterval: any;
  public isPlaying = false;
  private stepInterval = 1500; // milliseconds between animation steps
  
  // UI state
  public currentStep = 0;
  public maxSteps = 5;
  public stepTitles = [
    'Original Data',
    'Standardization',
    'Find Principal Components',
    'Project Data onto PCs',
    'Variance Explained',
    'Reconstruction'
  ];
  public currentStepDescription = '';

  // Sample data generation parameters
  private numPoints = 80;
  private rawData: Array<Point2D> = [];
  private stdData: Array<Point2D> = [];
  private projectedData: Array<ProjectedPoint> = [];
  private reconstructedData: Array<Point2D> = [];
  
  // PCA calculation results
  private mean: Point2D = {x: 0, y: 0};
  private stdDev: Point2D = {x: 1, y: 1};
  private covarianceMatrix: [[number, number], [number, number]] = [[1, 0], [0, 1]];
  private eigenvalues: [number, number] = [1, 1];
  private eigenvectors: [[number, number], [number, number]] = [[1, 0], [0, 1]];
  private explainedVariance: [number, number] = [0.5, 0.5];
  private cumulativeVariance: [number, number] = [0.5, 1.0];

  constructor(private zone: NgZone) {}

  ngOnInit(): void {
    this.generateCorrelatedData();
    this.calculatePCA();
    this.updateStepDescription();
  }

  ngAfterViewInit(): void {
    this.initializeVisualization();
    
    // Scroll to top of the component after initialization
    setTimeout(() => {
      if (this.simulationContainer && this.simulationContainer.nativeElement) {
        this.simulationContainer.nativeElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        window.scrollTo(0, 0);
      }
    }, 100);
  }

  // Generate correlated random data with specific variance directions
  private generateCorrelatedData(): void {
    // Define the covariance matrix (with correlation = 0.8)
    const varX = 10;
    const varY = 5;
    const corr = 0.8;
    const covXY = corr * Math.sqrt(varX * varY);
    
    // Generate random data from this distribution
    const mean = [50, 50];
    const data: Array<Point2D> = [];
    
    for (let i = 0; i < this.numPoints; i++) {
      // Generate two uncorrelated standard normal variables
      let z1 = this.boxMullerTransform();
      let z2 = this.boxMullerTransform();
      
      // Transform to correlated variables using Cholesky decomposition
      let x = mean[0] + Math.sqrt(varX) * z1;
      let y = mean[1] + covXY/Math.sqrt(varX) * z1 + Math.sqrt(varY - covXY*covXY/varX) * z2;
      
      data.push({x, y});
    }
    
    this.rawData = data;
  }
  
  // Box-Muller transform to generate normal random variables
  private boxMullerTransform(): number {
    const u1 = Math.random();
    const u2 = Math.random();
    
    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z;
  }

  // Calculate PCA on the generated data
  private calculatePCA(): void {
    // Step 1: Calculate mean
    this.mean = {
      x: d3.mean(this.rawData, d => d.x) || 0,
      y: d3.mean(this.rawData, d => d.y) || 0
    };
    
    // Step 2: Calculate standard deviation
    this.stdDev = {
      x: d3.deviation(this.rawData, d => d.x) || 1,
      y: d3.deviation(this.rawData, d => d.y) || 1
    };
    
    // Step 3: Standardize the data
    this.stdData = this.rawData.map(d => ({
      x: (d.x - this.mean.x) / this.stdDev.x,
      y: (d.y - this.mean.y) / this.stdDev.y
    }));
    
    // Step 4: Calculate covariance matrix
    let covXX = 0, covXY = 0, covYY = 0;
    for (const point of this.stdData) {
      covXX += point.x * point.x;
      covXY += point.x * point.y;
      covYY += point.y * point.y;
    }
    covXX /= this.stdData.length - 1;
    covXY /= this.stdData.length - 1;
    covYY /= this.stdData.length - 1;
    
    this.covarianceMatrix = [[covXX, covXY], [covXY, covYY]];
    
    // Step 5: Calculate eigenvalues and eigenvectors (analytical solution for 2x2 matrix)
    const a = this.covarianceMatrix[0][0];
    const b = this.covarianceMatrix[0][1];
    const c = this.covarianceMatrix[1][0];
    const d = this.covarianceMatrix[1][1];
    
    const trace = a + d;
    const determinant = a * d - b * c;
    
    // Eigenvalues
    const lambda1 = trace/2 + Math.sqrt((trace*trace)/4 - determinant);
    const lambda2 = trace/2 - Math.sqrt((trace*trace)/4 - determinant);
    this.eigenvalues = [lambda1, lambda2];
    
    // Eigenvectors (normalized)
    let v1x, v1y, v2x, v2y;
    
    if (b !== 0) {
      v1x = lambda1 - d;
      v1y = b;
      v2x = lambda2 - d;
      v2y = b;
    } else if (c !== 0) {
      v1x = c;
      v1y = lambda1 - a;
      v2x = c;
      v2y = lambda2 - a;
    } else {
      // Diagonal matrix case
      v1x = 1;
      v1y = 0;
      v2x = 0;
      v2y = 1;
    }
    
    // Normalize eigenvectors
    const v1Norm = Math.sqrt(v1x*v1x + v1y*v1y);
    const v2Norm = Math.sqrt(v2x*v2x + v2y*v2y);
    
    v1x /= v1Norm;
    v1y /= v1Norm;
    v2x /= v2Norm;
    v2y /= v2Norm;
    
    this.eigenvectors = [[v1x, v1y], [v2x, v2y]];
    
    // Step 6: Calculate explained variance
    const totalVariance = lambda1 + lambda2;
    this.explainedVariance = [
      lambda1 / totalVariance,
      lambda2 / totalVariance
    ];
    
    this.cumulativeVariance = [
      this.explainedVariance[0],
      this.explainedVariance[0] + this.explainedVariance[1]
    ];
    
    // Step 7: Project data onto principal components
    this.projectedData = this.stdData.map(d => {
      const pc1 = this.eigenvectors[0][0] * d.x + this.eigenvectors[0][1] * d.y;
      const pc2 = this.eigenvectors[1][0] * d.x + this.eigenvectors[1][1] * d.y;
      return { pc1, pc2 };
    });
    
    // Step 8: Reconstruction from principal components
    this.reconstructedData = this.projectedData.map(d => {
      const x = this.eigenvectors[0][0] * d.pc1 + this.eigenvectors[1][0] * d.pc2;
      const y = this.eigenvectors[0][1] * d.pc1 + this.eigenvectors[1][1] * d.pc2;
      
      // Un-standardize to original scale
      return {
        x: x * this.stdDev.x + this.mean.x,
        y: y * this.stdDev.y + this.mean.y
      };
    });
  }

  private initializeVisualization(): void {
    this.zone.runOutsideAngular(() => {
      this.drawScatterPlot();
      this.drawScreePlot();
      this.drawVariancePlot();
      this.handleCurrentStep();
    });
  }

  private drawScatterPlot(): void {
    const container = d3.select(this.scatterPlotContainer.nativeElement);
    container.selectAll("*").remove();
    
    const svg = container.append("svg")
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("viewBox", `0 0 ${this.width} ${this.height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("class", "scatter-svg");
    
    const g = svg.append("g")
      .attr("transform", `translate(${this.margin.left},${this.margin.top})`);
    
    // Set up scales for original data
    const xMax = d3.max(this.rawData, d => d.x) || 0;
    const xMin = d3.min(this.rawData, d => d.x) || 0;
    const yMax = d3.max(this.rawData, d => d.y) || 0;
    const yMin = d3.min(this.rawData, d => d.y) || 0;
    
    const xPadding = (xMax - xMin) * 0.1;
    const yPadding = (yMax - yMin) * 0.1;
    
    const xScale = d3.scaleLinear()
      .domain([xMin - xPadding, xMax + xPadding])
      .range([0, this.innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([yMin - yPadding, yMax + yPadding])
      .range([this.innerHeight, 0]);
    
    // Add axes
    const xAxis = g.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.innerHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll("text")
      .style("fill", "#e1e7f5");
    
    const yAxis = g.append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(yScale))
      .selectAll("text")
      .style("fill", "#e1e7f5");
    
    // Add labels
    svg.append("text")
      .attr("class", "x-label")
      .attr("text-anchor", "middle")
      .attr("x", this.width / 2)
      .attr("y", this.height - 10)
      .style("fill", "#e1e7f5")
      .text("Feature 1");
    
    svg.append("text")
      .attr("class", "y-label")
      .attr("text-anchor", "middle")
      .attr("transform", `translate(15, ${this.height / 2}) rotate(-90)`)
      .style("fill", "#e1e7f5")
      .text("Feature 2");
    
    svg.append("text")
      .attr("class", "plot-title")
      .attr("text-anchor", "middle")
      .attr("x", this.width / 2)
      .attr("y", 20)
      .style("fill", "#ffffff")
      .text("Data Visualization");
    
    // Add labels for legend
    const legend = g.append("g")
      .attr("class", "scatter-legend")
      .attr("transform", `translate(${this.innerWidth - 100}, 20)`)
      .style("opacity", 0);
    
    legend.append("rect")
      .attr("width", 100)
      .attr("height", 60)
      .attr("rx", 5)
      .attr("ry", 5)
      .style("fill", "#162a4a")
      .style("stroke", "#2a4980");
    
    legend.append("circle")
      .attr("cx", 20)
      .attr("cy", 20)
      .attr("r", 5)
      .style("fill", "#4285f4");
    
    legend.append("text")
      .attr("x", 35)
      .attr("y", 25)
      .style("fill", "#e1e7f5")
      .text("Original");
    
    legend.append("circle")
      .attr("cx", 20)
      .attr("cy", 40)
      .attr("r", 5)
      .style("fill", "#ff6b6b");
    
    legend.append("text")
      .attr("x", 35)
      .attr("y", 45)
      .style("fill", "#e1e7f5")
      .text("Reconstructed");
      
    // Draw data points
    g.selectAll(".point")
      .data(this.rawData)
      .enter()
      .append("circle")
      .attr("class", "point")
      .attr("cx", (d: Point2D) => xScale(d.x))
      .attr("cy", (d: Point2D) => yScale(d.y))
      .attr("r", 4)
      .style("fill", "#4285f4")
      .style("opacity", 0.7);
    
    // Store important elements and scales for later use
    this.scatterPlotElements = {
      svg, g, xScale, yScale, legend,
      xAxis: g.select(".x-axis"),
      yAxis: g.select(".y-axis")
    };
  }

  private scatterPlotElements: any;

  private drawScreePlot(): void {
    const container = d3.select(this.screePlotContainer.nativeElement);
    container.selectAll("*").remove();
    
    const svg = container.append("svg")
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("viewBox", `0 0 ${this.width} ${this.height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("class", "scree-svg");
    
    const g = svg.append("g")
      .attr("transform", `translate(${this.margin.left},${this.margin.top})`);
    
    // Set up scales
    const xScale = d3.scaleBand()
      .domain(["PC1", "PC2"])
      .range([0, this.innerWidth])
      .padding(0.3);
    
    const yScale = d3.scaleLinear()
      .domain([0, Math.max(...this.eigenvalues) * 1.1])
      .range([this.innerHeight, 0]);
    
    // Add axes
    g.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.innerHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll("text")
      .style("fill", "#e1e7f5");
    
    g.append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(yScale))
      .selectAll("text")
      .style("fill", "#e1e7f5");
    
    // Add labels
    svg.append("text")
      .attr("class", "x-label")
      .attr("text-anchor", "middle")
      .attr("x", this.width / 2)
      .attr("y", this.height - 10)
      .style("fill", "#e1e7f5")
      .text("Principal Components");
    
    svg.append("text")
      .attr("class", "y-label")
      .attr("text-anchor", "middle")
      .attr("transform", `translate(15, ${this.height / 2}) rotate(-90)`)
      .style("fill", "#e1e7f5")
      .text("Eigenvalue");
    
    svg.append("text")
      .attr("class", "plot-title")
      .attr("text-anchor", "middle")
      .attr("x", this.width / 2)
      .attr("y", 20)
      .style("fill", "#ffffff")
      .text("Scree Plot");
    
    // Initially hidden, will be shown in step 4
    this.screePlotElements = {
      svg, g, xScale, yScale,
      data: [
        { name: "PC1", value: this.eigenvalues[0] },
        { name: "PC2", value: this.eigenvalues[1] }
      ]
    };
  }

  private screePlotElements: any;

  private drawVariancePlot(): void {
    const container = d3.select(this.variancePlotContainer.nativeElement);
    container.selectAll("*").remove();
    
    const svg = container.append("svg")
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("viewBox", `0 0 ${this.width} ${this.height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("class", "variance-svg");
    
    const g = svg.append("g")
      .attr("transform", `translate(${this.margin.left},${this.margin.top})`);
    
    // Set up scales
    const xScale = d3.scaleBand()
      .domain(["PC1", "PC2"])
      .range([0, this.innerWidth])
      .padding(0.3);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([this.innerHeight, 0]);
    
    // Add axes
    g.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.innerHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll("text")
      .style("fill", "#e1e7f5");
    
    g.append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(yScale).tickFormat(d3.format(".0%")))
      .selectAll("text")
      .style("fill", "#e1e7f5");
    
    // Add labels
    svg.append("text")
      .attr("class", "x-label")
      .attr("text-anchor", "middle")
      .attr("x", this.width / 2)
      .attr("y", this.height - 10)
      .style("fill", "#e1e7f5")
      .text("Principal Components");
    
    svg.append("text")
      .attr("class", "y-label")
      .attr("text-anchor", "middle")
      .attr("transform", `translate(15, ${this.height / 2}) rotate(-90)`)
      .style("fill", "#e1e7f5")
      .text("Explained Variance");
    
    svg.append("text")
      .attr("class", "plot-title")
      .attr("text-anchor", "middle")
      .attr("x", this.width / 2)
      .attr("y", 20)
      .style("fill", "#ffffff")
      .text("Explained Variance");
    
    // Initially hidden, will be shown in step 4
    this.variancePlotElements = {
      svg, g, xScale, yScale,
      data: [
        { name: "PC1", value: this.explainedVariance[0] },
        { name: "PC2", value: this.explainedVariance[1] }
      ],
      cumulative: [
        { name: "PC1", value: this.cumulativeVariance[0] },
        { name: "PC2", value: this.cumulativeVariance[1] }
      ]
    };
  }

  private variancePlotElements: any;

  // Handle visualization steps
  public handleCurrentStep(): void {
    const { svg, g, xScale, yScale } = this.scatterPlotElements;
    
    switch (this.currentStep) {
      case 0: // Original Data
        this.drawOriginalData();
        break;
      case 1: // Standardization
        this.animateStandardization();
        break;
      case 2: // Find Principal Components
        this.showPrincipalComponents();
        break;
      case 3: // Project Data
        this.projectOntoComponents();
        break;
      case 4: // Variance Explained
        this.showVarianceExplained();
        break;
      case 5: // Reconstruction
        this.showReconstruction();
        break;
    }
  }

  private drawOriginalData(): void {
    const { svg, g, xScale, yScale } = this.scatterPlotElements;
    
    // Reset any previous transformations
    g.selectAll(".pc-line, .projection-line, .mean-point, .ellipse").remove();
    
    // Update points to original data
    g.selectAll(".point")
      .data(this.rawData)
      .transition()
      .duration(500)
      .attr("cx", (d: Point2D) => xScale(d.x))
      .attr("cy", (d: Point2D) => yScale(d.y))
      .style("fill", "#4285f4");
    
    // Update axis labels
    svg.select(".x-label").text("Feature 1");
    svg.select(".y-label").text("Feature 2");
    svg.select(".plot-title").text("Original Data");
  }

  private animateStandardization(): void {
    const { svg, g, xScale, yScale, xAxis, yAxis } = this.scatterPlotElements;
    
    // Remove any previous additions
    g.selectAll(".pc-line, .projection-line, .ellipse").remove();
    
    // 1. Show the mean point
    g.append("circle")
      .attr("class", "mean-point")
      .attr("cx", xScale(this.mean.x))
      .attr("cy", yScale(this.mean.y))
      .attr("r", 6)
      .style("fill", "#ff9d45")
      .style("stroke", "#ffffff")
      .style("stroke-width", 2);
    
    g.append("text")
      .attr("class", "mean-label")
      .attr("x", xScale(this.mean.x) + 10)
      .attr("y", yScale(this.mean.y) - 10)
      .style("fill", "#e1e7f5")
      .text("Mean");
    
    // 2. Animate movement to center (subtract mean)
    setTimeout(() => {
      // Update scatter points to move to standardized positions
      // First translate (center)
      const xMean = this.mean.x;
      const yMean = this.mean.y;
      
      g.selectAll(".point")
        .data(this.rawData)
        .transition()
        .duration(1000)
        .attr("cx", (d: Point2D) => xScale(d.x - xMean))
        .attr("cy", (d: Point2D) => yScale(d.y - yMean));
      
      // Update mean point
      g.select(".mean-point")
        .transition()
        .duration(1000)
        .attr("cx", xScale(0))
        .attr("cy", yScale(0));
      
      g.select(".mean-label")
        .transition()
        .duration(1000)
        .attr("x", xScale(0) + 10)
        .attr("y", yScale(0) - 10);
      
      // 3. Then scale (divide by standard deviation)
      setTimeout(() => {
        const tempData = this.rawData.map(d => ({
          x: (d.x - xMean),
          y: (d.y - yMean)
        }));
        
        // Update x and y scales for standardized data
        const stdMaxX = Math.max(
          Math.abs(d3.min(tempData, d => d.x) || 0),
          Math.abs(d3.max(tempData, d => d.x) || 0)
        ) / this.stdDev.x * 1.5;
        
        const stdMaxY = Math.max(
          Math.abs(d3.min(tempData, d => d.y) || 0),
          Math.abs(d3.max(tempData, d => d.y) || 0)
        ) / this.stdDev.y * 1.5;
        
        const newXScale = d3.scaleLinear()
          .domain([-stdMaxX, stdMaxX])
          .range([0, this.innerWidth]);
        
        const newYScale = d3.scaleLinear()
          .domain([-stdMaxY, stdMaxY])
          .range([this.innerHeight, 0]);
        
        // Update axes
        xAxis.transition()
          .duration(1000)
          .call(d3.axisBottom(newXScale));
        
        yAxis.transition()
          .duration(1000)
          .call(d3.axisLeft(newYScale));
        
        // Update points to standardized positions
        g.selectAll(".point")
          .data(this.stdData)
          .transition()
          .duration(1000)
          .attr("cx", (d: Point2D) => newXScale(d.x))
          .attr("cy", (d: Point2D) => newYScale(d.y));
        
        // Update labels
        svg.select(".x-label")
          .text("Standardized Feature 1");
        
        svg.select(".y-label")
          .text("Standardized Feature 2");
        
        svg.select(".plot-title")
          .text("Standardized Data");
        
        // Store the new scales
        this.scatterPlotElements.xScale = newXScale;
        this.scatterPlotElements.yScale = newYScale;
      }, 1200);
    }, 800);
  }

  private showPrincipalComponents(): void {
    const { g, xScale, yScale, svg } = this.scatterPlotElements;
    
    // Update the plot title
    svg.select(".plot-title")
      .text("Principal Components");
    
    // Add covariance ellipse to visualize the variance directions
    const [eigenvector1, eigenvector2] = this.eigenvectors;
    const [eigenvalue1, eigenvalue2] = this.eigenvalues;
    
    // Scale eigenvectors by their eigenvalues for visualization
    const scaleFactor = 2;
    const scaledEigenvector1 = [
      eigenvector1[0] * Math.sqrt(eigenvalue1) * scaleFactor,
      eigenvector1[1] * Math.sqrt(eigenvalue1) * scaleFactor
    ];
    
    const scaledEigenvector2 = [
      eigenvector2[0] * Math.sqrt(eigenvalue2) * scaleFactor,
      eigenvector2[1] * Math.sqrt(eigenvalue2) * scaleFactor
    ];
    
    // Draw PC1 direction (major axis of ellipse)
    g.append("line")
      .attr("class", "pc-line pc1-line")
      .attr("x1", xScale(-scaledEigenvector1[0]))
      .attr("y1", yScale(-scaledEigenvector1[1]))
      .attr("x2", xScale(scaledEigenvector1[0]))
      .attr("y2", yScale(scaledEigenvector1[1]))
      .style("stroke", "#7c4dff")
      .style("stroke-width", 3)
      .style("stroke-dasharray", "5,0")
      .style("opacity", 0)
      .transition()
      .duration(800)
      .style("opacity", 1);
    
    // Add label for PC1
    g.append("text")
      .attr("class", "pc-label pc1-label")
      .attr("x", xScale(scaledEigenvector1[0] * 0.7))
      .attr("y", yScale(scaledEigenvector1[1] * 0.7) - 10)
      .style("fill", "#7c4dff")
      .style("font-weight", "bold")
      .style("opacity", 0)
      .text("PC1")
      .transition()
      .duration(800)
      .style("opacity", 1);
    
    // Draw PC2 direction (minor axis of ellipse)
    setTimeout(() => {
      g.append("line")
        .attr("class", "pc-line pc2-line")
        .attr("x1", xScale(-scaledEigenvector2[0]))
        .attr("y1", yScale(-scaledEigenvector2[1]))
        .attr("x2", xScale(scaledEigenvector2[0]))
        .attr("y2", yScale(scaledEigenvector2[1]))
        .style("stroke", "#00c9ff")
        .style("stroke-width", 2)
        .style("stroke-dasharray", "5,0")
        .style("opacity", 0)
        .transition()
        .duration(800)
        .style("opacity", 1);
      
      // Add label for PC2
      g.append("text")
        .attr("class", "pc-label pc2-label")
        .attr("x", xScale(scaledEigenvector2[0] * 0.7))
        .attr("y", yScale(scaledEigenvector2[1] * 0.7) - 10)
        .style("fill", "#00c9ff")
        .style("font-weight", "bold")
        .style("opacity", 0)
        .text("PC2")
        .transition()
        .duration(800)
        .style("opacity", 1);
      
      // Add covariance ellipse
      setTimeout(() => {
        const ellipseData = this.generateEllipsePoints(
          0, 0,
          Math.sqrt(eigenvalue1) * scaleFactor,
          Math.sqrt(eigenvalue2) * scaleFactor,
          Math.atan2(eigenvector1[1], eigenvector1[0])
        );
        
        const lineGenerator = d3.line<Point2D>()
          .x(d => xScale(d.x))
          .y(d => yScale(d.y));
        
        g.append("path")
          .attr("class", "ellipse")
          .attr("d", lineGenerator(ellipseData))
          .style("stroke", "#24b47e")
          .style("stroke-width", 2)
          .style("fill", "#24b47e")
          .style("fill-opacity", 0.1)
          .style("opacity", 0)
          .transition()
          .duration(800)
          .style("opacity", 1);
      }, 800);
    }, 800);
  }

  private generateEllipsePoints(cx: number, cy: number, rx: number, ry: number, rotation: number, pointCount = 50): Array<Point2D> {
    const points: Array<Point2D> = [];
    const cosR = Math.cos(rotation);
    const sinR = Math.sin(rotation);
    
    for (let i = 0; i < pointCount; i++) {
      const angle = (i / pointCount) * 2 * Math.PI;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      
      // Ellipse point in its own coordinate system
      const xPrime = rx * cosA;
      const yPrime = ry * sinA;
      
      // Rotate and translate
      const x = cx + xPrime * cosR - yPrime * sinR;
      const y = cy + xPrime * sinR + yPrime * cosR;
      
      points.push({x, y});
    }
    
    return points;
  }

  private projectOntoComponents(): void {
    const { g, xScale, yScale, svg } = this.scatterPlotElements;
    
    // Update the plot title
    svg.select(".plot-title")
      .text("Projection onto Principal Components");
    
    // First, update the original points to a lighter color
    g.selectAll(".point")
      .transition()
      .duration(500)
      .style("fill", "#8bb4fa")
      .style("opacity", 0.4);
    
    // Get principal component vectors
    const [eigenvector1, eigenvector2] = this.eigenvectors;
    
    // Add projected points (one at a time with animated projections)
    setTimeout(() => {
      const pointsToShow = 15; // Show projections for a subset of points
      const pointIndices = Array.from({length: this.numPoints}, (_, i) => i)
        .sort(() => Math.random() - 0.5)
        .slice(0, pointsToShow);
      
      let pointIndex = 0;
      
      const showNextPoint = () => {
        if (pointIndex >= pointIndices.length) return;
        
        const i = pointIndices[pointIndex];
        const point = this.stdData[i];
        const projected = this.projectedData[i];
        
        // Calculate the projected point coordinates in original space
        const pc1Projected = {
          x: projected.pc1 * eigenvector1[0],
          y: projected.pc1 * eigenvector1[1]
        };
        
        // Draw projection line
        g.append("line")
          .attr("class", "projection-line")
          .attr("x1", xScale(point.x))
          .attr("y1", yScale(point.y))
          .attr("x2", xScale(point.x))
          .attr("y2", yScale(point.y))
          .style("stroke", "#ff6b6b")
          .style("stroke-width", 1.5)
          .style("stroke-dasharray", "3,3")
          .transition()
          .duration(500)
          .attr("x2", xScale(pc1Projected.x))
          .attr("y2", yScale(pc1Projected.y));
        
        // Add projected point
        g.append("circle")
          .attr("class", "projected-point")
          .attr("cx", xScale(point.x))
          .attr("cy", yScale(point.y))
          .attr("r", 4)
          .style("fill", "#7c4dff")
          .style("opacity", 0)
          .transition()
          .delay(500)
          .duration(300)
          .attr("cx", xScale(pc1Projected.x))
          .attr("cy", yScale(pc1Projected.y))
          .style("opacity", 1);
        
        pointIndex++;
        setTimeout(showNextPoint, 100);
      };
      
      showNextPoint();
      
      // After showing individual projections, show all projected points in new coordinate system
      setTimeout(() => {
        // Clear previous elements
        g.selectAll(".projection-line, .projected-point").remove();
        
        // Change the axes to PC1 and PC2
        const pc1Range = d3.extent(this.projectedData, d => d.pc1) as [number, number];
        const pc2Range = d3.extent(this.projectedData, d => d.pc2) as [number, number];
        
        const padding = 0.5;
        const newXDomain = [
          (pc1Range[0] || -1) - padding, 
          (pc1Range[1] || 1) + padding
        ];
        const newYDomain = [
          (pc2Range[0] || -1) - padding, 
          (pc2Range[1] || 1) + padding
        ];
        
        // Create new scales for PC coordinates
        const pcXScale = d3.scaleLinear()
          .domain(newXDomain)
          .range([0, this.innerWidth]);
        
        const pcYScale = d3.scaleLinear()
          .domain(newYDomain)
          .range([this.innerHeight, 0]);
        
        // Update axes with transition
        g.select(".x-axis")
          .transition()
          .duration(1000)
          .call(d3.axisBottom(pcXScale));
        
        g.select(".y-axis")
          .transition()
          .duration(1000)
          .call(d3.axisLeft(pcYScale));
        
        // Update labels
        svg.select(".x-label")
          .text("Principal Component 1");
        
        svg.select(".y-label")
          .text("Principal Component 2");
        
        // Hide original points
        g.selectAll(".point")
          .transition()
          .duration(500)
          .style("opacity", 0);
        
        // Hide PC lines and ellipse
        g.selectAll(".pc-line, .pc-label, .ellipse")
          .transition()
          .duration(500)
          .style("opacity", 0);
        
        // Add new points in PC space
        setTimeout(() => {
          g.selectAll(".pc-point")
            .data(this.projectedData)
            .enter()
            .append("circle")
            .attr("class", "pc-point")
            .attr("cx", (d: ProjectedPoint) => pcXScale(0))
            .attr("cy", (d: ProjectedPoint) => pcYScale(0))
            .attr("r", 4)
            .style("fill", "#7c4dff")
            .style("opacity", 0)
            .transition()
            .duration(1000)
            .attr("cx", (d: ProjectedPoint) => pcXScale(d.pc1))
            .attr("cy", (d: ProjectedPoint) => pcYScale(d.pc2))
            .style("opacity", 0.7);
          
          // Store PC scales for later use
          this.scatterPlotElements.pcXScale = pcXScale;
          this.scatterPlotElements.pcYScale = pcYScale;
          
        }, 600);
      }, pointsToShow * 100 + 1000);
    }, 500);
  }

  private showVarianceExplained(): void {
    const { svg } = this.scatterPlotElements;
    
    // Update the main plot title
    svg.select(".plot-title")
      .text("Variance Explained by Principal Components");
    
    // Show scree plot
    const { g, xScale, yScale, data } = this.screePlotElements;
    
    // Draw bars
    g.selectAll(".scree-bar")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "scree-bar")
      .attr("x", (d: NamedValue) => xScale(d.name))
      .attr("y", yScale(0))
      .attr("width", xScale.bandwidth())
      .attr("height", 0)
      .attr("fill", (d: NamedValue, i: number) => i === 0 ? "#7c4dff" : "#00c9ff")
      .transition()
      .duration(1000)
      .attr("y", (d: NamedValue) => yScale(d.value))
      .attr("height", (d: NamedValue) => this.innerHeight - yScale(d.value));
    
    // Add value labels
    g.selectAll(".scree-label")
      .data(data)
      .enter()
      .append("text")
      .attr("class", "scree-label")
      .attr("x", (d: NamedValue) => xScale(d.name) + xScale.bandwidth() / 2)
      .attr("y", (d: NamedValue) => yScale(d.value) - 10)
      .attr("text-anchor", "middle")
      .style("fill", "#e1e7f5")
      .style("font-size", "12px")
      .style("opacity", 0)
      .text((d: NamedValue) => d.value.toFixed(2))
      .transition()
      .delay(1000)
      .duration(500)
      .style("opacity", 1);
    
    // Show variance plot
    setTimeout(() => {
      const { g, xScale, yScale, data, cumulative } = this.variancePlotElements;
      
      // Draw bars for explained variance
      g.selectAll(".variance-bar")
        .data(data)
        .enter()
        .append("rect")
        .attr("class", "variance-bar")
        .attr("x", (d: NamedValue) => xScale(d.name))
        .attr("y", yScale(0))
        .attr("width", xScale.bandwidth())
        .attr("height", 0)
        .attr("fill", (d: NamedValue, i: number) => i === 0 ? "#7c4dff" : "#00c9ff")
        .transition()
        .duration(1000)
        .attr("y", (d: NamedValue) => yScale(d.value))
        .attr("height", (d: NamedValue) => this.innerHeight - yScale(d.value));
      
      // Add value labels
      g.selectAll(".variance-label")
        .data(data)
        .enter()
        .append("text")
        .attr("class", "variance-label")
        .attr("x", (d: NamedValue) => xScale(d.name) + xScale.bandwidth() / 2)
        .attr("y", (d: NamedValue) => yScale(d.value) - 10)
        .attr("text-anchor", "middle")
        .style("fill", "#e1e7f5")
        .style("font-size", "12px")
        .style("opacity", 0)
        .text((d: NamedValue) => (d.value * 100).toFixed(1) + "%")
        .transition()
        .delay(1000)
        .duration(500)
        .style("opacity", 1);
      
      // Add cumulative line
      const lineGenerator = d3.line<NamedValue>()
        .x(d => xScale(d.name) + xScale.bandwidth() / 2)
        .y(d => yScale(d.value));
      
      setTimeout(() => {
        g.append("path")
          .datum(cumulative)
          .attr("class", "cumulative-line")
          .attr("d", lineGenerator)
          .style("fill", "none")
          .style("stroke", "#24b47e")
          .style("stroke-width", 3)
          .style("stroke-dasharray", (d: NamedValue[]) => d.map((p: NamedValue) => p.value * this.innerHeight).join(","))
          .style("stroke-dashoffset", this.innerHeight)
          .transition()
          .duration(1000)
          .style("stroke-dashoffset", 0);
        
        // Add cumulative points
        g.selectAll(".cumulative-point")
          .data(cumulative)
          .enter()
          .append("circle")
          .attr("class", "cumulative-point")
          .attr("cx", (d: NamedValue) => xScale(d.name) + xScale.bandwidth() / 2)
          .attr("cy", (d: NamedValue) => yScale(d.value))
          .attr("r", 5)
          .style("fill", "#24b47e")
          .style("opacity", 0)
          .transition()
          .delay((d: NamedValue, i: number) => 1000 + i * 300)
          .duration(300)
          .style("opacity", 1);
        
        // Add cumulative labels
        g.selectAll(".cumulative-label")
          .data(cumulative)
          .enter()
          .append("text")
          .attr("class", "cumulative-label")
          .attr("x", (d: NamedValue) => xScale(d.name) + xScale.bandwidth() / 2)
          .attr("y", (d: NamedValue) => yScale(d.value) - 15)
          .attr("text-anchor", "middle")
          .style("fill", "#24b47e")
          .style("opacity", 0)
          .text((d: NamedValue) => (d.value * 100).toFixed(1) + "%")
          .transition()
          .delay((d: NamedValue, i: number) => 1000 + i * 300)
          .duration(300)
          .style("opacity", 1);
        
        // Add legend
        const legend = g.append("g")
          .attr("class", "legend")
          .attr("transform", `translate(${this.innerWidth - 150}, 20)`);
        
        legend.append("rect")
          .attr("x", 0)
          .attr("y", 0)
          .attr("width", 150)
          .attr("height", 70)
          .style("fill", "#162a4a")
          .style("stroke", "#2a4980")
          .style("opacity", 0)
          .transition()
          .duration(500)
          .style("opacity", 1);
        
        legend.append("rect")
          .attr("x", 10)
          .attr("y", 10)
          .attr("width", 15)
          .attr("height", 15)
          .style("fill", "#7c4dff")
          .style("opacity", 0)
          .transition()
          .delay(500)
          .duration(300)
          .style("opacity", 1);
        
        legend.append("text")
          .attr("x", 35)
          .attr("y", 22)
          .style("fill", "#e1e7f5")
          .style("opacity", 0)
          .text("Individual")
          .transition()
          .delay(500)
          .duration(300)
          .style("opacity", 1);
        
        legend.append("circle")
          .attr("cx", 17.5)
          .attr("cy", 42.5)
          .attr("r", 7.5)
          .style("fill", "#24b47e")
          .style("opacity", 0)
          .transition()
          .delay(800)
          .duration(300)
          .style("opacity", 1);
        
        legend.append("text")
          .attr("x", 35)
          .attr("y", 47)
          .style("fill", "#e1e7f5")
          .style("opacity", 0)
          .text("Cumulative")
          .transition()
          .delay(800)
          .duration(300)
          .style("opacity", 1);
        
      }, 1500);
    }, 1000);
  }

  private showReconstruction(): void {
    const { svg, g, xScale, yScale, legend } = this.scatterPlotElements;
    
    // Clear current visualization
    g.selectAll("*").remove();
    
    // Reset axes to original scale
    const xMax = d3.max(this.rawData, d => d.x) || 0;
    const xMin = d3.min(this.rawData, d => d.x) || 0;
    const yMax = d3.max(this.rawData, d => d.y) || 0;
    const yMin = d3.min(this.rawData, d => d.y) || 0;
    
    const xPadding = (xMax - xMin) * 0.1;
    const yPadding = (yMax - yMin) * 0.1;
    
    const newXScale = d3.scaleLinear()
      .domain([xMin - xPadding, xMax + xPadding])
      .range([0, this.innerWidth]);
    
    const newYScale = d3.scaleLinear()
      .domain([yMin - yPadding, yMax + yPadding])
      .range([this.innerHeight, 0]);
    
    // Add axes
    const xAxis = g.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.innerHeight})`)
      .call(d3.axisBottom(newXScale))
      .selectAll("text")
      .style("fill", "#e1e7f5");
    
    const yAxis = g.append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(newYScale))
      .selectAll("text")
      .style("fill", "#e1e7f5");
    
    // Update labels
    svg.select(".x-label").text("Feature 1");
    svg.select(".y-label").text("Feature 2");
    svg.select(".plot-title").text("Original vs. Reconstructed Data");
    
    // Draw original data points
    g.selectAll(".original-point")
      .data(this.rawData)
      .enter()
      .append("circle")
      .attr("class", "original-point")
      .attr("cx", (d: Point2D) => newXScale(d.x))
      .attr("cy", (d: Point2D) => newYScale(d.y))
      .attr("r", 4)
      .style("fill", "#4285f4")
      .style("opacity", 0.7);
    
    // Show legend
    const legendGroup = g.append("g")
      .attr("class", "legend")
      .attr("transform", `translate(${this.innerWidth - 150}, 20)`);
    
    legendGroup.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", 150)
      .attr("height", 70)
      .style("fill", "#162a4a")
      .style("stroke", "#2a4980");
    
    legendGroup.append("circle")
      .attr("cx", 17.5)
      .attr("cy", 17.5)
      .attr("r", 4)
      .style("fill", "#4285f4");
    
    legendGroup.append("text")
      .attr("x", 35)
      .attr("y", 22)
      .style("fill", "#e1e7f5")
      .text("Original");
    
    legendGroup.append("circle")
      .attr("cx", 17.5)
      .attr("cy", 47.5)
      .attr("r", 4)
      .style("fill", "#ff6b6b")
      .style("opacity", 0);
    
    legendGroup.append("text")
      .attr("x", 35)
      .attr("y", 52)
      .style("fill", "#e1e7f5")
      .style("opacity", 0)
      .text("Reconstructed");
    
    // Show reconstructed points with a delay
    setTimeout(() => {
      // Animate the legend for reconstructed points
      legend.select("circle:nth-of-type(2)")
        .transition()
        .duration(500)
        .style("opacity", 1);
      
      legend.select("text:nth-of-type(2)")
        .transition()
        .duration(500)
        .style("opacity", 1);
      
      // Add reconstructed points one by one
      let pointIndex = 0;
      
      const addNextPoint = () => {
        if (pointIndex >= this.rawData.length) return;
        
        const originalPoint = this.rawData[pointIndex];
        const reconstructedPoint = this.reconstructedData[pointIndex];
        
        // Add reconstructed point
        g.append("circle")
          .attr("class", "reconstructed-point")
          .attr("cx", newXScale(reconstructedPoint.x))
          .attr("cy", newYScale(reconstructedPoint.y))
          .attr("r", 0)
          .style("fill", "#ff6b6b")
          .transition()
          .duration(300)
          .attr("r", 4)
          .style("opacity", 0.7);
        
        // Add line connecting original and reconstructed
        g.append("line")
          .attr("class", "reconstruction-line")
          .attr("x1", newXScale(originalPoint.x))
          .attr("y1", newYScale(originalPoint.y))
          .attr("x2", newXScale(originalPoint.x))
          .attr("y2", newYScale(originalPoint.y))
          .style("stroke", "#8a9ab0")
          .style("stroke-width", 1)
          .style("stroke-dasharray", "3,3")
          .transition()
          .duration(300)
          .attr("x2", newXScale(reconstructedPoint.x))
          .attr("y2", newYScale(reconstructedPoint.y));
        
        pointIndex += 5; // Skip some points for clarity
        setTimeout(addNextPoint, 50);
      };
      
      addNextPoint();
    }, 1000);
    
    // Store the new scales
    this.scatterPlotElements.xScale = newXScale;
    this.scatterPlotElements.yScale = newYScale;
  }

  public nextStep(): void {
    this.stopAutoPlay();
    if (this.currentStep < this.maxSteps) {
      this.currentStep++;
      this.updateStepDescription();
      this.handleCurrentStep();
    }
  }

  public prevStep(): void {
    this.stopAutoPlay();
    if (this.currentStep > 0) {
      this.currentStep--;
      this.updateStepDescription();
      this.handleCurrentStep();
    }
  }

  public toggleAutoPlay(): void {
    if (this.isPlaying) {
      this.stopAutoPlay();
    } else {
      this.startAutoPlay();
    }
  }

  private startAutoPlay(): void {
    this.isPlaying = true;
    this.autoPlayInterval = setInterval(() => {
      this.zone.run(() => {
        if (this.currentStep < this.maxSteps) {
          this.currentStep++;
          this.updateStepDescription();
          this.handleCurrentStep();
        } else {
          this.stopAutoPlay();
        }
      });
    }, this.stepInterval);
  }

  private stopAutoPlay(): void {
    if (this.autoPlayInterval) {
      clearInterval(this.autoPlayInterval);
      this.autoPlayInterval = null;
    }
    this.isPlaying = false;
  }

  public updateStepDescription(): void {
    const descriptions = [
      "This is the original dataset in its raw form. The data points show a pattern with correlation between the two features. PCA will help us identify the main directions of variation.",
      
      "Before applying PCA, we standardize the data by subtracting the mean and dividing by the standard deviation. This ensures that features with larger scales don't dominate the analysis.",
      
      "PCA finds the directions (Principal Components) where the data varies the most. PC1 captures the maximum variance, while PC2 is orthogonal to PC1 and captures the second most variance. These directions are the eigenvectors of the covariance matrix.",
      
      "We can project our data onto the principal components, creating a new coordinate system. This transforms our data from the original feature space to the principal component space.",
      
      "The eigenvalues tell us how much variance each principal component explains. The scree plot shows the eigenvalues, while the variance plot shows the proportion of total variance explained by each component, both individually and cumulatively.",
      
      "We can reconstruct the original data using only the principal components we selected. This demonstrates how PCA can compress data while preserving important patterns. The difference between original and reconstructed points shows information loss."
    ];
    
    this.currentStepDescription = descriptions[this.currentStep];
  }
}