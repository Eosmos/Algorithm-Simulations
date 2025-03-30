import { Component, OnInit, ElementRef, ViewChild, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';

// Define interfaces for our data types
interface DataPoint {
  x1: number;
  x2: number;
  y: number;
}

interface TestPoint {
  x1: number;
  x2: number;
}

interface WeightsType {
  w0: number;
  w1: number;
  w2: number;
}

@Component({
  selector: 'app-logistic-regression-simulation',
  templateUrl: './logistic-regression-simulation.component.html',
  styleUrls: ['./logistic-regression-simulation.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule, DecimalPipe]
})
export class LogisticRegressionSimulationComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('dataPlot', { static: false }) dataPlotElement!: ElementRef;
  @ViewChild('sigmoidPlot', { static: false }) sigmoidPlotElement!: ElementRef;
  @ViewChild('costPlot', { static: false }) costPlotElement!: ElementRef;
  @ViewChild('coefficientsPlot', { static: false }) coefficientsPlotElement!: ElementRef;

  // Data
  private data: DataPoint[] = [];
  private testPoint: TestPoint = {x1: 5, x2: 5};
  
  // Model parameters
  private weights: WeightsType = {w0: 0, w1: 0, w2: 0}; // bias, weight1, weight2
  public learningRate = 0.1;
  public iterations = 100;
  private currentIteration = 0;
  
  // Animation control
  private animationId: number | null = null;
  public isPlaying = false;
  public speed = 200; // ms between iterations
  public showProbabilities = false;
  
  // History tracking
  private weightsHistory: WeightsType[] = [];
  private costHistory: number[] = [];
  
  // Visualization dimensions
  private margin = {top: 40, right: 40, bottom: 50, left: 60};
  private width = 500 - this.margin.left - this.margin.right;
  private height = 380 - this.margin.top - this.margin.bottom;
  
  // D3 elements
  private svgs: {[key: string]: d3.Selection<SVGGElement, unknown, null, undefined>} = {};
  private decisionBoundary: d3.Selection<SVGLineElement, unknown, null, undefined> | null = null;
  private probabilities: d3.Selection<SVGGElement, unknown, null, undefined> | null = null;
  
  // Current step explanation
  public currentStep = 0;
  public steps = [
    'Initialize random weights for the model',
    'Calculate initial probabilities using sigmoid function',
    'Compute cost (error) using binary cross-entropy',
    'Update weights using gradient descent',
    'Recalculate probabilities with new weights',
    'Visualize new decision boundary',
    'Repeat until convergence or max iterations'
  ];
  
  constructor() { }

  ngOnInit(): void {
    this.generateData();
    this.initializeWeights();
  }

  ngAfterViewInit(): void {
    // Allow Angular to complete rendering cycle before initializing D3
    setTimeout(() => {
      this.initializeVisualization();
    }, 0);
  }

  ngOnDestroy(): void {
    this.stopAnimation();
  }

  private generateData(): void {
    // Generate two clusters of points for binary classification
    const n = 100; // 50 points per class
    
    // Class 0: cluster around (3, 3)
    for (let i = 0; i < n/2; i++) {
      this.data.push({
        x1: 3 + d3.randomNormal(0, 0.8)(),
        x2: 3 + d3.randomNormal(0, 0.8)(),
        y: 0
      });
    }
    
    // Class 1: cluster around (7, 7)
    for (let i = 0; i < n/2; i++) {
      this.data.push({
        x1: 7 + d3.randomNormal(0, 0.8)(),
        x2: 7 + d3.randomNormal(0, 0.8)(),
        y: 1
      });
    }

    // Add some slightly overlapping points for realism
    for (let i = 0; i < 10; i++) {
      this.data.push({
        x1: 5 + d3.randomNormal(0, 0.5)(),
        x2: 5 + d3.randomNormal(0, 0.5)(),
        y: Math.random() > 0.5 ? 1 : 0
      });
    }
  }

  private initializeWeights(): void {
    // Initialize weights randomly
    this.weights = {
      w0: Math.random() * 2 - 1, // bias
      w1: Math.random() * 2 - 1, // weight for x1
      w2: Math.random() * 2 - 1  // weight for x2
    };
    
    // Track initial weights
    this.weightsHistory = [{ ...this.weights }];
  }

  private initializeVisualization(): void {
    this.initializeDataPlot();
    this.initializeSigmoidPlot();
    this.initializeCostPlot();
    this.initializeCoefficientsPlot();
  }

  private initializeDataPlot(): void {
    // Create SVG for data visualization
    this.svgs['dataPlot'] = d3.select(this.dataPlotElement.nativeElement)
      .append('svg')
      .attr('width', this.width + this.margin.left + this.margin.right)
      .attr('height', this.height + this.margin.top + this.margin.bottom)
      .append('g')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`) as d3.Selection<SVGGElement, unknown, null, undefined>;
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, 10])
      .range([0, this.width]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 10])
      .range([this.height, 0]);
    
    // Add axes
    this.svgs['dataPlot'].append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.height})`)
      .call(d3.axisBottom(xScale));
    
    this.svgs['dataPlot'].append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(yScale));
    
    // Add axes labels
    this.svgs['dataPlot'].append("text")
      .attr("transform", `translate(${this.width/2},${this.height + 35})`)
      .style("text-anchor", "middle")
      .text("Feature X₁");
    
    this.svgs['dataPlot'].append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -45)
      .attr("x", -this.height/2)
      .style("text-anchor", "middle")
      .text("Feature X₂");
    
    // Add title
    this.svgs['dataPlot'].append("text")
      .attr("x", this.width / 2)
      .attr("y", -20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text("Decision Boundary Evolution");
    
    // Draw data points
    this.svgs['dataPlot'].selectAll("circle.data-point")
      .data(this.data)
      .enter()
      .append("circle")
      .attr("class", "data-point")
      .attr("cx", (d: DataPoint) => xScale(d.x1))
      .attr("cy", (d: DataPoint) => yScale(d.x2))
      .attr("r", 4.5)
      .style("fill", (d: DataPoint) => d.y === 1 ? "#ff7f0e" : "#1f77b4")
      .style("opacity", 0.8);
    
    // Create a group for probability circles
    this.probabilities = this.svgs['dataPlot'].append("g")
      .attr("class", "probabilities") as d3.Selection<SVGGElement, unknown, null, undefined>;
      
    // Initialize decision boundary line
    this.decisionBoundary = this.svgs['dataPlot'].append("line")
      .attr("class", "decision-boundary")
      .style("stroke", "#d62728")
      .style("stroke-width", 2.5)
      .style("stroke-dasharray", "5,3") as d3.Selection<SVGLineElement, unknown, null, undefined>;
    
    this.updateDecisionBoundary(xScale, yScale);
    
    // Add interactive test point
    const testPointCircle = this.svgs['dataPlot'].append("circle")
      .attr("class", "test-point")
      .attr("r", 7)
      .style("fill", "#2ca02c")
      .style("opacity", 0.7)
      .style("stroke", "#000")
      .style("stroke-width", 1)
      .style("cursor", "move");
    
    // Add test point label
    const testPointLabel = this.svgs['dataPlot'].append("text")
      .attr("class", "test-point-label")
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .style("text-anchor", "middle")
      .text("Test Point");
    
    // Make test point draggable
    const drag = d3.drag<SVGCircleElement, unknown>()
      .on("drag", (event) => {
        // Constrain to plot area
        const x = Math.max(0, Math.min(this.width, event.x));
        const y = Math.max(0, Math.min(this.height, event.y));
        
        this.testPoint.x1 = xScale.invert(x);
        this.testPoint.x2 = yScale.invert(y);
        
        testPointCircle
          .attr("cx", x)
          .attr("cy", y);
        
        testPointLabel
          .attr("x", x)
          .attr("y", y - 15);
        
        // Update probability in sigmoid plot
        this.updateTestPointProbability();
      });
    
    testPointCircle.call(drag as any);
    
    // Position test point initially
    testPointCircle
      .attr("cx", xScale(this.testPoint.x1))
      .attr("cy", yScale(this.testPoint.x2));
    
    testPointLabel
      .attr("x", xScale(this.testPoint.x1))
      .attr("y", yScale(this.testPoint.x2) - 15);
  }

  private initializeSigmoidPlot(): void {
    // Create SVG for sigmoid function visualization
    this.svgs['sigmoidPlot'] = d3.select(this.sigmoidPlotElement.nativeElement)
      .append('svg')
      .attr('width', this.width + this.margin.left + this.margin.right)
      .attr('height', this.height + this.margin.top + this.margin.bottom)
      .append('g')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`) as d3.Selection<SVGGElement, unknown, null, undefined>;
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([-8, 8])
      .range([0, this.width]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([this.height, 0]);
    
    // Add axes
    this.svgs['sigmoidPlot'].append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.height})`)
      .call(d3.axisBottom(xScale));
    
    this.svgs['sigmoidPlot'].append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(yScale).tickFormat(d3.format(".1f")));
    
    // Add axes labels
    this.svgs['sigmoidPlot'].append("text")
      .attr("transform", `translate(${this.width/2},${this.height + 35})`)
      .style("text-anchor", "middle")
      .text("z = w₀ + w₁x₁ + w₂x₂");
    
    this.svgs['sigmoidPlot'].append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -45)
      .attr("x", -this.height/2)
      .style("text-anchor", "middle")
      .text("Probability P(y=1|x)");
    
    // Add title
    this.svgs['sigmoidPlot'].append("text")
      .attr("x", this.width / 2)
      .attr("y", -20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text("Sigmoid Function (Logistic Function)");
    
    // Add sigmoid function formula
    this.svgs['sigmoidPlot'].append("text")
      .attr("x", this.width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-style", "italic")
      .style("font-size", "12px")
      .text("P(y=1|x) = 1 / (1 + e^(-z))");
    
    // Draw sigmoid function
    const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
    const points = Array.from({length: 160}, (_, i) => {
      const z = -8 + i * 0.1;
      return {z, p: sigmoid(z)};
    });
    
    const line = d3.line<{z: number, p: number}>()
      .x(d => xScale(d.z))
      .y(d => yScale(d.p))
      .curve(d3.curveMonotoneX);
    
    this.svgs['sigmoidPlot'].append("path")
      .attr("class", "sigmoid-curve")
      .datum(points)
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 2.5)
      .attr("d", line);
    
    // Add threshold line at p=0.5
    this.svgs['sigmoidPlot'].append("line")
      .attr("class", "threshold-line")
      .style("stroke", "#d62728")
      .style("stroke-dasharray", "3,3")
      .attr("x1", 0)
      .attr("y1", yScale(0.5))
      .attr("x2", this.width)
      .attr("y2", yScale(0.5));
    
    // Add vertical line at z=0
    this.svgs['sigmoidPlot'].append("line")
      .attr("class", "zero-line")
      .style("stroke", "#d62728")
      .style("stroke-dasharray", "3,3")
      .attr("x1", xScale(0))
      .attr("y1", 0)
      .attr("x2", xScale(0))
      .attr("y2", this.height);
    
    // Add labels for key points
    this.svgs['sigmoidPlot'].append("text")
      .attr("x", xScale(0) + 5)
      .attr("y", 40)
      .style("font-size", "11px")
      .text("z = 0");
    
    this.svgs['sigmoidPlot'].append("text")
      .attr("x", this.width - 70)
      .attr("y", yScale(0.5) - 5)
      .style("font-size", "11px")
      .text("P(y=1|x) = 0.5");
    
    // Add interactive point for test data
    this.svgs['sigmoidPlot'].append("circle")
      .attr("id", "probability-point")
      .attr("r", 6)
      .style("fill", "#2ca02c")
      .style("opacity", 0.7)
      .style("stroke", "#000")
      .style("stroke-width", 1)
      .attr("cx", xScale(0))
      .attr("cy", yScale(sigmoid(0)));
    
    // Add text to display probability
    this.svgs['sigmoidPlot'].append("text")
      .attr("id", "probability-text")
      .attr("x", 10)
      .attr("y", 60)
      .style("font-size", "14px")
      .style("fill", "#2ca02c")
      .style("font-weight", "bold")
      .text(`Probability: ${sigmoid(0).toFixed(4)}`);
  }

  private initializeCostPlot(): void {
    // Create SVG for cost function visualization
    this.svgs['costPlot'] = d3.select(this.costPlotElement.nativeElement)
      .append('svg')
      .attr('width', this.width + this.margin.left + this.margin.right)
      .attr('height', this.height + this.margin.top + this.margin.bottom)
      .append('g')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`) as d3.Selection<SVGGElement, unknown, null, undefined>;
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, this.iterations])
      .range([0, this.width]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])  // This will be updated as we collect cost history
      .range([this.height, 0]);
    
    // Add axes
    this.svgs['costPlot'].append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.height})`)
      .call(d3.axisBottom(xScale));
    
    this.svgs['costPlot'].append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(yScale));
    
    // Add axes labels
    this.svgs['costPlot'].append("text")
      .attr("transform", `translate(${this.width/2},${this.height + 35})`)
      .style("text-anchor", "middle")
      .text("Iteration");
    
    this.svgs['costPlot'].append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -45)
      .attr("x", -this.height/2)
      .style("text-anchor", "middle")
      .text("Cost (Binary Cross-Entropy)");
    
    // Add title
    this.svgs['costPlot'].append("text")
      .attr("x", this.width / 2)
      .attr("y", -20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text("Cost Function Optimization");
    
    // Add cost function formula
    this.svgs['costPlot'].append("text")
      .attr("x", this.width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-style", "italic")
      .style("font-size", "12px")
      .text("J(w) = -1/m ∑[y log(p) + (1-y)log(1-p)]");
    
    // Initialize cost line
    this.svgs['costPlot'].append("path")
      .attr("id", "cost-line")
      .attr("fill", "none")
      .attr("stroke", "#ff7f0e")
      .attr("stroke-width", 2.5);
  }

  private initializeCoefficientsPlot(): void {
    // Create SVG for coefficients visualization
    this.svgs['coefficientsPlot'] = d3.select(this.coefficientsPlotElement.nativeElement)
      .append('svg')
      .attr('width', this.width + this.margin.left + this.margin.right)
      .attr('height', this.height + this.margin.top + this.margin.bottom)
      .append('g')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`) as d3.Selection<SVGGElement, unknown, null, undefined>;
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, this.iterations])
      .range([0, this.width]);
    
    const yScale = d3.scaleLinear()
      .domain([-5, 5])  // Will be updated as weights change
      .range([this.height, 0]);
    
    // Add axes
    this.svgs['coefficientsPlot'].append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${this.height})`)
      .call(d3.axisBottom(xScale));
    
    this.svgs['coefficientsPlot'].append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(yScale));
    
    // Add axes labels
    this.svgs['coefficientsPlot'].append("text")
      .attr("transform", `translate(${this.width/2},${this.height + 35})`)
      .style("text-anchor", "middle")
      .text("Iteration");
    
    this.svgs['coefficientsPlot'].append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -45)
      .attr("x", -this.height/2)
      .style("text-anchor", "middle")
      .text("Weight Value");
    
    // Add title
    this.svgs['coefficientsPlot'].append("text")
      .attr("x", this.width / 2)
      .attr("y", -20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text("Coefficient Evolution");
    
    // Initialize coefficient lines
    this.svgs['coefficientsPlot'].append("path")
      .attr("id", "w0-line")
      .attr("fill", "none")
      .attr("stroke", "#1f77b4")
      .attr("stroke-width", 2);
    
    this.svgs['coefficientsPlot'].append("path")
      .attr("id", "w1-line")
      .attr("fill", "none")
      .attr("stroke", "#ff7f0e")
      .attr("stroke-width", 2);
    
    this.svgs['coefficientsPlot'].append("path")
      .attr("id", "w2-line")
      .attr("fill", "none")
      .attr("stroke", "#2ca02c")
      .attr("stroke-width", 2);
    
    // Add legend
    const legend = this.svgs['coefficientsPlot'].append("g")
      .attr("class", "legend")
      .attr("transform", `translate(${this.width - 100}, 20)`);
    
    // Legend items
    const legendItems = [
      { label: "Bias (w₀)", color: "#1f77b4" },
      { label: "Weight 1 (w₁)", color: "#ff7f0e" },
      { label: "Weight 2 (w₂)", color: "#2ca02c" }
    ];
    
    legendItems.forEach((item, i) => {
      const legendItem = legend.append("g")
        .attr("transform", `translate(0, ${i * 20})`);
      
      legendItem.append("rect")
        .attr("width", 10)
        .attr("height", 10)
        .attr("fill", item.color);
      
      legendItem.append("text")
        .attr("x", 15)
        .attr("y", 9)
        .style("font-size", "12px")
        .text(item.label);
    });
  }

  private updateDecisionBoundary(xScale: d3.ScaleLinear<number, number>, yScale: d3.ScaleLinear<number, number>): void {
    // The decision boundary is where w0 + w1*x1 + w2*x2 = 0
    // We can rewrite this as x2 = -(w0 + w1*x1) / w2
    
    // For the line, we need two points
    const x1Min = 0;
    const x1Max = 10;
    
    if (!this.decisionBoundary) return;
    
    // If w2 is close to zero, decision boundary is vertical
    if (Math.abs(this.weights.w2) < 0.001) {
      // Avoid division by zero
      const x1Boundary = this.weights.w1 !== 0 ? -this.weights.w0 / this.weights.w1 : 5;
      this.decisionBoundary
        .attr("x1", xScale(x1Boundary))
        .attr("y1", yScale(0))
        .attr("x2", xScale(x1Boundary))
        .attr("y2", yScale(10));
    } else {
      const x2ForX1Min = -(this.weights.w0 + this.weights.w1 * x1Min) / this.weights.w2;
      const x2ForX1Max = -(this.weights.w0 + this.weights.w1 * x1Max) / this.weights.w2;
      
      this.decisionBoundary
        .attr("x1", xScale(x1Min))
        .attr("y1", yScale(x2ForX1Min))
        .attr("x2", xScale(x1Max))
        .attr("y2", yScale(x2ForX1Max));
    }
    
    // Update probabilities visualization if enabled
    if (this.showProbabilities) {
      this.updateProbabilitiesViz(xScale, yScale);
    }
  }

  private updateProbabilitiesViz(xScale: d3.ScaleLinear<number, number>, yScale: d3.ScaleLinear<number, number>): void {
    // Get probability predictions for each data point
    const predictions = this.calculatePredictions();
    
    if (!this.probabilities) return;
    
    // Remove existing probability circles
    this.probabilities.selectAll("circle").remove();
    
    // Add probability circles
    for (let i = 0; i < this.data.length; i++) {
      const point = this.data[i];
      const prob = predictions[i];
      
      // Size circle based on confidence (distance from 0.5)
      const radius = 2 + Math.abs(prob - 0.5) * 8;
      
      this.probabilities.append("circle")
        .attr("cx", xScale(point.x1))
        .attr("cy", yScale(point.x2))
        .attr("r", radius)
        .style("fill", prob >= 0.5 ? "#ff7f0e" : "#1f77b4")
        .style("opacity", 0.3)
        .append("title")
        .text(`P(y=1|x) = ${prob.toFixed(3)}`);
    }
  }

  private updateTestPointProbability(): void {
    const z = this.weights.w0 + this.weights.w1 * this.testPoint.x1 + this.weights.w2 * this.testPoint.x2;
    const probability = this.sigmoid(z);
    
    // Update point position in sigmoid plot
    const xScale = d3.scaleLinear()
      .domain([-8, 8])
      .range([0, this.width]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([this.height, 0]);
    
    const point = d3.select("#probability-point");
    if (!point.empty()) {
      point
        .attr("cx", xScale(Math.max(-8, Math.min(8, z)))) // Constrain to visible area
        .attr("cy", yScale(probability));
    }
    
    // Update probability text
    const text = d3.select("#probability-text");
    if (!text.empty()) {
      text.text(`Probability: ${probability.toFixed(4)} (z = ${z.toFixed(2)})`);
    }
  }

  private sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
  }

  private calculatePredictions(): number[] {
    return this.data.map(point => {
      const z = this.weights.w0 + this.weights.w1 * point.x1 + this.weights.w2 * point.x2;
      return this.sigmoid(z);
    });
  }

  private calculateCost(): number {
    const predictions = this.calculatePredictions();
    let cost = 0;
    
    for (let i = 0; i < this.data.length; i++) {
      const y = this.data[i].y;
      const p = predictions[i];
      
      // Handle edge cases to avoid log(0)
      const p_safe = Math.max(Math.min(p, 0.9999), 0.0001);
      
      // Binary cross-entropy
      cost += y * Math.log(p_safe) + (1 - y) * Math.log(1 - p_safe);
    }
    
    cost = -cost / this.data.length;
    return cost;
  }

  private updateWeights(): void {
    const predictions = this.calculatePredictions();
    const m = this.data.length;
    
    // Calculate gradients
    let gradient_w0 = 0;
    let gradient_w1 = 0;
    let gradient_w2 = 0;
    
    for (let i = 0; i < m; i++) {
      const error = predictions[i] - this.data[i].y;
      gradient_w0 += error;
      gradient_w1 += error * this.data[i].x1;
      gradient_w2 += error * this.data[i].x2;
    }
    
    // Update weights using gradient descent
    this.weights.w0 -= this.learningRate * gradient_w0 / m;
    this.weights.w1 -= this.learningRate * gradient_w1 / m;
    this.weights.w2 -= this.learningRate * gradient_w2 / m;
    
    // Add current weights to history
    this.weightsHistory.push({ ...this.weights });
  }

  private updateCostPlot(): void {
    const cost = this.calculateCost();
    this.costHistory.push(cost);
    
    if (!this.svgs['costPlot']) return;
    
    // Update plot
    const xScale = d3.scaleLinear()
      .domain([0, this.iterations])
      .range([0, this.width]);
    
    // Dynamic y-scale based on cost values
    const maxCost = Math.max(...this.costHistory) * 1.1; // Add 10% padding
    const yScale = d3.scaleLinear()
      .domain([0, maxCost])
      .range([this.height, 0]);
    
    // Update y-axis
    this.svgs['costPlot'].select<SVGGElement>(".y-axis")
      .transition()
      .duration(200)
      .call(d3.axisLeft(yScale) as any);
    
    // Update line
    const line = d3.line<number>()
      .x((_, i) => xScale(i))
      .y(d => yScale(d))
      .curve(d3.curveMonotoneX);
    
    this.svgs['costPlot'].select("#cost-line")
      .datum(this.costHistory)
      .attr("d", line as any); // Type cast to handle d3 line function
  }

  private updateCoefficientsPlot(): void {
    // Extract history for each coefficient
    const w0History = this.weightsHistory.map(w => w.w0);
    const w1History = this.weightsHistory.map(w => w.w1);
    const w2History = this.weightsHistory.map(w => w.w2);
    
    if (!this.svgs['coefficientsPlot']) return;
    
    // Calculate dynamic y-domain
    const allValues = [...w0History, ...w1History, ...w2History];
    const minValue = Math.min(...allValues) * 1.1;
    const maxValue = Math.max(...allValues) * 1.1;
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, this.iterations])
      .range([0, this.width]);
    
    const yScale = d3.scaleLinear()
      .domain([minValue, maxValue])
      .range([this.height, 0]);
    
    // Update y-axis
    this.svgs['coefficientsPlot'].select<SVGGElement>(".y-axis")
      .transition()
      .duration(200)
      .call(d3.axisLeft(yScale) as any);
    
    // Create line generator
    const line = d3.line<number>()
      .x((_, i) => xScale(i))
      .y(d => yScale(d))
      .curve(d3.curveMonotoneX);
    
    // Update lines
    this.svgs['coefficientsPlot'].select("#w0-line")
      .datum(w0History)
      .attr("d", line as any);
    
    this.svgs['coefficientsPlot'].select("#w1-line")
      .datum(w1History)
      .attr("d", line as any);
    
    this.svgs['coefficientsPlot'].select("#w2-line")
      .datum(w2History)
      .attr("d", line as any);
  }

  // Animation control methods
  public togglePlayPause(): void {
    if (this.isPlaying) {
      this.pauseAnimation();
    } else {
      this.playAnimation();
    }
  }

  public playAnimation(): void {
    if (this.isPlaying) return;
    
    this.isPlaying = true;
    this.runAnimation();
  }

  public pauseAnimation(): void {
    this.isPlaying = false;
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  public stopAnimation(): void {
    // This method was missing but referenced in ngOnDestroy
    this.isPlaying = false;
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  public resetSimulation(): void {
    this.pauseAnimation();
    this.currentIteration = 0;
    this.initializeWeights();
    this.weightsHistory = [{ ...this.weights }];
    this.costHistory = [];
    
    // Update visualizations
    const xScale = d3.scaleLinear()
      .domain([0, 10])
      .range([0, this.width]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 10])
      .range([this.height, 0]);
    
    this.updateDecisionBoundary(xScale, yScale);
    this.updateTestPointProbability();
    
    // Reset cost plot
    if (this.svgs['costPlot']) {
      this.svgs['costPlot'].select("#cost-line").attr("d", "");
    }
    
    // Reset coefficients plot
    if (this.svgs['coefficientsPlot']) {
      this.svgs['coefficientsPlot'].select("#w0-line").attr("d", "");
      this.svgs['coefficientsPlot'].select("#w1-line").attr("d", "");
      this.svgs['coefficientsPlot'].select("#w2-line").attr("d", "");
    }
    
    // Reset current step
    this.currentStep = 0;
  }

  public stepForward(): void {
    if (this.currentIteration >= this.iterations) {
      return;
    }
    
    this.performIteration();
    this.currentIteration++;
  }

  private runAnimation(): void {
    let lastTime = 0;
    
    const animate = (time: number) => {
      if (!this.isPlaying) return;
      
      const deltaTime = time - lastTime;
      
      if (deltaTime >= this.speed && this.currentIteration < this.iterations) {
        lastTime = time;
        
        // Perform one iteration
        this.performIteration();
        this.currentIteration++;
        
        // Cycle through explanation steps
        this.currentStep = (this.currentStep + 1) % this.steps.length;
      }
      
      if (this.currentIteration >= this.iterations) {
        this.pauseAnimation();
      } else {
        this.animationId = requestAnimationFrame(animate);
      }
    };
    
    this.animationId = requestAnimationFrame(animate);
  }

  private performIteration(): void {
    // Update weights
    this.updateWeights();
    
    // Calculate cost
    const cost = this.calculateCost();
    this.costHistory.push(cost);
    
    // Update visualizations
    const xScale = d3.scaleLinear()
      .domain([0, 10])
      .range([0, this.width]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 10])
      .range([this.height, 0]);
    
    this.updateDecisionBoundary(xScale, yScale);
    this.updateTestPointProbability();
    this.updateCostPlot();
    this.updateCoefficientsPlot();
  }

  public toggleProbabilities(): void {
    this.showProbabilities = !this.showProbabilities;
    
    const xScale = d3.scaleLinear()
      .domain([0, 10])
      .range([0, this.width]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 10])
      .range([this.height, 0]);
    
    if (this.showProbabilities) {
      this.updateProbabilitiesViz(xScale, yScale);
    } else if (this.probabilities) {
      this.probabilities.selectAll("circle").remove();
    }
  }
}