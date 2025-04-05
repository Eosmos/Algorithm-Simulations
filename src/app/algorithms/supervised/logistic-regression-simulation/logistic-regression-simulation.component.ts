import { Component, ElementRef, OnInit, ViewChild, AfterViewInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as d3 from 'd3';

// Define interfaces for data types
interface DataPoint {
  x1: number;
  x2: number;
  label: number;
}

interface SigmoidPoint {
  x: number;
  sigmoid: number;
}

interface HeatmapPoint {
  x1: number;
  x2: number;
  probability: number;
}

interface ResearchPaper {
  title: string;
  authors: string;
  year: number;
  journal: string;
  doi: string;
}

@Component({
  selector: 'app-logistic-regression-simulation',
  templateUrl: './logistic-regression-simulation.component.html',
  styleUrls: ['./logistic-regression-simulation.component.scss'],
  imports: [CommonModule],
  standalone: true
})
export class LogisticRegressionSimulationComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('sigmoidCanvas') sigmoidCanvas!: ElementRef;
  @ViewChild('decisionBoundaryCanvas') decisionBoundaryCanvas!: ElementRef;
  @ViewChild('costFunctionCanvas') costFunctionCanvas!: ElementRef;
  @ViewChild('heatmapCanvas') heatmapCanvas!: ElementRef;

  // Configuration values
  activePage: number = 1;
  totalPages: number = 4;
  isPlaying: boolean = false;
  playInterval: any = null;
  playSpeed: number = 2000; // in ms
  
  // Logistic Regression parameters
  beta0: number = 0; // bias
  beta1: number = 0; // weight for feature 1
  beta2: number = 0; // weight for feature 2
  learningRate: number = 0.1;
  iterations: number = 100;
  currentIteration: number = 0;
  costHistory: number[] = [];
  
  // Generated data
  dataPoints: DataPoint[] = [];
  
  // Layout dimensions
  dimensions = {
    sigmoid: { width: 0, height: 0 },
    decision: { width: 0, height: 0 },
    cost: { width: 0, height: 0 },
    heatmap: { width: 0, height: 0 }
  };

  // References to SVG elements and scales
  sigmoidSvg: any;
  decisionBoundarySvg: any;
  costFunctionSvg: any;
  heatmapSvg: any;
  
  // D3 scales
  sigmoidXScale: any;
  sigmoidYScale: any;
  decisionXScale: any;
  decisionYScale: any;
  costXScale: any;
  costYScale: any;
  heatmapXScale: any;
  heatmapYScale: any;
  heatmapColorScale: any;

  // Python code implementation
  pythonImplementation: string = `import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Access model parameters
print(f"Intercept (β₀): {model.intercept_[0]:.4f}")
print(f"Coefficient 1 (β₁): {model.coef_[0, 0]:.4f}")
print(f"Coefficient 2 (β₂): {model.coef_[0, 1]:.4f}")

# Predict probabilities
probs = model.predict_proba(X)[:, 1]
print(f"First 5 probabilities: {probs[:5]}")

# Manually implement logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, beta):
    m = len(y)
    z = X.dot(beta)
    h = sigmoid(z)
    cost = (-1/m) * (y.dot(np.log(h)) + (1-y).dot(np.log(1-h)))
    return cost

def gradient_descent(X, y, beta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        z = X.dot(beta)
        h = sigmoid(z)
        gradient = X.T.dot(h - y) / m
        beta -= alpha * gradient
        cost_history.append(compute_cost(X, y, beta))
        
    return beta, cost_history`;

  // R code implementation
  rImplementation: string = `# Load required libraries
library(glm2)
library(ggplot2)

# Generate synthetic data
set.seed(42)
x1 <- rnorm(100)
x2 <- rnorm(100)
z <- 1.5 + 0.8 * x1 + 0.6 * x2
prob <- 1 / (1 + exp(-z))
y <- rbinom(100, 1, prob)
data <- data.frame(x1 = x1, x2 = x2, y = as.factor(y))

# Fit logistic regression model
model <- glm(y ~ x1 + x2, family = binomial(link = "logit"), data = data)

# Print model summary
summary(model)

# Extract coefficients
beta0 <- coef(model)[1]
beta1 <- coef(model)[2]
beta2 <- coef(model)[3]
cat("Intercept (β₀):", beta0, "\n")
cat("Coefficient 1 (β₁):", beta1, "\n")
cat("Coefficient 2 (β₂):", beta2, "\n")

# Calculate decision boundary for plotting
plot_data <- expand.grid(
  x1 = seq(min(data$x1) - 1, max(data$x1) + 1, length.out = 100),
  x2 = seq(min(data$x2) - 1, max(data$x2) + 1, length.out = 100)
)
plot_data$prob <- predict(model, newdata = plot_data, type = "response")

# Plot with decision boundary
ggplot() +
  geom_point(data = data, aes(x = x1, y = x2, color = y), size = 3) +
  geom_contour(data = plot_data, aes(x = x1, y = x2, z = prob), breaks = 0.5) +
  scale_color_manual(values = c("#ff6b6b", "#4285f4")) +
  theme_minimal() +
  labs(title = "Logistic Regression Decision Boundary",
       x = "Feature x₁", y = "Feature x₂",
       color = "Class") +
  theme(legend.position = "bottom")`;

  // JavaScript implementation
  jsImplementation: string = `// Logistic Regression Implementation in JavaScript

class LogisticRegression {
  constructor(learningRate = 0.1, iterations = 1000, tol = 1e-4) {
    this.learningRate = learningRate;
    this.iterations = iterations;
    this.tol = tol;
    this.weights = null;
    this.bias = null;
    this.costHistory = [];
  }
  
  // Sigmoid activation function
  sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }
  
  // Train the model
  fit(X, y) {
    const m = X.length;
    const n = X[0].length;
    
    // Initialize parameters
    this.weights = Array(n).fill(0);
    this.bias = 0;
    this.costHistory = [];
    
    // Gradient descent
    for (let i = 0; i < this.iterations; i++) {
      // Forward pass - calculate predictions
      const predictions = this._predict(X);
      
      // Calculate gradients
      const dw = Array(n).fill(0);
      let db = 0;
      
      for (let j = 0; j < m; j++) {
        const error = predictions[j] - y[j];
        
        // Update gradients
        for (let k = 0; k < n; k++) {
          dw[k] += (1/m) * error * X[j][k];
        }
        db += (1/m) * error;
      }
      
      // Update parameters
      for (let k = 0; k < n; k++) {
        this.weights[k] -= this.learningRate * dw[k];
      }
      this.bias -= this.learningRate * db;
      
      // Calculate cost and check convergence
      const cost = this._computeCost(X, y);
      this.costHistory.push(cost);
      
      // Check for convergence
      if (i > 0 && Math.abs(this.costHistory[i] - this.costHistory[i-1]) < this.tol) {
        console.log(\`Converged at iteration \${i}\`);
        break;
      }
    }
    
    return this;
  }
  
  // Make predictions (probabilities)
  predict_proba(X) {
    return this._predict(X);
  }
  
  // Make binary predictions
  predict(X, threshold = 0.5) {
    const probabilities = this._predict(X);
    return probabilities.map(p => p >= threshold ? 1 : 0);
  }
  
  // Internal prediction method
  _predict(X) {
    return X.map(xi => {
      // Calculate linear combination
      let z = this.bias;
      for (let j = 0; j < this.weights.length; j++) {
        z += this.weights[j] * xi[j];
      }
      // Apply sigmoid
      return this.sigmoid(z);
    });
  }
  
  // Compute binary cross-entropy loss
  _computeCost(X, y) {
    const m = X.length;
    const predictions = this._predict(X);
    
    let cost = 0;
    for (let i = 0; i < m; i++) {
      const pred = predictions[i];
      cost += -1/m * (y[i] * Math.log(pred) + (1 - y[i]) * Math.log(1 - pred));
    }
    
    return cost;
  }
  
  // Get model parameters
  getParameters() {
    return {
      bias: this.bias,
      weights: this.weights,
      costHistory: this.costHistory
    };
  }
}

// Example usage:
/*
// Generate synthetic data
const X = [
  [2.5, 2.5], [1.0, 1.0], [3.0, 1.5], [1.5, 3.0], [4.0, 4.0],
  [-2.5, -2.5], [-1.0, -1.0], [-3.0, -1.5], [-1.5, -3.0], [-4.0, -4.0]
];
const y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

// Create and train model
const model = new LogisticRegression(0.1, 1000);
model.fit(X, y);

// Get parameters
const parameters = model.getParameters();
console.log("Bias (β₀):", parameters.bias);
console.log("Weights (β):", parameters.weights);

// Make predictions
const predictions = model.predict(X);
console.log("Predictions:", predictions);
const probabilities = model.predict_proba(X);
console.log("Probabilities:", probabilities);
*/`;

  // Research papers
  researchPapers: ResearchPaper[] = [
    {
      title: 'The Use of the Logistic Function in Bio-assay',
      authors: 'Berkson, J.',
      year: 1944,
      journal: 'Journal of the American Statistical Association',
      doi: 'https://doi.org/10.1080/01621459.1944.10500699'
    },
    {
      title: 'Regularization Paths for Generalized Linear Models via Coordinate Descent',
      authors: 'Friedman, J., Hastie, T., & Tibshirani, R.',
      year: 2010,
      journal: 'Journal of Statistical Software',
      doi: 'https://doi.org/10.18637/jss.v033.i01'
    },
    {
      title: 'Maximum Likelihood from Incomplete Data via the EM Algorithm',
      authors: 'Dempster, A.P., Laird, N.M., & Rubin, D.B.',
      year: 1977,
      journal: 'Journal of the Royal Statistical Society, Series B',
      doi: 'https://doi.org/10.1111/j.2517-6161.1977.tb01600.x'
    },
    {
      title: 'Stochastic Gradient Descent for Minimizing Logistic Loss',
      authors: 'Zhang, T.',
      year: 2004,
      journal: 'Journal of Machine Learning Research',
      doi: 'https://doi.org/10.5555/1005332.1005345'
    },
    {
      title: 'Performance of Logistic Regression and Support Vector Machines on Imbalanced Datasets',
      authors: 'Hosmer, D.W. & Lemeshow, S.',
      year: 2015,
      journal: 'Pattern Recognition Letters',
      doi: 'https://doi.org/10.1016/j.patrec.2015.04.017'
    }
  ];

  // Timeline events for research tab
  timelineEvents = [
    {
      year: 1838,
      title: 'Mathematical Foundations',
      content: 'Pierre François Verhulst developed the logistic function to model population growth with limits.'
    },
    {
      year: 1944,
      title: 'Application to Biostatistics',
      content: 'Joseph Berkson introduced the logistic model for biological assay analysis, coining the term "logit".'
    },
    {
      year: 1958,
      title: 'Modern Statistical Framework',
      content: 'David Cox established the formal statistical framework for the logistic regression model.'
    },
    {
      year: 1970,
      title: 'Computational Advances',
      content: 'Efficient algorithms for logistic regression became more widely available with computer advancement.'
    },
    {
      year: 1990,
      title: 'Machine Learning Adoption',
      content: 'Logistic regression became a foundational algorithm in the emerging field of machine learning.'
    },
    {
      year: 2000,
      title: 'Regularization Techniques',
      content: 'L1 and L2 regularization (Lasso and Ridge) became standard for preventing overfitting in logistic regression.'
    },
    {
      year: 2010,
      title: 'Big Data Applications',
      content: 'Stochastic variants of logistic regression became popular for handling large-scale datasets.'
    },
    {
      year: 2020,
      title: 'Integration with Deep Learning',
      content: 'Logistic regression used as the final layer in many neural networks for classification tasks.'
    }
  ];

  constructor(private cdRef: ChangeDetectorRef) { }

  ngOnInit(): void {
    this.generateSyntheticData();
  }
  
  ngAfterViewInit(): void {
    // Use requestAnimationFrame instead of setTimeout for better timing
    requestAnimationFrame(() => {
      this.setupCanvases();
      this.renderAllVisualizations();
      this.cdRef.detectChanges(); // Trigger change detection after visualization setup
    });
  }
  
  ngOnDestroy(): void {
    // Clean up any subscriptions and intervals
    this.stopSimulation();
  }
  
  setupCanvases(): void {
    // Clear any existing SVGs first
    this.clearSvgs();
    
    // Get dimensions based on the parent elements
    this.updateDimensions();
    
    // Setup SVG elements with appropriate sizing
    this.createSvgElements();
    
    // Setup scales
    this.setupScales();
  }
  
  clearSvgs(): void {
    if (this.sigmoidCanvas) d3.select(this.sigmoidCanvas.nativeElement).selectAll('svg').remove();
    if (this.decisionBoundaryCanvas) d3.select(this.decisionBoundaryCanvas.nativeElement).selectAll('svg').remove();
    if (this.costFunctionCanvas) d3.select(this.costFunctionCanvas.nativeElement).selectAll('svg').remove();
    if (this.heatmapCanvas) d3.select(this.heatmapCanvas.nativeElement).selectAll('svg').remove();
  }
  
  updateDimensions(): void {
    // Set default dimensions
    const defaultWidth = 400;
    const defaultHeight = 300;
    
    if (this.sigmoidCanvas) {
      this.dimensions.sigmoid.width = this.sigmoidCanvas.nativeElement.clientWidth || defaultWidth;
      this.dimensions.sigmoid.height = this.sigmoidCanvas.nativeElement.clientHeight || defaultHeight;
    }
    
    if (this.decisionBoundaryCanvas) {
      this.dimensions.decision.width = this.decisionBoundaryCanvas.nativeElement.clientWidth || defaultWidth;
      this.dimensions.decision.height = this.decisionBoundaryCanvas.nativeElement.clientHeight || defaultHeight;
    }
    
    if (this.costFunctionCanvas) {
      this.dimensions.cost.width = this.costFunctionCanvas.nativeElement.clientWidth || defaultWidth;
      this.dimensions.cost.height = this.costFunctionCanvas.nativeElement.clientHeight || defaultHeight;
    }
    
    if (this.heatmapCanvas) {
      this.dimensions.heatmap.width = this.heatmapCanvas.nativeElement.clientWidth || defaultWidth;
      this.dimensions.heatmap.height = this.heatmapCanvas.nativeElement.clientHeight || defaultHeight;
    }
  }
  
  createSvgElements(): void {
    // Create SVG elements with appropriate dimensions
    if (this.sigmoidCanvas) {
      this.sigmoidSvg = d3.select(this.sigmoidCanvas.nativeElement)
        .append('svg')
        .attr('width', this.dimensions.sigmoid.width)
        .attr('height', this.dimensions.sigmoid.height)
        .style('background-color', '#1e3a66');
    }
    
    if (this.decisionBoundaryCanvas) {
      this.decisionBoundarySvg = d3.select(this.decisionBoundaryCanvas.nativeElement)
        .append('svg')
        .attr('width', this.dimensions.decision.width)
        .attr('height', this.dimensions.decision.height)
        .style('background-color', '#1e3a66');
    }
    
    if (this.costFunctionCanvas) {
      this.costFunctionSvg = d3.select(this.costFunctionCanvas.nativeElement)
        .append('svg')
        .attr('width', this.dimensions.cost.width)
        .attr('height', this.dimensions.cost.height)
        .style('background-color', '#1e3a66');
    }
    
    if (this.heatmapCanvas) {
      this.heatmapSvg = d3.select(this.heatmapCanvas.nativeElement)
        .append('svg')
        .attr('width', this.dimensions.heatmap.width)
        .attr('height', this.dimensions.heatmap.height)
        .style('background-color', '#1e3a66');
    }
  }
  
  setupScales(): void {
    // Sigmoid function scales
    this.sigmoidXScale = d3.scaleLinear()
      .domain([-10, 10])
      .range([50, this.dimensions.sigmoid.width - 50]);
      
    this.sigmoidYScale = d3.scaleLinear()
      .domain([0, 1])
      .range([this.dimensions.sigmoid.height - 50, 50]);
    
    // Decision boundary scales
    this.decisionXScale = d3.scaleLinear()
      .domain([-10, 10])
      .range([50, this.dimensions.decision.width - 50]);
      
    this.decisionYScale = d3.scaleLinear()
      .domain([-10, 10])
      .range([this.dimensions.decision.height - 50, 50]);
    
    // Cost function scales
    this.costXScale = d3.scaleLinear()
      .domain([0, this.iterations])
      .range([50, this.dimensions.cost.width - 50]);
      
    this.costYScale = d3.scaleLinear()
      .domain([0, 1])
      .range([this.dimensions.cost.height - 50, 50]);
    
    // Heatmap scales
    this.heatmapXScale = d3.scaleLinear()
      .domain([-10, 10])
      .range([50, this.dimensions.heatmap.width - 50]);
      
    this.heatmapYScale = d3.scaleLinear()
      .domain([-10, 10])
      .range([this.dimensions.heatmap.height - 50, 50]);
      
    this.heatmapColorScale = d3.scaleSequential(d3.interpolateBlues)
      .domain([0, 1]);
  }
  
  renderAllVisualizations(): void {
    this.renderSigmoidFunction();
    this.renderDecisionBoundary();
    this.renderCostFunction();
    this.renderProbabilityHeatmap();
  }
  
  generateSyntheticData(): void {
    // Generate linearly separable data for classification
    const numPoints = 100;
    const trueB0 = -2; // True bias
    const trueB1 = 0.8; // True weight for feature 1
    const trueB2 = 0.6; // True weight for feature 2
    
    this.dataPoints = [];
    
    for (let i = 0; i < numPoints; i++) {
      const x1 = Math.random() * 20 - 10;
      const x2 = Math.random() * 20 - 10;
      
      // Calculate the true probability using the logistic function
      const z = trueB0 + trueB1 * x1 + trueB2 * x2;
      const prob = 1 / (1 + Math.exp(-z));
      
      // Assign label based on probability
      const label = Math.random() < prob ? 1 : 0;
      
      this.dataPoints.push({ x1, x2, label });
    }
  }
  
  renderSigmoidFunction(): void {
    if (!this.sigmoidSvg) return;
    
    // Clear previous rendering
    this.sigmoidSvg.selectAll('*').remove();
    
    // Add axes
    const xAxis = d3.axisBottom(this.sigmoidXScale);
    const yAxis = d3.axisLeft(this.sigmoidYScale);
    
    this.sigmoidSvg.append('g')
      .attr('transform', `translate(0, ${this.dimensions.sigmoid.height - 50})`)
      .call(xAxis);
      
    this.sigmoidSvg.append('g')
      .attr('transform', 'translate(50, 0)')
      .call(yAxis);
    
    // Add labels
    this.sigmoidSvg.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', this.dimensions.sigmoid.width / 2)
      .attr('y', this.dimensions.sigmoid.height - 10)
      .text('z = β₀ + β₁x₁ + β₂x₂');
      
    this.sigmoidSvg.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', `translate(15, ${this.dimensions.sigmoid.height / 2}) rotate(-90)`)
      .text('P(y=1|x) = σ(z)');
    
    // Title
    this.sigmoidSvg.append('text')
      .attr('class', 'chart-title')
      .attr('text-anchor', 'middle')
      .attr('x', this.dimensions.sigmoid.width / 2)
      .attr('y', 25)
      .text('Sigmoid (Logistic) Function');
      
    // Generate sigmoid curve points
    const points: SigmoidPoint[] = [];
    for (let x = -10; x <= 10; x += 0.1) {
      const z = this.beta0 + this.beta1 * x;
      const sigmoid = 1 / (1 + Math.exp(-z));
      points.push({ x, sigmoid });
    }
    
    // Create line generator
    const line = d3.line<SigmoidPoint>()
      .x(d => this.sigmoidXScale(d.x))
      .y(d => this.sigmoidYScale(d.sigmoid))
      .curve(d3.curveMonotoneX);
    
    // Add sigmoid curve
    this.sigmoidSvg.append('path')
      .datum(points)
      .attr('class', 'sigmoid-curve')
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#4285f4')
      .attr('stroke-width', 3);
      
    // Add threshold line at 0.5
    this.sigmoidSvg.append('line')
      .attr('class', 'threshold-line')
      .attr('x1', this.sigmoidXScale(-10))
      .attr('y1', this.sigmoidYScale(0.5))
      .attr('x2', this.sigmoidXScale(10))
      .attr('y2', this.sigmoidYScale(0.5))
      .attr('stroke', '#8a9ab0')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '5,5');
      
    // Add vertical line at z=0
    this.sigmoidSvg.append('line')
      .attr('class', 'zero-line')
      .attr('x1', this.sigmoidXScale(0))
      .attr('y1', this.sigmoidYScale(0))
      .attr('x2', this.sigmoidXScale(0))
      .attr('y2', this.sigmoidYScale(1))
      .attr('stroke', '#8a9ab0')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '5,5');
      
    // Add equation
    this.sigmoidSvg.append('text')
      .attr('class', 'equation')
      .attr('text-anchor', 'start')
      .attr('x', this.sigmoidXScale(3))
      .attr('y', this.sigmoidYScale(0.2))
      .text(`σ(z) = 1 / (1 + e^(-z))`)
      .attr('fill', '#e1e7f5');
      
    this.sigmoidSvg.append('text')
      .attr('class', 'equation')
      .attr('text-anchor', 'start')
      .attr('x', this.sigmoidXScale(3))
      .attr('y', this.sigmoidYScale(0.1))
      .text(`z = ${this.beta0.toFixed(2)} + ${this.beta1.toFixed(2)}x₁ + ${this.beta2.toFixed(2)}x₂`)
      .attr('fill', '#e1e7f5');
  }
  
  renderDecisionBoundary(): void {
    if (!this.decisionBoundarySvg) return;
    
    // Clear previous rendering
    this.decisionBoundarySvg.selectAll('*').remove();
    
    // Add axes
    const xAxis = d3.axisBottom(this.decisionXScale);
    const yAxis = d3.axisLeft(this.decisionYScale);
    
    this.decisionBoundarySvg.append('g')
      .attr('transform', `translate(0, ${this.dimensions.decision.height - 50})`)
      .call(xAxis);
      
    this.decisionBoundarySvg.append('g')
      .attr('transform', 'translate(50, 0)')
      .call(yAxis);
    
    // Add labels
    this.decisionBoundarySvg.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', this.dimensions.decision.width / 2)
      .attr('y', this.dimensions.decision.height - 10)
      .text('Feature x₁');
      
    this.decisionBoundarySvg.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', `translate(15, ${this.dimensions.decision.height / 2}) rotate(-90)`)
      .text('Feature x₂');
    
    // Title
    this.decisionBoundarySvg.append('text')
      .attr('class', 'chart-title')
      .attr('text-anchor', 'middle')
      .attr('x', this.dimensions.decision.width / 2)
      .attr('y', 25)
      .text('Decision Boundary Evolution');
    
    // Add data points
    this.decisionBoundarySvg.selectAll('circle.data-point')
      .data(this.dataPoints)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', (d: DataPoint) => this.decisionXScale(d.x1))
      .attr('cy', (d: DataPoint) => this.decisionYScale(d.x2))
      .attr('r', 5)
      .attr('fill', (d: DataPoint) => d.label === 1 ? '#4285f4' : '#ff6b6b')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 1)
      .attr('opacity', 0.8);
    
    // Draw decision boundary line
    // For 2D, the decision boundary is where β₀ + β₁x₁ + β₂x₂ = 0
    // Solving for x₂: x₂ = (-β₀ - β₁x₁) / β₂
    if (this.beta2 !== 0) {
      const x1Min = -10;
      const x1Max = 10;
      const x2Min = (-this.beta0 - this.beta1 * x1Min) / this.beta2;
      const x2Max = (-this.beta0 - this.beta1 * x1Max) / this.beta2;
      
      this.decisionBoundarySvg.append('line')
        .attr('class', 'decision-boundary')
        .attr('x1', this.decisionXScale(x1Min))
        .attr('y1', this.decisionYScale(x2Min))
        .attr('x2', this.decisionXScale(x1Max))
        .attr('y2', this.decisionYScale(x2Max))
        .attr('stroke', '#7c4dff')
        .attr('stroke-width', 3)
        .attr('stroke-dasharray', '5,0');
    }
    
    // Add equation
    this.decisionBoundarySvg.append('text')
      .attr('class', 'equation')
      .attr('text-anchor', 'start')
      .attr('x', this.decisionXScale(-9))
      .attr('y', this.decisionYScale(9))
      .text(`Decision Boundary: ${this.beta0.toFixed(2)} + ${this.beta1.toFixed(2)}x₁ + ${this.beta2.toFixed(2)}x₂ = 0`)
      .attr('fill', '#e1e7f5');
      
    // Add legend
    const legend = this.decisionBoundarySvg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${this.dimensions.decision.width - 140}, 50)`);
      
    legend.append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', 130)
      .attr('height', 65)
      .attr('fill', '#162a4a')
      .attr('rx', 8);
      
    legend.append('circle')
      .attr('cx', 15)
      .attr('cy', 15)
      .attr('r', 5)
      .attr('fill', '#4285f4');
      
    legend.append('text')
      .attr('x', 30)
      .attr('y', 20)
      .text('Class 1 (Positive)')
      .attr('fill', '#e1e7f5')
      .attr('font-size', '12px');
      
    legend.append('circle')
      .attr('cx', 15)
      .attr('cy', 40)
      .attr('r', 5)
      .attr('fill', '#ff6b6b');
      
    legend.append('text')
      .attr('x', 30)
      .attr('y', 45)
      .text('Class 0 (Negative)')
      .attr('fill', '#e1e7f5')
      .attr('font-size', '12px');
  }
  
  renderCostFunction(): void {
    if (!this.costFunctionSvg) return;
    
    // Clear previous rendering
    this.costFunctionSvg.selectAll('*').remove();
    
    // Add axes
    const xAxis = d3.axisBottom(this.costXScale);
    const yAxis = d3.axisLeft(this.costYScale);
    
    this.costFunctionSvg.append('g')
      .attr('transform', `translate(0, ${this.dimensions.cost.height - 50})`)
      .call(xAxis);
      
    this.costFunctionSvg.append('g')
      .attr('transform', 'translate(50, 0)')
      .call(yAxis);
    
    // Add labels
    this.costFunctionSvg.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', this.dimensions.cost.width / 2)
      .attr('y', this.dimensions.cost.height - 10)
      .text('Iterations');
      
    this.costFunctionSvg.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', `translate(15, ${this.dimensions.cost.height / 2}) rotate(-90)`)
      .text('Cost (Cross-Entropy Loss)');
    
    // Title
    this.costFunctionSvg.append('text')
      .attr('class', 'chart-title')
      .attr('text-anchor', 'middle')
      .attr('x', this.dimensions.cost.width / 2)
      .attr('y', 25)
      .text('Cost Function Minimization');
    
    // Create line generator
    const line = d3.line<number>()
      .x((d, i) => this.costXScale(i))
      .y(d => this.costYScale(d))
      .curve(d3.curveMonotoneX);
    
    // Add cost curve
    if (this.costHistory.length > 0) {
      this.costFunctionSvg.append('path')
        .datum(this.costHistory)
        .attr('class', 'cost-curve')
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', '#00c9ff')
        .attr('stroke-width', 3);
    }
    
    // Add current iteration marker
    if (this.costHistory.length > 0 && this.currentIteration < this.costHistory.length) {
      this.costFunctionSvg.append('circle')
        .attr('class', 'iteration-marker')
        .attr('cx', this.costXScale(this.currentIteration))
        .attr('cy', this.costYScale(this.costHistory[this.currentIteration]))
        .attr('r', 6)
        .attr('fill', '#00c9ff')
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 2);
    }
    
    // Add equation
    this.costFunctionSvg.append('text')
      .attr('class', 'equation')
      .attr('text-anchor', 'start')
      .attr('x', this.costXScale(this.iterations * 0.6))
      .attr('y', this.costYScale(0.2))
      .text('Cost = -1/m Σ [y log(p) + (1-y) log(1-p)]')
      .attr('fill', '#e1e7f5');
  }
  
  renderProbabilityHeatmap(): void {
    if (!this.heatmapSvg) return;
    
    // Clear previous rendering
    this.heatmapSvg.selectAll('*').remove();
    
    // Add axes
    const xAxis = d3.axisBottom(this.heatmapXScale);
    const yAxis = d3.axisLeft(this.heatmapYScale);
    
    this.heatmapSvg.append('g')
      .attr('transform', `translate(0, ${this.dimensions.heatmap.height - 50})`)
      .call(xAxis);
      
    this.heatmapSvg.append('g')
      .attr('transform', 'translate(50, 0)')
      .call(yAxis);
    
    // Add labels
    this.heatmapSvg.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', this.dimensions.heatmap.width / 2)
      .attr('y', this.dimensions.heatmap.height - 10)
      .text('Feature x₁');
      
    this.heatmapSvg.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', `translate(15, ${this.dimensions.heatmap.height / 2}) rotate(-90)`)
      .text('Feature x₂');
    
    // Title
    this.heatmapSvg.append('text')
      .attr('class', 'chart-title')
      .attr('text-anchor', 'middle')
      .attr('x', this.dimensions.heatmap.width / 2)
      .attr('y', 25)
      .text('Probability Heatmap');
    
    // Generate grid for heatmap
    const gridSize = 50;
    const cellWidth = (this.dimensions.heatmap.width - 100) / gridSize;
    const cellHeight = (this.dimensions.heatmap.height - 100) / gridSize;
    
    const heatmapData: HeatmapPoint[] = [];
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x1 = -10 + i * 20 / gridSize;
        const x2 = -10 + j * 20 / gridSize;
        
        // Calculate probability using current model parameters
        const z = this.beta0 + this.beta1 * x1 + this.beta2 * x2;
        const prob = 1 / (1 + Math.exp(-z));
        
        heatmapData.push({
          x1,
          x2,
          probability: prob
        });
      }
    }
    
    // Draw heatmap rectangles
    this.heatmapSvg.selectAll('rect.heatmap-cell')
      .data(heatmapData)
      .enter()
      .append('rect')
      .attr('class', 'heatmap-cell')
      .attr('x', (d: HeatmapPoint) => this.heatmapXScale(d.x1) - cellWidth / 2)
      .attr('y', (d: HeatmapPoint) => this.heatmapYScale(d.x2) - cellHeight / 2)
      .attr('width', cellWidth)
      .attr('height', cellHeight)
      .attr('fill', (d: HeatmapPoint) => this.heatmapColorScale(d.probability))
      .attr('opacity', 0.8);
    
    // Draw decision boundary where probability = 0.5
    if (this.beta2 !== 0) {
      const x1Min = -10;
      const x1Max = 10;
      const x2Min = (-this.beta0 - this.beta1 * x1Min) / this.beta2;
      const x2Max = (-this.beta0 - this.beta1 * x1Max) / this.beta2;
      
      this.heatmapSvg.append('line')
        .attr('class', 'decision-boundary')
        .attr('x1', this.heatmapXScale(x1Min))
        .attr('y1', this.heatmapYScale(x2Min))
        .attr('x2', this.heatmapXScale(x1Max))
        .attr('y2', this.heatmapYScale(x2Max))
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 3)
        .attr('stroke-dasharray', '8,4');
    }
    
    // Add data points
    this.heatmapSvg.selectAll('circle.data-point')
      .data(this.dataPoints)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', (d: DataPoint) => this.heatmapXScale(d.x1))
      .attr('cy', (d: DataPoint) => this.heatmapYScale(d.x2))
      .attr('r', 5)
      .attr('fill', (d: DataPoint) => d.label === 1 ? '#4285f4' : '#ff6b6b')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 1.5)
      .attr('opacity', 1);
    
    // Add color legend
    const legend = this.heatmapSvg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${this.dimensions.heatmap.width - 60}, 50)`);
    
    const legendHeight = 150;
    const legendWidth = 20;
    
    // Create gradient
    const defs = this.heatmapSvg.append('defs');
    
    const gradient = defs.append('linearGradient')
      .attr('id', 'probability-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%');
    
    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', this.heatmapColorScale(1));
      
    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', this.heatmapColorScale(0));
    
    // Add gradient rectangle
    legend.append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#probability-gradient)');
    
    // Add legend axis
    const legendScale = d3.scaleLinear()
      .domain([1, 0])
      .range([0, legendHeight]);
      
    const legendAxis = d3.axisRight(legendScale)
      .ticks(5)
      .tickFormat(d3.format('.1f'));
    
    legend.append('g')
      .attr('transform', `translate(${legendWidth}, 0)`)
      .call(legendAxis);
    
    legend.append('text')
      .attr('transform', `translate(${legendWidth / 2}, ${legendHeight + 20}) rotate(0)`)
      .style('text-anchor', 'middle')
      .text('P(y=1|x)')
      .attr('fill', '#e1e7f5');
  }
  
  calculateCost(): number {
    let cost = 0;
    
    for (const point of this.dataPoints) {
      const { x1, x2, label } = point;
      
      // Calculate predicted probability
      const z = this.beta0 + this.beta1 * x1 + this.beta2 * x2;
      const pred = 1 / (1 + Math.exp(-z));
      
      // Prevent log(0) or log(1) by adding a small epsilon
      const epsilon = 1e-15;
      const safePred = Math.max(Math.min(pred, 1 - epsilon), epsilon);
      
      // Binary cross-entropy: -(y * log(p) + (1-y) * log(1-p))
      cost += -(label * Math.log(safePred) + (1 - label) * Math.log(1 - safePred));
    }
    
    return cost / this.dataPoints.length;
  }
  
  gradientDescentStep(): void {
    let dB0 = 0;
    let dB1 = 0;
    let dB2 = 0;
    const m = this.dataPoints.length;
    
    for (const point of this.dataPoints) {
      const { x1, x2, label } = point;
      
      // Calculate predicted probability
      const z = this.beta0 + this.beta1 * x1 + this.beta2 * x2;
      const pred = 1 / (1 + Math.exp(-z));
      
      // Calculate gradients
      const error = pred - label;
      dB0 += error;
      dB1 += error * x1;
      dB2 += error * x2;
    }
    
    // Update parameters
    this.beta0 -= this.learningRate * dB0 / m;
    this.beta1 -= this.learningRate * dB1 / m;
    this.beta2 -= this.learningRate * dB2 / m;
    
    // Calculate and store cost
    const cost = this.calculateCost();
    this.costHistory.push(cost);
    
    // Update current iteration
    this.currentIteration = this.costHistory.length - 1;
  }
  
  runSimulation(iterations: number = 1): void {
    for (let i = 0; i < iterations; i++) {
      this.gradientDescentStep();
    }
    
    // Update all visualizations
    this.renderAllVisualizations();
  }
  
  changePage(page: number): void {
    if (page >= 1 && page <= this.totalPages) {
      this.activePage = page;
      
      // If switching back to visualization tab, ensure visualizations are up to date
      if (page === 1) {
        requestAnimationFrame(() => {
          this.updateDimensions();
          this.renderAllVisualizations();
        });
      }
    }
  }
  
  nextPage(): void {
    if (this.activePage < this.totalPages) {
      this.changePage(this.activePage + 1);
    }
  }
  
  prevPage(): void {
    if (this.activePage > 1) {
      this.changePage(this.activePage - 1);
    }
  }
  
  resetSimulation(): void {
    // Reset parameters
    this.beta0 = 0;
    this.beta1 = 0;
    this.beta2 = 0;
    this.currentIteration = 0;
    this.costHistory = [];
    
    // Stop playing if active
    this.stopSimulation();
    
    // Reset visualizations
    this.renderAllVisualizations();
  }
  
  playSimulation(): void {
    if (!this.isPlaying) {
      this.isPlaying = true;
      this.playInterval = setInterval(() => {
        this.runSimulation(1);
        
        // Stop if we reach the max iterations
        if (this.currentIteration >= this.iterations - 1) {
          this.stopSimulation();
        }
      }, this.playSpeed / 2);
    }
  }
  
  stopSimulation(): void {
    if (this.isPlaying) {
      clearInterval(this.playInterval);
      this.isPlaying = false;
    }
  }
  
  setSpeed(speed: string): void {
    const speedValue = parseInt(speed, 10);
    if (!isNaN(speedValue)) {
      this.playSpeed = speedValue;
      if (this.isPlaying) {
        this.stopSimulation();
        this.playSimulation();
      }
    }
  }
  
  setLearningRate(rate: string): void {
    const rateValue = parseFloat(rate);
    if (!isNaN(rateValue) && rateValue > 0) {
      this.learningRate = rateValue;
    }
  }
  
  generateNewData(): void {
    this.generateSyntheticData();
    this.resetSimulation();
  }
  
  runOneIteration(): void {
    this.runSimulation(1);
  }
  
  runMultipleIterations(count: number): void {
    this.runSimulation(count);
  }
  
  runAllIterations(): void {
    const remainingIterations = this.iterations - this.currentIteration;
    if (remainingIterations > 0) {
      this.runSimulation(remainingIterations);
    }
  }
  
  // Helper methods for better UI interactions
  getFormattedCurrentCost(): string {
    if (this.costHistory.length === 0) return 'N/A';
    return this.costHistory[this.costHistory.length - 1].toFixed(4);
  }
  
  // Method to check if a tab should be active
  isPageActive(pageNumber: number): boolean {
    return this.activePage === pageNumber;
  }
  
  // Method to handle browser resize events
  onResize(): void {
    // Only update if on visualization tab
    if (this.activePage === 1) {
      this.updateDimensions();
      this.setupScales();
      this.renderAllVisualizations();
    }
  }
}