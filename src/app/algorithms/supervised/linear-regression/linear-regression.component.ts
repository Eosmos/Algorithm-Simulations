import { Component, OnInit, ElementRef, ViewChild, AfterViewInit, HostListener, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ParametricGeometry } from 'three/examples/jsm/geometries/ParametricGeometry.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { gsap } from 'gsap';

@Component({
  selector: 'app-linear-regression',
  templateUrl: './linear-regression.component.html',
  styleUrls: ['./linear-regression.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class LinearRegressionComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('scatterPlot') private scatterPlotRef!: ElementRef;
  @ViewChild('costFunction') private costFunctionRef!: ElementRef;
  @ViewChild('dataDistribution') private dataDistRef!: ElementRef;
  @ViewChild('residualsPlot') private residualsRef!: ElementRef;
  
  // Configuration options
  private margin = { top: 20, right: 20, bottom: 50, left: 50 };
  private width = 0;
  private height = 0;
  
  // Parameters visible to the view
  public beta0 = 0; // Intercept
  public beta1 = 0; // Slope
  public learningRate = 0.01;
  public iterations = 0;
  public maxIterations = 100;
  public isPlaying = false;
  public animationSpeed = 50;
  public currentCost = 0;
  public showGradientPath = true;
  public showErrorLines = true;
  public tutorialMode = false;
  public tutorialStep = 0;
  public tutorialSteps = [
    "Welcome to Linear Regression! We'll learn how to fit a line to data points.",
    "These scattered points represent our data. Each point has x and y coordinates.",
    "Linear regression tries to find the best-fitting line through these points.",
    "The line equation is y = β₀ + β₁x, where β₀ is the y-intercept and β₁ is the slope.",
    "To find the best values for β₀ and β₁, we need to minimize the 'cost function'.",
    "The cost function measures the total error between our line and the actual data points.",
    "We use 'gradient descent' to find the minimum of this cost function.",
    "Gradient descent works by taking steps in the direction that decreases the cost the most.",
    "The learning rate controls how big these steps are. Too small = slow. Too large = unstable.",
    "Let's watch gradient descent in action! Press Play to start."
  ];
  public currentTutorialContent = this.tutorialSteps[0];
  
  // Data & animation
  private data: {x: number, y: number}[] = [];
  private rawData: {x: number, y: number}[] = [];
  private normalizedData: {x: number, y: number}[] = [];
  private animationId: number | null = null;
  private lastStepTime = 0;
  public datasetOptions = ["Simple Linear", "Polynomial", "Multiple Clusters", "Real World Example"];
  public selectedDataset = "Simple Linear";
  public explanationVisible = true;
  
  // Three.js variables - using non-null assertion operator (!) to avoid TypeScript errors
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private labelRenderer!: CSS2DRenderer;
  private controls!: OrbitControls;
  private costSurface!: THREE.Mesh;
  private currentPositionMarker!: THREE.Mesh;
  private gradientPathLine!: THREE.Line;
  private costHistory: {beta0: number, beta1: number, cost: number}[] = [];
  
  // D3 elements
  private svg: d3.Selection<any, unknown, null, undefined> | null = null;
  private xScale!: d3.ScaleLinear<number, number>;
  private yScale!: d3.ScaleLinear<number, number>;
  private distSvg: d3.Selection<any, unknown, null, undefined> | null = null;
  private resSvg: d3.Selection<any, unknown, null, undefined> | null = null;
  
  // Metrics
  public r2Score = 0;
  public meanAbsoluteError = 0;
  public rootMeanSquaredError = 0;
  public animationState = {
    state: 'initial', // 'initial', 'running', 'paused', 'completed'
    autoMode: false,
    storyMode: false,
    storyStep: 0
  };
  
  constructor() {
    // Defer data generation to ngOnInit to avoid initialization issues
  }
  
  ngOnInit(): void {
    this.generateData();
  }
  
  ngAfterViewInit(): void {
    // Defer initialization with a longer timeout to ensure DOM is fully ready
    setTimeout(() => {
      try {
        this.initializeVisualization();
      } catch (error) {
        console.error('Error in initialization:', error);
      }
      
      // Add window resize handling
      window.addEventListener('resize', this.handleResize.bind(this));
    }, 500); // Increased timeout for better stability
  }
  
  ngOnDestroy(): void {
    window.removeEventListener('resize', this.handleResize.bind(this));
    this.stopAnimation();
    
    // Clean up Three.js resources
    if (this.renderer) {
      this.renderer.dispose();
    }
    if (this.scene) {
      this.scene.clear();
    }
  }
  
  private initializeVisualization(): void {
    // Initialize the plots in proper order
    this.initScatterPlot();
    
    try {
      this.initCostFunctionVisualization();
    } catch (error) {
      console.error('Error initializing cost function visualization:', error);
    }
    
    try {
      this.initDataDistributionPlot();
    } catch (error) {
      console.error('Error initializing data distribution plot:', error);
    }
    
    try {
      this.initResidualsPlot();
    } catch (error) {
      console.error('Error initializing residuals plot:', error);
    }
    
    // Only update UI elements after 3D scene is initialized
    if (this.currentPositionMarker) {
      this.updateRegressionLine();
      this.updateMarkerPosition();
      this.updateMetrics();
    }
  }
  
  /**
   * Handle window resize events to maintain responsive visualizations
   */
  @HostListener('window:resize')
  handleResize(): void {
    // Redraw all visualizations on window resize
    this.initializeVisualization();
  }
  
  /**
   * Generate synthetic data with a linear relationship plus noise
   */
  private generateData(type: string = 'Simple Linear'): void {
    this.data = [];
    this.rawData = [];
    
    const dataSize = 40;
    
    switch(type) {
      case 'Simple Linear':
        const trueIntercept = 5;
        const trueSlope = 2;
        const noiseFactor = 1.5;
        
        // Generate data with linear relationship
        for (let i = 0; i < dataSize; i++) {
          const x = Math.random() * 10;
          const y = trueIntercept + trueSlope * x + (Math.random() - 0.5) * noiseFactor * 2;
          this.rawData.push({ x, y });
        }
        break;
        
      case 'Polynomial':
        // Generate data with nonlinear relationship
        for (let i = 0; i < dataSize; i++) {
          const x = Math.random() * 10;
          const y = 2 + 0.5 * Math.pow(x, 2) + (Math.random() - 0.5) * 3;
          this.rawData.push({ x, y });
        }
        break;
        
      case 'Multiple Clusters':
        // Generate clustered data
        for (let i = 0; i < dataSize/2; i++) {
          const x = Math.random() * 3 + 1;
          const y = 3 + x + (Math.random() - 0.5) * 1.5;
          this.rawData.push({ x, y });
        }
        
        for (let i = 0; i < dataSize/2; i++) {
          const x = Math.random() * 3 + 6;
          const y = 10 + x * 0.5 + (Math.random() - 0.5) * 1.5;
          this.rawData.push({ x, y });
        }
        break;
      
      case 'Real World Example':
        // Simulated house size (sq ft) vs price data
        for (let i = 0; i < dataSize; i++) {
          // House sizes between 1000 and 4000 sq ft
          const x = 1000 + Math.random() * 3000;
          // House prices with some relationship to size plus noise
          const y = 100000 + x * 100 + (Math.random() - 0.5) * 100000;
          this.rawData.push({ x, y });
        }
        break;
    }
    
    // Normalize data to be between 0-10 range for visualization
    this.normalizeData();
    
    // Initialize parameters
    this.beta0 = 0;
    this.beta1 = 0;
    this.iterations = 0;
    this.isPlaying = false;
    this.costHistory = [];
    this.animationState.state = 'initial';
    this.animationState.storyStep = 0;
  }
  
  /**
   * Normalize data to fit within 0-10 range for visualization purposes
   * while preserving the relationship
   */
  private normalizeData(): void {
    if (this.rawData.length === 0) return;
    
    // Find min/max for x and y
    const xValues = this.rawData.map(d => d.x);
    const yValues = this.rawData.map(d => d.y);
    
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    // Normalize to 0-10 range for better visualization
    this.data = this.rawData.map(d => ({
      x: ((d.x - xMin) / (xMax - xMin)) * 10,
      y: ((d.y - yMin) / (yMax - yMin)) * 10
    }));
    
    // Store normalized values for reference
    this.normalizedData = [...this.data];
  }
  
  // D3 visualization methods
  private initScatterPlot(): void {
    if (!this.scatterPlotRef) return;
    
    const container = this.scatterPlotRef.nativeElement;
    container.innerHTML = '';
    
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight || 400;
    
    this.width = containerWidth - this.margin.left - this.margin.right;
    this.height = containerHeight - this.margin.top - this.margin.bottom;
    
    this.svg = d3.select(container)
      .append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)
      .append('g')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`) as any;
    
    if (!this.svg) return;
    
    // Create scales
    this.xScale = d3.scaleLinear()
      .domain([0, d3.max(this.data, d => d.x) || 10])
      .range([0, this.width]);
    
    this.yScale = d3.scaleLinear()
      .domain([0, d3.max(this.data, d => d.y) || 10])
      .range([this.height, 0]);
    
    // Add grid
    this.svg.append('g')
      .attr('class', 'grid-lines')
      .selectAll('line')
      .data(this.xScale.ticks(10))
      .enter()
      .append('line')
      .attr('x1', d => this.xScale(d))
      .attr('y1', 0)
      .attr('x2', d => this.xScale(d))
      .attr('y2', this.height)
      .attr('stroke', '#e0e0e0')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3');
      
    this.svg.append('g')
      .attr('class', 'grid-lines')
      .selectAll('line')
      .data(this.yScale.ticks(10))
      .enter()
      .append('line')
      .attr('x1', 0)
      .attr('y1', d => this.yScale(d))
      .attr('x2', this.width)
      .attr('y2', d => this.yScale(d))
      .attr('stroke', '#e0e0e0')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3');
    
    // Add X and Y axes
    this.svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${this.height})`)
      .call(d3.axisBottom(this.xScale))
      .append('text')
      .attr('class', 'axis-label')
      .attr('x', this.width / 2)
      .attr('y', 40)
      .attr('fill', '#000')
      .attr('text-anchor', 'middle')
      .text(() => {
        if (this.selectedDataset === 'Real World Example') {
          return 'House Size (sq ft)';
        }
        return 'X';
      });
    
    this.svg.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(this.yScale))
      .append('text')
      .attr('class', 'axis-label')
      .attr('transform', 'rotate(-90)')
      .attr('x', -this.height / 2)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .attr('fill', '#000')
      .text(() => {
        if (this.selectedDataset === 'Real World Example') {
          return 'House Price ($)';
        }
        return 'Y';
      });
    
    // Add error lines (vertical distances to the line)
    if (this.showErrorLines) {
      this.svg.append('g')
        .attr('class', 'error-lines')
        .selectAll('line')
        .data(this.data)
        .enter()
        .append('line')
        .attr('x1', d => this.xScale(d.x))
        .attr('y1', d => this.yScale(d.y))
        .attr('x2', d => this.xScale(d.x))
        .attr('y2', d => this.yScale(this.beta0 + this.beta1 * d.x))
        .attr('stroke', '#e74c3c')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '3,3')
        .style('opacity', 0);
    }
    
    // Add data points with transition and hover effects
    this.svg.selectAll('.data-point')
      .data(this.data)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', d => this.xScale(d.x))
      .attr('cy', this.height) // Start from bottom
      .attr('r', 0) // Start with radius 0
      .style('fill', '#3498db')
      .style('stroke', '#2980b9')
      .style('stroke-width', 1.5)
      .style('opacity', 0.8)
      .on('mouseover', (event, d) => {
        const tooltip = d3.select('body')
          .append('div')
          .attr('class', 'tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0,0,0,0.7)')
          .style('color', 'white')
          .style('padding', '10px')
          .style('border-radius', '5px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .style('opacity', 0);
        
        let tooltipText = '';
        if (this.selectedDataset === 'Real World Example') {
          const originalX = this.rawData.find((item, index) => 
            Math.abs(this.normalizedData[index].x - d.x) < 0.001
          )?.x;
          const originalY = this.rawData.find((item, index) => 
            Math.abs(this.normalizedData[index].y - d.y) < 0.001
          )?.y;
          
          tooltipText = `House Size: ${Math.round(originalX || 0)} sq ft<br>Price: $${Math.round(originalY || 0).toLocaleString()}`;
        } else {
          tooltipText = `X: ${d.x.toFixed(2)}<br>Y: ${d.y.toFixed(2)}`;
        }
        
        tooltip.html(tooltipText)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 15) + 'px')
          .transition()
          .duration(200)
          .style('opacity', 1);
          
        d3.select(event.currentTarget)
          .transition()
          .duration(200)
          .attr('r', 8)
          .style('fill', '#e74c3c');
      })
      .on('mouseout', (event) => {
        d3.select('.tooltip').remove();
        
        d3.select(event.currentTarget)
          .transition()
          .duration(200)
          .attr('r', 6)
          .style('fill', '#3498db');
      })
      .transition()
      .duration(800)
      .delay((d, i) => i * 30)
      .attr('cy', d => this.yScale(d.y))
      .attr('r', 6);
    
    // Add regression line
    this.svg.append('line')
      .attr('class', 'regression-line')
      .attr('x1', 0)
      .attr('y1', this.yScale(this.beta0))
      .attr('x2', this.width)
      .attr('y2', this.yScale(this.beta0 + this.beta1 * this.xScale.invert(this.width)))
      .style('stroke', '#e74c3c')
      .style('stroke-width', 3)
      .style('opacity', 0)
      .transition()
      .duration(1000)
      .style('opacity', 1);
      
    // Add area to represent confidence interval (simplified)
    if (this.iterations > 10) {
      // Calculate variance for simplified confidence band
      const meanX = d3.mean(this.data, d => d.x) || 0;
      const variance = this.computeCost(this.beta0, this.beta1) * 2;
      
      const areaGenerator = d3.area<number>()
        .x(d => this.xScale(d))
        .y0(d => this.yScale(this.beta0 + this.beta1 * d - variance * (1 + Math.pow(d - meanX, 2) / 10)))
        .y1(d => this.yScale(this.beta0 + this.beta1 * d + variance * (1 + Math.pow(d - meanX, 2) / 10)));
      
      const xDomain = d3.range(0, 10, 0.1);
      
      this.svg.append('path')
        .datum(xDomain)
        .attr('class', 'confidence-area')
        .attr('d', areaGenerator as any)
        .attr('fill', '#e74c3c')
        .attr('opacity', 0.1);
    }
  }
  
  private initDataDistributionPlot(): void {
    if (!this.dataDistRef) return;
    
    const container = this.dataDistRef.nativeElement;
    container.innerHTML = '';
    
    const width = container.clientWidth;
    const height = container.clientHeight || 150;
    
    this.distSvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height) as any;
      
    const margin = {top: 20, right: 20, bottom: 30, left: 40};
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    if (this.distSvg) {
      const g = this.distSvg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
      
      // Extract x and y values
      const xValues = this.data.map(d => d.x);
      
      // Create x scale with kernel density estimation
      const xScale = d3.scaleLinear()
        .domain([0, d3.max(xValues) || 10])
        .range([0, chartWidth]);
      
      // Create kernel density estimation for x values
      const kde = kernelDensityEstimator(kernelEpanechnikov(1), xScale.ticks(40));
      const density = kde(xValues);
      
      // Create y scale for the density
      const yScale = d3.scaleLinear()
        .domain([0, d3.max(density, d => d[1]) || 0.1])
        .range([chartHeight, 0]);
      
      // Add X axis
      g.append('g')
        .attr('transform', `translate(0,${chartHeight})`)
        .call(d3.axisBottom(xScale))
        .append('text')
        .attr('fill', '#000')
        .attr('x', chartWidth / 2)
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .text('X Values');
      
      // Create a line generator
      const lineGenerator = d3.line<[number, number]>()
        .curve(d3.curveBasis)
        .x(d => xScale(d[0]))
        .y(d => yScale(d[1]));
      
      // Add area path
      g.append('path')
        .datum(density)
        .attr('fill', '#3498db')
        .attr('opacity', 0.5)
        .attr('stroke', '#2980b9')
        .attr('stroke-width', 1.5)
        .attr('d', lineGenerator as any);
        
      // Add distribution title
      g.append('text')
        .attr('x', chartWidth / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .text('X Value Distribution');
    }
      
    // Helper functions for kernel density estimation
    function kernelDensityEstimator(kernel: (v: number) => number, X: number[]) {
      return function(V: number[]) {
        return X.map(x => [x, d3.mean(V, v => kernel(x - v)) || 0]);
      };
    }
    
    function kernelEpanechnikov(k: number) {
      return function(v: number) {
        return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
      };
    }
  }
  
  private initResidualsPlot(): void {
    if (!this.residualsRef) return;
    
    const container = this.residualsRef.nativeElement;
    container.innerHTML = '';
    
    const width = container.clientWidth;
    const height = container.clientHeight || 150;
    
    this.resSvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height) as any;
      
    const margin = {top: 20, right: 20, bottom: 30, left: 40};
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    if (this.resSvg) {
      const g = this.resSvg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
      
      // Calculate residuals
      const residuals = this.data.map(d => {
        const predicted = this.beta0 + this.beta1 * d.x;
        return {
          x: d.x,
          residual: d.y - predicted
        };
      });
      
      // Create scales
      const xScale = d3.scaleLinear()
        .domain([0, d3.max(this.data, d => d.x) || 10])
        .range([0, chartWidth]);
        
      const yScale = d3.scaleLinear()
        .domain([
          d3.min(residuals, d => d.residual) || -1, 
          d3.max(residuals, d => d.residual) || 1
        ])
        .range([chartHeight, 0]);
      
      // Add grid line at y=0
      g.append('line')
        .attr('x1', 0)
        .attr('y1', yScale(0))
        .attr('x2', chartWidth)
        .attr('y2', yScale(0))
        .attr('stroke', '#666')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3');
      
      // Add axes
      g.append('g')
        .attr('transform', `translate(0,${yScale(0)})`)
        .call(d3.axisBottom(xScale))
        .append('text')
        .attr('fill', '#000')
        .attr('x', chartWidth / 2)
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .text('X Values');
        
      g.append('g')
        .call(d3.axisLeft(yScale))
        .append('text')
        .attr('fill', '#000')
        .attr('transform', 'rotate(-90)')
        .attr('y', -30)
        .attr('x', -chartHeight / 2)
        .attr('text-anchor', 'middle')
        .text('Residuals');
      
      // Add residual points
      g.selectAll('.residual-point')
        .data(residuals)
        .enter()
        .append('circle')
        .attr('class', 'residual-point')
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.residual))
        .attr('r', 4)
        .attr('fill', d => d.residual > 0 ? '#e74c3c' : '#2ecc71')
        .attr('opacity', 0.7)
        .attr('stroke', '#fff')
        .attr('stroke-width', 1);
        
      // Add distribution title
      g.append('text')
        .attr('x', chartWidth / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .text('Residuals Plot');
    }
  }
  
  /**
   * Initialize the 3D cost function visualization using Three.js
   */
  private initCostFunctionVisualization(): void {
    if (!this.costFunctionRef || !this.costFunctionRef.nativeElement) {
      console.error('Cost function reference is not available');
      return;
    }
    
    const container = this.costFunctionRef.nativeElement;
    container.innerHTML = '';
    
    const width = container.clientWidth || 600;
    const height = container.clientHeight || 400;
    
    try {
      // STEP 1: Create the scene and set basic properties
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0xf8f9fa);
      this.scene.fog = new THREE.Fog(0xf8f9fa, 20, 100);
      
      // STEP 2: Set up camera
      this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
      this.camera.position.set(8, 8, 8);
      this.camera.lookAt(0, 0, 0);
      
      // STEP 3: Create renderer
      this.renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true
      });
      this.renderer.setSize(width, height);
      this.renderer.setPixelRatio(window.devicePixelRatio);
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      container.appendChild(this.renderer.domElement);
      
      // STEP 4: Create CSS2D renderer for labels
      this.labelRenderer = new CSS2DRenderer();
      this.labelRenderer.setSize(width, height);
      this.labelRenderer.domElement.style.position = 'absolute';
      this.labelRenderer.domElement.style.top = '0px';
      this.labelRenderer.domElement.style.pointerEvents = 'none';
      container.appendChild(this.labelRenderer.domElement);
      
      // STEP 5: Set up orbit controls
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.maxDistance = 50;
      
      // STEP 6: Add basic lighting
      this.addLighting();
      
      // STEP 7: Add coordinate system axes
      this.addAxes();
      
      // STEP 8: Add cost function surface
      this.createCostSurface();
      
      // STEP 9: Add current position marker
      this.addPositionMarker();
      
      // STEP 10: Add gradient path line
      this.addGradientPathLine();
      
      // STEP 11: Add optimal solution marker (if data is available)
      this.addOptimalSolutionMarker();
      
      // STEP 12: Start animation loop
      this.animate();
      
    } catch (error) {
      console.error('Error in initCostFunctionVisualization:', error);
      throw error;
    }
  }
  
  /**
   * Add lighting to the scene
   */
  private addLighting(): void {
    try {
      // Ambient light
      const ambientLight = new THREE.AmbientLight(0x404040, 2);
      this.scene.add(ambientLight);
      
      // Directional light (main light)
      const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
      directionalLight.position.set(5, 10, 7.5);
      directionalLight.castShadow = true;
      directionalLight.shadow.mapSize.width = 2048;
      directionalLight.shadow.mapSize.height = 2048;
      directionalLight.shadow.camera.near = 0.5;
      directionalLight.shadow.camera.far = 500;
      this.scene.add(directionalLight);
      
      // Spot light for better shadows
      const spotLight = new THREE.SpotLight(0xffffff, 0.5);
      spotLight.position.set(-10, 20, 10);
      spotLight.castShadow = true;
      spotLight.angle = Math.PI / 6;
      spotLight.penumbra = 0.2;
      this.scene.add(spotLight);
    } catch (error) {
      console.error('Error adding lighting:', error);
    }
  }
  
  /**
   * Add axis helper and labels to the scene
   */
  private addAxes(): void {
    try {
      // Add coordinate system axes
      const axesHelper = new THREE.AxesHelper(10);
      this.scene.add(axesHelper);
      
      // Create axis labels
      this.addAxisLabels();
    } catch (error) {
      console.error('Error adding axes:', error);
    }
  }
  
  /**
   * Add text labels for the 3D axes
   */
  private addAxisLabels(): void {
    try {
      // Create canvas elements for the text labels
      const createTextTexture = (text: string): THREE.Texture => {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 128;
        const context = canvas.getContext('2d');
        if (!context) {
          console.warn('Could not get canvas context');
          return new THREE.Texture();
        }
        
        context.fillStyle = '#ffffff';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.font = 'Bold 60px Arial';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillStyle = '#000000';
        context.fillText(text, canvas.width / 2, canvas.height / 2);
        
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        return texture;
      };
      
      // Create sprites for each axis label
      // Using non-null assertion to avoid TypeScript errors
      try {
        const beta0Label = new THREE.Sprite(
          new THREE.SpriteMaterial({ map: createTextTexture('β₀'), transparent: true })
        );
        beta0Label.position.set(6, 0, 0);
        beta0Label.scale.set(1, 0.5, 1);
        this.scene!.add(beta0Label);
      } catch (error) {
        console.error('Error creating beta0 label:', error);
      }
      
      try {
        const beta1Label = new THREE.Sprite(
          new THREE.SpriteMaterial({ map: createTextTexture('β₁'), transparent: true })
        );
        beta1Label.position.set(0, 0, 6);
        beta1Label.scale.set(1, 0.5, 1);
        this.scene!.add(beta1Label);
      } catch (error) {
        console.error('Error creating beta1 label:', error);
      }
      
      try {
        const costLabel = new THREE.Sprite(
          new THREE.SpriteMaterial({ map: createTextTexture('Cost'), transparent: true })
        );
        costLabel.position.set(0, 6, 0);
        costLabel.scale.set(1, 0.5, 1);
        this.scene!.add(costLabel);
      } catch (error) {
        console.error('Error creating cost label:', error);
      }
    } catch (error) {
      console.error('Error in addAxisLabels:', error);
    }
  }
  
  /**
   * Add position marker (red sphere) to the scene
   */
  private addPositionMarker(): void {
    try {
      const markerGeometry = new THREE.SphereGeometry(0.25, 32, 32);
      const markerMaterial = new THREE.MeshPhongMaterial({ 
        color: 0xe74c3c, 
        emissive: 0xe74c3c,
        emissiveIntensity: 0.3,
        specular: 0xffffff,
        shininess: 30
      });
      
      this.currentPositionMarker = new THREE.Mesh(markerGeometry, markerMaterial);
      this.currentPositionMarker.castShadow = true;
      
      // Set initial position directly
      const cost = this.computeCost(this.beta0, this.beta1);
      this.currentPositionMarker.position.set(this.beta0, cost * 0.1, this.beta1);
      
      // Add the marker to the scene
      this.scene!.add(this.currentPositionMarker);
      
      // Add position label
      try {
        const posDiv = document.createElement('div');
        posDiv.className = 'label';
        posDiv.textContent = 'Current Position';
        posDiv.style.backgroundColor = 'transparent';
        posDiv.style.color = '#e74c3c';
        posDiv.style.fontSize = '12px';
        posDiv.style.padding = '2px';
        posDiv.style.textShadow = '0 0 3px rgba(0,0,0,0.5)';
        
        const posLabel = new CSS2DObject(posDiv);
        posLabel.position.set(0, 0.5, 0);
        
        // Add the label to the marker - using non-null assertion
        this.currentPositionMarker!.add(posLabel);
      } catch (error) {
        console.error('Error creating position label:', error);
      }
    } catch (error) {
      console.error('Error in addPositionMarker:', error);
    }
  }
  
  /**
   * Add gradient path line to the scene
   */
  private addGradientPathLine(): void {
    try {
      // Initialize with just the starting point
      const points = [
        new THREE.Vector3(this.beta0, this.computeCost(this.beta0, this.beta1) * 0.1, this.beta1)
      ];
      
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      
      const material = new THREE.LineBasicMaterial({
        color: 0xe74c3c,
        linewidth: 3,
      });
      
      this.gradientPathLine = new THREE.Line(geometry, material);
      this.scene!.add(this.gradientPathLine);
    } catch (error) {
      console.error('Error adding gradient path line:', error);
    }
  }
  
  /**
   * Add optimal solution marker to the scene
   */
  private addOptimalSolutionMarker(): void {
    if (this.data.length === 0) return;
    
    try {
      // Find optimal parameters analytically
      const analyticalParams = this.calculateAnalyticalSolution();
      
      const optimalGeometry = new THREE.SphereGeometry(0.2, 32, 32);
      const optimalMaterial = new THREE.MeshPhongMaterial({ 
        color: 0x2ecc71, 
        emissive: 0x2ecc71,
        emissiveIntensity: 0.3,
        specular: 0xffffff,
        shininess: 30
      });
      
      const optimalMarker = new THREE.Mesh(optimalGeometry, optimalMaterial);
      optimalMarker.position.set(
        analyticalParams.beta0,
        this.computeCost(analyticalParams.beta0, analyticalParams.beta1) * 0.1,
        analyticalParams.beta1
      );
      optimalMarker.castShadow = true;
      this.scene!.add(optimalMarker);
      
      // Add label for optimal point
      const optimalDiv = document.createElement('div');
      optimalDiv.className = 'label';
      optimalDiv.textContent = 'Global Minimum';
      optimalDiv.style.backgroundColor = 'transparent';
      optimalDiv.style.color = '#2ecc71';
      optimalDiv.style.fontSize = '12px';
      optimalDiv.style.fontWeight = 'bold';
      optimalDiv.style.padding = '2px';
      optimalDiv.style.textShadow = '0 0 3px rgba(0,0,0,0.5)';
      
      const optimalLabel = new CSS2DObject(optimalDiv);
      optimalLabel.position.set(0, 0.5, 0);
      optimalMarker.add(optimalLabel);
    } catch (error) {
      console.error('Error adding optimal solution marker:', error);
    }
  }
  
  /**
   * Create the cost surface
   */
  private createCostSurface(): void {
    try {
      // Calculate cost function range for better visualization
      let maxCost = 0;
      for (let i = -5; i <= 5; i++) {
        for (let j = -2.5; j <= 2.5; j++) {
          const cost = this.computeCost(i, j);
          if (cost > maxCost) maxCost = cost;
        }
      }
      
      // Scaling factor for the cost visualization
      const costScale = 5 / maxCost;
      
      // Create parametric function for the cost surface
      const parametricFunction = (u: number, v: number, target: THREE.Vector3) => {
        const beta0Range = 10; // Range for beta0
        const beta1Range = 5;  // Range for beta1
        
        const beta0 = (u - 0.5) * beta0Range;
        const beta1 = (v - 0.5) * beta1Range;
        
        const cost = this.computeCost(beta0, beta1);
        
        target.set(beta0, cost * costScale, beta1); // Scaled for better visualization
      };
      
      // Create the surface geometry
      const geometry = new ParametricGeometry(parametricFunction, 50, 50);
      
      // Create gradient material for better visualization
      const material = new THREE.MeshPhongMaterial({
        color: 0x3498db,
        side: THREE.DoubleSide,
        shininess: 50,
        transparent: true,
        opacity: 0.8,
        flatShading: false,
        wireframe: false
      });
      
      this.costSurface = new THREE.Mesh(geometry, material);
      this.costSurface.receiveShadow = true;
      this.costSurface.castShadow = true;
      this.scene!.add(this.costSurface);
      
      // Add wireframe for better understanding of the surface shape
      const wireframeMaterial = new THREE.MeshBasicMaterial({
        color: 0x000000,
        wireframe: true,
        transparent: true,
        opacity: 0.1
      });
      
      const wireframe = new THREE.Mesh(geometry.clone(), wireframeMaterial);
      this.scene!.add(wireframe);
    } catch (error) {
      console.error('Error creating cost surface:', error);
    }
  }
  
  /**
   * Three.js animation loop
   */
  private animate(): void {
    this.animationId = requestAnimationFrame(() => this.animate());
    
    // Only apply gradient descent if playing
    if (this.isPlaying) {
      const currentTime = performance.now();
      
      // Check if enough time has passed since the last step
      if (currentTime - this.lastStepTime > this.animationSpeed) {
        this.gradientDescentStep();
        this.lastStepTime = currentTime;
      }
    }
    
    // Update controls and render scene
    // Using non-null assertions to avoid TypeScript errors
    if (this.controls) {
      this.controls.update();
    }
    
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
    
    if (this.labelRenderer && this.scene && this.camera) {
      this.labelRenderer.render(this.scene, this.camera);
    }
  }
  
  /**
   * Stop the animation loop
   */
  private stopAnimation(): void {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }
  
  /**
   * Compute the cost (mean squared error) for given parameters
   */
  public computeCost(b0: number, b1: number): number {
    if (this.data.length === 0) return 0;
    
    let sum = 0;
    for (const point of this.data) {
      const prediction = b0 + b1 * point.x;
      const error = prediction - point.y;
      sum += Math.pow(error, 2);
    }
    return sum / (2 * this.data.length);
  }
  
  /**
   * Perform one step of gradient descent
   */
  private gradientDescentStep(): void {
    if (this.iterations >= this.maxIterations) {
      this.isPlaying = false;
      this.animationState.state = 'completed';
      return;
    }
    
    // Calculate gradients
    let sumGradientBeta0 = 0;
    let sumGradientBeta1 = 0;
    
    for (const point of this.data) {
      const prediction = this.beta0 + this.beta1 * point.x;
      const error = prediction - point.y;
      
      sumGradientBeta0 += error;
      sumGradientBeta1 += error * point.x;
    }
    
    const gradientBeta0 = sumGradientBeta0 / this.data.length;
    const gradientBeta1 = sumGradientBeta1 / this.data.length;
    
    // Update parameters
    this.beta0 -= this.learningRate * gradientBeta0;
    this.beta1 -= this.learningRate * gradientBeta1;
    
    // Update visualizations
    this.updateRegressionLine();
    this.updateMarkerPosition();
    this.initResidualsPlot();
    
    this.iterations++;
    
    // Update story mode if needed
    if (this.animationState.storyMode && this.iterations % 10 === 0) {
      this.nextStoryStep();
    }
  }
  
  /**
   * Update the position of the marker on the cost surface
   */
  private updateMarkerPosition(): void {
    if (!this.currentPositionMarker) return;
    
    const cost = this.computeCost(this.beta0, this.beta1);
    this.currentCost = cost;
    
    // Update marker position with GSAP animation - using non-null assertion
    const newY = cost * 0.1;
    gsap.to(this.currentPositionMarker!.position, {
      x: this.beta0,
      y: newY,
      z: this.beta1,
      duration: 0.5,
      ease: "power2.out"
    });
    
    // Add current position to history
    this.costHistory.push({ beta0: this.beta0, beta1: this.beta1, cost });
    
    // Update the line showing the gradient descent path
    this.updateGradientPathLine();
    
    // Update error lines in scatter plot
    this.updateErrorLines();
    
    // Update metrics
    this.updateMetrics();
  }
  
  /**
   * Update the gradient path line with new points
   */
  private updateGradientPathLine(): void {
    if (!this.showGradientPath || this.costHistory.length < 2 || !this.gradientPathLine) return;
    
    try {
      const points = this.costHistory.map(point => 
        new THREE.Vector3(point.beta0, point.cost * 0.1, point.beta1)
      );
      
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      this.gradientPathLine.geometry.dispose();
      this.gradientPathLine.geometry = geometry;
    } catch (error) {
      console.error('Error updating gradient path line:', error);
    }
  }
  
  /**
   * Update error lines in scatter plot
   */
  private updateErrorLines(): void {
    if (!this.svg || !this.showErrorLines) return;
    
    this.svg.selectAll('.error-lines line')
      .data(this.data)
      .transition()
      .duration(100)
      .attr('y2', d => this.yScale(this.beta0 + this.beta1 * d.x))
      .style('opacity', 1);
  }
  
  /**
   * Update the regression line in the scatter plot
   */
  private updateRegressionLine(): void {
    if (!this.svg) return;
    
    this.svg.select('.regression-line')
      .transition()
      .duration(100)
      .attr('y1', this.yScale(this.beta0))
      .attr('y2', this.yScale(this.beta0 + this.beta1 * this.xScale.invert(this.width)));
  }
  
  /**
   * Update metrics (R², MAE, RMSE)
   */
  private updateMetrics(): void {
    this.r2Score = this.calculateR2();
    this.meanAbsoluteError = this.calculateMAE();
    this.rootMeanSquaredError = this.calculateRMSE();
  }
  
  /**
   * Calculate R² (coefficient of determination)
   */
  private calculateR2(): number {
    if (this.data.length === 0) return 0;
    
    const yMean = d3.mean(this.data, d => d.y) || 0;
    
    let totalSS = 0; // Total sum of squares
    let residualSS = 0; // Residual sum of squares
    
    for (const point of this.data) {
      const predicted = this.beta0 + this.beta1 * point.x;
      totalSS += Math.pow(point.y - yMean, 2);
      residualSS += Math.pow(point.y - predicted, 2);
    }
    
    if (totalSS === 0) return 0;
    return 1 - (residualSS / totalSS);
  }
  
  /**
   * Calculate mean absolute error
   */
  private calculateMAE(): number {
    if (this.data.length === 0) return 0;
    
    let sum = 0;
    for (const point of this.data) {
      const predicted = this.beta0 + this.beta1 * point.x;
      sum += Math.abs(point.y - predicted);
    }
    return sum / this.data.length;
  }
  
  /**
   * Calculate root mean squared error
   */
  private calculateRMSE(): number {
    if (this.data.length === 0) return 0;
    
    let sum = 0;
    for (const point of this.data) {
      const predicted = this.beta0 + this.beta1 * point.x;
      sum += Math.pow(point.y - predicted, 2);
    }
    return Math.sqrt(sum / this.data.length);
  }
  
  /**
   * Calculate the analytical solution for linear regression
   */
  private calculateAnalyticalSolution(): { beta0: number, beta1: number } {
    if (this.data.length === 0) return { beta0: 0, beta1: 0 };
    
    const xMean = d3.mean(this.data, d => d.x) || 0;
    const yMean = d3.mean(this.data, d => d.y) || 0;
    
    let numerator = 0;
    let denominator = 0;
    
    for (const point of this.data) {
      numerator += (point.x - xMean) * (point.y - yMean);
      denominator += Math.pow(point.x - xMean, 2);
    }
    
    if (denominator === 0) return { beta0: yMean, beta1: 0 };
    
    const beta1 = numerator / denominator;
    const beta0 = yMean - beta1 * xMean;
    
    return { beta0, beta1 };
  }
  
  /**
   * Advance the story mode
   */
  private nextStoryStep(): void {
    this.animationState.storyStep++;
    
    // Could implement custom story steps here
    // For now, just toggle camera position for demonstration
    // Using non-null assertions for camera access
    if (this.animationState.storyStep % 2 === 0) {
      gsap.to(this.camera!.position, {
        x: 8,
        y: 8, 
        z: 8,
        duration: 1.5,
        ease: "power2.inOut",
        onUpdate: () => this.camera!.lookAt(0, 0, 0)
      });
    } else {
      gsap.to(this.camera!.position, {
        x: 0,
        y: 10, 
        z: 0,
        duration: 1.5,
        ease: "power2.inOut",
        onUpdate: () => this.camera!.lookAt(0, 0, 0)
      });
    }
  }

  // Tutorial methods
  public toggleTutorial(): void {
    this.tutorialMode = !this.tutorialMode;
    this.tutorialStep = 0;
    this.currentTutorialContent = this.tutorialSteps[0];
  }

  public nextTutorial(): void {
    if (this.tutorialStep < this.tutorialSteps.length - 1) {
      this.tutorialStep++;
      this.currentTutorialContent = this.tutorialSteps[this.tutorialStep];
    } else {
      this.tutorialMode = false;
    }
  }

  public prevTutorial(): void {
    if (this.tutorialStep > 0) {
      this.tutorialStep--;
      this.currentTutorialContent = this.tutorialSteps[this.tutorialStep];
    }
  }

  // User interaction methods
  public toggleGradientPath(): void {
    this.showGradientPath = !this.showGradientPath;
    if (this.gradientPathLine) {
      this.gradientPathLine.visible = this.showGradientPath;
    }
  }

  public toggleErrorLines(): void {
    this.showErrorLines = !this.showErrorLines;
    
    if (!this.svg) return;
    
    this.svg.selectAll('.error-lines line')
      .style('opacity', this.showErrorLines ? 1 : 0);
  }

  public playFullSimulation(): void {
    // Reset first
    this.reset();
    
    // Then start playing
    this.isPlaying = true;
    this.lastStepTime = performance.now();
    this.animationState.state = 'running';
    this.animationState.autoMode = true;
  }

  public reset(): void {
    this.beta0 = 0;
    this.beta1 = 0;
    this.iterations = 0;
    this.isPlaying = false;
    this.costHistory = [];
    this.animationState.state = 'initial';
    this.animationState.storyStep = 0;
    
    this.updateRegressionLine();
    
    // Only call if 3D scene is initialized
    if (this.currentPositionMarker) {
      this.updateMarkerPosition();
    }
    
    try {
      this.initResidualsPlot();
    } catch (error) {
      console.error('Error updating residuals plot:', error);
    }
  }

  public playPause(): void {
    this.isPlaying = !this.isPlaying;
    this.lastStepTime = performance.now();
    
    this.animationState.state = this.isPlaying ? 'running' : 'paused';
  }

  public stepForward(): void {
    if (this.iterations < this.maxIterations) {
      this.gradientDescentStep();
    }
  }

  /**
   * Toggle explanation visibility
   */
  public toggleExplanation(): void {
    this.explanationVisible = !this.explanationVisible;
  }

  /**
   * Change the dataset
   */
  public changeDataset(dataset: string): void {
    this.selectedDataset = dataset;
    this.generateData(dataset);
    
    // Don't initialize visualization in the middle of changeDataset
    // Allow Angular's change detection to complete first
    setTimeout(() => {
      this.initializeVisualization();
    });
  }
  
  /**
   * Activate story mode that takes the user through a guided journey
   */
  public startStoryMode(): void {
    this.reset();
    this.animationState.storyMode = true;
    this.animationState.storyStep = 0;
    
    // Position camera for first view - using non-null assertions
    gsap.to(this.camera!.position, {
      x: 0,
      y: 10, 
      z: 0,
      duration: 1.5,
      ease: "power2.inOut",
      onUpdate: () => this.camera!.lookAt(0, 0, 0),
      onComplete: () => {
        // Start playing after camera is positioned
        this.isPlaying = true;
        this.lastStepTime = performance.now();
        this.animationState.state = 'running';
      }
    });
  }
}