import { Component, OnInit, OnDestroy, ElementRef, ViewChild, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';
import * as THREE from 'three';
// For Angular 19, THREE.js imports need to be handled differently
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface Point {
  x: number;
  y: number;
  label: number | undefined;
  isSupport?: boolean;
}

interface SvmModel {
  w?: number[];
  b?: number;
  supportVectors?: Point[];
  alpha?: number[];
  kernelParams?: any;
  kernelType: string;
  C: number;
  gamma?: number;
  degree?: number;
}

enum SimulationStage {
  DATASET_INTRO = 0,
  HYPERPLANE_SEARCH = 1,
  MARGIN_MAXIMIZATION = 2,
  SUPPORT_VECTORS = 3,
  SOFT_MARGIN = 4,
  KERNEL_TRICK = 5,
  PREDICTIONS = 6
}

enum AnimationState {
  STOPPED = 0,
  PLAYING = 1,
  PAUSED = 2
}

@Component({
  selector: 'app-svm-simulation',
  templateUrl: './svm-simulation.component.html',
  styleUrls: ['./svm-simulation.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class SvmSimulationComponent implements OnInit, OnDestroy {
  @ViewChild('d3Container', { static: true }) private d3Container!: ElementRef;
  @ViewChild('threeContainer', { static: true }) private threeContainer!: ElementRef;

  // Data and Model
  private linearData: Point[] = [];
  private nonLinearData: Point[] = [];
  private currentData: Point[] = [];
  public model: SvmModel = {
    kernelType: 'linear',
    C: 1.0,
    gamma: 0.1,
    degree: 3
  };

  // D3 Elements
  private svg: any;
  private xScale: any;
  private yScale: any;
  private width = 600;
  private height = 400;
  private margin = { top: 20, right: 20, bottom: 30, left: 40 };

  // THREE.js Elements
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private animationFrameId: number = 0;

  // Simulation Control
  public stage: SimulationStage = SimulationStage.DATASET_INTRO;
  public animationState: AnimationState = AnimationState.STOPPED;
  public playIntervalId: any = null;
  public simSpeed = 3000; // ms between stages during auto-play
  
  // Tab control
  public selectedTab = 'explanation';
  public selectTab = (tabName: string): void => {
    this.selectedTab = tabName;
  };
  
  // Info and display
  public researchPapers = [
    {
      title: "A training algorithm for optimal margin classifiers",
      authors: "Boser, B. E., Guyon, I. M., & Vapnik, V. N.",
      year: 1992,
      description: "Introduced the kernel trick for SVMs"
    },
    {
      title: "Support-vector networks",
      authors: "Cortes, C., & Vapnik, V.",
      year: 1995,
      description: "Introduced the soft margin classifier"
    }
  ];
  
  public stageInfo = [
    {
      title: "Dataset Introduction",
      description: "SVM works with labeled data to find the optimal boundary between classes. Here we see our training dataset with two classes."
    },
    {
      title: "Hyperplane Search",
      description: "SVM looks for a hyperplane (a line in 2D) that separates the data. There are many possible hyperplanes, but which one is best?"
    },
    {
      title: "Margin Maximization",
      description: "The best hyperplane has the maximum margin - the widest street between the classes. This often leads to better generalization."
    },
    {
      title: "Support Vectors",
      description: "The critical points that define the margin are called support vectors. These are the only points that matter for the decision boundary."
    },
    {
      title: "Soft Margin",
      description: "With the C parameter, we can allow some points to be misclassified to get a wider margin. This makes the model more robust to noise."
    },
    {
      title: "Kernel Trick",
      description: "For non-linear data, SVM uses the kernel trick to implicitly map data to a higher dimension where it becomes linearly separable."
    },
    {
      title: "Making Predictions",
      description: "Once trained, SVM classifies new points based on which side of the hyperplane they fall on. Let's try some test points!"
    }
  ];
  
  public kernelOptions = [
    { value: 'linear', label: 'Linear Kernel' },
    { value: 'polynomial', label: 'Polynomial Kernel' },
    { value: 'rbf', label: 'RBF (Gaussian) Kernel' }
  ];
  
  public datasetOptions = [
    { value: 'linear', label: 'Linearly Separable' },
    { value: 'nonlinear', label: 'Non-Linear (Circle)' },
    { value: 'overlapping', label: 'Overlapping Classes' }
  ];

  public selectedDataset = 'linear';
  public SVMEquation = "f(x) = sign(w^T x + b)";
  
  constructor(private ngZone: NgZone) {}

  ngOnInit(): void {
    this.initializeData();
    this.initializeD3();
    
    // Only initialize THREE.js for kernel visualization
    if (this.stage === SimulationStage.KERNEL_TRICK) {
      this.initializeThreeJs();
    }
    
    this.renderCurrentStage();
  }

  ngOnDestroy(): void {
    if (this.playIntervalId) {
      clearInterval(this.playIntervalId);
    }
    
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    
    if (this.renderer) {
      this.renderer.dispose();
    }
  }

  initializeData(): void {
    // Generate linearly separable data
    this.linearData = [
      ...Array(50).fill(0).map(() => ({
        x: Math.random() * 4 - 5,
        y: Math.random() * 4 - 2,
        label: -1
      })),
      ...Array(50).fill(0).map(() => ({
        x: Math.random() * 4 + 1,
        y: Math.random() * 4 + 0,
        label: 1
      }))
    ];

    // Generate non-linear circular data
    this.nonLinearData = [];
    // Generate points in a circle (label 1)
    for (let i = 0; i < 60; i++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = 1 + Math.random() * 0.5;
      this.nonLinearData.push({
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        label: 1
      });
    }
    // Generate points outside the circle (label -1)
    for (let i = 0; i < 60; i++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = 3 + Math.random() * 1.5;
      this.nonLinearData.push({
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        label: -1
      });
    }

    this.currentData = this.linearData;
  }

  initializeD3(): void {
    const element = this.d3Container.nativeElement;
    
    // Clear any existing SVG
    d3.select(element).select('svg').remove();
    
    // Create new SVG
    this.svg = d3.select(element)
      .append('svg')
      .attr('width', this.width + this.margin.left + this.margin.right)
      .attr('height', this.height + this.margin.top + this.margin.bottom)
      .append('g')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

    // Setup scales
    const xExtent = d3.extent(this.currentData, d => d.x) as [number, number];
    const yExtent = d3.extent(this.currentData, d => d.y) as [number, number];
    
    // Add some padding to the extents
    const xPadding = (xExtent[1] - xExtent[0]) * 0.2;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.2;
    
    this.xScale = d3.scaleLinear()
      .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
      .range([0, this.width]);
    
    this.yScale = d3.scaleLinear()
      .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
      .range([this.height, 0]);

    // Add axes
    this.svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${this.height})`)
      .call(d3.axisBottom(this.xScale));
    
    this.svg.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(this.yScale));
      
    // Add axes labels
    this.svg.append('text')
      .attr('class', 'x-label')
      .attr('text-anchor', 'end')
      .attr('x', this.width)
      .attr('y', this.height + this.margin.top + 10)
      .text('Feature X');
      
    this.svg.append('text')
      .attr('class', 'y-label')
      .attr('text-anchor', 'end')
      .attr('transform', 'rotate(-90)')
      .attr('y', -this.margin.left + 10)
      .attr('x', -this.margin.top)
      .text('Feature Y');
  }

  initializeThreeJs(): void {
    const element = this.threeContainer.nativeElement;
    
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x162a4a); // Updated to dark-blue-bg
    
    // Create camera
    this.camera = new THREE.PerspectiveCamera(75, element.clientWidth / element.clientHeight, 0.1, 1000);
    this.camera.position.z = 5;
    
    // Create renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(element.clientWidth, element.clientHeight);
    element.appendChild(this.renderer.domElement);
    
    // Add controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    
    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xcccccc, 0.5);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1).normalize();
    this.scene.add(directionalLight);
    
    // Add coordinate axes
    const axesHelper = new THREE.AxesHelper(5);
    this.scene.add(axesHelper);
    
    // Start animation loop
    this.animate();
  }

  animate(): void {
    this.ngZone.runOutsideAngular(() => {
      this.animationFrameId = requestAnimationFrame(() => this.animate());
      if (this.controls) {
        this.controls.update();
      }
      this.renderer.render(this.scene, this.camera);
    });
  }

  renderCurrentStage(): void {
    switch (this.stage) {
      case SimulationStage.DATASET_INTRO:
        this.renderDataset();
        break;
      case SimulationStage.HYPERPLANE_SEARCH:
        this.renderDataset();
        this.animateHyperplaneSearch();
        break;
      case SimulationStage.MARGIN_MAXIMIZATION:
        this.renderDataset();
        this.renderMarginMaximization();
        break;
      case SimulationStage.SUPPORT_VECTORS:
        this.renderDataset();
        this.renderSupportVectors();
        break;
      case SimulationStage.SOFT_MARGIN:
        this.renderDataset();
        this.renderSoftMargin();
        break;
      case SimulationStage.KERNEL_TRICK:
        if (!this.scene) {
          this.initializeThreeJs();
        }
        this.renderKernelTrick();
        break;
      case SimulationStage.PREDICTIONS:
        this.renderDataset();
        this.renderPredictions();
        break;
    }
  }
  
  // Utility function to safely cast arrays for d3 line functions
  private safeLineData(data: any[][]): [number, number][] {
    return data.map(point => [Number(point[0]), Number(point[1])] as [number, number]);
  }

  renderDataset(): void {
    // Clear any existing points
    this.svg.selectAll('.data-point').remove();
    
    // Add data points
    this.svg.selectAll('.data-point')
      .data(this.currentData)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', (d: Point) => this.xScale(d.x))
      .attr('cy', (d: Point) => this.yScale(d.y))
      .attr('r', 5)
      .attr('fill', (d: Point) => d.label === 1 ? '#4285F4' : '#FF6B6B') // Updated colors
      .attr('stroke', (d: Point) => d.isSupport ? '#FFFFFF' : 'none') // Updated colors
      .attr('stroke-width', (d: Point) => d.isSupport ? 2 : 0)
      .attr('opacity', 0)
      .transition()
      .duration(800)
      .attr('opacity', 1);
  }

  animateHyperplaneSearch(): void {
    // Clear existing hyperplanes
    this.svg.selectAll('.hyperplane').remove();
    
    // Simulate searching for hyperplanes
    const totalLines = 10;
    const lineDuration = 300;
    
    for (let i = 0; i < totalLines; i++) {
      // Create random lines that somewhat separate the data
      const slope = Math.random() * 2 - 1;
      const intercept = Math.random() * 4 - 2;
      
      const lineFunc = d3.line<[number, number]>()
        .x(d => this.xScale(d[0]))
        .y(d => this.yScale(slope * d[0] + intercept));
      
      const xDomain = this.xScale.domain();
      const lineData: [number, number][] = [xDomain[0], xDomain[1]].map(x => [x, slope * x + intercept]);
      
      setTimeout(() => {
        this.svg.append('path')
          .attr('class', 'hyperplane')
          .attr('d', lineFunc(lineData))
          .attr('stroke', '#8A9AB0') // Updated color
          .attr('stroke-width', 1)
          .attr('opacity', 0.5)
          .attr('stroke-dasharray', '5,5');
      }, i * lineDuration);
    }
    
    // After showing random lines, show the "best" line
    setTimeout(() => {
      this.svg.selectAll('.hyperplane').remove();
      
      // For linear data, use a simple linear separator
      if (this.selectedDataset === 'linear') {
        // Create a "best" hyperplane (for simple linear data)
        const lineFunc = d3.line<[number, number]>()
          .x(d => this.xScale(d[0]))
          .y(d => this.yScale(0.8 * d[0] - 0.5));
        
        const xDomain = this.xScale.domain();
        const lineData: [number, number][] = [xDomain[0], xDomain[1]].map(x => [x, 0.8 * x - 0.5]);
        
        this.svg.append('path')
          .attr('class', 'hyperplane')
          .attr('d', lineFunc(lineData))
          .attr('stroke', '#FFFFFF') // Updated color
          .attr('stroke-width', 2);
          
        this.model.w = [0.8, 1];
        this.model.b = -0.5;
      } else {
        // For non-linear data, we'll show this better in the kernel trick stage
        this.svg.append('text')
          .attr('class', 'hyperplane-text')
          .attr('text-anchor', 'middle')
          .attr('x', this.width / 2)
          .attr('y', this.height / 2)
          .text("Can't find a good linear separator!")
          .attr('fill', '#FF6B6B'); // Updated color
      }
    }, totalLines * lineDuration + 500);
  }

  renderMarginMaximization(): void {
    // Clear existing elements
    this.svg.selectAll('.hyperplane, .margin-boundary, .margin').remove();
    
    // We need a hyperplane first
    if (!this.model.w || !this.model.b) {
      this.model.w = [0.8, 1];
      this.model.b = -0.5;
    }
    
    // Create the hyperplane
    const w = this.model.w;
    const b = this.model.b;
    
    const lineFunc = d3.line<[number, number]>()
      .x(d => this.xScale(d[0]))
      .y(d => this.yScale(-w[0]/w[1] * d[0] - b/w[1]));
    
    const xDomain = this.xScale.domain();
    const lineData: [number, number][] = [xDomain[0], xDomain[1]].map(x => [x, -w[0]/w[1] * x - b/w[1]]);
    
    // Draw main hyperplane
    this.svg.append('path')
      .attr('class', 'hyperplane')
      .attr('d', lineFunc(lineData))
      .attr('stroke', '#FFFFFF') // Updated color
      .attr('stroke-width', 2);
    
    // Draw margin boundaries
    const margin = 1 / Math.sqrt(w[0]*w[0] + w[1]*w[1]);
    
    // Upper margin boundary
    const upperLineFunc = d3.line<[number, number]>()
      .x(d => this.xScale(d[0]))
      .y(d => this.yScale(-w[0]/w[1] * d[0] - (b+1)/w[1]));
    
    const upperLineData: [number, number][] = [xDomain[0], xDomain[1]].map(x => [x, -w[0]/w[1] * x - (b+1)/w[1]]);
    
    this.svg.append('path')
      .attr('class', 'margin-boundary')
      .attr('d', upperLineFunc(upperLineData))
      .attr('stroke', '#4285F4') // Updated color
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '3,3');
    
    // Lower margin boundary
    const lowerLineFunc = d3.line<[number, number]>()
      .x(d => this.xScale(d[0]))
      .y(d => this.yScale(-w[0]/w[1] * d[0] - (b-1)/w[1]));
    
    const lowerLineData: [number, number][] = [xDomain[0], xDomain[1]].map(x => [x, -w[0]/w[1] * x - (b-1)/w[1]]);
    
    this.svg.append('path')
      .attr('class', 'margin-boundary')
      .attr('d', lowerLineFunc(lowerLineData))
      .attr('stroke', '#FF6B6B') // Updated color
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '3,3');
    
    // Add margin indicator
    const centerX = this.xScale(0);
    const centerY = this.yScale(-b/w[1]);
    
    this.svg.append('line')
      .attr('class', 'margin')
      .attr('x1', centerX)
      .attr('y1', centerY)
      .attr('x2', centerX)
      .attr('y2', this.yScale(-(b+1)/w[1]))
      .attr('stroke', '#FFFFFF') // Updated color
      .attr('stroke-width', 1)
      .attr('marker-end', 'url(#arrow)');
      
    // Add margin label
    this.svg.append('text')
      .attr('class', 'margin-label')
      .attr('x', centerX + 5)
      .attr('y', (centerY + this.yScale(-(b+1)/w[1])) / 2)
      .text(`Margin: ${margin.toFixed(2)}`)
      .attr('font-size', '12px')
      .attr('fill', '#E1E7F5'); // Updated color
      
    // Add arrowhead marker definition
    this.svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#FFFFFF'); // Updated color
  }

  renderSupportVectors(): void {
    // First, we need margin maximization to be shown
    this.renderMarginMaximization();
    
    // Identify support vectors (points close to margin boundaries)
    const w = this.model.w;
    const b = this.model.b;
    
    // Make sure w and b are defined
    if (w && b) {
      this.currentData.forEach(point => {
        const distance = Math.abs(w[0]*point.x + w[1]*point.y + b) / Math.sqrt(w[0]*w[0] + w[1]*w[1]);
        point.isSupport = distance < 1.1; // A bit of tolerance
      });
      
      // Update the visualization to highlight support vectors
      this.svg.selectAll('.data-point')
        .data(this.currentData)
        .attr('stroke', (d: Point) => d.isSupport ? '#FFFFFF' : 'none') // Updated color
        .attr('stroke-width', (d: Point) => d.isSupport ? 2 : 0)
        .attr('r', (d: Point) => d.isSupport ? 7 : 5);
    }
    
    // Add annotation for support vectors
    this.svg.selectAll('.support-vector-label').remove();
    
    const supportVectors = this.currentData.filter(p => p.isSupport);
    if (supportVectors.length > 0) {
      // Just label a few support vectors
      for (let i = 0; i < Math.min(3, supportVectors.length); i++) {
        const sv = supportVectors[i];
        this.svg.append('text')
          .attr('class', 'support-vector-label')
          .attr('x', this.xScale(sv.x) + 10)
          .attr('y', this.yScale(sv.y) - 10)
          .text('Support Vector')
          .attr('font-size', '10px')
          .attr('fill', '#E1E7F5'); // Updated color
      }
    }
  }

  renderSoftMargin(): void {
    // Clone the linear data but add some overlapping points
    const softMarginData = JSON.parse(JSON.stringify(this.linearData));
    
    // Add some overlapping points that would be misclassified
    const overlappingPoints = [
      { x: -2, y: 1, label: -1 },
      { x: -1, y: 1.5, label: -1 },
      { x: 0, y: -1, label: 1 },
      { x: 1, y: -1.5, label: 1 }
    ];
    
    // Add the overlapping points
    overlappingPoints.forEach(p => softMarginData.push(p));
    
    // Update current data
    this.currentData = softMarginData;
    
    // Render dataset with new points
    this.renderDataset();
    
    // Create slider for C parameter
    this.svg.append('text')
      .attr('class', 'c-parameter-label')
      .attr('x', 20)
      .attr('y', 30)
      .text('C Parameter: 1.0')
      .attr('font-size', '14px')
      .attr('fill', '#E1E7F5'); // Updated color
    
    // Show two hyperplanes - one with high C and one with low C
    setTimeout(() => {
      // For high C (strict margin)
      const highCLineFunc = d3.line<[number, number]>()
        .x(d => this.xScale(d[0]))
        .y(d => this.yScale(0.9 * d[0] - 0.2));
      
      const xDomain = this.xScale.domain();
      const highCLineData: [number, number][] = [xDomain[0], xDomain[1]].map(x => [x, 0.9 * x - 0.2]);
      
      this.svg.append('path')
        .attr('class', 'hyperplane high-c')
        .attr('d', highCLineFunc(highCLineData))
        .attr('stroke', '#FF6B6B') // Updated color
        .attr('stroke-width', 2)
        .attr('opacity', 0)
        .transition()
        .duration(1000)
        .attr('opacity', 1);
        
      this.svg.append('text')
        .attr('class', 'high-c-label')
        .attr('x', this.xScale(xDomain[1] - 1))
        .attr('y', this.yScale(0.9 * (xDomain[1] - 1) - 0.2) - 10)
        .text('High C (Strict)')
        .attr('font-size', '12px')
        .attr('fill', '#FF6B6B') // Updated color
        .attr('text-anchor', 'end')
        .attr('opacity', 0)
        .transition()
        .duration(1000)
        .attr('opacity', 1);
    }, 1000);
    
    setTimeout(() => {
      // For low C (soft margin)
      const lowCLineFunc = d3.line<[number, number]>()
        .x(d => this.xScale(d[0]))
        .y(d => this.yScale(0.7 * d[0] - 0.5));
      
      const xDomain = this.xScale.domain();
      const lowCLineData: [number, number][] = [xDomain[0], xDomain[1]].map(x => [x, 0.7 * x - 0.5]);
      
      this.svg.append('path')
        .attr('class', 'hyperplane low-c')
        .attr('d', lowCLineFunc(lowCLineData))
        .attr('stroke', '#24B47E') // Updated color
        .attr('stroke-width', 2)
        .attr('opacity', 0)
        .transition()
        .duration(1000)
        .attr('opacity', 1);
        
      this.svg.append('text')
        .attr('class', 'low-c-label')
        .attr('x', this.xScale(xDomain[1] - 1))
        .attr('y', this.yScale(0.7 * (xDomain[1] - 1) - 0.5) + 20)
        .text('Low C (Soft Margin)')
        .attr('font-size', '12px')
        .attr('fill', '#24B47E') // Updated color
        .attr('text-anchor', 'end')
        .attr('opacity', 0)
        .transition()
        .duration(1000)
        .attr('opacity', 1);
        
      // Show misclassified points
      this.svg.selectAll('.misclassified')
        .data(overlappingPoints)
        .enter()
        .append('circle')
        .attr('class', 'misclassified')
        .attr('cx', (d: Point) => this.xScale(d.x))
        .attr('cy', (d: Point) => this.yScale(d.y))
        .attr('r', 5)
        .attr('fill', 'none')
        .attr('stroke', '#FF9D45') // Updated color
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '3,3')
        .attr('opacity', 0)
        .transition()
        .duration(1000)
        .attr('opacity', 1);
    }, 2000);
  }

  renderKernelTrick(): void {
    // In this stage, we show both 2D and 3D visualizations
    
    // Clear existing content
    this.svg.selectAll('*').remove();
    
    // First, draw the non-linear data in 2D
    this.currentData = this.nonLinearData;
    this.renderDataset();
    
    // Add explanation text
    this.svg.append('text')
      .attr('class', 'kernel-explanation')
      .attr('x', this.width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .text('Non-linear data (2D View)')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .attr('fill', '#E1E7F5'); // Updated color
    
    // Add text for the kernel function
    if (this.model.kernelType === 'rbf') {
      this.svg.append('text')
        .attr('class', 'kernel-function')
        .attr('x', 20)
        .attr('y', 60)
        .text(`RBF Kernel: K(x,y) = exp(-γ||x-y||²), γ = ${this.model.gamma}`)
        .attr('font-size', '12px')
        .attr('fill', '#E1E7F5'); // Updated color
    }
    
    // Now handle the 3D visualization 
    if (this.scene) {
      // Clear existing objects
      while(this.scene.children.length > 0) { 
        this.scene.remove(this.scene.children[0]); 
      }
      
      // Add coordinate system
      const axesHelper = new THREE.AxesHelper(5);
      this.scene.add(axesHelper);
      
      // Add lights
      const ambientLight = new THREE.AmbientLight(0xcccccc, 0.5);
      this.scene.add(ambientLight);
      
      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(1, 1, 1).normalize();
      this.scene.add(directionalLight);
      
      // Create points in 3D
      const pointsGroup = new THREE.Group();
      
      // Create materials for points
      const classMaterial1 = new THREE.MeshStandardMaterial({ color: 0x4285F4 }); // Updated color
      const classMaterial2 = new THREE.MeshStandardMaterial({ color: 0xFF6B6B }); // Updated color
      
      // Add all data points to 3D visualization
      this.nonLinearData.forEach(point => {
        // For RBF kernel, we map points onto a paraboloid
        // z = x² + y²
        const x = point.x;
        const y = point.y;
        const z = x*x + y*y;
        
        const geometry = new THREE.SphereGeometry(0.1);
        const material = point.label === 1 ? classMaterial1 : classMaterial2;
        const sphere = new THREE.Mesh(geometry, material);
        
        sphere.position.set(x, y, z);
        pointsGroup.add(sphere);
      });
      
      this.scene.add(pointsGroup);
      
      // Add the separating plane in 3D (z = constant)
      const planeSize = 10;
      const planeHeight = 2; // Adjusted based on the parabola
      
      const planeGeometry = new THREE.PlaneGeometry(planeSize, planeSize);
      const planeMaterial = new THREE.MeshStandardMaterial({ 
        color: 0xFFFFFF,
        transparent: true,
        opacity: 0.7,
        side: THREE.DoubleSide
      });
      
      const plane = new THREE.Mesh(planeGeometry, planeMaterial);
      plane.rotation.x = Math.PI / 2; // Make it horizontal
      plane.position.set(0, 0, planeHeight);
      
      // Animate the plane moving up and down
      let step = 0;
      const animate3D = () => {
        step += 0.01;
        if (plane) {
          plane.position.z = planeHeight + Math.sin(step) * 0.5;
        }
        requestAnimationFrame(animate3D);
      };
      
      this.scene.add(plane);
      animate3D();
      
      // Add a text label to explain what's happening
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 128;
      const context = canvas.getContext('2d');
      if (context) {
        context.fillStyle = 'rgba(22, 42, 74, 0.8)'; // Updated color
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = '#E1E7F5'; // Updated color
        context.font = '16px Inter, Roboto, sans-serif';
        context.fillText('3D Projection via RBF Kernel', 10, 30);
        context.fillText('Linear separator is possible in', 10, 60);
        context.fillText('higher dimensions!', 10, 80);
      }
      
      const texture = new THREE.CanvasTexture(canvas);
      const labelMaterial = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        side: THREE.DoubleSide
      });
      const labelGeometry = new THREE.PlaneGeometry(2, 1);
      const label = new THREE.Mesh(labelGeometry, labelMaterial);
      label.position.set(3, 1, 3);
      label.lookAt(0, 0, 0);
      
      this.scene.add(label);
      
      // Set the camera to a good viewing position
      this.camera.position.set(5, 5, 5);
      this.camera.lookAt(0, 0, 2);
    }
    
    // If using D3, also add a non-linear decision boundary to the 2D view
    setTimeout(() => {
      // Create a grid of points to evaluate the RBF decision function
      const gridSize = 50;
      const xDomain = this.xScale.domain();
      const yDomain = this.yScale.domain();
      const xStep = (xDomain[1] - xDomain[0]) / gridSize;
      const yStep = (yDomain[1] - yDomain[0]) / gridSize;
      
      // Create a set of contour points
      const contourPoints: [number, number][] = [];
      
      // Simple RBF decision function simulation
      // For a circle dataset, use distance from origin
      for (let i = 0; i <= gridSize; i++) {
        for (let j = 0; j <= gridSize; j++) {
          const x = xDomain[0] + i * xStep;
          const y = yDomain[0] + j * yStep;
          
          // For circular data, we can estimate where the boundary should be
          const distFromOrigin = Math.sqrt(x*x + y*y);
          if (Math.abs(distFromOrigin - 1.75) < 0.1) {
            contourPoints.push([x, y]);
          }
        }
      }
      
      // Draw the decision boundary
      if (contourPoints.length > 0) {
        // Use D3 line interpolation to create a smooth boundary
        const lineFunction = d3.line<[number, number]>()
          .x(d => this.xScale(d[0]))
          .y(d => this.yScale(d[1]))
          .curve(d3.curveBasisClosed);
        
        // Sort points to form a circle
        contourPoints.sort((a, b) => {
          const angleA = Math.atan2(a[1], a[0]);
          const angleB = Math.atan2(b[1], b[0]);
          return angleA - angleB;
        });
        
        this.svg.append('path')
          .attr('class', 'decision-boundary')
          .attr('d', lineFunction(contourPoints))
          .attr('stroke', '#FFFFFF') // Updated color
          .attr('stroke-width', 2)
          .attr('fill', 'none')
          .attr('opacity', 0)
          .transition()
          .duration(1000)
          .attr('opacity', 1);
          
        this.svg.append('text')
          .attr('class', 'decision-boundary-label')
          .attr('x', this.width / 2)
          .attr('y', this.height - 20)
          .attr('text-anchor', 'middle')
          .text('Non-linear Decision Boundary (RBF Kernel)')
          .attr('font-size', '14px')
          .attr('fill', '#E1E7F5') // Updated color
          .attr('opacity', 0)
          .transition()
          .duration(1000)
          .attr('opacity', 1);
      }
    }, 1000);
  }

  renderPredictions(): void {
    // First, render the dataset and current model
    this.renderDataset();
    this.renderMarginMaximization();
    
    // Add some test points that will be classified
    const testPoints: Point[] = [
      { x: -3, y: 1, label: undefined },
      { x: 0, y: 2, label: undefined },
      { x: 2, y: -1, label: undefined }
    ];
    
    // Add test points with question marks
    this.svg.selectAll('.test-point')
      .data(testPoints)
      .enter()
      .append('circle')
      .attr('class', 'test-point')
      .attr('cx', (d: Point) => this.xScale(d.x))
      .attr('cy', (d: Point) => this.yScale(d.y))
      .attr('r', 8)
      .attr('fill', '#FF9D45') // Updated color
      .attr('stroke', '#FFFFFF') // Updated color
      .attr('stroke-width', 1);
      
    this.svg.selectAll('.test-label')
      .data(testPoints)
      .enter()
      .append('text')
      .attr('class', 'test-label')
      .attr('x', (d: Point) => this.xScale(d.x))
      .attr('y', (d: Point) => this.yScale(d.y) + 5)
      .attr('text-anchor', 'middle')
      .text('?')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .attr('fill', '#FFFFFF'); // Updated color
    
    // Animate the classification process
    setTimeout(() => {
      // Classify each test point
      testPoints.forEach((point, i) => {
        setTimeout(() => {
          // Simple classifier based on the model
          if (this.model.w && this.model.b) {
            const wx = this.model.w[0] * point.x + this.model.w[1] * point.y + this.model.b;
            point.label = wx >= 0 ? 1 : -1;
            
            // Update the test point
            this.svg.selectAll('.test-point')
              .filter((_d: any, j: number) => j === i)
              .transition()
              .duration(500)
              .attr('fill', point.label === 1 ? '#4285F4' : '#FF6B6B'); // Updated colors
              
            // Update the label
            this.svg.selectAll('.test-label')
              .filter((_d: any, j: number) => j === i)
              .transition()
              .duration(500)
              .text(point.label === 1 ? '+1' : '-1');
              
            // Show prediction line
            const lineData: [number, number][] = [
              [point.x, point.y], 
              [point.x, -this.model.w[0]/this.model.w[1] * point.x - this.model.b/this.model.w[1]]
            ];
            
            const lineFunc = d3.line<[number, number]>()
              .x(d => this.xScale(d[0]))
              .y(d => this.yScale(d[1]));
              
            this.svg.append('path')
              .attr('class', 'prediction-line')
              .attr('d', lineFunc(lineData))
              .attr('stroke', '#FFFFFF') // Updated color
              .attr('stroke-width', 1)
              .attr('stroke-dasharray', '3,3')
              .attr('opacity', 0)
              .transition()
              .duration(500)
              .attr('opacity', 1);
          }
        }, i * 1000);
      });
    }, 1000);
  }

  // UI Control methods
  changeDataset(dataset: string): void {
    this.selectedDataset = dataset;
    
    switch (dataset) {
      case 'linear':
        this.currentData = this.linearData;
        this.model.kernelType = 'linear';
        break;
      case 'nonlinear':
        this.currentData = this.nonLinearData;
        this.model.kernelType = 'rbf';
        break;
      case 'overlapping':
        // Create overlapping data if not already done
        const overlapData = JSON.parse(JSON.stringify(this.linearData));
        // Add some overlapping points
        overlapData.push(...[
          { x: -2, y: 1, label: -1 },
          { x: -1, y: 1.5, label: -1 },
          { x: 0, y: -1, label: 1 },
          { x: 1, y: -1.5, label: 1 }
        ]);
        this.currentData = overlapData;
        break;
    }
    
    // Reset and reinitialize with new data
    this.initializeD3();
    this.renderCurrentStage();
  }

  changeKernel(kernel: string): void {
    this.model.kernelType = kernel;
    
    // Update the equation display
    switch (kernel) {
      case 'linear':
        this.SVMEquation = "f(x) = sign(w^T x + b)";
        break;
      case 'polynomial':
        this.SVMEquation = `f(x) = sign(∑ α_i y_i (γ⟨x_i, x⟩ + r)^${this.model.degree} + b)`;
        break;
      case 'rbf':
        this.SVMEquation = "f(x) = sign(∑ α_i y_i exp(-γ||x_i - x||²) + b)";
        break;
    }
    
    // If in kernel trick stage, refresh the visualization
    if (this.stage === SimulationStage.KERNEL_TRICK) {
      this.renderKernelTrick();
    }
  }

  changeParam(param: string, value: number): void {
    switch (param) {
      case 'C':
        this.model.C = value;
        break;
      case 'gamma':
        this.model.gamma = value;
        break;
      case 'degree':
        this.model.degree = value;
        break;
    }
    
    // Update visualizations if necessary
    if (this.stage === SimulationStage.SOFT_MARGIN && param === 'C') {
      this.renderSoftMargin();
    }
    
    if (this.stage === SimulationStage.KERNEL_TRICK && (param === 'gamma' || param === 'degree')) {
      this.renderKernelTrick();
    }
  }

  previousStage(): void {
    if (this.stage > 0) {
      this.stage--;
      this.renderCurrentStage();
    }
  }

  nextStage(): void {
    if (this.stage < SimulationStage.PREDICTIONS) {
      this.stage++;
      
      // If we're moving to kernel trick, initialize THREE.js if needed
      if (this.stage === SimulationStage.KERNEL_TRICK && !this.scene) {
        this.initializeThreeJs();
      }
      
      this.renderCurrentStage();
    }
  }

  playSimulation(): void {
    if (this.animationState === AnimationState.PLAYING) {
      // Pause
      if (this.playIntervalId) {
        clearInterval(this.playIntervalId);
        this.playIntervalId = null;
      }
      this.animationState = AnimationState.PAUSED;
    } else {
      // Play or resume
      this.animationState = AnimationState.PLAYING;
      
      const advanceStage = () => {
        if (this.stage < SimulationStage.PREDICTIONS) {
          this.stage++;
          this.renderCurrentStage();
        } else {
          // End of simulation
          clearInterval(this.playIntervalId);
          this.playIntervalId = null;
          this.animationState = AnimationState.STOPPED;
        }
      };
      
      this.playIntervalId = setInterval(advanceStage, this.simSpeed);
    }
  }

  stopSimulation(): void {
    if (this.playIntervalId) {
      clearInterval(this.playIntervalId);
      this.playIntervalId = null;
    }
    
    this.animationState = AnimationState.STOPPED;
    this.stage = SimulationStage.DATASET_INTRO;
    this.renderCurrentStage();
  }

  resetSimulation(): void {
    this.stopSimulation();
    
    // Reset data and model parameters
    this.selectedDataset = 'linear';
    this.currentData = this.linearData;
    
    this.model = {
      kernelType: 'linear',
      C: 1.0,
      gamma: 0.1,
      degree: 3
    };
    
    // Reset visualization
    this.initializeD3();
    this.renderCurrentStage();
  }
}