import { Component, OnInit, ElementRef, ViewChild, AfterViewInit, HostListener } from '@angular/core';
import * as d3 from 'd3';
import * as THREE from 'three';
// Update imports to match Angular project structure
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ParametricGeometry } from 'three/examples/jsm/geometries/ParametricGeometry.js';

@Component({
  selector: 'app-linear-regression',
  templateUrl: './linear-regression.component.html',
  styleUrls: ['./linear-regression.component.scss']
})
// Note: You must also import FormsModule in your AppModule or the feature module
// where this component is declared for ngModel to work
export class LinearRegressionComponent implements OnInit, AfterViewInit {
  @ViewChild('scatterPlot') private scatterPlotRef!: ElementRef;
  @ViewChild('costFunction') private costFunctionRef!: ElementRef;
  
  // Configuration options
  private margin = { top: 20, right: 20, bottom: 30, left: 40 };
  private width = 600 - this.margin.left - this.margin.right;
  private height = 400 - this.margin.top - this.margin.bottom;
  
  // Parameters visible to the view
  public beta0 = 0; // Intercept
  public beta1 = 0; // Slope
  public learningRate = 0.01; // Make sure this is treated as a number when used
  public iterations = 0;
  public maxIterations = 100;
  public isPlaying = false;
  public animationSpeed = 50; // milliseconds between steps
  public currentCost = 0;
  public showGradientPath = true;
  
  // Data & animation
  private data: {x: number, y: number}[] = [];
  private animationId: number | null = null;
  private lastStepTime = 0;
  
  // Three.js variables
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private costSurface!: THREE.Mesh;
  private currentPositionMarker!: THREE.Mesh;
  private gradientPathLine!: THREE.Line;
  private costHistory: {beta0: number, beta1: number, cost: number}[] = [];
  
  constructor() {
    this.generateData();
  }
  
  ngOnInit(): void {}
  
  ngAfterViewInit(): void {
    setTimeout(() => {
      this.initScatterPlot();
      this.initCostFunctionVisualization();
      this.updateRegressionLine();
      
      // Add window resize handling
      window.addEventListener('resize', this.handleResize.bind(this));
    });
  }
  
  /**
   * Handle window resize events to maintain responsive visualizations
   */
  @HostListener('window:resize')
  handleResize(): void {
    if (this.renderer) {
      const width = this.costFunctionRef.nativeElement.clientWidth;
      const height = this.costFunctionRef.nativeElement.clientHeight;
      
      this.renderer.setSize(width, height);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
    
    // Could also update D3 visualization here if needed
  }
  
  /**
   * Generate synthetic data with a linear relationship plus noise
   */
  private generateData(): void {
    const trueIntercept = 5;
    const trueSlope = 2;
    const noiseFactor = 1.5;
    
    for (let i = 0; i < 30; i++) {
      const x = Math.random() * 10;
      const y = trueIntercept + trueSlope * x + (Math.random() - 0.5) * noiseFactor;
      this.data.push({ x, y });
    }
  }
  
  /**
   * Initialize the D3.js scatter plot with data points and regression line
   */
  private initScatterPlot(): void {
    const container = this.scatterPlotRef.nativeElement;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight || 400;
    
    const width = containerWidth - this.margin.left - this.margin.right;
    const height = containerHeight - this.margin.top - this.margin.bottom;
    
    const svg = d3.select(container)
      .append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)
      .append('g')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(this.data, d => d.x) || 10])
      .range([0, width]);
    
    const yScale = d3.scaleLinear()
      .domain([0, d3.max(this.data, d => d.y) || 10])
      .range([height, 0]);
    
    // Add X and Y axes
    svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale))
      .append('text')
      .attr('class', 'axis-label')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('fill', '#000')
      .text('X');
    
    svg.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale))
      .append('text')
      .attr('class', 'axis-label')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -30)
      .attr('fill', '#000')
      .text('Y');
    
    // Add data points with transition
    svg.selectAll('.dot')
      .data(this.data)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(d.x))
      .attr('cy', height) // Start from bottom
      .attr('r', 0) // Start with radius 0
      .style('fill', '#3498db')
      .style('opacity', 0.7)
      .transition()
      .duration(1000)
      .delay((d, i) => i * 20)
      .attr('cy', d => yScale(d.y))
      .attr('r', 5);
    
    // Add regression line
    svg.append('line')
      .attr('class', 'regression-line')
      .attr('x1', 0)
      .attr('y1', yScale(this.beta0))
      .attr('x2', width)
      .attr('y2', yScale(this.beta0 + this.beta1 * xScale.invert(width)))
      .style('stroke', '#e74c3c')
      .style('stroke-width', 2.5)
      .style('stroke-dasharray', '5,5')
      .style('opacity', 0)
      .transition()
      .duration(1000)
      .style('opacity', 1)
      .style('stroke-dasharray', '0');
      
    // Add grid lines for better readability
    svg.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height})`)
      .call((g) => {
        d3.axisBottom(xScale)
          .tickSize(-height)
          .tickFormat(() => '')
          (g);
      })
      .selectAll('line')
      .style('stroke', '#e6e6e6');
      
    svg.append('g')
      .attr('class', 'grid')
      .call((g) => {
        d3.axisLeft(yScale)
          .tickSize(-width)
          .tickFormat(() => '')
          (g);
      })
      .selectAll('line')
      .style('stroke', '#e6e6e6');
  }
  
  /**
   * Initialize the 3D cost function visualization using Three.js
   */
  private initCostFunctionVisualization(): void {
    const container = this.costFunctionRef.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight || 400;
    
    // Set up Three.js scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf7f7f7);
    
    // Camera setup
    this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    this.camera.position.set(8, 8, 8);
    this.camera.lookAt(0, 0, 0);
    
    // Renderer with anti-aliasing for smoother edges
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);
    
    // Controls for interacting with the 3D view
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    
    // Add lights for better 3D rendering
    const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 7.5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 1024;
    directionalLight.shadow.mapSize.height = 1024;
    this.scene.add(directionalLight);
    
    // Add cost function surface
    this.createCostSurface();
    
    // Add coordinate system axes
    const axesHelper = new THREE.AxesHelper(5);
    this.scene.add(axesHelper);
    
    // Add labels for axes
    this.addAxisLabels();
    
    // Add current position marker (red sphere)
    const markerGeometry = new THREE.SphereGeometry(0.2, 32, 32);
    const markerMaterial = new THREE.MeshPhongMaterial({ 
      color: 0xe74c3c, 
      emissive: 0xe74c3c,
      emissiveIntensity: 0.3,
      specular: 0xffffff,
      shininess: 30
    });
    this.currentPositionMarker = new THREE.Mesh(markerGeometry, markerMaterial);
    this.currentPositionMarker.castShadow = true;
    this.updateMarkerPosition();
    this.scene.add(this.currentPositionMarker);
    
    // Create gradient path line
    this.createGradientPathLine();
    
    // Start animation loop
    this.animate();
  }
  
  /**
   * Add text labels for the 3D axes
   */
  private addAxisLabels(): void {
    // Create canvas elements for the text labels
    const createTextTexture = (text: string): THREE.Texture => {
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 128;
      const context = canvas.getContext('2d')!;
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
    const beta0Label = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: createTextTexture('β₀'), transparent: true })
    );
    beta0Label.position.set(6, 0, 0);
    beta0Label.scale.set(1, 0.5, 1);
    this.scene.add(beta0Label);
    
    const beta1Label = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: createTextTexture('β₁'), transparent: true })
    );
    beta1Label.position.set(0, 0, 6);
    beta1Label.scale.set(1, 0.5, 1);
    this.scene.add(beta1Label);
    
    const costLabel = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: createTextTexture('Cost'), transparent: true })
    );
    costLabel.position.set(0, 6, 0);
    costLabel.scale.set(1, 0.5, 1);
    this.scene.add(costLabel);
  }
  
  /**
   * Create the 3D surface representing the cost function
   */
  private createCostSurface(): void {
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
      shininess: 40,
      transparent: true,
      opacity: 0.85,
      flatShading: false
    });
    
    this.costSurface = new THREE.Mesh(geometry, material);
    this.costSurface.receiveShadow = true;
    this.scene.add(this.costSurface);
    
    // Add wireframe for better understanding of the surface shape
    const wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 0x000000,
      wireframe: true,
      transparent: true,
      opacity: 0.15
    });
    
    const wireframe = new THREE.Mesh(geometry.clone(), wireframeMaterial);
    this.scene.add(wireframe);
    
    // Add contour lines on the surface
    this.addContourLines(costScale);
  }
  
  /**
   * Add contour lines to the cost surface for better visualization
   */
  private addContourLines(costScale: number): void {
    const contourLevels = 10;
    const beta0Range = 10;
    const beta1Range = 5;
    const resolution = 30;
    
    for (let level = 1; level <= contourLevels; level++) {
      const costLevel = (level / contourLevels) * 5; // Maximum cost * costScale
      const points: THREE.Vector3[] = [];
      
      // Sample points on the surface at this cost level
      for (let i = 0; i <= resolution; i++) {
        const beta0 = (i / resolution - 0.5) * beta0Range;
        
        for (let j = 0; j <= resolution; j++) {
          const beta1 = (j / resolution - 0.5) * beta1Range;
          const cost = this.computeCost(beta0, beta1) * costScale;
          
          // If this point is close to our target contour level
          if (Math.abs(cost - costLevel) < 0.1) {
            points.push(new THREE.Vector3(beta0, costLevel, beta1));
          }
        }
      }
      
      if (points.length > 1) {
        const contourGeometry = new THREE.BufferGeometry().setFromPoints(points);
        const contourMaterial = new THREE.PointsMaterial({ 
          color: 0x000000,
          size: 0.05,
          transparent: true,
          opacity: 0.5
        });
        const contourLine = new THREE.Points(contourGeometry, contourMaterial);
        this.scene.add(contourLine);
      }
    }
  }
  
  /**
   * Create the line showing the gradient descent path
   */
  private createGradientPathLine(): void {
    // Initialize with just the starting point
    const points = [
      new THREE.Vector3(this.beta0, this.computeCost(this.beta0, this.beta1) * 0.1, this.beta1)
    ];
    
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    
    const material = new THREE.LineBasicMaterial({
      color: 0xe74c3c,
      linewidth: 2,
    });
    
    this.gradientPathLine = new THREE.Line(geometry, material);
    this.scene.add(this.gradientPathLine);
  }
  
  /**
   * Update the gradient path line with new points
   */
  private updateGradientPathLine(): void {
    if (!this.showGradientPath || this.costHistory.length < 2) return;
    
    const points = this.costHistory.map(point => 
      new THREE.Vector3(point.beta0, point.cost * 0.1, point.beta1)
    );
    
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    this.gradientPathLine.geometry.dispose();
    this.gradientPathLine.geometry = geometry;
  }
  
  /**
   * Compute the cost (mean squared error) for given parameters
   */
  public computeCost(b0: number, b1: number): number {
    let sum = 0;
    for (const point of this.data) {
      const prediction = b0 + b1 * point.x;
      sum += Math.pow(prediction - point.y, 2);
    }
    return sum / (2 * this.data.length);
  }
  
  /**
   * Update the position of the marker on the cost surface
   */
  private updateMarkerPosition(): void {
    const cost = this.computeCost(this.beta0, this.beta1);
    this.currentCost = cost;
    
    // Update marker position (with a slight animation)
    const newY = cost * 0.1;
    this.currentPositionMarker.position.x = this.beta0;
    this.currentPositionMarker.position.z = this.beta1;
    
    // Add current position to history
    this.costHistory.push({ beta0: this.beta0, beta1: this.beta1, cost });
    
    // Update the line showing the gradient descent path
    this.updateGradientPathLine();
    
    // Animate the marker moving up/down
    const targetY = { y: this.currentPositionMarker.position.y };
    const newPosition = { y: newY };
    
    // Setting up a simple animation timing
    const startY = this.currentPositionMarker.position.y;
    const deltaY = newY - startY;
    let startTime: number | null = null;
    
    const animateY = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;
      const progress = Math.min(elapsed / 300, 1); // 300ms animation
      
      this.currentPositionMarker.position.y = startY + deltaY * progress;
      
      if (progress < 1) {
        requestAnimationFrame(animateY);
      }
    };
    
    requestAnimationFrame(animateY);
  }
  
  /**
   * Three.js animation loop
   */
  private animate(): void {
    requestAnimationFrame(() => this.animate());
    
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
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
  
  /**
   * Perform one step of gradient descent
   */
  private gradientDescentStep(): void {
    if (this.iterations >= this.maxIterations) {
      this.isPlaying = false;
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
    
    this.iterations++;
  }
  
  /**
   * Update the regression line in the scatter plot
   */
  private updateRegressionLine(): void {
    const svg = d3.select(this.scatterPlotRef.nativeElement).select('svg').select('g');
    
    // Get the current width and height
    const width = this.scatterPlotRef.nativeElement.clientWidth - this.margin.left - this.margin.right;
    const height = this.scatterPlotRef.nativeElement.clientHeight - this.margin.top - this.margin.bottom;
    
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(this.data, d => d.x) || 10])
      .range([0, width]);
    
    const yScale = d3.scaleLinear()
      .domain([0, d3.max(this.data, d => d.y) || 10])
      .range([height, 0]);
    
    svg.select('.regression-line')
      .transition()
      .duration(100)
      .attr('y1', yScale(this.beta0))
      .attr('y2', yScale(this.beta0 + this.beta1 * xScale.invert(width)));
  }
  
  /**
   * Toggle between play and pause states
   */
  public playPause(): void {
    this.isPlaying = !this.isPlaying;
    this.lastStepTime = performance.now();
  }
  
  /**
   * Perform a single step of gradient descent
   */
  public stepForward(): void {
    if (this.iterations < this.maxIterations) {
      this.gradientDescentStep();
    }
  }
  
  /**
   * Reset the simulation to initial state
   */
  public reset(): void {
    this.beta0 = 0;
    this.beta1 = 0;
    this.iterations = 0;
    this.isPlaying = false;
    this.costHistory = [];
    
    this.updateRegressionLine();
    this.updateMarkerPosition();
  }
  
  /**
   * Toggle the visibility of the gradient path
   */
  public toggleGradientPath(): void {
    this.showGradientPath = !this.showGradientPath;
    this.gradientPathLine.visible = this.showGradientPath;
  }
  
  /**
   * Play the entire simulation from start to finish
   */
  public playFullSimulation(): void {
    // Reset first
    this.reset();
    
    // Then start playing
    this.isPlaying = true;
    this.lastStepTime = performance.now();
  }
}