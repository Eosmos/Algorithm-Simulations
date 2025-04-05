// linear-regression-simulator.component.ts

import { Component, OnInit, AfterViewInit, ElementRef, ViewChild, NgZone, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';
import * as THREE from 'three';
// Import OrbitControls with proper type support
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

@Component({
  selector: 'app-linear-regression-simulator',
  templateUrl: './linear-regression-simulator.component.html',
  styleUrls: ['./linear-regression-simulator.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class LinearRegressionSimulatorComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('d3Container') private d3Container!: ElementRef;
  @ViewChild('threeContainer') private threeContainer!: ElementRef;
  
  // Simulation state
  isPlaying = false;
  currentStep = 0;
  animationSpeed = 1000; // ms per step
  private animationTimer: any;
  
  // Linear regression parameters
  beta0 = 0; // Intercept
  beta1 = 0; // Slope
  learningRate = 0.01;
  iterations = 100;
  
  // Data
  private data: {x: number, y: number}[] = [];
  
  // D3 elements
  private svg: any;
  private xScale: any;
  private yScale: any;
  private regressionLine: any;
  private residuals: any;
  
  // THREE.js elements
  private scene: THREE.Scene | null = null;
  private camera: THREE.PerspectiveCamera | null = null;
  private renderer: THREE.WebGLRenderer | null = null;
  private controls: OrbitControls | null = null;
  private costSurface: THREE.Mesh | null = null;
  private gradientPath: THREE.Line | null = null;
  private costFunctionHistory: number[] = [];
  private bestFitMarker: THREE.Mesh | null = null;
  
  // UI state
  activeTab = 'visualization';
  showFormulas = false;
  currentVisualization = '2d'; // '2d' or '3d'
  
  // Algorithm information
  algorithmName = 'Linear Regression';
  category = 'Supervised Learning';
  description = 'A supervised learning algorithm that models the relationship between a dependent variable and one or more independent variables using a linear equation.';
  
  constructor(private ngZone: NgZone) {}
  
  ngOnInit(): void {
    this.generateRandomData(50);
    this.resetSimulation();
  }
  
  ngAfterViewInit(): void {
    setTimeout(() => {
      this.initializeD3();
      this.initializeThreeJs();
    }, 0);
    
    // Handle window resize
    window.addEventListener('resize', this.onWindowResize.bind(this));
  }
  
  ngOnDestroy(): void {
    this.stopSimulation();
    window.removeEventListener('resize', this.onWindowResize.bind(this));
    
    // Clean up Three.js resources
    if (this.renderer) {
      this.renderer.dispose();
      if (this.threeContainer.nativeElement.contains(this.renderer.domElement)) {
        this.threeContainer.nativeElement.removeChild(this.renderer.domElement);
      }
    }
  }
  
  private onWindowResize(): void {
    // Update D3 visualization
    if (this.svg) {
      const element = this.d3Container.nativeElement;
      this.svg.attr('width', element.clientWidth);
      this.svg.attr('height', element.clientHeight);
      this.updateD3Visualization();
    }
    
    // Update Three.js visualization
    if (this.camera && this.renderer) {
      const element = this.threeContainer.nativeElement;
      this.camera.aspect = element.clientWidth / element.clientHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(element.clientWidth, element.clientHeight);
    }
  }
  
  private generateRandomData(n: number): void {
    this.data = [];
    const trueSlope = 2;
    const trueIntercept = 5;
    
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 10;
      // y = mx + b + some noise
      const y = trueSlope * x + trueIntercept + (Math.random() - 0.5) * 4;
      this.data.push({x, y});
    }
  }
  
  private initializeD3(): void {
    if (!this.d3Container) return;
    
    const element = this.d3Container.nativeElement;
    const width = element.clientWidth;
    const height = element.clientHeight;
    const margin = {top: 20, right: 20, bottom: 60, left: 60};
    
    // Clear previous SVG
    d3.select(element).selectAll("svg").remove();
    
    this.svg = d3.select(element)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
    
    // Define scales
    const xExtent = d3.extent(this.data, d => d.x) as [number, number];
    const yExtent = d3.extent(this.data, d => d.y) as [number, number];
    
    this.xScale = d3.scaleLinear()
      .domain([xExtent[0] - 1, xExtent[1] + 1])
      .range([margin.left, width - margin.right]);
    
    this.yScale = d3.scaleLinear()
      .domain([yExtent[0] - 1, yExtent[1] + 1])
      .range([height - margin.bottom, margin.top]);
    
    // Add axes
    this.svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(this.xScale));
    
    this.svg.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(this.yScale));
    
    // Add axis labels
    this.svg.append('text')
      .attr('class', 'x-axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', width / 2)
      .attr('y', height - margin.bottom / 3)
      .text('X (Input Feature)');
    
    this.svg.append('text')
      .attr('class', 'y-axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', `translate(${margin.left / 3},${height / 2}) rotate(-90)`)
      .text('Y (Target Variable)');
    
    // Add grid lines
    this.svg.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(this.xScale)
        .tickSize(-(height - margin.top - margin.bottom))
        .tickFormat(null as any)
      );
    
    this.svg.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(this.yScale)
        .tickSize(-(width - margin.left - margin.right))
        .tickFormat(null as any)
      );
    
    // Group for residuals
    this.residuals = this.svg.append('g')
      .attr('class', 'residuals');
    
    // Add data points
    this.svg.append('g')
      .attr('class', 'data-points')
      .selectAll('circle')
      .data(this.data)
      .enter()
      .append('circle')
      .attr('cx', (d: {x: number, y: number}) => this.xScale(d.x))
      .attr('cy', (d: {x: number, y: number}) => this.yScale(d.y))
      .attr('r', 5)
      .attr('fill', '#4285f4')
      .attr('opacity', 0.7);
    
    // Add regression line (initially at y = 0)
    this.regressionLine = this.svg.append('line')
      .attr('class', 'regression-line')
      .attr('x1', this.xScale(xExtent[0] - 1))
      .attr('y1', this.yScale(this.beta0 + this.beta1 * (xExtent[0] - 1)))
      .attr('x2', this.xScale(xExtent[1] + 1))
      .attr('y2', this.yScale(this.beta0 + this.beta1 * (xExtent[1] + 1)))
      .attr('stroke', '#ff9d45')
      .attr('stroke-width', 2);
    
    // Add cost display
    this.svg.append('text')
      .attr('class', 'cost-display')
      .attr('x', width - margin.right)
      .attr('y', margin.top)
      .attr('text-anchor', 'end')
      .text(`MSE: ${this.computeMSE(this.beta0, this.beta1).toFixed(2)}`);
    
    // Update visualization
    this.updateD3Visualization();
  }
  
  private updateD3Visualization(): void {
    if (!this.svg || !this.xScale || !this.yScale) return;
    
    const xExtent = d3.extent(this.data, d => d.x) as [number, number];
    
    // Update regression line
    this.regressionLine
      .attr('x1', this.xScale(xExtent[0] - 1))
      .attr('y1', this.yScale(this.beta0 + this.beta1 * (xExtent[0] - 1)))
      .attr('x2', this.xScale(xExtent[1] + 1))
      .attr('y2', this.yScale(this.beta0 + this.beta1 * (xExtent[1] + 1)));
    
    // Update residuals
    this.residuals.selectAll('line').remove();
    this.data.forEach(point => {
      const predictedY = this.beta0 + this.beta1 * point.x;
      
      this.residuals.append('line')
        .attr('x1', this.xScale(point.x))
        .attr('y1', this.yScale(point.y))
        .attr('x2', this.xScale(point.x))
        .attr('y2', this.yScale(predictedY))
        .attr('stroke', '#ff6b6b')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3');
    });
    
    // Update cost display
    this.svg.select('.cost-display')
      .text(`MSE: ${this.computeMSE(this.beta0, this.beta1).toFixed(2)}`);
  }
  
  private initializeThreeJs(): void {
    if (!this.threeContainer) return;
    
    const element = this.threeContainer.nativeElement;
    const width = element.clientWidth;
    const height = element.clientHeight;
    
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#0c1428');
    
    // Create camera
    this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    this.camera.position.set(3, 3, 3);
    this.camera.lookAt(0, 0, 0);
    
    // Create renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    element.appendChild(this.renderer.domElement);
    
    // Add controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    this.scene.add(directionalLight);
    
    // Create cost function surface
    this.createCostSurface();
    
    // Add gradient descent path
    this.createGradientPath();
    
    // Add axes helper
    const axesHelper = new THREE.AxesHelper(2);
    this.scene.add(axesHelper);
    
    // Add axes labels
    this.addAxisLabels();
    
    // Start animation loop
    this.ngZone.runOutsideAngular(() => this.animate());
  }
  
  private addAxisLabels(): void {
    if (!this.scene) return;
    
    const createTextSprite = (text: string, position: THREE.Vector3, color: string) => {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      if (!context) return;
      
      canvas.width = 256;
      canvas.height = 128;
      
      context.font = 'Bold 24px Arial';
      context.fillStyle = color;
      context.textAlign = 'center';
      context.fillText(text, 128, 64);
      
      const texture = new THREE.Texture(canvas);
      texture.needsUpdate = true;
      
      const material = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(material);
      sprite.position.copy(position);
      sprite.scale.set(0.5, 0.25, 1);
      
      this.scene?.add(sprite);
    };
    
    // Add axis labels
    createTextSprite('β₀ (Intercept)', new THREE.Vector3(2.5, 0, 0), '#ffffff');
    createTextSprite('MSE (Cost)', new THREE.Vector3(0, 2.5, 0), '#ffffff');
    createTextSprite('β₁ (Slope)', new THREE.Vector3(0, 0, 2.5), '#ffffff');
  }
  
  private createCostSurface(): void {
    if (!this.scene) return;
    
    // Create a surface representing the MSE cost function
    const geometry = new THREE.PlaneGeometry(4, 4, 50, 50);
    
    // Deform the plane to represent the cost function
    const positions = (geometry.attributes['position'].array as Float32Array);
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i]; // beta0
      const z = positions[i + 2]; // beta1
      
      // Compute the MSE for beta0 = x and beta1 = z
      const mse = this.computeMSE(x, z);
      positions[i + 1] = mse * 0.1; // Scale down the MSE for better visualization
    }
    
    // Update normals after changing vertices
    geometry.computeVertexNormals();
    
    // Create material with gradient based on height
    const vertexShader = `
      varying vec3 vPosition;
      void main() {
        vPosition = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;
    
    const fragmentShader = `
      varying vec3 vPosition;
      void main() {
        float height = vPosition.y;
        vec3 color;
        if (height < 0.05) {
          color = mix(vec3(0.0, 0.5, 1.0), vec3(0.0, 0.8, 1.0), height / 0.05);
        } else if (height < 0.1) {
          color = mix(vec3(0.0, 0.8, 1.0), vec3(0.0, 1.0, 0.8), (height - 0.05) / 0.05);
        } else if (height < 0.2) {
          color = mix(vec3(0.0, 1.0, 0.8), vec3(0.5, 1.0, 0.0), (height - 0.1) / 0.1);
        } else if (height < 0.3) {
          color = mix(vec3(0.5, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (height - 0.2) / 0.1);
        } else if (height < 0.4) {
          color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (height - 0.3) / 0.1);
        } else {
          color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), min(1.0, (height - 0.4) / 0.2));
        }
        gl_FragColor = vec4(color, 0.85);
      }
    `;
    
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      transparent: true,
      side: THREE.DoubleSide,
    });
    
    // Create mesh
    this.costSurface = new THREE.Mesh(geometry, material);
    this.scene.add(this.costSurface);
    
    // Add wireframe for better visualization
    const wireframeMaterial = new THREE.LineBasicMaterial({ 
      color: 0xffffff, 
      transparent: true, 
      opacity: 0.2 
    });
    const wireframe = new THREE.LineSegments(
      new THREE.WireframeGeometry(geometry), 
      wireframeMaterial
    );
    this.costSurface.add(wireframe);
    
    // Add a marker for optimal point
    const optimalParams = this.findOptimalParameters();
    const optimalMSE = this.computeMSE(optimalParams.beta0, optimalParams.beta1);
    
    const markerGeometry = new THREE.SphereGeometry(0.05, 16, 16);
    const markerMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    this.bestFitMarker = new THREE.Mesh(markerGeometry, markerMaterial);
    this.bestFitMarker.position.set(
      optimalParams.beta0, 
      optimalMSE * 0.1, 
      optimalParams.beta1
    );
    this.scene.add(this.bestFitMarker);
  }
  
  private findOptimalParameters(): { beta0: number, beta1: number } {
    // Compute optimal parameters using normal equation
    // X' * X * β = X' * y
    const X = this.data.map(d => [1, d.x]);
    const y = this.data.map(d => d.y);
    
    // Compute X' * X
    const XtX = [
      [0, 0], 
      [0, 0]
    ];
    
    for (const row of X) {
      XtX[0][0] += row[0] * row[0];
      XtX[0][1] += row[0] * row[1];
      XtX[1][0] += row[1] * row[0];
      XtX[1][1] += row[1] * row[1];
    }
    
    // Compute X' * y
    const Xty = [0, 0];
    for (let i = 0; i < X.length; i++) {
      Xty[0] += X[i][0] * y[i];
      Xty[1] += X[i][1] * y[i];
    }
    
    // Compute inverse of X' * X
    const det = XtX[0][0] * XtX[1][1] - XtX[0][1] * XtX[1][0];
    const invXtX = [
      [XtX[1][1] / det, -XtX[0][1] / det],
      [-XtX[1][0] / det, XtX[0][0] / det]
    ];
    
    // Compute β = (X' * X)^-1 * X' * y
    const beta0 = invXtX[0][0] * Xty[0] + invXtX[0][1] * Xty[1];
    const beta1 = invXtX[1][0] * Xty[0] + invXtX[1][1] * Xty[1];
    
    return { beta0, beta1 };
  }
  
  private createGradientPath(): void {
    if (!this.scene) return;
    
    // Create a line showing the gradient descent path
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.LineBasicMaterial({ 
      color: 0xff9d45, 
      linewidth: 3
    });
    
    // Initialize with a single point (will be updated during animation)
    const mse = this.computeMSE(this.beta0, this.beta1);
    const points = [new THREE.Vector3(this.beta0, mse * 0.1, this.beta1)];
    geometry.setFromPoints(points);
    
    this.gradientPath = new THREE.Line(geometry, material);
    this.scene.add(this.gradientPath);
    
    // Initialize cost function history
    this.costFunctionHistory = [mse];
  }
  
  private updateGradientPath(): void {
    if (!this.gradientPath || !this.scene) return;
    
    // Calculate current MSE
    const mse = this.computeMSE(this.beta0, this.beta1);
    this.costFunctionHistory.push(mse);
    
    // Add new point to the path
    const points = [];
    
    // Reconstruct all points
    for (let i = 0; i <= this.currentStep; i++) {
      // Simple linear interpolation for beta values
      const t = i / this.iterations;
      const initialBeta0 = 0;
      const initialBeta1 = 0;
      const currentBeta0 = this.beta0;
      const currentBeta1 = this.beta1;
      
      const beta0 = initialBeta0 + (currentBeta0 - initialBeta0) * (i / this.currentStep || 0);
      const beta1 = initialBeta1 + (currentBeta1 - initialBeta1) * (i / this.currentStep || 0);
      
      const stepMSE = this.costFunctionHistory[i] || mse;
      points.push(new THREE.Vector3(beta0, stepMSE * 0.1, beta1));
    }
    
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    this.gradientPath.geometry.dispose();
    this.gradientPath.geometry = geometry;
  }
  
  private animate(): void {
    if (!this.scene || !this.camera || !this.renderer || !this.controls) return;
    
    requestAnimationFrame(() => this.animate());
    
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
  
  private computeMSE(beta0: number, beta1: number): number {
    let mse = 0;
    for (const point of this.data) {
      const prediction = beta0 + beta1 * point.x;
      const error = prediction - point.y;
      mse += error * error;
    }
    return mse / this.data.length;
  }
  
  private computeGradients(): {beta0Grad: number, beta1Grad: number} {
    let beta0Grad = 0;
    let beta1Grad = 0;
    
    for (const point of this.data) {
      const prediction = this.beta0 + this.beta1 * point.x;
      const error = prediction - point.y;
      
      beta0Grad += error;
      beta1Grad += error * point.x;
    }
    
    return {
      beta0Grad: beta0Grad * (2 / this.data.length),
      beta1Grad: beta1Grad * (2 / this.data.length)
    };
  }
  
  private performGradientDescentStep(): void {
    // Compute gradients
    const {beta0Grad, beta1Grad} = this.computeGradients();
    
    // Update parameters
    this.beta0 -= this.learningRate * beta0Grad;
    this.beta1 -= this.learningRate * beta1Grad;
    
    // Update visualizations
    this.updateD3Visualization();
    this.updateGradientPath();
    
    // Update current step
    this.currentStep++;
    
    // Stop if we've reached the maximum number of iterations
    if (this.currentStep >= this.iterations) {
      this.stopSimulation();
    }
  }
  
  // Public methods for UI interaction
  playSimulation(): void {
    if (this.isPlaying) return;
    
    this.isPlaying = true;
    this.animationTimer = setInterval(() => {
      this.performGradientDescentStep();
    }, this.animationSpeed);
  }
  
  pauseSimulation(): void {
    if (!this.isPlaying) return;
    
    this.isPlaying = false;
    clearInterval(this.animationTimer);
  }
  
  stopSimulation(): void {
    this.pauseSimulation();
    this.currentStep = 0;
  }
  
  resetSimulation(): void {
    this.stopSimulation();
    this.beta0 = 0;
    this.beta1 = 0;
    this.costFunctionHistory = [this.computeMSE(this.beta0, this.beta1)];
    
    if (this.svg) this.updateD3Visualization();
    
    // Reset gradient path
    if (this.gradientPath && this.scene) {
      this.scene.remove(this.gradientPath);
      this.createGradientPath();
    }
  }
  
  stepForward(): void {
    if (this.currentStep < this.iterations) {
      this.performGradientDescentStep();
    }
  }
  
  stepBackward(): void {
    if (this.currentStep > 0) {
      this.currentStep--;
      // For demo purposes, we'll just reset and step forward to this point
      const tempBeta0 = this.beta0;
      const tempBeta1 = this.beta1;
      this.beta0 = 0;
      this.beta1 = 0;
      
      const stepsToTake = this.currentStep;
      this.currentStep = 0;
      
      for (let i = 0; i < stepsToTake; i++) {
        const {beta0Grad, beta1Grad} = this.computeGradients();
        this.beta0 -= this.learningRate * beta0Grad;
        this.beta1 -= this.learningRate * beta1Grad;
        this.currentStep++;
      }
      
      this.updateD3Visualization();
      this.updateGradientPath();
    }
  }
  
  // Angular template-safe handlers
  setLearningRate(rate: number): void {
    this.learningRate = rate;
  }
  
  setIterations(iterations: number): void {
    this.iterations = iterations;
  }
  
  setAnimationSpeed(speed: number): void {
    this.animationSpeed = speed;
    if (this.isPlaying) {
      this.pauseSimulation();
      this.playSimulation();
    }
  }
  
  // Getters for template binding
  get simulationProgress(): number {
    return (this.currentStep / this.iterations) * 100;
  }
  
  get currentParameters(): {beta0: number, beta1: number} {
    return {
      beta0: parseFloat(this.beta0.toFixed(4)),
      beta1: parseFloat(this.beta1.toFixed(4))
    };
  }
  
  get currentMSE(): number {
    return parseFloat(this.computeMSE(this.beta0, this.beta1).toFixed(4));
  }
  
  get equationDisplay(): string {
    const b0 = this.currentParameters.beta0.toFixed(2);
    const b1 = this.currentParameters.beta1.toFixed(2);
    const sign = this.currentParameters.beta1 >= 0 ? '+' : '';
    return `y = ${b0} ${sign} ${b1}x`;
  }
  
  // Methods for switching views
  switchVisualization(type: '2d' | '3d'): void {
    this.currentVisualization = type;
  }
  
  switchTab(tab: string): void {
    this.activeTab = tab;
    
    // When switching to visualization tab, ensure the visualizations are properly sized
    if (tab === 'visualization') {
      setTimeout(() => this.onWindowResize(), 0);
    }
  }
  
  toggleFormulas(): void {
    this.showFormulas = !this.showFormulas;
  }
}