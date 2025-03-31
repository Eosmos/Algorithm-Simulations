import { Component, ElementRef, OnInit, ViewChild, AfterViewInit, OnDestroy, ChangeDetectionStrategy, ChangeDetectorRef, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import * as d3 from 'd3';

interface State {
  position: number;
  velocity: number;
}

interface Action {
  force: number;
}

interface Trajectory {
  states: State[];
  actions: Action[];
  rewards: number[];
  returns: number[];
}

interface PolicyParams {
  meanWeights: number[];
  stdDev: number;
}

@Component({
  selector: 'app-policy-gradient-simulation',
  templateUrl: './policy-gradient-simulation.component.html',
  styleUrls: ['./policy-gradient-simulation.component.scss'],
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class PolicyGradientSimulationComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('simulationContainer') simulationContainer!: ElementRef;
  @ViewChild('policyVisualization') policyVisualization!: ElementRef;
  @ViewChild('rewardChart') rewardChart!: ElementRef;
  @ViewChild('parameterLandscape') parameterLandscape!: ElementRef;

  // Three.js variables
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private car!: THREE.Object3D; 
  private mountain!: THREE.Group;
  private goal!: THREE.Group;

  // Simulation parameters
  public isPlaying = false;
  public isAutoPlaying = false;
  private animationFrameId: number | null = null;
  private simulationStep = 0;
  public episodeCount = 0;
  public maxEpisodes = 100;
  private maxStepsPerEpisode = 200;
  
  // Mountain Car environment parameters
  public minPosition = -1.2;
  public maxPosition = 0.6;
  public maxVelocity = 0.07;
  public goalPosition = 0.5;
  private gravity = 0.0025;
  private hillFrequency = 3.0;
  
  // Current state
  public currentState: State = { position: -0.5, velocity: 0 };
  
  // Policy parameters (theta)
  public policyParams: PolicyParams = {
    meanWeights: [0, 0], // [position weight, velocity weight]
    stdDev: 0.5
  };
  
  // Learning parameters
  public learningRate = 0.01;
  public gamma = 0.99; // discount factor
  
  // Component properties for improved UX
  public activeView: 'all' | 'environment' | 'policy' | 'rewards' | 'landscape' = 'all';
  public simulationInProgress = false;
  public simulationStatus = '';
  public showActivityIndicator = false;
  
  // UI controls
  public playSpeed = 1;
  public showExplanations = true;
  
  // Active step in visualization
  public activeStep: 'environment' | 'policy' | 'gradient' | 'update' = 'environment';
  
  // Collected trajectories
  private trajectories: Trajectory[] = [];
  private currentTrajectory: Trajectory = {
    states: [],
    actions: [],
    rewards: [],
    returns: []
  };
  
  // History for charts
  private episodeRewards: number[] = [];
  private policyParamHistory: { episode: number; meanW1: number; meanW2: number; stdDev: number }[] = [];
  
  // Store current gradient for policy update
  private currentGradient: number[] = [0, 0];
  
  constructor(private cdr: ChangeDetectorRef, private ngZone: NgZone) {}

  ngOnInit(): void {
    // Initialize arrays with zeros
    this.episodeRewards = Array(this.maxEpisodes).fill(0);
    
    // Initialize policy parameter history
    this.policyParamHistory.push({
      episode: 0,
      meanW1: this.policyParams.meanWeights[0],
      meanW2: this.policyParams.meanWeights[1],
      stdDev: this.policyParams.stdDev
    });
    
    // Make sure explanations are visible by default
    this.showExplanations = true;
  }

  ngAfterViewInit(): void {
    // Add a small delay to ensure DOM is fully rendered
    setTimeout(() => {
      this.initThreeJs();
      this.initD3Charts();
      this.resetSimulation();
    }, 100);
  }

  ngOnDestroy(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }
    
    // Clean up Three.js resources
    this.renderer.dispose();
  }

  // Helper method to format position for display, avoiding change detection issues
  public formatPosition(value: number): string {
    return value.toFixed(2);
  }
  
  // Helper method to format velocity for display, avoiding change detection issues
  public formatVelocity(value: number): string {
    return value.toFixed(2);
  }

  // Set the active view
  public setActiveView(view: 'all' | 'environment' | 'policy' | 'rewards' | 'landscape'): void {
    this.activeView = view;
    // Update visualizations when view changes
    setTimeout(() => {
      this.updateVisualizations();
      this.onWindowResize();
    }, 100);
    this.cdr.detectChanges();
  }
  
  // Toggle explanations
  public toggleExplanations(): void {
    this.showExplanations = !this.showExplanations;
    console.log('Explanations toggled: ', this.showExplanations);
    
    // Force update the DOM
    setTimeout(() => {
      this.cdr.detectChanges();
    }, 0);
  }
  
  // Increase simulation speed
  public increaseSpeed(): void {
    this.playSpeed = Math.min(4, this.playSpeed * 2);
    this.cdr.detectChanges();
  }
  
  // Decrease simulation speed
  public decreaseSpeed(): void {
    this.playSpeed = Math.max(0.25, this.playSpeed / 2);
    this.cdr.detectChanges();
  }

  private initThreeJs(): void {
    try {
      // Get container dimensions
      const container = this.simulationContainer.nativeElement;
      
      // Log container dimensions to debug
      console.log('Container dimensions:', container.clientWidth, container.clientHeight);
      
      if (!container || container.clientWidth <= 0 || container.clientHeight <= 0) {
        console.error('Invalid container dimensions for Three.js');
        // Try setting some default dimensions
        container.style.width = '100%';
        container.style.height = '300px';
      }

      const width = Math.max(container.clientWidth, 300);
      const height = Math.max(container.clientHeight, 200);

      // Create scene
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0x0c1428); // Dark blue background

      // Create camera
      this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
      this.camera.position.set(0, 2, 5);

      // Create renderer
      this.renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true,
        canvas: container.querySelector('canvas') || undefined
      });
      this.renderer.setSize(width, height);
      this.renderer.setPixelRatio(window.devicePixelRatio);
      this.renderer.shadowMap.enabled = true;
      
      // Only append if canvas doesn't already exist
      if (!container.querySelector('canvas')) {
        container.appendChild(this.renderer.domElement);
      }

      // Add lights
      const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
      this.scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
      directionalLight.position.set(5, 10, 7.5);
      directionalLight.castShadow = true;
      this.scene.add(directionalLight);

      // Create mountains
      this.createMountainTerrain();

      // Create car
      this.createCar();

      // Create goal flag
      this.createGoalFlag();

      // Add orbit controls
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.minDistance = 3;
      this.controls.maxDistance = 10;
      this.controls.maxPolarAngle = Math.PI / 2;

      // Handle window resize
      window.addEventListener('resize', () => this.onWindowResize());

      // Start animation loop
      this.animate();
      
      console.log('Three.js scene initialized successfully');
    } catch (error) {
      console.error('Error initializing Three.js:', error);
    }
  }

  private onWindowResize(): void {
    if (this.simulationContainer) {
      const container = this.simulationContainer.nativeElement;
      const width = container.clientWidth;
      const height = container.clientHeight;

      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    }
  }

  private createMountainTerrain(): void {
    this.mountain = new THREE.Group();
    
    // Create mountain shape with more resolution and width
    const lineGeometry = new THREE.BufferGeometry();
    const points = [];
    
    // Use more points for smoother curve
    for (let x = -2.5; x <= 2.5; x += 0.05) {
      const y = Math.sin(x * this.hillFrequency) * 0.45 - 0.45;
      points.push(new THREE.Vector3(x, y, 0));
    }
    
    lineGeometry.setFromPoints(points);
    
    const lineMaterial = new THREE.LineBasicMaterial({ 
      color: 0x4285f4, // Primary blue
      linewidth: 5
    });
    
    const mountainLine = new THREE.Line(lineGeometry, lineMaterial);
    this.mountain.add(mountainLine);
    
    // Create mountain surface with higher resolution
    const surfaceGeometry = new THREE.PlaneGeometry(5, 2, 100, 1);
    const positions = surfaceGeometry.attributes['position'].array;
    
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i];
      positions[i + 1] = Math.sin(x * this.hillFrequency) * 0.45 - 0.45;
    }
    
    surfaceGeometry.computeVertexNormals();
    
    const surfaceMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x2a4980, // Light blue
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.7,
      wireframe: false,
      metalness: 0.2,
      roughness: 0.8
    });
    
    const mountainSurface = new THREE.Mesh(surfaceGeometry, surfaceMaterial);
    mountainSurface.rotation.x = -Math.PI / 2;
    mountainSurface.position.z = 0;
    this.mountain.add(mountainSurface);
    
    // Add a grid to help with perspective and scale
    const gridHelper = new THREE.GridHelper(5, 10, 0x888888, 0x444444);
    gridHelper.position.y = -0.5;
    gridHelper.rotation.x = Math.PI / 2;
    this.mountain.add(gridHelper);
    
    this.scene.add(this.mountain);
  }

  private createCar(): void {
    // Make a more visible car with multiple components
    const carGroup = new THREE.Group();
    
    // Car body
    const bodyGeometry = new THREE.BoxGeometry(0.2, 0.1, 0.1);
    const bodyMaterial = new THREE.MeshStandardMaterial({ 
      color: 0xff9d45, // Orange
      emissive: 0x551700,
      emissiveIntensity: 0.3,
      metalness: 0.8,
      roughness: 0.2
    });
    
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.castShadow = true;
    body.receiveShadow = true;
    carGroup.add(body);
    
    // Wheels
    const wheelGeometry = new THREE.CylinderGeometry(0.05, 0.05, 0.02, 16);
    const wheelMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x333333, 
      roughness: 0.8 
    });
    
    // Front left wheel
    const wheelFL = new THREE.Mesh(wheelGeometry, wheelMaterial);
    wheelFL.rotation.z = Math.PI / 2;
    wheelFL.position.set(-0.08, -0.05, 0.06);
    carGroup.add(wheelFL);
    
    // Front right wheel
    const wheelFR = new THREE.Mesh(wheelGeometry, wheelMaterial);
    wheelFR.rotation.z = Math.PI / 2;
    wheelFR.position.set(-0.08, -0.05, -0.06);
    carGroup.add(wheelFR);
    
    // Rear left wheel
    const wheelRL = new THREE.Mesh(wheelGeometry, wheelMaterial);
    wheelRL.rotation.z = Math.PI / 2;
    wheelRL.position.set(0.08, -0.05, 0.06);
    carGroup.add(wheelRL);
    
    // Rear right wheel
    const wheelRR = new THREE.Mesh(wheelGeometry, wheelMaterial);
    wheelRR.rotation.z = Math.PI / 2;
    wheelRR.position.set(0.08, -0.05, -0.06);
    carGroup.add(wheelRR);
    
    // Add a light on top of the car to make it more visible
    const carLight = new THREE.PointLight(0xff9d45, 0.5, 1);
    carLight.position.set(0, 0.1, 0);
    carGroup.add(carLight);
    
    this.car = carGroup;
    
    // Set initial position
    this.updateCarPosition();
    
    this.scene.add(this.car);
  }

  private createGoalFlag(): void {
    this.goal = new THREE.Group();
    
    // Flag pole - make it taller and more visible
    const poleGeometry = new THREE.CylinderGeometry(0.01, 0.01, 0.6, 8);
    const poleMaterial = new THREE.MeshStandardMaterial({ 
      color: 0xe1e7f5, // Light gray
      metalness: 0.8,
      roughness: 0.2
    });
    
    const pole = new THREE.Mesh(poleGeometry, poleMaterial);
    pole.position.set(this.goalPosition, Math.sin(this.goalPosition * this.hillFrequency) * 0.45 - 0.15, 0);
    
    // Flag - make it larger and more visible
    const flagGeometry = new THREE.PlaneGeometry(0.25, 0.15);
    const flagMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x24b47e, // Green
      side: THREE.DoubleSide,
      emissive: 0x24b47e,
      emissiveIntensity: 0.3
    });
    
    const flag = new THREE.Mesh(flagGeometry, flagMaterial);
    flag.position.set(0.12, 0.2, 0);
    
    // Add some waving animation to the flag
    const waveGeometry = new THREE.BufferGeometry();
    const waveVertices = [];
    const divisions = 20;
    
    for (let i = 0; i <= divisions; i++) {
      for (let j = 0; j <= divisions; j++) {
        const x = (i / divisions) * 0.25;
        const y = (j / divisions) * 0.15;
        const z = 0.02 * Math.sin(i * Math.PI / 2);
        waveVertices.push(x, y, z);
      }
    }
    
    // Create faces
    const indices = [];
    
    for (let i = 0; i < divisions; i++) {
      for (let j = 0; j < divisions; j++) {
        const a = i * (divisions + 1) + j;
        const b = i * (divisions + 1) + j + 1;
        const c = (i + 1) * (divisions + 1) + j;
        const d = (i + 1) * (divisions + 1) + j + 1;
        
        indices.push(a, b, c);
        indices.push(c, b, d);
      }
    }
    
    waveGeometry.setIndex(indices);
    waveGeometry.setAttribute('position', new THREE.Float32BufferAttribute(waveVertices, 3));
    waveGeometry.computeVertexNormals();
    
    const waveMaterial = new THREE.MeshStandardMaterial({
      color: 0x24b47e,
      side: THREE.DoubleSide,
      wireframe: false
    });
    
    const waveFlag = new THREE.Mesh(waveGeometry, waveMaterial);
    waveFlag.position.set(0, 0.15, 0);
    
    // Add a light to highlight the goal flag
    const flagLight = new THREE.PointLight(0x24b47e, 0.8, 1);
    flagLight.position.set(0.1, 0.3, 0);
    
    this.goal.add(pole);
    //this.goal.add(flag); // Use either the flat flag or the wave flag
    this.goal.add(waveFlag);
    this.goal.add(flagLight);
    
    this.scene.add(this.goal);
  }

  private updateCarPosition(): void {
    // Map position from state space to world space
    const x = this.currentState.position;
    const y = Math.sin(x * this.hillFrequency) * 0.45 - 0.35; // Slightly above the mountain surface
    
    if (this.car) {
      this.car.position.set(x, y, 0);
    }
    
    // Mark for check to avoid ExpressionChangedAfterItHasBeenCheckedError
    this.cdr.markForCheck();
  }

  private animate(): void {
    // Run animation loop outside Angular zone for better performance
    this.ngZone.runOutsideAngular(() => {
      this.animationFrameId = requestAnimationFrame(() => this.animate());
      
      if (this.controls) {
        this.controls.update();
      }
      
      // Add some subtle movement to the flag to draw attention
      if (this.goal) {
        this.goal.children.forEach(child => {
          if (child.type === 'Mesh' && child !== this.goal.children[0]) {
            child.rotation.z = Math.sin(Date.now() * 0.003) * 0.1;
          }
        });
      }
      
      // Make sure renderer and scene exist
      if (this.renderer && this.scene && this.camera) {
        try {
          this.renderer.render(this.scene, this.camera);
        } catch (error) {
          console.error('Error rendering scene:', error);
        }
      }
    });
  }

  private initD3Charts(): void {
    this.initRewardChart();
    this.initPolicyVisualization();
    this.initParameterLandscape();
  }

  private initRewardChart(): void {
    const container = this.rewardChart.nativeElement;
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select(container)
      .append('svg')
      .attr('width', container.clientWidth)
      .attr('height', 200)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // X axis
    const x = d3.scaleLinear()
      .domain([0, this.maxEpisodes])
      .range([0, width]);
    
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .attr('class', 'x-axis')
      .call(d3.axisBottom(x));
    
    // Y axis
    const y = d3.scaleLinear()
      .domain([-200, 0]) // Negative rewards (typical for mountain car)
      .range([height, 0]);
    
    svg.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(y));
    
    // Add labels
    svg.append('text')
      .attr('class', 'x-label')
      .attr('text-anchor', 'middle')
      .attr('x', width / 2)
      .attr('y', height + 40)
      .text('Episode');
    
    svg.append('text')
      .attr('class', 'y-label')
      .attr('text-anchor', 'middle')
      .attr('transform', `translate(-40,${height / 2})rotate(-90)`)
      .text('Total Reward');
    
    // Add line
    const line = d3.line<number>()
      .x((d, i) => x(i))
      .y(d => y(d))
      .curve(d3.curveMonotoneX);
    
    svg.append('path')
      .datum(this.episodeRewards)
      .attr('class', 'reward-line')
      .attr('fill', 'none')
      .attr('stroke', '#7c4dff') // Purple from the design guide
      .attr('stroke-width', 2)
      .attr('d', line);
  }

  private initPolicyVisualization(): void {
    const container = this.policyVisualization.nativeElement;
    const width = container.clientWidth;
    const height = 200;
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    
    // Create SVG
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    // X axis (force values)
    const x = d3.scaleLinear()
      .domain([-1, 1]) // Force range
      .range([0, chartWidth]);
    
    svg.append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .attr('class', 'x-axis')
      .call(d3.axisBottom(x));
    
    // Y axis (probability density)
    const y = d3.scaleLinear()
      .domain([0, 1.5]) // Probability density range
      .range([chartHeight, 0]);
    
    svg.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(y));
    
    // Add labels
    svg.append('text')
      .attr('class', 'x-label')
      .attr('text-anchor', 'middle')
      .attr('x', chartWidth / 2)
      .attr('y', chartHeight + 40)
      .text('Action (Force)');
    
    svg.append('text')
      .attr('class', 'y-label')
      .attr('text-anchor', 'middle')
      .attr('transform', `translate(-40,${chartHeight / 2})rotate(-90)`)
      .text('Probability Density');
    
    // Add Gaussian curve
    const area = d3.area<{x: number, y: number}>()
      .x(d => x(d.x))
      .y0(chartHeight)
      .y1(d => y(d.y));
    
    // Create data points for the Gaussian
    const gaussianData = this.getGaussianData();
    
    svg.append('path')
      .datum(gaussianData)
      .attr('class', 'gaussian-area')
      .attr('fill', '#4285f480') // Semi-transparent blue
      .attr('d', area);
    
    // Add mean indicator
    const meanForce = this.calculateMeanForce();
    
    svg.append('line')
      .attr('class', 'mean-line')
      .attr('x1', x(meanForce))
      .attr('x2', x(meanForce))
      .attr('y1', y(0))
      .attr('y2', y(this.gaussianPdf(meanForce, meanForce, this.policyParams.stdDev)))
      .attr('stroke', '#ff9d45') // Orange
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5');
  }

  private initParameterLandscape(): void {
    const container = this.parameterLandscape.nativeElement;
    const width = container.clientWidth;
    const height = 250;
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    
    // Create SVG
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    // X axis (weight 1 - position weight)
    const x = d3.scaleLinear()
      .domain([-1, 1])
      .range([0, chartWidth]);
    
    svg.append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .attr('class', 'x-axis')
      .call(d3.axisBottom(x));
    
    // Y axis (weight 2 - velocity weight)
    const y = d3.scaleLinear()
      .domain([-1, 1])
      .range([chartHeight, 0]);
    
    svg.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(y));
    
    // Add labels
    svg.append('text')
      .attr('class', 'x-label')
      .attr('text-anchor', 'middle')
      .attr('x', chartWidth / 2)
      .attr('y', chartHeight + 40)
      .text('Position Weight');
    
    svg.append('text')
      .attr('class', 'y-label')
      .attr('text-anchor', 'middle')
      .attr('transform', `translate(-40,${chartHeight / 2})rotate(-90)`)
      .text('Velocity Weight');
    
    // Create contour data
    const contourData: [number, number, number][] = [];
    for (let i = -1; i <= 1; i += 0.05) {
      for (let j = -1; j <= 1; j += 0.05) {
        // Simplified objective function landscape (for visualization)
        const value = Math.sin(i * 3) * Math.cos(j * 3) - 0.3 * (i * i + j * j);
        contourData.push([i, j, value]);
      }
    }
    
    // Convert to format needed by d3.contour
    const n = Math.sqrt(contourData.length);
    const contourValues = new Array(n * n);
    let k = 0;
    
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        contourValues[k] = contourData[k][2];
        k++;
      }
    }
    
    // Create contours
    const contours = d3.contours()
      .size([n, n])
      .thresholds(10)
      (contourValues);
    
    // Create color scale
    const colorScale = d3.scaleSequential(d3.interpolateBlues)
      .domain([-1, 1]);
    
    // Add contour paths
    svg.append('g')
      .attr('class', 'contours')
      .selectAll('path')
      .data(contours)
      .enter().append('path')
      .attr('d', d3.geoPath(d3.geoIdentity().scale(chartWidth / n)))
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', '#4285f4')
      .attr('stroke-opacity', 0.3)
      .attr('stroke-width', 0.5);
    
    // Add current policy position point
    svg.append('circle')
      .attr('class', 'policy-point')
      .attr('cx', x(this.policyParams.meanWeights[0]))
      .attr('cy', y(this.policyParams.meanWeights[1]))
      .attr('r', 6)
      .attr('fill', '#ff9d45'); // Orange
      
    // Add arrowhead marker
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#ff6b6b'); // Red
      
    // Add gradient arrow (will be updated during simulation)
    svg.append('line')
      .attr('class', 'gradient-arrow')
      .attr('x1', x(this.policyParams.meanWeights[0]))
      .attr('y1', y(this.policyParams.meanWeights[1]))
      .attr('x2', x(this.policyParams.meanWeights[0]))
      .attr('y2', y(this.policyParams.meanWeights[1]))
      .attr('stroke', '#ff6b6b') // Red
      .attr('stroke-width', 2)
      .attr('marker-end', 'url(#arrow)')
      .attr('opacity', 0); // Hidden initially
  }

  // Gaussian probability density function (PDF)
  private gaussianPdf(x: number, mean: number, stdDev: number): number {
    const variance = stdDev * stdDev;
    return (1 / (Math.sqrt(2 * Math.PI * variance))) * 
           Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
  }

  // Get data points for Gaussian curve visualization
  private getGaussianData(): {x: number, y: number}[] {
    const meanForce = this.calculateMeanForce();
    const stdDev = this.policyParams.stdDev;
    const data: {x: number, y: number}[] = [];
    
    for (let x = -1; x <= 1; x += 0.01) {
      const y = this.gaussianPdf(x, meanForce, stdDev);
      data.push({ x, y });
    }
    
    return data;
  }

  // Calculate mean force based on current state and policy parameters
  private calculateMeanForce(): number {
    const positionWeight = this.policyParams.meanWeights[0];
    const velocityWeight = this.policyParams.meanWeights[1];
    
    // Linear combination of state features
    let meanForce = positionWeight * this.currentState.position + 
                   velocityWeight * this.currentState.velocity;
    
    // Clip to action range
    return Math.max(-1, Math.min(1, meanForce));
  }

  // Sample action from policy
  private sampleAction(): Action {
    const meanForce = this.calculateMeanForce();
    const stdDev = this.policyParams.stdDev;
    
    // Box-Muller transform for Gaussian sampling
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    
    // Apply mean and standard deviation
    let force = meanForce + stdDev * z0;
    
    // Clip to action range
    force = Math.max(-1, Math.min(1, force));
    
    return { force };
  }

  // Step the environment forward
  private stepEnvironment(action: Action): { nextState: State; reward: number; done: boolean } {
    const { position, velocity } = this.currentState;
    
    // Calculate next velocity
    let nextVelocity = velocity + 
                       0.001 * action.force - 
                       this.gravity * Math.cos(3 * position);
    
    // Clip velocity
    nextVelocity = Math.max(-this.maxVelocity, Math.min(this.maxVelocity, nextVelocity));
    
    // Calculate next position
    let nextPosition = position + nextVelocity;
    
    // Check boundary conditions
    if (nextPosition < this.minPosition) {
      nextPosition = this.minPosition;
      nextVelocity = 0;
    }
    
    // Check if goal reached
    const done = nextPosition >= this.goalPosition;
    
    // Reward is -1 for each step until goal
    const reward = done ? 0 : -1;
    
    return {
      nextState: { position: nextPosition, velocity: nextVelocity },
      reward,
      done
    };
  }

  // Reset the environment
  private resetEnvironment(): void {
    // Use NgZone to ensure this runs in the Angular zone
    this.ngZone.run(() => {
      // Random initial position between -0.6 and -0.4
      const initialPosition = -0.5 + (Math.random() * 0.2 - 0.1);
      this.currentState = { position: initialPosition, velocity: 0 };
      this.updateCarPosition();
      
      // Manually trigger change detection
      this.cdr.detectChanges();
    });
  }

  // Reset the entire simulation
  public resetSimulation(): void {
    this.isPlaying = false;
    this.isAutoPlaying = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    this.episodeCount = 0;
    this.simulationStep = 0;
    
    // Reset policy parameters
    this.policyParams = {
      meanWeights: [0, 0],
      stdDev: 0.5
    };
    
    // Reset episode history
    this.episodeRewards = Array(this.maxEpisodes).fill(0);
    this.policyParamHistory = [{
      episode: 0,
      meanW1: this.policyParams.meanWeights[0],
      meanW2: this.policyParams.meanWeights[1],
      stdDev: this.policyParams.stdDev
    }];
    
    // Reset trajectories
    this.trajectories = [];
    this.currentTrajectory = {
      states: [],
      actions: [],
      rewards: [],
      returns: []
    };
    
    // Reset environment
    this.resetEnvironment();
    
    // Reset active step
    this.activeStep = 'environment';
    
    // Update visualizations
    this.updateVisualizations();
  }

  // Play/pause the simulation
  public togglePlay(): void {
    this.isPlaying = !this.isPlaying;
    this.isAutoPlaying = false;
    
    if (this.isPlaying) {
      this.simulateStep();
    }
  }

  // Auto-play the simulation
  public toggleAutoPlay(): void {
    this.isAutoPlaying = !this.isAutoPlaying;
    this.isPlaying = this.isAutoPlaying;
    this.simulationInProgress = this.isAutoPlaying;
    
    // Reset simulation if we've reached the end
    if (this.episodeCount >= this.maxEpisodes && this.isAutoPlaying) {
      this.resetSimulation();
    }
    
    if (this.isAutoPlaying) {
      // Show a progress indicator while auto-playing
      this.showActivityIndicator = true;
      this.simulationStatus = 'Simulation running...';
      
      // Focus on the environment view when auto-playing
      if (this.activeView === 'all') {
        this.setActiveView('environment');
      }
      
      // Use faster speed by default for auto-play
      if (this.playSpeed < 2) {
        this.playSpeed = 2;
      }
      
      this.simulateStep();
    } else {
      this.showActivityIndicator = false;
      this.simulationStatus = '';
    }
  }

  // Simulate one step
  private simulateStep(): void {
    if (!this.isPlaying) return;
    
    // Execute steps based on current active phase
    switch (this.activeStep) {
      case 'environment':
        this.environmentStep();
        break;
      case 'policy':
        this.policyEvaluationStep();
        break;
      case 'gradient':
        this.gradientCalculationStep();
        break;
      case 'update':
        this.policyUpdateStep();
        break;
    }
    
    // Schedule next step
    setTimeout(() => {
      if (this.isPlaying) {
        this.simulateStep();
      }
    }, 1000 / (this.playSpeed));
  }

  // Environment interaction step
  private environmentStep(): void {
    // Sample action from policy
    const action = this.sampleAction();
    
    // Apply action to environment
    const { nextState, reward, done } = this.stepEnvironment(action);
    
    // Record in trajectory
    this.currentTrajectory.states.push({ ...this.currentState });
    this.currentTrajectory.actions.push(action);
    this.currentTrajectory.rewards.push(reward);
    
    // Update state
    this.currentState = nextState;
    this.updateCarPosition();
    
    // Update visualizations with animation
    this.updatePolicyVisualization();
    
    // Update status
    this.simulationStatus = `Moving car with force: ${action.force.toFixed(2)}, New position: ${this.formatPosition(this.currentState.position)}`;
    
    // Check if episode is done
    if (done || this.currentTrajectory.states.length >= this.maxStepsPerEpisode) {
      // Show a notification when goal is reached
      if (done) {
        this.simulationStatus = 'Goal reached! Calculating policy update...';
      } else {
        this.simulationStatus = 'Max steps reached. Calculating policy update...';
      }
      
      // Move to policy evaluation step
      this.activeStep = 'policy';
      
      // Highlight the policy view
      if (this.isAutoPlaying && this.activeView !== 'all') {
        this.setActiveView('policy');
      }
      
      // If auto-playing, continue to next step after a short delay
      if (this.isAutoPlaying) {
        setTimeout(() => {
          this.simulateStep();
        }, 500); // Slight delay to show the transition
      }
    } else if (this.isAutoPlaying) {
      // Continue environment steps more quickly
      setTimeout(() => {
        this.simulateStep();
      }, 50 / this.playSpeed);
    }
  }

  // Calculate returns for the current trajectory
  private policyEvaluationStep(): void {
    // Update status
    this.simulationStatus = 'Evaluating policy returns...';
    
    const rewards = this.currentTrajectory.rewards;
    const returns: number[] = new Array(rewards.length).fill(0);
    
    // Calculate returns (discounted sum of future rewards)
    let runningReturn = 0;
    for (let t = rewards.length - 1; t >= 0; t--) {
      runningReturn = rewards[t] + this.gamma * runningReturn;
      returns[t] = runningReturn;
    }
    
    this.currentTrajectory.returns = returns;
    
    // Record episode total reward
    const totalReward = rewards.reduce((sum, r) => sum + r, 0);
    this.episodeRewards[this.episodeCount] = totalReward;
    
    // Update reward chart
    this.updateRewardChart();
    
    // Move to gradient calculation step
    this.activeStep = 'gradient';
    
    // Highlight the rewards view
    if (this.isAutoPlaying && this.activeView !== 'all') {
      this.setActiveView('rewards');
    }
    
    // If auto-playing, continue to next step after a short delay
    if (this.isAutoPlaying) {
      setTimeout(() => {
        this.simulateStep();
      }, 800 / this.playSpeed);
    }
  }

  // Calculate policy gradient
  private gradientCalculationStep(): void {
    // Update status
    this.simulationStatus = 'Calculating policy gradient...';
    
    // Initialize gradient values
    const gradient = [0, 0];
    
    // Calculate log probability gradients for each action in the trajectory
    for (let t = 0; t < this.currentTrajectory.states.length; t++) {
      const state = this.currentTrajectory.states[t];
      const action = this.currentTrajectory.actions[t];
      const returnValue = this.currentTrajectory.returns[t];
      
      // Calculate action log probability gradient
      // For Gaussian policy: ∇_θ log π_θ(a|s) = (a - μ(s)) * ∇_θ μ(s) / σ^2
      // where μ(s) = θ^T φ(s) and φ(s) = [position, velocity]
      
      // Calculate mean force for this state
      const meanForce = this.policyParams.meanWeights[0] * state.position + 
                       this.policyParams.meanWeights[1] * state.velocity;
      
      // Calculate gradient of log probability
      const gradientScale = (action.force - meanForce) / 
                         (this.policyParams.stdDev * this.policyParams.stdDev);
      
      // Accumulate gradients (scaled by return)
      gradient[0] += gradientScale * state.position * returnValue;
      gradient[1] += gradientScale * state.velocity * returnValue;
    }
    
    // Normalize by trajectory length
    gradient[0] /= this.currentTrajectory.states.length;
    gradient[1] /= this.currentTrajectory.states.length;
    
    // Visualize the gradient
    this.updateGradientVisualization(gradient);
    
    // Store gradient for policy update
    this.currentGradient = gradient;
    
    // Show gradient magnitude in status
    const gradientMagnitude = Math.sqrt(gradient[0] * gradient[0] + gradient[1] * gradient[1]);
    this.simulationStatus = `Gradient calculated: magnitude = ${gradientMagnitude.toFixed(3)}`;
    
    // Move to policy update step
    this.activeStep = 'update';
    
    // Highlight the parameter landscape view
    if (this.isAutoPlaying && this.activeView !== 'all') {
      this.setActiveView('landscape');
    }
    
    // If auto-playing, continue to next step after a short delay
    if (this.isAutoPlaying) {
      setTimeout(() => {
        this.simulateStep();
      }, 1000 / this.playSpeed);
    }
  }

  // Update policy parameters based on gradient
  private policyUpdateStep(): void {
    // Update status
    this.simulationStatus = 'Updating policy parameters...';
    
    // Store old parameters for animation
    const oldParams = {
      weightPos: this.policyParams.meanWeights[0], 
      weightVel: this.policyParams.meanWeights[1],
      stdDev: this.policyParams.stdDev
    };
    
    // Apply gradient update
    this.policyParams.meanWeights[0] += this.learningRate * this.currentGradient[0];
    this.policyParams.meanWeights[1] += this.learningRate * this.currentGradient[1];
    
    // Optional: Decrease standard deviation over time for exploration control
    if (this.episodeCount > 30) {
      this.policyParams.stdDev = Math.max(0.1, this.policyParams.stdDev * 0.995);
    }
    
    // Store parameter history
    this.policyParamHistory.push({
      episode: this.episodeCount + 1,
      meanW1: this.policyParams.meanWeights[0],
      meanW2: this.policyParams.meanWeights[1],
      stdDev: this.policyParams.stdDev
    });
    
    // Show parameter changes
    this.simulationStatus = `Updated policy: θ₁: ${oldParams.weightPos.toFixed(3)} → ${this.policyParams.meanWeights[0].toFixed(3)}, ` +
                           `θ₂: ${oldParams.weightVel.toFixed(3)} → ${this.policyParams.meanWeights[1].toFixed(3)}`;
    
    // Update visualizations with animation
    this.updatePolicyVisualization();
    this.updateParameterLandscape();
    
    // Increment episode counter
    this.episodeCount++;
    
    // Reset trajectory
    this.trajectories.push(this.currentTrajectory);
    this.currentTrajectory = {
      states: [],
      actions: [],
      rewards: [],
      returns: []
    };
    
    // Reset environment
    this.resetEnvironment();
    
    // Reset active step
    this.activeStep = 'environment';
    
    // Go back to environment view for next episode
    if (this.isAutoPlaying && this.activeView !== 'all') {
      setTimeout(() => {
        this.setActiveView('environment');
      }, 1000 / this.playSpeed);
    }
    
    // Check if we've reached max episodes
    if (this.episodeCount >= this.maxEpisodes) {
      this.isPlaying = false;
      this.isAutoPlaying = false;
      this.simulationInProgress = false;
      this.simulationStatus = 'Simulation complete! The policy has been optimized.';
      return;
    }
    
    // If auto-playing, continue to next step after a delay to show the transition
    if (this.isAutoPlaying) {
      setTimeout(() => {
        this.simulateStep();
      }, 1500 / this.playSpeed);
    }
  }

  // Update visualizations
  private updateVisualizations(): void {
    this.updateCarPosition();
    this.updatePolicyVisualization();
    this.updateRewardChart();
    this.updateParameterLandscape();
  }

  // Update the policy visualization
  private updatePolicyVisualization(): void {
    const container = this.policyVisualization.nativeElement;
    const svg = d3.select(container).select('svg g');
    
    // Update Gaussian data
    const gaussianData = this.getGaussianData();
    
    // Update area
    const width = container.clientWidth;
    const height = 200;
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    // X and Y scales
    const x = d3.scaleLinear()
      .domain([-1, 1])
      .range([0, chartWidth]);
    
    const y = d3.scaleLinear()
      .domain([0, 1.5])
      .range([chartHeight, 0]);
    
    // Update area chart
    const area = d3.area<{x: number, y: number}>()
      .x(d => x(d.x))
      .y0(chartHeight)
      .y1(d => y(d.y));
    
    svg.select('.gaussian-area')
      .datum(gaussianData)
      .attr('d', area);
    
    // Update mean line
    const meanForce = this.calculateMeanForce();
    
    svg.select('.mean-line')
      .attr('x1', x(meanForce))
      .attr('x2', x(meanForce))
      .attr('y1', y(0))
      .attr('y2', y(this.gaussianPdf(meanForce, meanForce, this.policyParams.stdDev)));
  }

  // Update the reward chart
  private updateRewardChart(): void {
    const container = this.rewardChart.nativeElement;
    const svg = d3.select(container).select('svg g');
    
    // Chart dimensions
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;
    
    // X and Y scales
    const x = d3.scaleLinear()
      .domain([0, this.maxEpisodes])
      .range([0, width]);
    
    const y = d3.scaleLinear()
      .domain([-200, 0])
      .range([height, 0]);
    
    // Update line
    const line = d3.line<number>()
      .x((d, i) => x(i))
      .y(d => y(d))
      .curve(d3.curveMonotoneX);
    
    svg.select('.reward-line')
      .datum(this.episodeRewards.slice(0, this.episodeCount + 1))
      .attr('d', line);
  }

  // Update the parameter landscape
  private updateParameterLandscape(): void {
    const container = this.parameterLandscape.nativeElement;
    const svg = d3.select(container).select('svg g');
    
    // Chart dimensions
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 250 - margin.top - margin.bottom;
    
    // X and Y scales
    const x = d3.scaleLinear()
      .domain([-1, 1])
      .range([0, width]);
    
    const y = d3.scaleLinear()
      .domain([-1, 1])
      .range([height, 0]);
    
    // Update policy point
    svg.select('.policy-point')
      .attr('cx', x(this.policyParams.meanWeights[0]))
      .attr('cy', y(this.policyParams.meanWeights[1]));
    
    // Hide gradient arrow
    svg.select('.gradient-arrow')
      .attr('opacity', 0);
  }

  // Visualize the policy gradient
  private updateGradientVisualization(gradient: number[]): void {
    const container = this.parameterLandscape.nativeElement;
    const svg = d3.select(container).select('svg g');
    
    // Chart dimensions
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 250 - margin.top - margin.bottom;
    
    // X and Y scales
    const x = d3.scaleLinear()
      .domain([-1, 1])
      .range([0, width]);
    
    const y = d3.scaleLinear()
      .domain([-1, 1])
      .range([height, 0]);
    
    // Calculate arrow parameters
    const startX = this.policyParams.meanWeights[0];
    const startY = this.policyParams.meanWeights[1];
    
    // Scale gradient for visualization
    const gradientMagnitude = Math.sqrt(gradient[0] * gradient[0] + gradient[1] * gradient[1]);
    let scaleFactor = 0.1;
    
    if (gradientMagnitude > 0) {
      const normalizedGradient = [
        gradient[0] / gradientMagnitude,
        gradient[1] / gradientMagnitude
      ];
      
      const endX = startX + normalizedGradient[0] * scaleFactor;
      const endY = startY + normalizedGradient[1] * scaleFactor;
      
      // Update gradient arrow
      svg.select('.gradient-arrow')
        .attr('x1', x(startX))
        .attr('y1', y(startY))
        .attr('x2', x(endX))
        .attr('y2', y(endY))
        .attr('opacity', 1);
    }
  }
}