import { Component, OnInit, AfterViewInit, ElementRef, ViewChild, NgZone, OnDestroy, ChangeDetectorRef } from '@angular/core';
import * as THREE from 'three';
import * as d3 from 'd3';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { NgIf, NgFor, NgClass } from '@angular/common';

@Component({
  selector: 'app-policy-gradient',
  templateUrl: './policy-gradient.component.html',
  styleUrls: ['./policy-gradient.component.scss'],
  standalone: true,
  imports: [NgIf, NgFor, NgClass]
})
export class PolicyGradientComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('pendulumCanvas') pendulumCanvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('parameterLandscape') parameterLandscapeRef!: ElementRef<HTMLDivElement>;
  @ViewChild('trajectoryVisualization') trajectoryVisualizationRef!: ElementRef<HTMLDivElement>;
  @ViewChild('policyOutput') policyOutputRef!: ElementRef<HTMLDivElement>;
  
  // Component state
  activeTab: string = 'overview';
  isSimulationRunning: boolean = false;
  isAutoPlay: boolean = false;
  learningRate: number = 0.01;
  discountFactor: number = 0.99;
  episodeCount: number = 0;
  totalReward: number = 0;
  currentStep: number = 0;
  
  // Initialization flags
  private pendulumInitialized: boolean = false;
  private visualizationsInitialized: boolean = false;
  
  // Papers and references
  papers = [
    {
      title: "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning",
      author: "Williams, R. J.",
      year: 1992,
      journal: "Machine Learning",
      volume: "8(3–4)",
      pages: "229–256",
      url: "https://link.springer.com/article/10.1007/BF00992696"
    },
    {
      title: "Policy Gradient Methods for Reinforcement Learning with Function Approximation",
      author: "Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y.",
      year: 1999,
      journal: "Advances in Neural Information Processing Systems (NIPS)",
      volume: "12",
      url: "https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html"
    }
  ];
  
  // Simulation variables
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private pendulum!: THREE.Group;
  private policy: PolicyNetwork = new PolicyNetwork(2, 1);
  private animationId: number | null = null;
  private trajectories: Trajectory[] = [];
  private currentTrajectory: Trajectory | null = null;
  private paramSvg: any;
  private trajectorySvg: any;
  private policySvg: any;
  private parameterHistory: {x: number, y: number, reward: number}[] = [];
  
  constructor(private ngZone: NgZone, private changeDetector: ChangeDetectorRef) {}

  ngOnInit(): void {
    // Initialize data structures
    this.resetSimulation();
  }

  ngAfterViewInit(): void {
    // Let Angular complete its initialization before we check for DOM elements
    this.changeDetector.detectChanges();
    
    // If simulation tab is active by default, try to initialize visualizations
    if (this.activeTab === 'simulation') {
      setTimeout(() => {
        this.initializeVisualizationsIfNeeded();
      }, 500);
    }
  }
  
  ngOnDestroy(): void {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
    }
    
    // Clean up THREE.js resources
    if (this.renderer) {
      this.renderer.dispose();
      this.renderer.forceContextLoss();
    }
  }
  
  // Tab navigation
  selectTab(tab: string): void {
    this.activeTab = tab;
    
    // On first selection of simulation tab, initialize visualizations
    if (tab === 'simulation') {
      setTimeout(() => {
        this.initializeVisualizationsIfNeeded();
      }, 100);
    }
  }
  
  // Initialize all visualizations if they haven't been initialized yet
  private initializeVisualizationsIfNeeded(): void {
    try {
      if (!this.pendulumInitialized && this.pendulumCanvasRef?.nativeElement) {
        this.initPendulumScene();
        this.pendulumInitialized = true;
      }
      
      if (!this.visualizationsInitialized) {
        if (this.parameterLandscapeRef?.nativeElement) {
          this.initParameterLandscape();
        }
        
        if (this.trajectoryVisualizationRef?.nativeElement) {
          this.initTrajectoryVisualization();
        }
        
        if (this.policyOutputRef?.nativeElement) {
          this.initPolicyOutput();
        }
        
        if (this.pendulumInitialized && 
            this.parameterLandscapeRef?.nativeElement && 
            this.trajectoryVisualizationRef?.nativeElement && 
            this.policyOutputRef?.nativeElement) {
          this.visualizationsInitialized = true;
          this.renderPendulum();
          this.resetSimulation();
        }
      }
    } catch (error) {
      console.error('Error initializing visualizations:', error);
    }
  }
  
  // Simulation controls
  toggleSimulation(): void {
    this.isSimulationRunning = !this.isSimulationRunning;
    
    if (this.isSimulationRunning) {
      this.runSimulation();
    }
  }
  
  toggleAutoPlay(): void {
    this.isAutoPlay = !this.isAutoPlay;
  }
  
  resetSimulation(): void {
    this.isSimulationRunning = false;
    this.episodeCount = 0;
    this.totalReward = 0;
    this.currentStep = 0;
    this.trajectories = [];
    this.currentTrajectory = null;
    this.parameterHistory = [];
    
    // Reset policy to random initialization
    this.policy = new PolicyNetwork(2, 1);
    
    // Reset visualizations - only if they've been initialized
    if (this.pendulum) this.resetPendulum();
    if (this.paramSvg) this.updateParameterLandscape();
    if (this.trajectorySvg) this.updateTrajectoryVisualization();
    if (this.policySvg) this.updatePolicyOutput();
  }
  
  // Learning rate control
  updateLearningRate(event: Event): void {
    this.learningRate = parseFloat((event.target as HTMLInputElement).value);
  }
  
  // Discount factor control
  updateDiscountFactor(event: Event): void {
    this.discountFactor = parseFloat((event.target as HTMLInputElement).value);
  }
  
  // Main simulation loop
  private runSimulation(): void {
    if (!this.isSimulationRunning) return;
    
    this.ngZone.runOutsideAngular(() => {
      const simulateStep = () => {
        if (!this.isSimulationRunning) return;
        
        // If no current trajectory or trajectory is complete, start a new episode
        if (!this.currentTrajectory || this.currentTrajectory.isComplete) {
          this.startNewEpisode();
        }
        
        // Perform a step in the current episode
        this.performStep();
        
        // Update visualizations
        this.updatePendulum();
        this.updateParameterLandscape();
        this.updateTrajectoryVisualization();
        this.updatePolicyOutput();
        
        // If current trajectory is complete and autoplay is enabled, continue simulation
        if (this.currentTrajectory && this.currentTrajectory.isComplete && this.isAutoPlay) {
          // Train policy on completed trajectory
          this.trainPolicy();
          // Continue simulation
          setTimeout(() => {
            if (this.isSimulationRunning) {
              this.animationId = requestAnimationFrame(simulateStep);
            }
          }, 500); // Brief pause between episodes
        } else if (this.currentTrajectory && !this.currentTrajectory.isComplete) {
          // Continue current episode
          this.animationId = requestAnimationFrame(simulateStep);
        } else {
          // Manual mode, wait for user to continue
          this.isSimulationRunning = false;
          this.ngZone.run(() => {}); // Trigger change detection
        }
      };
      
      this.animationId = requestAnimationFrame(simulateStep);
    });
  }
  
  // Start a new episode
  private startNewEpisode(): void {
    // Train policy on previous trajectory if exists
    if (this.currentTrajectory && this.currentTrajectory.isComplete) {
      this.trainPolicy();
    }
    
    // Create new trajectory
    this.currentTrajectory = new Trajectory();
    this.trajectories.push(this.currentTrajectory);
    this.episodeCount++;
    
    // Reset pendulum to initial state
    this.resetPendulum();
  }
  
  // Perform a single step in the current episode
  private performStep(): void {
    if (!this.currentTrajectory) return;
    
    // Get current state
    const angle = this.pendulum.rotation.z;
    const velocity = this.pendulum.userData['angularVelocity'];
    const state = [Math.cos(angle), Math.sin(angle), velocity];
    
    // Get action from policy
    const actionDistribution = this.policy.forward(state);
    const action = actionDistribution.sample();
    
    // Apply action to pendulum
    this.pendulum.userData['torque'] = action * 2.0; // Scale action to reasonable torque
    
    // Update pendulum physics
    this.updatePendulumPhysics();
    
    // Calculate reward (higher when pendulum is upright)
    const uprightPosition = Math.cos(angle);
    const reward = uprightPosition - 0.1 * Math.abs(velocity) - 0.001 * Math.abs(action);
    
    // Add step to trajectory
    this.currentTrajectory.addStep(state, action, reward);
    this.totalReward += reward;
    this.currentStep++;
    
    // Check if episode is complete (fixed length episodes for simplicity)
    if (this.currentTrajectory.steps.length >= 200) {
      this.currentTrajectory.complete();
    }
  }
  
  // Train policy on completed trajectory
  private trainPolicy(): void {
    if (!this.currentTrajectory || !this.currentTrajectory.isComplete) return;
    
    // Calculate discounted returns for each step
    const returns = this.calculateReturns(this.currentTrajectory.steps.map(step => step.reward));
    
    // Update policy using policy gradient
    for (let t = 0; t < this.currentTrajectory.steps.length; t++) {
      const step = this.currentTrajectory.steps[t];
      // Update policy parameters based on the policy gradient
      this.policy.update(step.state, step.action, returns[t], this.learningRate);
    }
    
    // Store parameter history for visualization
    const params = this.policy.getParameters();
    this.parameterHistory.push({
      x: params[0],
      y: params[1],
      reward: this.currentTrajectory.getTotalReward()
    });
  }
  
  // Calculate discounted returns
  private calculateReturns(rewards: number[]): number[] {
    const returns: number[] = new Array(rewards.length).fill(0);
    
    let G = 0;
    for (let t = rewards.length - 1; t >= 0; t--) {
      G = rewards[t] + this.discountFactor * G;
      returns[t] = G;
    }
    
    return returns;
  }
  
  // Initialize and update visualizations
  private initPendulumScene(): void {
    // Ensure we have the canvas reference
    if (!this.pendulumCanvasRef || !this.pendulumCanvasRef.nativeElement) {
      console.error('Pendulum canvas reference is not available');
      return;
    }
    
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#0c1428'); // Darkest blue from design system
    
    // Create camera
    this.camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    this.camera.position.z = 5;
    
    // Create renderer
    this.renderer = new THREE.WebGLRenderer({ canvas: this.pendulumCanvasRef.nativeElement, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(400, 400);
    
    // Add controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enablePan = false;
    this.controls.enableZoom = false;
    this.controls.enableRotate = false;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    this.scene.add(directionalLight);
    
    // Create pendulum
    this.createPendulum();
    
    // Add world reference frame
    const axesHelper = new THREE.AxesHelper(3);
    this.scene.add(axesHelper);
  }
  
  private createPendulum(): void {
    // Create pendulum group
    this.pendulum = new THREE.Group();
    
    // Add pivot point (small sphere)
    const pivotGeometry = new THREE.SphereGeometry(0.1, 16, 16);
    const pivotMaterial = new THREE.MeshStandardMaterial({ color: '#4285f4' }); // Primary blue
    const pivot = new THREE.Mesh(pivotGeometry, pivotMaterial);
    this.pendulum.add(pivot);
    
    // Add rod
    const rodGeometry = new THREE.CylinderGeometry(0.05, 0.05, 2, 16);
    const rodMaterial = new THREE.MeshStandardMaterial({ color: '#8bb4fa' }); // Light blue
    const rod = new THREE.Mesh(rodGeometry, rodMaterial);
    rod.position.y = -1; // Center of rod is at -1
    this.pendulum.add(rod);
    
    // Add bob (weight at end)
    const bobGeometry = new THREE.SphereGeometry(0.2, 32, 32);
    const bobMaterial = new THREE.MeshStandardMaterial({ color: '#ff9d45' }); // Orange (RL color)
    const bob = new THREE.Mesh(bobGeometry, bobMaterial);
    bob.position.y = -2; // Bob is at end of rod
    this.pendulum.add(bob);
    
    // Add to scene
    this.scene.add(this.pendulum);
    
    // Initialize pendulum state
    this.pendulum.userData = {
      angularVelocity: 0,
      torque: 0
    };
    
    // Reset pendulum to initial position
    this.resetPendulum();
  }
  
  private resetPendulum(): void {
    // Only proceed if pendulum exists
    if (!this.pendulum) return;
    
    // Set pendulum to hanging down position with slight offset
    this.pendulum.rotation.z = Math.PI + (Math.random() - 0.5) * 0.5;
    
    // Reset physics values
    this.pendulum.userData['angularVelocity'] = 0;
    this.pendulum.userData['torque'] = 0;
  }
  
  private updatePendulumPhysics(): void {
    // Simple pendulum physics
    const gravity = 9.8;
    const mass = 1.0;
    const length = 2.0;
    const friction = 0.1;
    
    // Calculate gravitational torque
    const angle = this.pendulum.rotation.z;
    const gravTorque = mass * gravity * length * Math.sin(angle);
    
    // Calculate total torque (gravity + applied)
    const totalTorque = -gravTorque + this.pendulum.userData['torque'];
    
    // Update angular velocity (F = ma)
    this.pendulum.userData['angularVelocity'] += totalTorque * 0.01; // Small time step
    
    // Apply friction
    this.pendulum.userData['angularVelocity'] *= (1 - friction * 0.01);
    
    // Update angle
    this.pendulum.rotation.z += this.pendulum.userData['angularVelocity'] * 0.01;
  }
  
  private renderPendulum(): void {
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }
  
  private updatePendulum(): void {
    // Only proceed if the pendulum exists
    if (!this.pendulum) return;
    
    // Render pendulum with current state
    this.renderPendulum();
  }
  
  private initParameterLandscape(): void {
    // Ensure we have the container reference
    if (!this.parameterLandscapeRef || !this.parameterLandscapeRef.nativeElement) {
      console.error('Parameter landscape reference is not available');
      return;
    }
    
    const container = this.parameterLandscapeRef.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Create SVG
    this.paramSvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
    
    // Add title
    this.paramSvg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('class', 'visualization-title')
      .text('Policy Parameter Landscape');
    
    // Add axes
    this.paramSvg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${height - 30})`);
    
    this.paramSvg.append('g')
      .attr('class', 'y-axis')
      .attr('transform', 'translate(40, 0)');
    
    // Add initial point
    this.paramSvg.append('g')
      .attr('class', 'points');
    
    // Update visualization
    this.updateParameterLandscape();
  }
  
  private updateParameterLandscape(): void {
    if (!this.paramSvg) return;
    
    const container = this.parameterLandscapeRef.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight || 280;
    
    // Update SVG dimensions
    this.paramSvg
      .attr('width', width)
      .attr('height', height);
    
    // If no history, just return
    if (this.parameterHistory.length === 0) return;
    
    // Create scales
    const xExtent = d3.extent(this.parameterHistory, d => d.x) as [number, number];
    const safeXMin = xExtent[0] !== undefined ? xExtent[0] : 0;
    const safeXMax = xExtent[1] !== undefined ? xExtent[1] : 1;
    const margin = 0.5;
    
    const xScale = d3.scaleLinear()
      .domain([safeXMin - margin, safeXMax + margin])
      .range([40, width - 20]);
    
    const yExtent = d3.extent(this.parameterHistory, d => d.y) as [number, number];
    const safeYMin = yExtent[0] !== undefined ? yExtent[0] : 0;
    const safeYMax = yExtent[1] !== undefined ? yExtent[1] : 1;
    
    const yScale = d3.scaleLinear()
      .domain([safeYMin - margin, safeYMax + margin])
      .range([height - 30, 30]);
    
    // Update axes
    this.paramSvg.select('.x-axis')
      .attr('transform', `translate(0, ${height - 30})`)
      .call(d3.axisBottom(xScale));
    
    this.paramSvg.select('.y-axis')
      .call(d3.axisLeft(yScale));
    
    // Create color scale for rewards
    const rewards = this.parameterHistory.map(d => d.reward);
    const minReward = Math.min(...rewards, 0);
    const maxReward = Math.max(...rewards, 1);
    
    const colorScale = d3.scaleSequential(d3.interpolateOranges)
      .domain([minReward, maxReward]);
    
    // Clear previous points to prevent accumulation
    this.paramSvg.select('.points').selectAll('*').remove();
    
    // Update points
    const points = this.paramSvg.select('.points')
      .selectAll('circle')
      .data(this.parameterHistory);
    
    points.enter()
      .append('circle')
      .attr('r', 5)
      .attr('cx', (d: {x: number}) => xScale(d.x))
      .attr('cy', (d: {y: number}) => yScale(d.y))
      .attr('fill', (d: {reward: number}) => colorScale(d.reward));
    
    // Remove existing path to prevent accumulation
    this.paramSvg.select('.param-path').remove();
    
    // Add line connecting points
    const line = d3.line<{x: number, y: number, reward: number}>()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y));
    
    // Add path if we have enough points
    if (this.parameterHistory.length > 1) {
      this.paramSvg.append('path')
        .attr('class', 'param-path')
        .datum(this.parameterHistory)
        .attr('fill', 'none')
        .attr('stroke', '#e1e7f5')
        .attr('stroke-width', 1.5)
        .attr('d', line);
    }
  }
  
  private initTrajectoryVisualization(): void {
    // Ensure we have the container reference
    if (!this.trajectoryVisualizationRef || !this.trajectoryVisualizationRef.nativeElement) {
      console.error('Trajectory visualization reference is not available');
      return;
    }
    
    const container = this.trajectoryVisualizationRef.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Create SVG
    this.trajectorySvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
    
    // Add title
    this.trajectorySvg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('class', 'visualization-title')
      .text('Episode Rewards');
    
    // Add axes
    this.trajectorySvg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${height - 30})`);
    
    this.trajectorySvg.append('g')
      .attr('class', 'y-axis')
      .attr('transform', 'translate(40, 0)');
    
    // Update visualization
    this.updateTrajectoryVisualization();
  }
  
  private updateTrajectoryVisualization(): void {
    if (!this.trajectorySvg) return;
    
    const container = this.trajectoryVisualizationRef.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Update SVG dimensions
    this.trajectorySvg
      .attr('width', width)
      .attr('height', height);
    
    // If no trajectories, just return
    if (this.trajectories.length === 0) return;
    
    // Extract rewards per episode
    const episodeRewards = this.trajectories.map(t => t.getTotalReward());
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, Math.max(10, episodeRewards.length - 1)])
      .range([40, width - 20]);
    
    // Ensure we have valid min/max for y scale
    const minReward = episodeRewards.length > 0 ? Math.min(...episodeRewards) : 0;
    const maxReward = episodeRewards.length > 0 ? Math.max(...episodeRewards) : 10;
    const yPadding = (maxReward - minReward) * 0.1 || 10;
    
    const yScale = d3.scaleLinear()
      .domain([minReward - yPadding, maxReward + yPadding])
      .range([height - 30, 30]);
    
    // Update axes
    this.trajectorySvg.select('.x-axis')
      .call(d3.axisBottom(xScale).ticks(Math.min(10, episodeRewards.length)));
    
    this.trajectorySvg.select('.y-axis')
      .call(d3.axisLeft(yScale));
    
    // Remove existing path
    this.trajectorySvg.select('.reward-path').remove();
    
    // Create line
    const line = d3.line<number>()
      .x((d, i) => xScale(i))
      .y(d => yScale(d));
    
    // Add line if we have enough points
    if (episodeRewards.length > 1) {
      this.trajectorySvg.append('path')
        .attr('class', 'reward-path')
        .datum(episodeRewards)
        .attr('fill', 'none')
        .attr('stroke', '#ff9d45')
        .attr('stroke-width', 2)
        .attr('d', line);
    }
    
    // Update points
    const points = this.trajectorySvg.selectAll('.reward-point')
      .data(episodeRewards);
    
    points.enter()
      .append('circle')
      .attr('class', 'reward-point')
      .attr('r', 4)
      .merge(points)
      .attr('cx', (d: number, i: number) => xScale(i))
      .attr('cy', (d: number) => yScale(d))
      .attr('fill', '#ff9d45');
    
    points.exit().remove();
  }
  
  private initPolicyOutput(): void {
    // Ensure we have the container reference
    if (!this.policyOutputRef || !this.policyOutputRef.nativeElement) {
      console.error('Policy output reference is not available');
      return;
    }
    
    const container = this.policyOutputRef.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Create SVG
    this.policySvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
    
    // Add title
    this.policySvg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('class', 'visualization-title')
      .text('Policy Output Distribution');
    
    // Create plot areas
    this.policySvg.append('g')
      .attr('class', 'distribution-plot')
      .attr('transform', `translate(${width/2}, ${height/2})`);
    
    // Update visualization
    this.updatePolicyOutput();
  }
  
  private updatePolicyOutput(): void {
    if (!this.policySvg) return;
    
    const container = this.policyOutputRef.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight || 280;
    
    // Update SVG dimensions
    this.policySvg
      .attr('width', width)
      .attr('height', height);
    
    // Get current state for policy
    let state = [1, 0, 0]; // Default state (pendulum at bottom)
    if (this.pendulum) {
      const angle = this.pendulum.rotation.z;
      const velocity = this.pendulum.userData?.['angularVelocity'] || 0;
      state = [Math.cos(angle), Math.sin(angle), velocity];
    }
    
    // Get policy distribution for current state
    const distribution = this.policy.forward(state);
    
    // Create a range of actions to visualize the distribution
    const actions = Array.from({length: 101}, (_, i) => -2 + i * 0.04);
    const probabilities = actions.map(a => distribution.pdf(a));
    
    // Get max probability for scaling
    const maxProbability = Math.max(...probabilities, 0.1);
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([-2, 2])
      .range([50, width - 50]);
    
    const yScale = d3.scaleLinear()
      .domain([0, maxProbability * 1.1])
      .range([height - 50, 50]);
    
    // Create line
    const line = d3.line()
      .x((d: [number, number]) => xScale(d[0]))
      .y((d: [number, number]) => yScale(d[1]));
    
    // Clear previous content to prevent accumulation
    this.policySvg.select('.distribution-plot').selectAll('*').remove();
    
    // Update distribution plot
    const plot = this.policySvg.select('.distribution-plot');
    
    // Add axes
    plot.append('g')
      .attr('transform', `translate(0, ${height - 50})`)
      .call(d3.axisBottom(xScale));
    
    plot.append('g')
      .attr('transform', 'translate(50, 0)')
      .call(d3.axisLeft(yScale));
    
    // Add axis labels
    plot.append('text')
      .attr('x', width / 2)
      .attr('y', height - 10)
      .attr('text-anchor', 'middle')
      .attr('class', 'axis-label')
      .text('Action (Torque)');
    
    plot.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('class', 'axis-label')
      .text('Probability Density');
    
    // Add distribution line
    const points: [number, number][] = actions.map((a, i) => [a, probabilities[i]]);
    
    plot.append('path')
      .datum(points)
      .attr('fill', 'none')
      .attr('stroke', '#ff9d45')
      .attr('stroke-width', 2)
      .attr('d', line);
    
    // Add mean line
    const mean = distribution.mean;
    plot.append('line')
      .attr('x1', xScale(mean))
      .attr('y1', yScale(0))
      .attr('x2', xScale(mean))
      .attr('y2', yScale(maxProbability * 0.8))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5');
    
    // Add mean label
    plot.append('text')
      .attr('x', xScale(mean))
      .attr('y', yScale(maxProbability * 0.85))
      .attr('dy', -10)
      .attr('text-anchor', 'middle')
      .attr('class', 'mean-label')
      .text(`Mean: ${mean.toFixed(2)}`);
  }
  
  // Window resize handling
  private resizeVisualizations(): void {
    if (!this.visualizationsInitialized) return;
    
    if (this.renderer) {
      const canvas = this.pendulumCanvasRef.nativeElement;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      
      this.renderer.setSize(width, height);
      this.renderPendulum();
    }
    
    if (this.paramSvg) this.updateParameterLandscape();
    if (this.trajectorySvg) this.updateTrajectoryVisualization();
    if (this.policySvg) this.updatePolicyOutput();
  }
  
  // Window resize event handler
  onResize(): void {
    if (this.activeTab === 'simulation') {
      this.resizeVisualizations();
    }
  }
}

// Helper classes
class GaussianDistribution {
  constructor(public mean: number, public stdDev: number) {}
  
  sample(): number {
    // Box-Muller transform for Gaussian sampling
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return this.mean + this.stdDev * z0;
  }
  
  pdf(x: number): number {
    // Probability density function for Gaussian
    const variance = this.stdDev * this.stdDev;
    return (1.0 / Math.sqrt(2.0 * Math.PI * variance)) * 
           Math.exp(-0.5 * Math.pow((x - this.mean) / this.stdDev, 2));
  }
  
  logProb(x: number): number {
    // Log probability for numerical stability
    const variance = this.stdDev * this.stdDev;
    return -0.5 * Math.log(2.0 * Math.PI * variance) -
           0.5 * Math.pow((x - this.mean) / this.stdDev, 2);
  }
}

class PolicyNetwork {
  // Simple linear policy: action = w * state + b
  private weights: number[];
  private bias: number;
  
  constructor(inputDim: number, outputDim: number) {
    // Initialize weights randomly
    this.weights = Array.from({length: inputDim}, () => (Math.random() - 0.5) * 0.1);
    this.bias = (Math.random() - 0.5) * 0.1;
  }
  
  forward(state: number[]): GaussianDistribution {
    // Calculate mean of Gaussian distribution
    let mean = this.bias;
    for (let i = 0; i < this.weights.length && i < state.length; i++) {
      mean += this.weights[i] * state[i];
    }
    
    // Fixed standard deviation (could be learned as well)
    const stdDev = 0.5;
    
    return new GaussianDistribution(mean, stdDev);
  }
  
  update(state: number[], action: number, returnValue: number, learningRate: number): void {
    // Calculate action distribution
    const distribution = this.forward(state);
    
    // Calculate log probability of the taken action
    const logProb = distribution.logProb(action);
    
    // Calculate gradients
    const actionDiff = action - distribution.mean;
    const gradientFactor = actionDiff / (distribution.stdDev * distribution.stdDev);
    
    // Update weights and bias
    for (let i = 0; i < this.weights.length && i < state.length; i++) {
      this.weights[i] += learningRate * returnValue * gradientFactor * state[i];
    }
    this.bias += learningRate * returnValue * gradientFactor;
  }
  
  getParameters(): number[] {
    // Return the first two weights for visualization purposes
    return [this.weights[0] || 0, this.weights[1] || 0];
  }
}

class Trajectory {
  steps: {state: number[], action: number, reward: number}[] = [];
  isComplete: boolean = false;
  
  addStep(state: number[], action: number, reward: number): void {
    this.steps.push({state, action, reward});
  }
  
  complete(): void {
    this.isComplete = true;
  }
  
  getTotalReward(): number {
    return this.steps.reduce((sum, step) => sum + step.reward, 0);
  }
}