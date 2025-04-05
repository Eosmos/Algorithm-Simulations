import { Component, OnInit, AfterViewInit, ElementRef, ViewChild, OnDestroy, NgZone } from '@angular/core';
import { DecimalPipe, NgFor, NgIf } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';
import * as THREE from 'three';

@Component({
  selector: 'app-gan-simulation',
  templateUrl: './gan-simulation.component.html',
  styleUrls: ['./gan-simulation.component.scss'],
  standalone: true,
  imports: [NgIf, NgFor, FormsModule, DecimalPipe]
})
export class GanSimulationComponent implements OnInit, AfterViewInit, OnDestroy {
  // Class properties
  @ViewChild('ganSimulation') ganSimulationRef!: ElementRef;
  @ViewChild('ganDistribution') ganDistributionRef!: ElementRef;
  @ViewChild('ganTrainingProgress') ganTrainingProgressRef!: ElementRef;

  // Simulation state
  isPlaying: boolean = false;
  currentStep: number = 0;
  maxSteps: number = 200;
  animationSpeed: number = 500; // ms between steps
  animationId: number | null = null;
  pendingStep: number | null = null;

  // Loading states
  isLoading: boolean = true;
  loadingProgress: number = 0;
  loadingMessage: string = 'Initializing...';
  visualizationReady: boolean = false;
  initializationTimeExceeded: boolean = false;

  // Timeouts and intervals
  private initializationTimeout: any = null;

  // Performance flags
  isAnimationActive: boolean = true;

  // GAN parameters
  epochs: number = 0;
  discriminatorLoss: number = 1.0;
  generatorLoss: number = 1.0;
  generationQuality: number = 0;

  // Visualization objects
  scene!: THREE.Scene;
  camera!: THREE.PerspectiveCamera;
  renderer!: THREE.WebGLRenderer;

  // For cleanup
  private resizeHandler: (() => void) | null = null;

  // Learning visualizations
  distributionSvg: any;
  progressSvg: any;

  // GAN data
  realSamples: { x: number, y: number }[] = [];
  fakeSamples: { x: number, y: number }[] = [];
  lossHistory: { epoch: number, discriminatorLoss: number, generatorLoss: number }[] = [];

  // Selected research paper for detailed view
  selectedPaper: { title: string, authors: string, year: number, description: string } | null = null;

  // Research papers
  researchPapers = [
    {
      title: 'Generative Adversarial Nets',
      authors: 'Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio',
      year: 2014,
      description: 'The original GAN paper introducing the adversarial training framework where two neural networks compete against each other.'
    },
    {
      title: 'Improved Techniques for Training GANs',
      authors: 'Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen',
      year: 2016,
      description: 'Introduces techniques to improve the stability of GAN training, including feature matching, minibatch discrimination, and historical averaging.'
    },
    {
      title: 'Wasserstein GAN',
      authors: 'Martin Arjovsky, Soumith Chintala, Léon Bottou',
      year: 2017,
      description: 'Proposes using Wasserstein distance (Earth Mover\'s distance) to train GANs, which helps with training stability and prevents mode collapse.'
    },
    {
      title: 'Progressive Growing of GANs for Improved Quality, Stability, and Variation',
      authors: 'Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen',
      year: 2018,
      description: 'Introduces a method to grow both the generator and discriminator progressively, starting from a low resolution and adding new layers for higher resolution outputs.'
    },
    {
      title: 'StyleGAN: A Style-Based Generator Architecture for GANs',
      authors: 'Tero Karras, Samuli Laine, Timo Aila',
      year: 2019,
      description: 'Introduces a style-based generator that enables more control over the generated image features and improved quality.'
    }
  ];

  // GAN applications
  applications = [
    {
      name: 'Image Generation',
      description: 'Creating photorealistic images from random noise, enabling creation of faces, landscapes, and objects that don\'t exist.'
    },
    {
      name: 'Image-to-Image Translation',
      description: 'Converting images from one domain to another, such as sketches to photos, day to night scenes, or style transfer between artistic genres.'
    },
    {
      name: 'Data Augmentation',
      description: 'Generating synthetic training examples to expand datasets for improved machine learning model training, especially valuable when real data is limited.'
    },
    {
      name: 'Super-Resolution',
      description: 'Enhancing low-resolution images to create detailed, high-resolution versions that preserve and extrapolate features.'
    },
    {
      name: 'Anomaly Detection',
      description: 'Identifying unusual patterns by training on normal data, where anomalies are poorly reconstructed or have distinctive discriminator scores.'
    },
    {
      name: 'Text-to-Image Synthesis',
      description: 'Generating images based on textual descriptions, creating visual content from natural language inputs.'
    }
  ];

  constructor(private ngZone: NgZone) { }

  // Loader attempts to ensure the component loads eventually
  private maxInitAttempts = 5;
  private currentInitAttempt = 0;
  private checkInterval: any = null;

  ngOnInit(): void {
    // Just prepare data - don't try to access DOM yet
    this.loadingMessage = 'Preparing data...';
    this.initializeGanData();
    this.loadingProgress = 20;

    // Set timeout for initialization failure
    this.initializationTimeout = setTimeout(() => {
      this.initializationTimeExceeded = true;
    }, 5000); // 5 seconds timeout for initialization
  }

  ngAfterViewInit(): void {
    // First create the placeholders to ensure we have something to display
    this.createPlaceholderVisualizations();

    // Start checking if DOM elements are available
    this.checkInterval = setInterval(() => {
      this.checkAndInitializeComponent();
    }, 200);

    // Ensure we set visualizationReady to true after a timeout
    // even if the other initialization steps fail
    setTimeout(() => {
      if (!this.visualizationReady) {
        console.log('Forcing visualization ready state after timeout');
        this.clearInitializationTimeout();
        this.visualizationReady = true;
        this.updateDistributionVisualization(this.currentStep);
      }
    }, 3000);
  }

  // Check if DOM elements are available and initialize component
  private checkAndInitializeComponent(): void {
    this.currentInitAttempt++;

    // Check if DOM elements are available via direct query if ViewChild didn't work
    const hasSimulationElement = this.ganSimulationRef || document.querySelector('#ganSimulation');
    const hasDistributionElement = this.ganDistributionRef || document.querySelector('#ganDistribution');
    const hasTrainingProgressElement = this.ganTrainingProgressRef || document.querySelector('#ganTrainingProgress');

    if (hasSimulationElement && hasDistributionElement && hasTrainingProgressElement) {
      // DOM elements are available, clear check interval and start loading
      clearInterval(this.checkInterval);

      // Store references if ViewChild didn't work
      if (!this.ganSimulationRef && hasSimulationElement) {
        this.ganSimulationRef = { nativeElement: document.querySelector('#ganSimulation') } as ElementRef;
      }
      if (!this.ganDistributionRef && hasDistributionElement) {
        this.ganDistributionRef = { nativeElement: document.querySelector('#ganDistribution') } as ElementRef;
      }
      if (!this.ganTrainingProgressRef && hasTrainingProgressElement) {
        this.ganTrainingProgressRef = { nativeElement: document.querySelector('#ganTrainingProgress') } as ElementRef;
      }

      console.log('DOM elements available, starting progressive loading');
      this.startProgressiveLoading();
    } else if (this.currentInitAttempt >= this.maxInitAttempts) {
      // Max attempts reached, try simplified initialization
      clearInterval(this.checkInterval);
      console.log('Max attempts reached, using simplified initialization');

      // Set loading complete even without full initialization
      setTimeout(() => {
        this.isLoading = false;
        this.loadingProgress = 100;
        this.loadingMessage = 'Initialization partial, some features may be limited';
        // Set visualization ready to true to remove the "Preparing" message
        this.visualizationReady = true;
        console.log('Component loaded with limited features');
      }, 500);
    }
  }

  ngOnDestroy(): void {
    // Stop any ongoing animations
    this.isAnimationActive = false;

    // Clear check interval if it exists
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }

    // Clear initialization timeout
    this.clearInitializationTimeout();

    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    // Clean up Three.js resources
    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.domElement && this.renderer.domElement.parentNode) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
    }

    // Clean up scene objects
    if (this.scene) {
      this.scene.clear();
    }

    // Remove D3 visualizations
    if (this.distributionSvg) {
      d3.select('#ganDistribution').selectAll('*').remove();
    }

    if (this.progressSvg) {
      d3.select('#ganTrainingProgress').selectAll('*').remove();
    }

    // Remove event listeners
    if (this.resizeHandler) {
      window.removeEventListener('resize', this.resizeHandler);
      this.resizeHandler = null;
    }
  }

  // Optional: Add a function to allow manual reload if needed
  retryLoading(): void {
    if (this.isLoading) {
      // Reset progress and restart loading
      this.loadingProgress = 0;
      this.loadingMessage = 'Retrying...';
      this.startProgressiveLoading();
    }
  }

  private startProgressiveLoading(): void {
    // Stage 2: Create scene (40%)
    this.loadingMessage = 'Setting up 3D scene...';
    setTimeout(() => {
      if (this.ganSimulationRef) {
        this.initializeThreeJsScene();
      } else {
        console.warn("Simulation reference still not available - skipping scene setup");
      }
      this.loadingProgress = 40;

      // Stage 3: Setup distributions (60%)
      this.loadingMessage = 'Creating visualizations...';
      setTimeout(() => {
        if (this.ganDistributionRef) {
          this.initializeDistributionVisualization();
        } else {
          console.warn("Distribution reference still not available - skipping distribution setup");
        }
        this.loadingProgress = 60;

        // Stage 4: Training progress (80%)
        this.loadingMessage = 'Processing training data...';
        setTimeout(() => {
          if (this.ganTrainingProgressRef) {
            this.initializeTrainingProgressVisualization();
          } else {
            console.warn("Training progress reference still not available - skipping progress setup");
          }
          this.loadingProgress = 80;

          // Stage 5: Final rendering (100%)
          this.loadingMessage = 'Finishing up...';
          setTimeout(() => {
            if (this.scene) {
              this.renderGanStructure();
            } else {
              console.warn("Scene still not available - skipping structure rendering");
            }
            this.loadingProgress = 100;

            // Complete loading after a short delay to ensure smooth transition
            setTimeout(() => {
              this.isLoading = false;
              // Set visualization ready to true so the "preparing" message disappears
              this.visualizationReady = true;
              // Clear initialization timeout since we're done
              this.clearInitializationTimeout();

              // Initialize starting visualization state
              this.currentStep = 0;
              this.epochs = 0;
              this.discriminatorLoss = 1.0;
              this.generatorLoss = 1.0;
              this.generationQuality = 0;

              // If there was a pending step, apply it now
              if (this.pendingStep !== null) {
                this.updateDistributionVisualization(this.pendingStep);
                this.pendingStep = null;
              }

              console.log('GAN simulation fully loaded and ready');
            }, 500);
          }, 300);
        }, 200);
      }, 200);
    }, 200);
  }

  private initializeGanData(): void {
    try {
      // Initialize real data samples (representing some target distribution)
      // Reduce sample size for better performance
      this.realSamples = this.generateBimodalGaussianSamples(50);

      // Initialize fake data samples (initially random)
      this.fakeSamples = this.generateRandomSamples(50);

      // Initialize loss history - reduce number of data points
      for (let i = 0; i < 25; i++) {
        const epoch = i * 8;
        const discLoss = 0.7 * Math.exp(-i / 25) + 0.3 * (0.5 + 0.5 * Math.sin(i / 5));
        const genLoss = 0.8 * Math.exp(-i / 30) + 0.3 * (0.5 + 0.5 * Math.cos(i / 4));

        this.lossHistory.push({
          epoch,
          discriminatorLoss: discLoss,
          generatorLoss: genLoss
        });
      }
    } catch (error) {
      console.error('Error initializing GAN data:', error);
    }
  }

  private generateBimodalGaussianSamples(n: number): { x: number, y: number }[] {
    const samples: { x: number, y: number }[] = [];

    for (let i = 0; i < n / 2; i++) {
      // First mode
      samples.push({
        x: this.gaussianRandom(0.3, 0.05),
        y: this.gaussianRandom(0.3, 0.05)
      });

      // Second mode
      samples.push({
        x: this.gaussianRandom(0.7, 0.05),
        y: this.gaussianRandom(0.7, 0.05)
      });
    }

    return samples;
  }

  private generateRandomSamples(n: number): { x: number, y: number }[] {
    const samples: { x: number, y: number }[] = [];

    for (let i = 0; i < n; i++) {
      samples.push({
        x: Math.random(),
        y: Math.random()
      });
    }

    return samples;
  }

  private gaussianRandom(mean: number, stdev: number): number {
    const u = 1 - Math.random();
    const v = Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stdev + mean;
  }

  private initializeThreeJsScene(): void {
    try {
      // Return early if element is not available yet
      if (!this.ganSimulationRef) {
        // Try direct DOM access as fallback
        const element = document.getElementById('ganSimulation');
        if (element) {
          this.ganSimulationRef = { nativeElement: element } as ElementRef;
        } else {
          console.warn('Simulation reference not available yet');
          return;
        }
      }

      const container = this.ganSimulationRef.nativeElement;

      // Create scene with simplified settings
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color('#0c1428');

      // Create camera with reduced field of view
      this.camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.1,
        100
      );
      this.camera.position.z = 5;

      // Create renderer with optimized settings
      this.renderer = new THREE.WebGLRenderer({
        antialias: false, // Disable antialiasing for performance
        powerPreference: 'high-performance',
        alpha: false
      });
      this.renderer.setSize(container.clientWidth, container.clientHeight);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // Limit pixel ratio
      container.appendChild(this.renderer.domElement);

      // Add minimal lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
      this.scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(1, 1, 1);
      this.scene.add(directionalLight);

      // Set up resize handler
      const resizeHandler = this.handleResize.bind(this);
      window.addEventListener('resize', resizeHandler);

      // Store reference to handler for cleanup
      this.resizeHandler = resizeHandler;

      // Start animation loop outside Angular's change detection
      this.ngZone.runOutsideAngular(() => {
        this.animate();
      });
    } catch (error) {
      console.error('Error initializing ThreeJS scene:', error);
      // Set visualization ready to true even if we encounter an error
      this.visualizationReady = true;
    }
  }

  // Window resize handler
  private handleResize(): void {
    if (!this.camera || !this.renderer || !this.ganSimulationRef) return;

    const container = this.ganSimulationRef.nativeElement;
    this.camera.aspect = container.clientWidth / container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(container.clientWidth, container.clientHeight);
  }

  private animate(): void {
    if (!this.isAnimationActive) return;

    // Use a performant animation frame requesting
    this.animationId = requestAnimationFrame(() => this.animate());

    // Only render if the component is likely visible
    if (document.hidden) {
      return;
    }

    // Throttle rendering for better performance
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  private renderGanStructure(): void {
    try {
      if (!this.scene) {
        console.warn('Scene not initialized');
        return;
      }

      // Clear existing objects except lights
      const lightsToKeep: THREE.Light[] = [];
      this.scene.traverse((object) => {
        if (object instanceof THREE.Light) {
          lightsToKeep.push(object.clone() as THREE.Light);
        }
      });

      this.scene.clear();

      // Re-add lights
      for (const light of lightsToKeep) {
        this.scene.add(light);
      }

      // Create generator network visualization
      const generator = this.createNetworkMesh('#7c4dff', -2, 0, 0);
      this.scene.add(generator);

      // Create discriminator network visualization
      const discriminator = this.createNetworkMesh('#4285f4', 2, 0, 0);
      this.scene.add(discriminator);

      // Create data flow visualization
      const dataFlow = this.createDataFlowMesh();
      this.scene.add(dataFlow);

      // Add visual labels (using positioning only since we can't add HTML directly)
      this.addVisualLabels();
    } catch (error) {
      console.error('Error rendering GAN structure:', error);
      // Set visualization ready to true even if we encounter an error
      this.visualizationReady = true;
    }
  }

  private createNetworkMesh(color: string, x: number, y: number, z: number): THREE.Object3D {
    const networkGroup = new THREE.Group();

    // Create layers with simplified geometry
    const numLayers = 3; // Reduced from 4
    const layerWidth = 0.8;
    const layerHeight = 1.5;
    const layerDepth = 0.1;
    const layerSpacing = 0.3;

    for (let i = 0; i < numLayers; i++) {
      const layerGeometry = new THREE.BoxGeometry(layerWidth, layerHeight * (i === 0 || i === numLayers - 1 ? 0.7 : 1), layerDepth);
      const layerMaterial = new THREE.MeshPhongMaterial({
        color: color,
        transparent: true,
        opacity: 0.8,
        emissive: color,
        emissiveIntensity: 0.2,
        flatShading: true  // More performant than smooth shading
      });

      const layer = new THREE.Mesh(layerGeometry, layerMaterial);
      layer.position.x = x;
      layer.position.y = y;
      layer.position.z = z - i * (layerDepth + layerSpacing);

      networkGroup.add(layer);

      // Add fewer neurons for better performance
      const neuronsPerLayer = i === 0 ? 2 : (i === numLayers - 1 ? 1 : 3);
      const neuronSize = 0.05;
      const neuronSpacing = layerHeight / (neuronsPerLayer + 1);

      for (let j = 0; j < neuronsPerLayer; j++) {
        const neuronGeometry = new THREE.SphereGeometry(neuronSize, 8, 8); // Reduced segments
        const neuronMaterial = new THREE.MeshPhongMaterial({
          color: '#ffffff',
          emissive: '#ffffff',
          emissiveIntensity: 0.5,
          flatShading: true
        });

        const neuron = new THREE.Mesh(neuronGeometry, neuronMaterial);
        neuron.position.x = x;
        neuron.position.y = y + (j + 1) * neuronSpacing - layerHeight / 2;
        neuron.position.z = z - i * (layerDepth + layerSpacing);

        networkGroup.add(neuron);
      }
    }

    return networkGroup;
  }

  private createDataFlowMesh(): THREE.Object3D {
    const flowGroup = new THREE.Group();

    // Noise input
    const noiseGeometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const noiseMaterial = new THREE.MeshPhongMaterial({
      color: '#ffffff',
      transparent: true,
      opacity: 0.6,
      wireframe: true
    });

    const noise = new THREE.Mesh(noiseGeometry, noiseMaterial);
    noise.position.set(-3.5, 0, 0);
    flowGroup.add(noise);

    // Flow arrows
    const arrowMaterial = new THREE.LineBasicMaterial({ color: '#8bb4fa' });

    // Noise to Generator
    const noiseToGenPoints = [];
    noiseToGenPoints.push(new THREE.Vector3(-3.2, 0, 0));
    noiseToGenPoints.push(new THREE.Vector3(-2.4, 0, 0));

    const noiseToGenGeometry = new THREE.BufferGeometry().setFromPoints(noiseToGenPoints);
    const noiseToGenLine = new THREE.Line(noiseToGenGeometry, arrowMaterial);
    flowGroup.add(noiseToGenLine);

    // Generator to Discriminator
    const genToDiscPoints = [];
    genToDiscPoints.push(new THREE.Vector3(-1.5, 0, 0));
    genToDiscPoints.push(new THREE.Vector3(0, -0.5, 0));
    genToDiscPoints.push(new THREE.Vector3(1.5, -0.5, 0));

    const genToDiscGeometry = new THREE.BufferGeometry().setFromPoints(genToDiscPoints);
    const genToDiscLine = new THREE.Line(genToDiscGeometry, arrowMaterial);
    flowGroup.add(genToDiscLine);

    // Real data to Discriminator
    const realDataGeometry = new THREE.CircleGeometry(0.3, 32);
    const realDataMaterial = new THREE.MeshPhongMaterial({
      color: '#24b47e',
      transparent: true,
      opacity: 0.8
    });

    const realData = new THREE.Mesh(realDataGeometry, realDataMaterial);
    realData.position.set(3.5, 0.5, 0);
    flowGroup.add(realData);

    const realToDiscPoints = [];
    realToDiscPoints.push(new THREE.Vector3(3.2, 0.5, 0));
    realToDiscPoints.push(new THREE.Vector3(2.4, 0, 0));

    const realToDiscGeometry = new THREE.BufferGeometry().setFromPoints(realToDiscPoints);
    const realToDiscLine = new THREE.Line(realToDiscGeometry, arrowMaterial);
    flowGroup.add(realToDiscLine);

    // Discriminator output
    const outputGeometry = new THREE.BoxGeometry(0.4, 0.2, 0.1);
    const outputMaterial = new THREE.MeshPhongMaterial({
      color: '#ff9d45',
      transparent: true,
      opacity: 0.8
    });

    const output = new THREE.Mesh(outputGeometry, outputMaterial);
    output.position.set(3.5, -0.5, 0);
    flowGroup.add(output);

    const discToOutputPoints = [];
    discToOutputPoints.push(new THREE.Vector3(2.4, -0.5, 0));
    discToOutputPoints.push(new THREE.Vector3(3.3, -0.5, 0));

    const discToOutputGeometry = new THREE.BufferGeometry().setFromPoints(discToOutputPoints);
    const discToOutputLine = new THREE.Line(discToOutputGeometry, arrowMaterial);
    flowGroup.add(discToOutputLine);

    return flowGroup;
  }

  private addVisualLabels(): void {
    // In a real implementation, we would use HTML overlays or textures
    // For this simplified version, we'll just log the positions
    console.log('Labels would be positioned at:');
    console.log('Generator: (-2, -1.5, 0)');
    console.log('Discriminator: (2, -1.5, 0)');
    console.log('Random Noise: (-3.5, 0, 0)');
    console.log('Generated Data: (0, -0.5, 0)');
    console.log('Real Data: (3.5, 0.5, 0)');
    console.log('Real or Fake?: (3.5, -0.5, 0)');
  }

  private initializeDistributionVisualization(): void {
    try {
      // Guard against null references
      if (!this.ganDistributionRef) {
        // Try direct DOM access as fallback
        const element = document.getElementById('ganDistribution');
        if (element) {
          this.ganDistributionRef = { nativeElement: element } as ElementRef;
        } else {
          console.warn('Distribution reference not available yet');
          return;
        }
      }

      const container = this.ganDistributionRef.nativeElement;
      const width = container.clientWidth || 300; // Fallback width if clientWidth is 0
      const height = container.clientHeight || 200; // Fallback height if clientHeight is 0
      const margin = { top: 20, right: 20, bottom: 30, left: 40 };

      // Create SVG
      this.distributionSvg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

      console.log('Distribution visualization initialized successfully');

      // Create scales
      const xScale = d3.scaleLinear()
        .domain([0, 1])
        .range([margin.left, width - margin.right]);

      const yScale = d3.scaleLinear()
        .domain([0, 1])
        .range([height - margin.bottom, margin.top]);

      // Add axes
      this.distributionSvg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(xScale));

      this.distributionSvg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale));

      // Add title
      this.distributionSvg.append('text')
        .attr('x', width / 2)
        .attr('y', margin.top / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('fill', '#e1e7f5')
        .text('Data Distribution Visualization');

      // Add real data points
      this.distributionSvg.selectAll('.real-point')
        .data(this.realSamples)
        .enter()
        .append('circle')
        .attr('class', 'real-point')
        .attr('cx', (d: { x: number, y: number }) => xScale(d.x))
        .attr('cy', (d: { x: number, y: number }) => yScale(d.y))
        .attr('r', 3)
        .style('fill', '#24b47e')
        .style('opacity', 0.7);

      // Add fake data points
      this.distributionSvg.selectAll('.fake-point')
        .data(this.fakeSamples)
        .enter()
        .append('circle')
        .attr('class', 'fake-point')
        .attr('cx', (d: { x: number, y: number }) => xScale(d.x))
        .attr('cy', (d: { x: number, y: number }) => yScale(d.y))
        .attr('r', 3)
        .style('fill', '#7c4dff')
        .style('opacity', 0.7);

      // Add legend
      const legend = this.distributionSvg.append('g')
        .attr('transform', `translate(${width - 100}, ${margin.top + 10})`);

      legend.append('circle')
        .attr('cx', 0)
        .attr('cy', 0)
        .attr('r', 3)
        .style('fill', '#24b47e');

      legend.append('text')
        .attr('x', 10)
        .attr('y', 5)
        .style('font-size', '12px')
        .style('fill', '#e1e7f5')
        .text('Real Data');

      legend.append('circle')
        .attr('cx', 0)
        .attr('cy', 20)
        .attr('r', 3)
        .style('fill', '#7c4dff');

      legend.append('text')
        .attr('x', 10)
        .attr('y', 25)
        .style('font-size', '12px')
        .style('fill', '#e1e7f5')
        .text('Fake Data');
    } catch (error) {
      console.error('Error initializing distribution visualization:', error);
      // Create placeholder visualizations on error
      this.createPlaceholderVisualizations();
    }
  }

  private initializeTrainingProgressVisualization(): void {
    try {
      // Guard against null references
      if (!this.ganTrainingProgressRef) {
        // Try direct DOM access as fallback
        const element = document.getElementById('ganTrainingProgress');
        if (element) {
          this.ganTrainingProgressRef = { nativeElement: element } as ElementRef;
        } else {
          console.warn('Training progress reference not available yet');
          return;
        }
      }

      const container = this.ganTrainingProgressRef.nativeElement;
      const width = container.clientWidth || 300; // Fallback width if clientWidth is 0
      const height = container.clientHeight || 200; // Fallback height if clientHeight is 0
      const margin = { top: 20, right: 50, bottom: 30, left: 50 };

      // Create SVG
      this.progressSvg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

      console.log('Training progress visualization initialized successfully');

      // Create scales
      const xScale = d3.scaleLinear()
        .domain([0, d3.max(this.lossHistory, (d) => d.epoch) || 200])
        .range([margin.left, width - margin.right]);

      const yScale = d3.scaleLinear()
        .domain([0, 1])
        .range([height - margin.bottom, margin.top]);

      // Add axes
      this.progressSvg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat((d) => `${d}`));

      this.progressSvg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale).ticks(5));

      // Add title
      this.progressSvg.append('text')
        .attr('x', width / 2)
        .attr('y', margin.top / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('fill', '#e1e7f5')
        .text('Training Progress');

      // Add X axis label
      this.progressSvg.append('text')
        .attr('x', width / 2)
        .attr('y', height - margin.bottom / 3)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('fill', '#e1e7f5')
        .text('Epochs');

      // Add Y axis label
      this.progressSvg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -(height / 2))
        .attr('y', margin.left / 3)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('fill', '#e1e7f5')
        .text('Loss');

      // Create line generators
      const discriminatorLine = d3.line<{ epoch: number, discriminatorLoss: number }>()
        .x((d) => xScale(d.epoch))
        .y((d) => yScale(d.discriminatorLoss))
        .curve(d3.curveMonotoneX);

      const generatorLine = d3.line<{ epoch: number, generatorLoss: number }>()
        .x((d) => xScale(d.epoch))
        .y((d) => yScale(d.generatorLoss))
        .curve(d3.curveMonotoneX);

      // Add discriminator loss line
      this.progressSvg.append('path')
        .datum(this.lossHistory)
        .attr('class', 'discriminator-line')
        .attr('fill', 'none')
        .attr('stroke', '#4285f4')
        .attr('stroke-width', 2)
        .attr('d', discriminatorLine);

      // Add generator loss line
      this.progressSvg.append('path')
        .datum(this.lossHistory)
        .attr('class', 'generator-line')
        .attr('fill', 'none')
        .attr('stroke', '#7c4dff')
        .attr('stroke-width', 2)
        .attr('d', generatorLine);

      // Add legend
      const legend = this.progressSvg.append('g')
        .attr('transform', `translate(${width - 130}, ${margin.top + 10})`);

      legend.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', 15)
        .attr('y2', 0)
        .style('stroke', '#4285f4')
        .style('stroke-width', 2);

      legend.append('text')
        .attr('x', 20)
        .attr('y', 5)
        .style('font-size', '12px')
        .style('fill', '#e1e7f5')
        .text('Discriminator Loss');

      legend.append('line')
        .attr('x1', 0)
        .attr('y1', 20)
        .attr('x2', 15)
        .attr('y2', 20)
        .style('stroke', '#7c4dff')
        .style('stroke-width', 2);

      legend.append('text')
        .attr('x', 20)
        .attr('y', 25)
        .style('font-size', '12px')
        .style('fill', '#e1e7f5')
        .text('Generator Loss');

      // Add current position indicator
      this.progressSvg.append('line')
        .attr('class', 'current-position')
        .attr('x1', xScale(0))
        .attr('y1', margin.top)
        .attr('x2', xScale(0))
        .attr('y2', height - margin.bottom)
        .style('stroke', '#ffffff')
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4 4')
        .style('opacity', 0.7);
    } catch (error) {
      console.error('Error initializing training progress visualization:', error);
      // Create placeholder visualizations on error
      this.createPlaceholderVisualizations();
    }
  }

  updateDistributionVisualization(step: number): void {
    try {
      // First check if visualization is ready
      if (!this.visualizationReady) {
        console.log('Visualization not ready yet, storing for later');
        this.pendingStep = step;
        return;
      }

      // If we don't have distribution svg, try to recreate placeholders
      if (!this.distributionSvg || !this.progressSvg) {
        console.log('SVG elements not available, recreating placeholders');
        this.createPlaceholderVisualizations();
        if (!this.distributionSvg || !this.progressSvg) {
          console.error('Could not create visualization placeholders');
          return;
        }
      }

      // Update fake samples based on simulation step
      // In a real implementation, this would use actual GAN outputs
      const progressRatio = Math.min(step / this.maxSteps, 1);

      // Gradually move fake samples toward real distribution
      this.fakeSamples = this.fakeSamples.map((sample, i) => {
        const targetSample = this.realSamples[i % this.realSamples.length];

        return {
          x: sample.x + (targetSample.x - sample.x) * progressRatio + (Math.random() - 0.5) * 0.05 * (1 - progressRatio),
          y: sample.y + (targetSample.y - sample.y) * progressRatio + (Math.random() - 0.5) * 0.05 * (1 - progressRatio)
        };
      });

      // Get container dimensions
      const container = this.ganDistributionRef.nativeElement;
      const width = container.clientWidth || 300;
      const height = container.clientHeight || 200;

      try {
        // Simple update for placeholders - just update positions directly
        this.distributionSvg.selectAll('.placeholder-point, .fake-point')
          .data(this.fakeSamples)
          .attr('cx', (d: { x: number, y: number }) => d.x * (width - 40) + 20)
          .attr('cy', (d: { x: number, y: number }) => d.y * (height - 40) + 20);
      } catch (e) {
        console.log('Error updating distribution points, using simpler approach');

        // If we can't update the existing points, recreate them
        this.distributionSvg.selectAll('.placeholder-point, .fake-point').remove();

        this.distributionSvg.selectAll('.placeholder-point')
          .data(this.fakeSamples)
          .enter()
          .append('circle')
          .attr('class', 'placeholder-point')
          .attr('cx', (d: { x: number, y: number }) => d.x * (width - 40) + 20)
          .attr('cy', (d: { x: number, y: number }) => d.y * (height - 40) + 20)
          .attr('r', 5)
          .style('fill', '#7c4dff');
      }

      // Update metrics display
      this.epochs = Math.floor(progressRatio * 200);
      this.discriminatorLoss = 0.7 * Math.exp(-this.epochs / 100) + 0.2;
      this.generatorLoss = 0.8 * Math.exp(-this.epochs / 120) + 0.2;
      this.generationQuality = progressRatio * 100;

      console.log('Successfully updated visualization for step', step);
    } catch (error) {
      console.error('Error updating distribution visualization:', error);
      // Don't stop the simulation if there's an error
      this.epochs = Math.floor((step / this.maxSteps) * 200);
      this.discriminatorLoss = 0.7 * Math.exp(-this.epochs / 100) + 0.2;
      this.generatorLoss = 0.8 * Math.exp(-this.epochs / 120) + 0.2;
      this.generationQuality = (step / this.maxSteps) * 100;
    }
  }

  play(): void {
    if (this.isPlaying || this.isLoading) return;

    // If visualizations aren't ready, force them to be created now
    if (!this.visualizationReady) {
      console.log('Forcing visualization initialization before play');
      this.visualizationReady = true;
      this.createPlaceholderVisualizations();
    }

    this.isPlaying = true;
    this.playStep();
  }

  pause(): void {
    this.isPlaying = false;
  }

  reset(): void {
    this.isPlaying = false;
    this.currentStep = 0;

    // Force visualization ready if not already
    if (!this.visualizationReady) {
      console.log('Forcing visualization ready on reset');
      this.visualizationReady = true;
      this.createPlaceholderVisualizations();
    }

    // Update visualization regardless
    this.updateDistributionVisualization(this.currentStep);
  }

  private playStep(): void {
    if (!this.isPlaying) return;

    this.currentStep = Math.min(this.currentStep + 1, this.maxSteps);

    // Always update the metrics even if visualization fails
    this.epochs = Math.floor((this.currentStep / this.maxSteps) * 200);
    this.discriminatorLoss = 0.7 * Math.exp(-this.epochs / 100) + 0.2;
    this.generatorLoss = 0.8 * Math.exp(-this.epochs / 120) + 0.2;
    this.generationQuality = (this.currentStep / this.maxSteps) * 100;

    // Try to update visualization
    try {
      this.updateDistributionVisualization(this.currentStep);
    } catch (error) {
      console.error('Error during play step visualization update:', error);
      // Continue playing even if visualization fails
    }

    if (this.currentStep < this.maxSteps) {
      setTimeout(() => this.playStep(), this.animationSpeed);
    } else {
      this.isPlaying = false;
    }
  }

  setStep(step: number): void {
    if (this.isLoading) {
      console.log('Cannot set step while loading');
      return;
    }

    this.currentStep = Math.min(Math.max(0, step), this.maxSteps);

    // Check if visualization is ready before updating
    if (this.visualizationReady && this.distributionSvg) {
      this.updateDistributionVisualization(this.currentStep);
    } else {
      console.log('Visualization not ready for step change, storing for later');
      this.pendingStep = this.currentStep;
    }
  }

  setSpeed(speed: number): void {
    this.animationSpeed = speed;
  }

  onSpeedChange(event: Event): void {
    const target = event.target as HTMLInputElement;
    this.setSpeed(1100 - Number(target.value));
  }

  showPaperDetails(paper: any): void {
    this.selectedPaper = paper;
  }

  closePaperDetails(): void {
    this.selectedPaper = null;
  }

  // Helper to clear initialization timeout
  private clearInitializationTimeout(): void {
    if (this.initializationTimeout) {
      clearTimeout(this.initializationTimeout);
      this.initializationTimeout = null;
    }
  }

  // Force completion when visualization is taking too long
  forceCompletionAndContinue(): void {
    console.log('User forced completion of visualization initialization');
    this.visualizationReady = true;
    this.clearInitializationTimeout();

    // Create placeholder visualizations if they don't exist
    this.createPlaceholderVisualizations();
  }

  // Create simple placeholder visualizations when proper ones can't be initialized
  private createPlaceholderVisualizations(): void {
    try {
      // First, try to get references to containers if they're not already available
      if (!this.ganDistributionRef) {
        const element = document.getElementById('ganDistribution');
        if (element) {
          this.ganDistributionRef = { nativeElement: element } as ElementRef;
        }
      }

      if (!this.ganTrainingProgressRef) {
        const element = document.getElementById('ganTrainingProgress');
        if (element) {
          this.ganTrainingProgressRef = { nativeElement: element } as ElementRef;
        }
      }

      if (!this.ganSimulationRef) {
        const element = document.getElementById('ganSimulation');
        if (element) {
          this.ganSimulationRef = { nativeElement: element } as ElementRef;
        }
      }

      // Create placeholder for distribution if needed
      if (!this.distributionSvg && this.ganDistributionRef && this.ganDistributionRef.nativeElement) {
        const container = this.ganDistributionRef.nativeElement;

        // Clear any existing content
        d3.select(container).selectAll('*').remove();

        this.distributionSvg = d3.select(container)
          .append('svg')
          .attr('width', container.clientWidth || 300)
          .attr('height', container.clientHeight || 200);

        const width = container.clientWidth || 300;
        const height = container.clientHeight || 200;
        const margin = { top: 20, right: 30, bottom: 40, left: 50 };

        // Create scales
        const xScale = d3.scaleLinear()
          .domain([0, 1])
          .range([margin.left, width - margin.right]);

        const yScale = d3.scaleLinear()
          .domain([0, 1])
          .range([height - margin.bottom, margin.top]);

        // Add axes
        this.distributionSvg.append('g')
          .attr('transform', `translate(0,${height - margin.bottom})`)
          .attr('class', 'x-axis')
          .style('color', '#8a9ab0')
          .call(d3.axisBottom(xScale)
            .ticks(5)
            .tickFormat(d3.format(".1f")));

        this.distributionSvg.append('g')
          .attr('transform', `translate(${margin.left},0)`)
          .attr('class', 'y-axis')
          .style('color', '#8a9ab0')
          .call(d3.axisLeft(yScale)
            .ticks(5)
            .tickFormat(d3.format(".1f")));

        // Add axis labels
        this.distributionSvg.append('text')
          .attr('x', width / 2)
          .attr('y', height - 5)
          .attr('text-anchor', 'middle')
          .style('fill', '#8a9ab0')
          .text('X');

        this.distributionSvg.append('text')
          .attr('transform', 'rotate(-90)')
          .attr('x', -height / 2)
          .attr('y', 15)
          .attr('text-anchor', 'middle')
          .style('fill', '#8a9ab0')
          .text('Y');

        // Generate bimodal distribution for real data
        const realSamples: Array<{ x: number, y: number }> = [];

        // First cluster
        for (let i = 0; i < 15; i++) {
          realSamples.push({
            x: 0.3 + Math.random() * 0.1,
            y: 0.3 + Math.random() * 0.1
          });
        }

        // Second cluster
        for (let i = 0; i < 15; i++) {
          realSamples.push({
            x: 0.7 + Math.random() * 0.1,
            y: 0.7 + Math.random() * 0.1
          });
        }

        this.realSamples = realSamples;

        // Generate random samples for fake data
        const fakeSamples: Array<{ x: number, y: number }> = [];
        for (let i = 0; i < 30; i++) {
          fakeSamples.push({
            x: 0.2 + Math.random() * 0.6,
            y: 0.2 + Math.random() * 0.6
          });
        }

        this.fakeSamples = fakeSamples;

        // Add real data points
        this.distributionSvg.selectAll('.real-point')
          .data(this.realSamples)
          .enter()
          .append('circle')
          .attr('class', 'real-point')
          .attr('cx', (d: { x: number, y: number }) => xScale(d.x))
          .attr('cy', (d: { x: number, y: number }) => yScale(d.y))
          .attr('r', 4)
          .style('fill', '#24b47e')
          .style('opacity', 0.7);

        // Add fake data points
        this.distributionSvg.selectAll('.fake-point')
          .data(this.fakeSamples)
          .enter()
          .append('circle')
          .attr('class', 'fake-point')
          .attr('cx', (d: { x: number, y: number }) => xScale(d.x))
          .attr('cy', (d: { x: number, y: number }) => yScale(d.y))
          .attr('r', 4)
          .style('fill', '#7c4dff')
          .style('opacity', 0.7);

        // Add legend
        const legend = this.distributionSvg.append('g')
          .attr('transform', `translate(${width - margin.right - 100}, ${margin.top})`);

        // Real data legend item
        legend.append('circle')
          .attr('cx', 0)
          .attr('cy', 0)
          .attr('r', 4)
          .style('fill', '#24b47e');

        legend.append('text')
          .attr('x', 10)
          .attr('y', 4)
          .style('font-size', '12px')
          .style('fill', '#e1e7f5')
          .text('Real Data');

        // Fake data legend item
        legend.append('circle')
          .attr('cx', 0)
          .attr('cy', 20)
          .attr('r', 4)
          .style('fill', '#7c4dff');

        legend.append('text')
          .attr('x', 10)
          .attr('y', 24)
          .style('font-size', '12px')
          .style('fill', '#e1e7f5')
          .text('Generated Data');

        console.log('Created enhanced distribution visualization');
      }

      // Create placeholder for progress if needed
      if (!this.progressSvg && this.ganTrainingProgressRef && this.ganTrainingProgressRef.nativeElement) {
        const container = this.ganTrainingProgressRef.nativeElement;

        // Clear any existing content
        d3.select(container).selectAll('*').remove();

        const width = container.clientWidth || 300;
        const height = container.clientHeight || 200;
        const margin = { top: 20, right: 80, bottom: 40, left: 50 };

        this.progressSvg = d3.select(container)
          .append('svg')
          .attr('width', width)
          .attr('height', height);

        // Generate more realistic loss data
        const epochs = 100;
        this.lossHistory = [];

        // Generator typically starts with higher loss than discriminator
        let genLoss = 0.9;
        let discLoss = 0.7;

        for (let i = 0; i < epochs; i++) {
          // Add some randomness and trends to the loss curves
          const progress = i / epochs;

          // Discriminator typically has an early advantage then stabilizes
          discLoss = Math.max(0.2, 0.7 * Math.exp(-progress * 3) + 0.2 + (Math.random() * 0.05));

          // Generator typically improves more gradually
          genLoss = Math.max(0.3, 0.9 * Math.exp(-progress * 2) + 0.3 + (Math.random() * 0.07));

          // Add slight oscillations typical in GAN training
          if (i > 20) {
            discLoss += 0.05 * Math.sin(i / 5);
            genLoss += 0.08 * Math.sin(i / 4 + 1);
          }

          this.lossHistory.push({
            epoch: i,
            discriminatorLoss: discLoss,
            generatorLoss: genLoss
          });
        }

        // Create scales
        const xScale = d3.scaleLinear()
          .domain([0, epochs - 1])
          .range([margin.left, width - margin.right]);

        const yScale = d3.scaleLinear()
          .domain([0, 1])
          .range([height - margin.bottom, margin.top]);

        // Add axes
        this.progressSvg.append('g')
          .attr('transform', `translate(0,${height - margin.bottom})`)
          .attr('class', 'x-axis')
          .style('color', '#8a9ab0')
          .call(d3.axisBottom(xScale)
            .ticks(5)
            .tickFormat(d3.format('d')));

        this.progressSvg.append('g')
          .attr('transform', `translate(${margin.left},0)`)
          .attr('class', 'y-axis')
          .style('color', '#8a9ab0')
          .call(d3.axisLeft(yScale)
            .ticks(5)
            .tickFormat(d3.format(".1f")));

        // Add axis labels
        this.progressSvg.append('text')
          .attr('x', width / 2)
          .attr('y', height - 5)
          .attr('text-anchor', 'middle')
          .style('fill', '#8a9ab0')
          .text('Epochs');

        this.progressSvg.append('text')
          .attr('transform', 'rotate(-90)')
          .attr('x', -height / 2)
          .attr('y', 15)
          .attr('text-anchor', 'middle')
          .style('fill', '#8a9ab0')
          .text('Loss');

        // Create line generators
        const discriminatorLine = d3.line<{ epoch: number, discriminatorLoss: number }>()
          .x(d => xScale(d.epoch))
          .y(d => yScale(d.discriminatorLoss))
          .curve(d3.curveMonotoneX);

        const generatorLine = d3.line<{ epoch: number, generatorLoss: number }>()
          .x(d => xScale(d.epoch))
          .y(d => yScale(d.generatorLoss))
          .curve(d3.curveMonotoneX);

        // Add discriminator loss line
        this.progressSvg.append('path')
          .datum(this.lossHistory)
          .attr('class', 'discriminator-line')
          .attr('fill', 'none')
          .attr('stroke', '#4285f4')
          .attr('stroke-width', 2)
          .attr('d', discriminatorLine);

        // Add generator loss line
        this.progressSvg.append('path')
          .datum(this.lossHistory)
          .attr('class', 'generator-line')
          .attr('fill', 'none')
          .attr('stroke', '#7c4dff')
          .attr('stroke-width', 2)
          .attr('d', generatorLine);

        // Add legend
        const legend = this.progressSvg.append('g')
          .attr('transform', `translate(${width - margin.right + 10}, ${margin.top + 10})`);

        // Discriminator legend
        legend.append('line')
          .attr('x1', 0)
          .attr('y1', 0)
          .attr('x2', 20)
          .attr('y2', 0)
          .style('stroke', '#4285f4')
          .style('stroke-width', 2);

        legend.append('text')
          .attr('x', 25)
          .attr('y', 4)
          .style('font-size', '12px')
          .style('fill', '#e1e7f5')
          .text('D Loss');

        // Generator legend
        legend.append('line')
          .attr('x1', 0)
          .attr('y1', 20)
          .attr('x2', 20)
          .attr('y2', 20)
          .style('stroke', '#7c4dff')
          .style('stroke-width', 2);

        legend.append('text')
          .attr('x', 25)
          .attr('y', 24)
          .style('font-size', '12px')
          .style('fill', '#e1e7f5')
          .text('G Loss');

        // Add current position marker
        this.progressSvg.append('line')
          .attr('class', 'position-marker')
          .attr('x1', xScale(0))
          .attr('y1', margin.top)
          .attr('x2', xScale(0))
          .attr('y2', height - margin.bottom)
          .attr('stroke', '#ffffff')
          .attr('stroke-width', 1)
          .attr('stroke-dasharray', '4,4');

        console.log('Created enhanced training progress visualization');
      }

      // Create a simple 3D scene if needed
      if (this.ganSimulationRef && this.ganSimulationRef.nativeElement && !this.scene) {
        const container = this.ganSimulationRef.nativeElement;

        // Clear any existing content
        while (container.firstChild) {
          container.removeChild(container.firstChild);
        }

        // Create a simple scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color('#0c1428');

        this.camera = new THREE.PerspectiveCamera(
          60,
          container.clientWidth / container.clientHeight,
          0.1,
          100
        );
        this.camera.position.z = 5;

        this.renderer = new THREE.WebGLRenderer({
          antialias: false,
          powerPreference: 'high-performance',
          alpha: false
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(this.renderer.domElement);

        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        // Add directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);

        // Create a proper GAN visualization instead of just a cube
        this.createGanNetworkVisualization();

        // Add animation
        const animate = () => {
          if (!this.isAnimationActive) return;

          requestAnimationFrame(animate);

          // Gentle rotation of the entire scene for better 3D perception
          if (this.scene.children.length > 0) {
            this.scene.rotation.y += 0.002;
          }

          this.renderer.render(this.scene, this.camera);
        };

        animate();

        console.log('Created 3D scene with GAN network visualization');
      }

      console.log('Created placeholder visualizations');

      // Initialize random data for visualization updates
      if (this.realSamples.length === 0 && this.fakeSamples.length === 0) {
        this.realSamples = [];
        this.fakeSamples = [];

        for (let i = 0; i < 20; i++) {
          this.realSamples.push({
            x: 0.3 + Math.random() * 0.4,
            y: 0.3 + Math.random() * 0.4
          });

          this.fakeSamples.push({
            x: Math.random(),
            y: Math.random()
          });
        }
      }

      // Set visualization ready flag to true after creating placeholders
      this.visualizationReady = true;

      // If there was a pending step, apply it now
      if (this.pendingStep !== null) {
        setTimeout(() => {
          if (this.pendingStep !== null) {
            this.updateDistributionVisualization(this.pendingStep);
            this.pendingStep = null;
          }
        }, 100);
      }
    } catch (error) {
      console.error('Error creating placeholder visualizations:', error);

      // Even if there's an error, set visualization ready to true
      this.visualizationReady = true;
    }
  }
  // Create a proper GAN network visualization
  private createGanNetworkVisualization(): void {
    try {
      // Create a group to hold our GAN model
      const ganGroup = new THREE.Group();

      // Generator Network (Purple)
      const generatorGroup = this.createNetworkModel(
        '#7c4dff', // Purple color for generator
        -2,        // Position on left side
        3,         // Number of layers
        [4, 6, 8]  // Neurons per layer
      );
      ganGroup.add(generatorGroup);

      // Discriminator Network (Blue)
      const discriminatorGroup = this.createNetworkModel(
        '#4285f4', // Blue color for discriminator
        2,         // Position on right side
        3,         // Number of layers
        [8, 6, 1]  // Neurons per layer (output is single neuron)
      );
      ganGroup.add(discriminatorGroup);

      // Add connections between the networks
      this.createConnections(ganGroup);

      // Add labels
      this.addNetworkLabels(ganGroup);

      // Add to scene
      this.scene.add(ganGroup);
    } catch (error) {
      console.error('Error creating GAN visualization:', error);
      // Fallback to simple representation if complex one fails
      this.createSimpleGanRepresentation();
    }
  }

  // Create a simple representation if the detailed one fails
  private createSimpleGanRepresentation(): void {
    // Generator block (left, purple)
    const generatorGeometry = new THREE.BoxGeometry(1.5, 1.5, 0.5);
    const generatorMaterial = new THREE.MeshPhongMaterial({ color: '#7c4dff' });
    const generator = new THREE.Mesh(generatorGeometry, generatorMaterial);
    generator.position.set(-2, 0, 0);
    this.scene.add(generator);

    // Discriminator block (right, blue)
    const discriminatorGeometry = new THREE.BoxGeometry(1.5, 1.5, 0.5);
    const discriminatorMaterial = new THREE.MeshPhongMaterial({ color: '#4285f4' });
    const discriminator = new THREE.Mesh(discriminatorGeometry, discriminatorMaterial);
    discriminator.position.set(2, 0, 0);
    this.scene.add(discriminator);

    // Connection line
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
    const points = [];
    points.push(new THREE.Vector3(-1, 0, 0));
    points.push(new THREE.Vector3(1, 0, 0));

    const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(lineGeometry, lineMaterial);
    this.scene.add(line);
  }

  // Create a neural network model
  private createNetworkModel(color: string, xPosition: number, numLayers: number, neuronsPerLayer: number[]): THREE.Group {
    const networkGroup = new THREE.Group();
    const layerSpacing = 0.5;
    const neuronSpacing = 0.3;
    const neuronRadius = 0.08;
    const layerWidth = 0.3;

    // Create layers
    for (let i = 0; i < numLayers; i++) {
      // Create layer box
      const layerHeight = Math.max(neuronsPerLayer[i] * neuronSpacing, 1);
      const layerGeometry = new THREE.BoxGeometry(layerWidth, layerHeight, 0.2);
      const layerMaterial = new THREE.MeshPhongMaterial({
        color: color,
        transparent: true,
        opacity: 0.5
      });

      const layer = new THREE.Mesh(layerGeometry, layerMaterial);
      layer.position.set(xPosition, 0, -i * layerSpacing);
      networkGroup.add(layer);

      // Create neurons for this layer
      const numNeurons = neuronsPerLayer[i];
      const totalHeight = (numNeurons - 1) * neuronSpacing;

      for (let j = 0; j < numNeurons; j++) {
        const neuronGeometry = new THREE.SphereGeometry(neuronRadius, 8, 8);
        const neuronMaterial = new THREE.MeshPhongMaterial({
          color: 0xffffff,
          emissive: color,
          emissiveIntensity: 0.3
        });

        const neuron = new THREE.Mesh(neuronGeometry, neuronMaterial);
        // Position neurons evenly, centered vertically
        const yPos = (j * neuronSpacing) - (totalHeight / 2);
        neuron.position.set(xPosition, yPos, -i * layerSpacing);
        networkGroup.add(neuron);

        // Add connections to next layer's neurons if not the last layer
        if (i < numLayers - 1) {
          const nextNumNeurons = neuronsPerLayer[i + 1];
          const nextTotalHeight = (nextNumNeurons - 1) * neuronSpacing;

          // Only create some connections to avoid visual clutter
          const numConnections = Math.min(2, nextNumNeurons);
          for (let k = 0; k < numConnections; k++) {
            const connectionIndex = Math.floor(k * nextNumNeurons / numConnections);
            const nextYPos = (connectionIndex * neuronSpacing) - (nextTotalHeight / 2);

            const lineMaterial = new THREE.LineBasicMaterial({
              color: 0x8f8f8f,
              transparent: true,
              opacity: 0.3
            });

            const points = [];
            points.push(new THREE.Vector3(xPosition, yPos, -i * layerSpacing));
            points.push(new THREE.Vector3(xPosition, nextYPos, -(i + 1) * layerSpacing));

            const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(lineGeometry, lineMaterial);
            networkGroup.add(line);
          }
        }
      }
    }

    return networkGroup;
  }

  // Create connections between generator and discriminator
  private createConnections(ganGroup: THREE.Group): void {
    // Noise input to Generator
    const noiseGeometry = new THREE.BoxGeometry(0.4, 0.4, 0.4);
    const noiseMaterial = new THREE.MeshPhongMaterial({
      color: 0xffffff,
      wireframe: true
    });
    const noise = new THREE.Mesh(noiseGeometry, noiseMaterial);
    noise.position.set(-3.5, 0, 0);
    ganGroup.add(noise);

    // Connection from noise to generator
    const noiseToGenMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
    const noiseToGenPoints = [
      new THREE.Vector3(-3.3, 0, 0),
      new THREE.Vector3(-2.5, 0, 0)
    ];
    const noiseToGenGeometry = new THREE.BufferGeometry().setFromPoints(noiseToGenPoints);
    const noiseToGenLine = new THREE.Line(noiseToGenGeometry, noiseToGenMaterial);
    ganGroup.add(noiseToGenLine);

    // Connection from generator to discriminator
    const genToDiscMaterial = new THREE.LineBasicMaterial({ color: 0x7c4dff });
    const genToDiscPoints = [
      new THREE.Vector3(-1.5, 0, 0),
      new THREE.Vector3(0, -0.5, 0),
      new THREE.Vector3(1.5, -0.5, 0)
    ];
    const genToDiscGeometry = new THREE.BufferGeometry().setFromPoints(genToDiscPoints);
    const genToDiscLine = new THREE.Line(genToDiscGeometry, genToDiscMaterial);
    ganGroup.add(genToDiscLine);

    // Real data
    const realDataGeometry = new THREE.CircleGeometry(0.3, 16);
    const realDataMaterial = new THREE.MeshPhongMaterial({ color: 0x24b47e });
    const realData = new THREE.Mesh(realDataGeometry, realDataMaterial);
    realData.position.set(0, 0.8, 0);
    realData.rotation.y = Math.PI / 2; // Make it face forward
    ganGroup.add(realData);

    // Connection from real data to discriminator
    const realToDiscMaterial = new THREE.LineBasicMaterial({ color: 0x24b47e });
    const realToDiscPoints = [
      new THREE.Vector3(0, 0.8, 0),
      new THREE.Vector3(1.5, 0.5, 0)
    ];
    const realToDiscGeometry = new THREE.BufferGeometry().setFromPoints(realToDiscPoints);
    const realToDiscLine = new THREE.Line(realToDiscGeometry, realToDiscMaterial);
    ganGroup.add(realToDiscLine);

    // Discriminator output
    const outputGeometry = new THREE.BoxGeometry(0.3, 0.3, 0.1);
    const outputMaterial = new THREE.MeshPhongMaterial({ color: 0xff9d45 });
    const output = new THREE.Mesh(outputGeometry, outputMaterial);
    output.position.set(3.5, 0, 0);
    ganGroup.add(output);

    // Connection from discriminator to output
    const discToOutputMaterial = new THREE.LineBasicMaterial({ color: 0x4285f4 });
    const discToOutputPoints = [
      new THREE.Vector3(2.5, 0, 0),
      new THREE.Vector3(3.3, 0, 0)
    ];
    const discToOutputGeometry = new THREE.BufferGeometry().setFromPoints(discToOutputPoints);
    const discToOutputLine = new THREE.Line(discToOutputGeometry, discToOutputMaterial);
    ganGroup.add(discToOutputLine);
  }

  // Add text labels to the network (using sprites since we can't directly add HTML to Three.js)
  private addNetworkLabels(ganGroup: THREE.Group): void {
    // Create sprite for Generator label
    const makeTextSprite = (text: string, x: number, y: number, z: number, color: string = '#ffffff'): void => {
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 128;

      const context = canvas.getContext('2d');
      if (context) {
        context.fillStyle = 'rgba(0,0,0,0)';
        context.fillRect(0, 0, canvas.width, canvas.height);

        context.font = 'Bold 24px Arial';
        context.fillStyle = color;
        context.textAlign = 'center';
        context.fillText(text, canvas.width / 2, canvas.height / 2);

        const texture = new THREE.Texture(canvas);
        texture.needsUpdate = true;

        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.position.set(x, y, z);
        sprite.scale.set(2, 1, 1);
        ganGroup.add(sprite);
      }
    };

    // Add labels for main components
    makeTextSprite("Generator", -2, -1.5, 0, '#7c4dff');
    makeTextSprite("Discriminator", 2, -1.5, 0, '#4285f4');
    makeTextSprite("Real Data", 0, 1.2, 0, '#24b47e');
    makeTextSprite("Noise", -3.5, -0.8, 0, '#ffffff');
    makeTextSprite("Real/Fake", 3.5, -0.5, 0, '#ff9d45');
  }
}