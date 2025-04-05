import { Component, ElementRef, OnInit, OnDestroy, ViewChild, AfterViewInit, HostListener } from '@angular/core';
import { NgFor, NgIf, NgClass } from '@angular/common';
import * as THREE from 'three';
import * as d3 from 'd3';
import { RnnVisualizationGuideComponent } from './rnn-simulation-note.component';
import { EquationDisplayComponent } from './equation-display.component';

// Angular 19 compatible import for OrbitControls
// Make sure to install @types/three package if not already installed
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

@Component({
  selector: 'app-rnn-simulation',
  templateUrl: './rnn-simulation.component.html',
  styleUrls: ['./rnn-simulation.component.scss'],
  imports: [NgFor, NgIf, NgClass, RnnVisualizationGuideComponent, EquationDisplayComponent],
  standalone: true
})
export class RnnSimulationComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('threeCanvas') private canvasRef!: ElementRef;
  @ViewChild('hiddenStateChart') private hiddenStateChartRef!: ElementRef;
  @ViewChild('outputChart') private outputChartRef!: ElementRef;
  @ViewChild('gradientChart') private gradientChartRef!: ElementRef;

  // Three.js properties
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private animationFrameId!: number;
  private clock = new THREE.Clock();

  // RNN model properties
  private hiddenUnits = 5;
  private inputUnits = 3;
  private outputUnits = 3;
  private timeSteps = 5;
  private neurons: THREE.Group[] = [];
  private connections: THREE.Line[] = [];
  private hiddenStates: number[][] = [];
  private outputs: number[][] = [];
  private gradients: number[][] = [];

  // Animation control
  public isPlaying = false;
  public currentTimeStep = 0;
  public animationSpeed = 1.0;
  public selectedView = 'unrolled'; // 'compact', 'unrolled', 'gradient'

  // Learning simulation
  public learningRate = 0.01;
  public epoch = 0;
  public maxEpochs = 100;
  public loss = 1.0;
  public inputSequence = 'Hello';
  public outputSequence = '';
  public showVanishingGradient = false;

  // Interface controls
  public activeTab = 'concept';
  public viewOptions = [
    { id: 'compact', label: 'Compact View' },
    { id: 'unrolled', label: 'Unrolled View' },
    { id: 'gradient', label: 'Gradient Flow' }
  ];
  public simulationModes = [
    { id: 'text-gen', label: 'Text Generation' },
    { id: 'time-series', label: 'Time Series Prediction' }
  ];
  public selectedMode = 'text-gen';
  
  // Tooltips
  public tooltips = {
    hiddenState: 'The hidden state vector acts as the network\'s memory, storing information from previous time steps.',
    outputLayer: 'The output layer produces predictions based on the current hidden state.',
    weights: 'The same weights are reused at each time step (parameter sharing).',
    vanishingGradient: 'As gradients flow backward through time, they can diminish exponentially, making it difficult for the network to learn long-range dependencies.'
  };

  constructor() {}

  ngOnInit(): void {
    // Initialize simulation data
    this.initializeHiddenStates();
    this.initializeOutputs();
    this.initializeGradients();
  }

  ngAfterViewInit(): void {
    this.initThreeJS();
    this.createRNNModel();
    this.updateView();
    this.animate();
    this.renderHiddenStateChart();
    this.renderOutputChart();
    this.renderGradientChart();
  }

  ngOnDestroy(): void {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    // Clean up Three.js resources
    this.disposeThreeJSResources();
  }

  @HostListener('window:resize')
  onWindowResize(): void {
    if (this.camera && this.renderer) {
      const canvas = this.canvasRef.nativeElement;
      this.camera.aspect = canvas.clientWidth / canvas.clientHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    }
    // Redraw D3 charts
    this.renderHiddenStateChart();
    this.renderOutputChart();
    this.renderGradientChart();
  }

  private initThreeJS(): void {
    const canvas = this.canvasRef.nativeElement;
    
    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#0c1428'); // Darkest Blue from design system
    
    // Camera setup
    this.camera = new THREE.PerspectiveCamera(
      75, 
      canvas.clientWidth / canvas.clientHeight, 
      0.1, 
      1000
    );
    this.camera.position.z = 15;
    
    // Renderer setup
    this.renderer = new THREE.WebGLRenderer({ 
      canvas, 
      antialias: true,
      alpha: true 
    });
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    
    // Controls setup
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight('#8a9ab0', 0.5); // Muted Blue
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight('#ffffff', 1);
    directionalLight.position.set(0, 10, 10);
    this.scene.add(directionalLight);
  }

  private createRNNModel(): void {
    // Clear existing model
    this.neurons = [];
    this.connections = [];
    
    this.scene.clear();
    
    // Add lights
    const ambientLight = new THREE.AmbientLight('#8a9ab0', 0.5);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight('#ffffff', 1);
    directionalLight.position.set(0, 10, 10);
    this.scene.add(directionalLight);
    
    // Create neurons for each layer and time step
    this.createNeurons();
    
    // Create connections between neurons
    this.createConnections();
  }

  private createNeurons(): void {
    // Materials
    const inputMaterial = new THREE.MeshPhongMaterial({ color: '#4285f4' }); // Primary Blue
    const hiddenMaterial = new THREE.MeshPhongMaterial({ color: '#7c4dff' }); // Purple
    const outputMaterial = new THREE.MeshPhongMaterial({ color: '#00c9ff' }); // Cyan
    
    // Geometry
    const neuronGeometry = new THREE.SphereGeometry(0.3, 16, 16);
    
    // Create neurons for each time step
    for (let t = 0; t < this.timeSteps; t++) {
      const timeStepGroup = new THREE.Group();
      timeStepGroup.name = `timeStep_${t}`;
      
      // Input layer
      for (let i = 0; i < this.inputUnits; i++) {
        const neuron = new THREE.Mesh(neuronGeometry, inputMaterial);
        neuron.position.set(t * 5, i * 1.5 - (this.inputUnits - 1) * 0.75, -3);
        neuron.name = `input_${t}_${i}`;
        timeStepGroup.add(neuron);
      }
      
      // Hidden layer
      for (let h = 0; h < this.hiddenUnits; h++) {
        const neuron = new THREE.Mesh(neuronGeometry, hiddenMaterial);
        neuron.position.set(t * 5, h * 1.5 - (this.hiddenUnits - 1) * 0.75, 0);
        neuron.name = `hidden_${t}_${h}`;
        timeStepGroup.add(neuron);
      }
      
      // Output layer
      for (let o = 0; o < this.outputUnits; o++) {
        const neuron = new THREE.Mesh(neuronGeometry, outputMaterial);
        neuron.position.set(t * 5, o * 1.5 - (this.outputUnits - 1) * 0.75, 3);
        neuron.name = `output_${t}_${o}`;
        timeStepGroup.add(neuron);
      }
      
      this.scene.add(timeStepGroup);
      this.neurons.push(timeStepGroup);
    }
  }

  private createConnections(): void {
    // Material for connections
    const feedforwardMaterial = new THREE.LineBasicMaterial({ color: '#8bb4fa', transparent: true, opacity: 0.5 }); // Light Blue
    const recurrentMaterial = new THREE.LineBasicMaterial({ color: '#ff9d45', transparent: true, opacity: 0.7 }); // Orange
    
    for (let t = 0; t < this.timeSteps; t++) {
      // Input to hidden connections
      for (let i = 0; i < this.inputUnits; i++) {
        for (let h = 0; h < this.hiddenUnits; h++) {
          const inputNeuron = this.getNeuronByName(`input_${t}_${i}`) as THREE.Mesh;
          const hiddenNeuron = this.getNeuronByName(`hidden_${t}_${h}`) as THREE.Mesh;
          
          if (inputNeuron && hiddenNeuron) {
            const points = [
              inputNeuron.position.clone(),
              hiddenNeuron.position.clone()
            ];
            
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, feedforwardMaterial);
            line.name = `conn_input_hidden_${t}_${i}_${h}`;
            this.scene.add(line);
            this.connections.push(line);
          }
        }
      }
      
      // Hidden to output connections
      for (let h = 0; h < this.hiddenUnits; h++) {
        for (let o = 0; o < this.outputUnits; o++) {
          const hiddenNeuron = this.getNeuronByName(`hidden_${t}_${h}`) as THREE.Mesh;
          const outputNeuron = this.getNeuronByName(`output_${t}_${o}`) as THREE.Mesh;
          
          if (hiddenNeuron && outputNeuron) {
            const points = [
              hiddenNeuron.position.clone(),
              outputNeuron.position.clone()
            ];
            
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, feedforwardMaterial);
            line.name = `conn_hidden_output_${t}_${h}_${o}`;
            this.scene.add(line);
            this.connections.push(line);
          }
        }
      }
      
      // Recurrent connections (t to t+1)
      if (t < this.timeSteps - 1) {
        for (let h1 = 0; h1 < this.hiddenUnits; h1++) {
          for (let h2 = 0; h2 < this.hiddenUnits; h2++) {
            const hiddenNeuron1 = this.getNeuronByName(`hidden_${t}_${h1}`) as THREE.Mesh;
            const hiddenNeuron2 = this.getNeuronByName(`hidden_${t+1}_${h2}`) as THREE.Mesh;
            
            if (hiddenNeuron1 && hiddenNeuron2) {
              // Create a curved path for recurrent connections
              const startPoint = hiddenNeuron1.position.clone();
              const endPoint = hiddenNeuron2.position.clone();
              const controlPoint = new THREE.Vector3(
                (startPoint.x + endPoint.x) / 2,
                (startPoint.y + endPoint.y) / 2,
                startPoint.z + 1.5
              );
              
              const curve = new THREE.QuadraticBezierCurve3(startPoint, controlPoint, endPoint);
              const points = curve.getPoints(20);
              const geometry = new THREE.BufferGeometry().setFromPoints(points);
              const line = new THREE.Line(geometry, recurrentMaterial);
              line.name = `conn_recurrent_${t}_${t+1}_${h1}_${h2}`;
              this.scene.add(line);
              this.connections.push(line);
            }
          }
        }
      }
    }
  }

  private getNeuronByName(name: string): THREE.Object3D | null {
    for (const group of this.neurons) {
      const neuron = group.children.find(child => child.name === name);
      if (neuron) {
        return neuron;
      }
    }
    return null;
  }

  private animate(): void {
    this.animationFrameId = requestAnimationFrame(() => this.animate());
    
    const delta = this.clock.getDelta();
    
    if (this.isPlaying) {
      // Move animation forward
      this.animateTimeStep(delta);
    }
    
    // Update controls
    this.controls.update();
    
    // Render the scene
    this.renderer.render(this.scene, this.camera);
  }

  private animateTimeStep(delta: number): void {
    // Advance the animation based on speed
    this.currentTimeStep += delta * this.animationSpeed;
    
    // Loop back to the beginning when finished
    if (this.currentTimeStep >= this.timeSteps) {
      this.currentTimeStep = 0;
      
      // Increment epoch for learning simulation
      if (this.epoch < this.maxEpochs) {
        this.epoch++;
        this.loss = Math.max(0.1, this.loss * 0.9); // Simulate decreasing loss
        
        // Generate output sequence
        if (this.selectedMode === 'text-gen') {
          this.outputSequence = this.simulateTextGeneration();
        }
      }
    }
    
    // Update visualization to show current time step
    this.updateNeuronActivations();
    this.updateConnectionStrengths();
    this.animateDataFlow(delta);
    this.renderHiddenStateChart();
    this.renderOutputChart();
    this.renderGradientChart();
  }

  private animateDataFlow(delta: number): void {
    const timeStep = Math.floor(this.currentTimeStep);
    const timeFraction = this.currentTimeStep - timeStep;
    
    // Animate data flowing through connections
    this.connections.forEach(connection => {
      const material = connection.material as THREE.LineBasicMaterial;
      const name = connection.name;
      
      if (name.includes(`_${timeStep}_`)) {
        // Active connections in the current time step
        const pulse = 0.7 + 0.3 * Math.sin(this.clock.getElapsedTime() * 5);
        material.opacity = pulse;
      } else if (name.includes('recurrent') && 
                name.includes(`_${timeStep-1}_${timeStep}_`)) {
        // Active recurrent connections between previous and current time step
        material.opacity = 0.9;
        material.color.set('#ff9d45'); // Bright orange
      } else if (name.includes('recurrent')) {
        // Other recurrent connections
        material.opacity = 0.4;
      }
    });
  }

  private updateNeuronActivations(): void {
    const timeStep = Math.floor(this.currentTimeStep);
    const fraction = this.currentTimeStep - timeStep;
    
    // Update neuron colors and sizes based on activation
    for (let t = 0; t < this.timeSteps; t++) {
      const group = this.neurons[t];
      
      // Emphasize current time step and fade others
      const opacity = t === timeStep ? 1.0 : 0.3;
      
      group.children.forEach(child => {
        const mesh = child as THREE.Mesh;
        const material = mesh.material as THREE.MeshPhongMaterial;
        
        material.opacity = opacity;
        material.transparent = opacity < 1.0;
        
        if (child.name.startsWith('hidden_')) {
          const hiddenIndex = parseInt(child.name.split('_')[2]);
          if (t <= timeStep) {
            // Get activation value from stored hidden states
            const activation = this.hiddenStates[t][hiddenIndex];
            
            // Scale for visualization (between 0.2 and 1.0)
            const scale = 0.2 + activation * 0.8;
            mesh.scale.set(scale, scale, scale);
            
            // Adjust emissive intensity based on activation
            material.emissive.set('#7c4dff');
            material.emissiveIntensity = activation * 0.5;
          }
        } else if (child.name.startsWith('output_')) {
          const outputIndex = parseInt(child.name.split('_')[2]);
          if (t <= timeStep) {
            // Get activation value from stored outputs
            const activation = this.outputs[t][outputIndex];
            
            // Scale for visualization
            const scale = 0.2 + activation * 0.8;
            mesh.scale.set(scale, scale, scale);
            
            // Adjust emissive intensity based on activation
            material.emissive.set('#00c9ff');
            material.emissiveIntensity = activation * 0.5;
          }
        }
      });
    }
  }

  private updateConnectionStrengths(): void {
    const timeStep = Math.floor(this.currentTimeStep);
    
    // Update connection opacity and thickness based on weights and gradient flow
    for (const connection of this.connections) {
      const name = connection.name;
      
      if (name.includes(`_${timeStep}_`)) {
        // Forward pass connection
        connection.visible = true;
        (connection.material as THREE.LineBasicMaterial).opacity = 0.8;
      } else if (this.selectedView === 'gradient' && name.includes('recurrent')) {
        // Show gradient flow in recurrent connections
        const parts = name.split('_');
        const fromTimeStep = parseInt(parts[2]);
        const toTimeStep = parseInt(parts[3]);
        
        if (fromTimeStep < timeStep && toTimeStep <= timeStep) {
          connection.visible = true;
          
          // Visualize vanishing gradient if enabled
          if (this.showVanishingGradient) {
            const distance = timeStep - fromTimeStep;
            // Exponential decay for vanishing gradient
            const opacity = Math.exp(-distance * 0.7);
            (connection.material as THREE.LineBasicMaterial).opacity = opacity;
          } else {
            (connection.material as THREE.LineBasicMaterial).opacity = 0.7;
          }
        } else {
          connection.visible = false;
        }
      } else if (name.includes('recurrent')) {
        // Recurrent connections
        const parts = name.split('_');
        const fromTimeStep = parseInt(parts[2]);
        const toTimeStep = parseInt(parts[3]);
        
        // Show only current recurrent connections
        connection.visible = (fromTimeStep === timeStep || fromTimeStep === timeStep - 1);
      } else {
        // Hide other connections
        connection.visible = false;
      }
    }
  }

  private updateView(): void {
    switch (this.selectedView) {
      case 'compact':
        this.setupCompactView();
        break;
      case 'unrolled':
        this.setupUnrolledView();
        break;
      case 'gradient':
        this.setupGradientView();
        break;
    }
  }

  private setupCompactView(): void {
    // Reset camera position
    this.camera.position.set(0, 0, 15);
    this.controls.update();
    
    // Show only one time step
    for (let t = 0; t < this.timeSteps; t++) {
      const group = this.neurons[t];
      group.visible = t === 0;
      
      // Position the visible group at the center
      if (t === 0) {
        group.position.set(0, 0, 0);
      }
    }
    
    // Show only connections for the visible time step
    for (const connection of this.connections) {
      const name = connection.name;
      connection.visible = name.includes('_0_');
    }
    
    // Add self-loop for recurrent connection visualization
    this.addRecurrentSelfLoop();
  }

  private setupUnrolledView(): void {
    // Calculate center of the model
    const centerX = ((this.timeSteps - 1) * 5) / 2;
    
    // Reset camera position to see the whole unrolled network
    this.camera.position.set(centerX, 0, 18);
    this.camera.lookAt(new THREE.Vector3(centerX, 0, 0));
    this.controls.update();
    
    // Show all time steps
    for (let t = 0; t < this.timeSteps; t++) {
      const group = this.neurons[t];
      group.visible = true;
      group.position.set(0, 0, 0);
    }
    
    // Show all connections
    for (const connection of this.connections) {
      connection.visible = true;
    }
    
    // Remove self-loop if it exists
    this.removeRecurrentSelfLoop();
    
    // Make sure the model fits in the view
    this.fitCameraToModel();
  }
  
  private fitCameraToModel(): void {
    // Create a bounding box encompassing all neurons
    const bbox = new THREE.Box3();
    
    // Add all visible objects to the bounding box
    this.neurons.forEach(group => {
      if (group.visible) {
        bbox.expandByObject(group);
      }
    });
    
    if (bbox.isEmpty()) return;
    
    // Get the size of the bounding box
    const size = new THREE.Vector3();
    bbox.getSize(size);
    
    // Get the center of the bounding box
    const center = new THREE.Vector3();
    bbox.getCenter(center);
    
    // Calculate the required camera distance to fit the entire model
    const fov = this.camera.fov * (Math.PI / 180);
    const maxDim = Math.max(size.x, size.y);
    const cameraZ = Math.max(15, (maxDim / 2) / Math.tan(fov / 2) * 1.2); // Minimum distance of 15, plus 20% margin
    
    // Update camera position
    this.camera.position.z = cameraZ;
    this.camera.lookAt(center);
    this.controls.target.copy(center);
    this.controls.update();
  }

  private setupGradientView(): void {
    // Similar to unrolled but with gradient visualization
    this.setupUnrolledView();
    
    // Enable vanishing gradient visualization
    this.showVanishingGradient = true;
  }

  private addRecurrentSelfLoop(): void {
    // Check if self-loop already exists
    if (this.scene.getObjectByName('recurrent_self_loop')) {
      return;
    }
    
    // Create a circular arrow to represent recurrent connection
    const curve = new THREE.EllipseCurve(
      0, 0,             // center
      1.5, 1,           // xRadius, yRadius
      Math.PI, 0,       // start angle, end angle
      true              // clockwise
    );
    
    const points = curve.getPoints(50);
    const geometry = new THREE.BufferGeometry().setFromPoints(
      points.map(p => new THREE.Vector3(p.x, p.y, 0))
    );
    
    const material = new THREE.LineBasicMaterial({ color: '#ff9d45', linewidth: 2 });
    const ellipse = new THREE.Line(geometry, material);
    ellipse.name = 'recurrent_self_loop';
    ellipse.position.set(1, 0, 0);
    ellipse.rotation.z = Math.PI / 2;
    
    this.scene.add(ellipse);
  }

  private removeRecurrentSelfLoop(): void {
    const selfLoop = this.scene.getObjectByName('recurrent_self_loop');
    if (selfLoop) {
      this.scene.remove(selfLoop);
    }
  }

  private initializeHiddenStates(): void {
    this.hiddenStates = [];
    
    for (let t = 0; t < this.timeSteps; t++) {
      const hiddenState = Array(this.hiddenUnits).fill(0).map(() => Math.random());
      // Normalize values between 0 and 1
      const sum = hiddenState.reduce((a, b) => a + b, 0);
      const normalizedState = hiddenState.map(v => v / sum);
      this.hiddenStates.push(normalizedState);
    }
  }

  private initializeOutputs(): void {
    this.outputs = [];
    
    for (let t = 0; t < this.timeSteps; t++) {
      const output = Array(this.outputUnits).fill(0).map(() => Math.random());
      // Apply softmax
      const expValues = output.map(v => Math.exp(v));
      const sum = expValues.reduce((a, b) => a + b, 0);
      const softmaxOutput = expValues.map(v => v / sum);
      this.outputs.push(softmaxOutput);
    }
  }

  private initializeGradients(): void {
    this.gradients = [];
    
    for (let t = 0; t < this.timeSteps; t++) {
      // Simulate vanishing gradient effect - gradients get smaller as we go back in time
      const factor = Math.exp(-(this.timeSteps - t - 1) * 0.5);
      const gradient = Array(this.hiddenUnits).fill(0).map(() => Math.random() * factor);
      this.gradients.push(gradient);
    }
  }

  private disposeThreeJSResources(): void {
    // Dispose of geometries, materials, textures, etc.
    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        if (object.geometry) {
          object.geometry.dispose();
        }
        
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach(material => material.dispose());
          } else {
            object.material.dispose();
          }
        }
      }
    });
    
    // Dispose of the renderer
    this.renderer.dispose();
  }

  private renderHiddenStateChart(): void {
    if (!this.hiddenStateChartRef) return;
    
    const container = this.hiddenStateChartRef.nativeElement;
    container.innerHTML = '';
    
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = container.clientHeight - margin.top - margin.bottom;
    
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Current time step rounded down
    const timeStep = Math.floor(this.currentTimeStep);
    
    if (timeStep >= this.hiddenStates.length) return;
    
    const data = this.hiddenStates[timeStep].map((value, index) => ({
      unit: index,
      value
    }));
    
    const x = d3.scaleBand()
      .domain(data.map(d => d.unit.toString()))
      .range([0, width])
      .padding(0.3);
    
    const y = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);
    
    svg.selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => x(d.unit.toString()) as number)
      .attr('width', x.bandwidth())
      .attr('y', d => y(d.value))
      .attr('height', d => height - y(d.value))
      .attr('fill', '#7c4dff')
      .attr('opacity', 0.7);
    
    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).tickFormat(d => `h${d}`));
    
    svg.append('g')
      .call(d3.axisLeft(y));
    
    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 0)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e1e7f5')
      .text(`Hidden State (t=${timeStep})`);
  }

  private renderOutputChart(): void {
    if (!this.outputChartRef) return;
    
    const container = this.outputChartRef.nativeElement;
    container.innerHTML = '';
    
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = container.clientHeight - margin.top - margin.bottom;
    
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Current time step rounded down
    const timeStep = Math.floor(this.currentTimeStep);
    
    if (timeStep >= this.outputs.length) return;
    
    const data = this.outputs[timeStep].map((value, index) => ({
      unit: index,
      value
    }));
    
    const x = d3.scaleBand()
      .domain(data.map(d => d.unit.toString()))
      .range([0, width])
      .padding(0.3);
    
    const y = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);
    
    svg.selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => x(d.unit.toString()) as number)
      .attr('width', x.bandwidth())
      .attr('y', d => y(d.value))
      .attr('height', d => height - y(d.value))
      .attr('fill', '#00c9ff')
      .attr('opacity', 0.7);
    
    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).tickFormat(d => `y${d}`));
    
    svg.append('g')
      .call(d3.axisLeft(y));
    
    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 0)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e1e7f5')
      .text(`Output Probabilities (t=${timeStep})`);
  }

  private renderGradientChart(): void {
    if (!this.gradientChartRef || this.selectedView !== 'gradient') return;
    
    const container = this.gradientChartRef.nativeElement;
    container.innerHTML = '';
    
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = container.clientHeight - margin.top - margin.bottom;
    
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Create data for line chart showing gradient magnitudes over time
    const data = [];
    for (let t = 0; t < this.timeSteps; t++) {
      // Calculate average gradient magnitude for this time step
      const avgGradient = this.gradients[t].reduce((sum, val) => sum + val, 0) / this.gradients[t].length;
      data.push({
        timeStep: t,
        gradient: avgGradient
      });
    }
    
    const x = d3.scaleLinear()
      .domain([0, this.timeSteps - 1])
      .range([0, width]);
    
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.gradient) as number])
      .range([height, 0]);
    
    // Add line
    const line = d3.line<{timeStep: number, gradient: number}>()
      .x(d => x(d.timeStep))
      .y(d => y(d.gradient));
    
    svg.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', '#ff9d45')
      .attr('stroke-width', 2)
      .attr('d', line);
    
    // Add circles at each data point
    svg.selectAll('.dot')
      .data(data)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => x(d.timeStep))
      .attr('cy', d => y(d.gradient))
      .attr('r', 5)
      .attr('fill', '#ff9d45');
    
    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(this.timeSteps).tickFormat(d => `t-${this.timeSteps - 1 - Number(d)}`));
    
    svg.append('g')
      .call(d3.axisLeft(y));
    
    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 0)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e1e7f5')
      .text('Gradient Magnitude through Time (Vanishing Gradient Effect)');
  }

  // Simulate the RNN's text generation capability
  private simulateTextGeneration(): string {
    // This is a simplified simulation; in a real RNN, we would have a vocabulary
    // and would sample from the output probability distribution
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ';
    let output = '';
    
    // Generate a few characters based on the input
    for (let i = 0; i < 5; i++) {
      // Get a random output from our simulation
      const timeStep = Math.min(i, this.timeSteps - 1);
      const outputProbs = this.outputs[timeStep];
      
      // Pick a character based on the output probabilities
      let sum = 0;
      const r = Math.random();
      
      for (let j = 0; j < outputProbs.length; j++) {
        sum += outputProbs[j];
        if (r <= sum) {
          // Map to a character (simplified)
          const charIndex = Math.floor(j * (chars.length / outputProbs.length));
          output += chars[charIndex];
          break;
        }
      }
    }
    
    return output;
  }

  // UI control methods
  public togglePlayPause(): void {
    this.isPlaying = !this.isPlaying;
    if (this.isPlaying) {
      this.clock.start();
    } else {
      this.clock.stop();
    }
  }

  public restart(): void {
    this.currentTimeStep = 0;
    this.epoch = 0;
    this.loss = 1.0;
    this.outputSequence = '';
    this.initializeHiddenStates();
    this.initializeOutputs();
    this.initializeGradients();
    this.updateNeuronActivations();
    this.updateConnectionStrengths();
    this.renderHiddenStateChart();
    this.renderOutputChart();
    this.renderGradientChart();
  }

  public changeView(viewId: string): void {
    this.selectedView = viewId;
    this.updateView();
  }

  public stepForward(): void {
    if (this.currentTimeStep < this.timeSteps - 0.1) {
      this.currentTimeStep += 0.1;
    } else {
      this.currentTimeStep = 0;
    }
    this.updateNeuronActivations();
    this.updateConnectionStrengths();
    this.renderHiddenStateChart();
    this.renderOutputChart();
    this.renderGradientChart();
  }

  public stepBackward(): void {
    if (this.currentTimeStep > 0.1) {
      this.currentTimeStep -= 0.1;
    } else {
      this.currentTimeStep = this.timeSteps - 0.1;
    }
    this.updateNeuronActivations();
    this.updateConnectionStrengths();
    this.renderHiddenStateChart();
    this.renderOutputChart();
    this.renderGradientChart();
  }

  public changeSpeed(speed: string): void {
    this.animationSpeed = parseFloat(speed);
  }

  public changeTab(tabId: string): void {
    this.activeTab = tabId;
  }

  public changeMode(modeId: string): void {
    this.selectedMode = modeId;
    this.restart();
  }

  public toggleVanishingGradient(): void {
    this.showVanishingGradient = !this.showVanishingGradient;
    this.updateConnectionStrengths();
  }
}