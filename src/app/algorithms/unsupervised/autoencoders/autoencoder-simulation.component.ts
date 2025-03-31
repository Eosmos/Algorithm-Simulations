import { Component, OnInit, AfterViewInit, ElementRef, ViewChild, NgZone, OnDestroy } from '@angular/core';
import * as THREE from 'three';
// Defining OrbitControls as an any type as a fallback solution
// In a real project, you'd install @types/three and configure paths correctly
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import * as d3 from 'd3';

interface NeuronPosition {
  x: number;
  y: number;
  z: number;
  layerIndex: number;
  neuronIndex: number;
}

interface DigitExample {
  pixels: number[];
  label: number;
  encodedValue: [number, number];
  reconstructedPixels: number[];
}

// D3 specific types
type D3Selection = d3.Selection<d3.BaseType, unknown, null, undefined>;
type D3SVGSelection = d3.Selection<SVGSVGElement, unknown, null, undefined>;
type D3CircleSelection = d3.Selection<SVGCircleElement, DigitExample, SVGGElement, unknown>;

@Component({
  selector: 'app-autoencoder-simulation',
  templateUrl: './autoencoder-simulation.component.html',
  styleUrls: ['./autoencoder-simulation.component.scss'],
  standalone: true
})
export class AutoencoderSimulationComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('threeCanvas') threeCanvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('latentSpaceCanvas') latentSpaceCanvasRef!: ElementRef<HTMLDivElement>;
  @ViewChild('inputCanvas') inputCanvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('outputCanvas') outputCanvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('noisyCanvas') noisyCanvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('denoisedCanvas') denoisedCanvasRef!: ElementRef<HTMLCanvasElement>;
  
  // Three.js properties
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: any; // Using any type for OrbitControls to avoid TypeScript error
  
  // 3D objects
  private neurons: THREE.Mesh[] = [];
  private connections: THREE.Line[] = [];
  private neuronPositions: NeuronPosition[] = [];
  private dataParticle!: THREE.Mesh;
  private dataTrail: THREE.Mesh[] = [];
  
  // D3 properties
  private latentSpaceSvg: D3SVGSelection | null = null;
  private latentSpaceXScale: d3.ScaleLinear<number, number> | null = null;
  private latentSpaceYScale: d3.ScaleLinear<number, number> | null = null;
  private latentSpacePoints: {
    points: D3CircleSelection | null;
    selectedPoint: d3.Selection<SVGCircleElement, unknown, null, undefined> | null;
  } = { points: null, selectedPoint: null };
  
  // Animation properties
  public isPlaying = false;
  private animationFrameId = 0;
  private currentStep = 0;
  private totalSteps = 200;
  
  // Autoencoder properties
  private encoderLayers = [784, 256, 128, 64, 16, 2]; // For MNIST digits (28x28 = 784)
  private decoderLayers = [2, 16, 64, 128, 256, 784];
  private activeNeurons: Set<number> = new Set();
  
  // Example data
  private digitExamples: DigitExample[] = [];
  
  // UI state
  public selectedExample = 0;
  public noiseLevel = 0.2;
  public latentSpaceDim1 = 0;
  public latentSpaceDim2 = 0;
  public openAccordion: string | null = null;
  
  constructor(private ngZone: NgZone) {}
  
  ngOnInit(): void {
    this.generateExampleData();
  }
  
  ngAfterViewInit(): void {
    this.initThreeJs();
    this.initLatentSpaceViz();
    this.initCanvases();
    this.startAnimation();
  }
  
  private initThreeJs(): void {
    // Set up scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0c1428);
    
    // Set up camera
    const aspectRatio = this.threeCanvasRef.nativeElement.clientWidth / this.threeCanvasRef.nativeElement.clientHeight;
    this.camera = new THREE.PerspectiveCamera(60, aspectRatio, 0.1, 1000);
    this.camera.position.set(0, 4, 15);
    
    // Set up renderer
    this.renderer = new THREE.WebGLRenderer({ 
      canvas: this.threeCanvasRef.nativeElement,
      antialias: true 
    });
    this.renderer.setSize(this.threeCanvasRef.nativeElement.clientWidth, this.threeCanvasRef.nativeElement.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    
    // Set up controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 5;
    this.controls.maxDistance = 30;
    
    // Add lights
    // Ambient light for overall illumination
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    this.scene.add(ambientLight);
    
    // Directional lights from different angles for better depth
    const frontLight = new THREE.DirectionalLight(0xffffff, 0.8);
    frontLight.position.set(0, 5, 10);
    this.scene.add(frontLight);
    
    const sideLight = new THREE.DirectionalLight(0xffffff, 0.3);
    sideLight.position.set(10, 5, 0);
    this.scene.add(sideLight);
    
    const backLight = new THREE.DirectionalLight(0xffffff, 0.2);
    backLight.position.set(0, 5, -10);
    this.scene.add(backLight);
    
    // Add subtle point lights at encoder and decoder ends for emphasis
    const encoderLight = new THREE.PointLight(0x4285f4, 1, 20);
    encoderLight.position.set(-10, 3, 0);
    this.scene.add(encoderLight);
    
    const decoderLight = new THREE.PointLight(0x7c4dff, 1, 20);
    decoderLight.position.set(10, 3, 0);
    this.scene.add(decoderLight);
    
    // Create neural network visualization
    this.createNeuralNetworkVisualization();
    
    // Create data particle for animation
    const particleGeometry = new THREE.SphereGeometry(0.2, 16, 16);
    const particleMaterial = new THREE.MeshPhongMaterial({ 
      color: 0x00c9ff,
      emissive: 0x00c9ff,
      emissiveIntensity: 0.5
    });
    this.dataParticle = new THREE.Mesh(particleGeometry, particleMaterial);
    this.dataParticle.visible = false;
    this.scene.add(this.dataParticle);
    
    // Add a glow effect to the particle
    const particleGlow = new THREE.PointLight(0x00c9ff, 1, 3);
    this.dataParticle.add(particleGlow);
    
    // Add visual aids to orient the viewer
    // Network labels
    const createLabel = (text: string, position: THREE.Vector3, color: string) => {
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 64;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'rgba(12, 20, 40, 0.8)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(2, 2, canvas.width-4, canvas.height-4);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 24px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, canvas.width/2, canvas.height/2);
      }
      
      const texture = new THREE.CanvasTexture(canvas);
      const material = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(material);
      sprite.position.copy(position);
      sprite.scale.set(4, 1, 1);
      this.scene.add(sprite);
    };
    
    // Calculate positions for encoder/decoder labels
    const allLayers = [...this.encoderLayers, ...this.decoderLayers.slice(1)];
    const layerSpacing = 3;
    const networkWidth = (allLayers.length - 1) * layerSpacing;
    const encoderCenter = -networkWidth/2 + (this.encoderLayers.length * layerSpacing)/2;
    const decoderCenter = -networkWidth/2 + ((this.encoderLayers.length + this.decoderLayers.length)/2) * layerSpacing;
    
    // Add encoder/decoder labels
    createLabel("ENCODER", new THREE.Vector3(encoderCenter, -5, 0), "#4285f4");
    createLabel("DECODER", new THREE.Vector3(decoderCenter, -5, 0), "#7c4dff");
    
    // Add event listener for window resize
    window.addEventListener('resize', this.onWindowResize.bind(this));
  }
  
  private createNeuralNetworkVisualization(): void {
    // Calculate total layers (encoder + decoder)
    const allLayers = [...this.encoderLayers, ...this.decoderLayers.slice(1)];
    
    // Define dimensions
    const layerSpacing = 3;
    const networkWidth = (allLayers.length - 1) * layerSpacing;
    const neuronSpacing = 0.5;
    
    // Create helper function to calculate neuron positions for a layer
    const calculateLayerPositions = (layerSize: number, layerIdx: number, isInputOrOutput: boolean) => {
      // Calculate x position (depth)
      const x = layerIdx * layerSpacing - networkWidth / 2;
      
      // For very large layers (input/output), use a grid arrangement
      if (layerSize > 100 && isInputOrOutput) {
        const gridSize = Math.ceil(Math.sqrt(layerSize));
        const positions: {x: number, y: number, z: number}[] = [];
        
        // Create a square grid of neurons
        for (let row = 0; row < gridSize && positions.length < layerSize; row++) {
          for (let col = 0; col < gridSize && positions.length < layerSize; col++) {
            // Calculate grid position with spacing
            const gridSpacing = 0.15;
            const gridWidth = (gridSize - 1) * gridSpacing;
            const posY = (row * gridSpacing) - (gridWidth / 2);
            const posZ = (col * gridSpacing) - (gridWidth / 2);
            
            positions.push({
              x: x,
              y: posY,
              z: posZ
            });
          }
        }
        return positions;
      } else {
        // For smaller layers, use a circle arrangement
        const positions: {x: number, y: number, z: number}[] = [];
        const radius = Math.min(2.5, Math.max(0.5, layerSize * 0.05)); // Scale radius based on layer size
        
        for (let i = 0; i < layerSize; i++) {
          // Skip some neurons for very large layers to improve performance
          if (layerSize > 50 && !isInputOrOutput && i % Math.ceil(layerSize / 30) !== 0) {
            continue;
          }
          
          // Position neurons in a circle
          const angle = (i / layerSize) * Math.PI * 2;
          const posY = Math.sin(angle) * radius;
          const posZ = Math.cos(angle) * radius;
          
          positions.push({
            x: x,
            y: posY,
            z: posZ
          });
        }
        return positions;
      }
    };
    
    // Create neurons for each layer
    for (let layerIdx = 0; layerIdx < allLayers.length; layerIdx++) {
      const layerSize = allLayers[layerIdx];
      const isBottleneck = layerIdx === this.encoderLayers.length - 1;
      const isInput = layerIdx === 0;
      const isOutput = layerIdx === allLayers.length - 1;
      const isInputOrOutput = isInput || isOutput;
      
      // Get positions for this layer
      const positions = calculateLayerPositions(layerSize, layerIdx, isInputOrOutput);
      
      // Choose layer color
      let layerColor: number;
      if (isBottleneck) {
        layerColor = 0x00c9ff; // Cyan for bottleneck
      } else if (layerIdx < this.encoderLayers.length) {
        layerColor = 0x4285f4; // Blue for encoder
      } else {
        layerColor = 0x7c4dff; // Purple for decoder
      }
      
      // Create neurons
      for (let idx = 0; idx < positions.length; idx++) {
        const pos = positions[idx];
        
        // Size depends on layer type
        let size: number;
        if (isBottleneck) {
          size = 0.2; // Larger for bottleneck
        } else if (isInputOrOutput) {
          size = 0.08; // Smaller for input/output (many neurons)
        } else {
          size = 0.12; // Medium for hidden layers
        }
        
        const geometry = new THREE.SphereGeometry(size, 16, 16);
        const material = new THREE.MeshPhongMaterial({ 
          color: layerColor,
          transparent: true,
          opacity: 0.8,
          shininess: 30
        });
        
        const neuron = new THREE.Mesh(geometry, material);
        neuron.position.set(pos.x, pos.y, pos.z);
        this.scene.add(neuron);
        this.neurons.push(neuron);
        
        // Store neuron position for later use
        this.neuronPositions.push({
          x: pos.x,
          y: pos.y,
          z: pos.z,
          layerIndex: layerIdx,
          neuronIndex: idx
        });
      }
      
      // Create layer label
      const layerLabel = layerIdx === this.encoderLayers.length - 1 ? 
        'Bottleneck (z)' : 
        `${layerSize} neurons`;
      
      const textMaterial = new THREE.MeshBasicMaterial({ color: 0xe1e7f5 });
      
      // Create a sprite for the text
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 64;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'rgba(12, 20, 40, 0.8)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = isBottleneck ? '#00c9ff' : 
                         layerIdx < this.encoderLayers.length ? '#4285f4' : '#7c4dff';
        ctx.lineWidth = 2;
        ctx.strokeRect(2, 2, canvas.width-4, canvas.height-4);
        ctx.fillStyle = '#e1e7f5';
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(layerLabel, canvas.width/2, canvas.height/2);
      }
      
      const texture = new THREE.CanvasTexture(canvas);
      const labelMaterial = new THREE.SpriteMaterial({ map: texture });
      const label = new THREE.Sprite(labelMaterial);
      
      // Position the label above or below the layer
      const labelPos = calculateLayerPositions(1, layerIdx, false)[0];
      label.position.set(labelPos.x, 3, 0);
      label.scale.set(2, 0.5, 1);
      this.scene.add(label);
    }
    
    // Create connections between layers
    for (let layerIdx = 0; layerIdx < allLayers.length - 1; layerIdx++) {
      const isEncoderConnection = layerIdx < this.encoderLayers.length - 1;
      const connectionColor = isEncoderConnection ? 0x4285f4 : 0x7c4dff;
      
      // Get positions for current and next layer
      const currentLayerPositions = this.neuronPositions.filter(np => np.layerIndex === layerIdx);
      const nextLayerPositions = this.neuronPositions.filter(np => np.layerIndex === layerIdx + 1);
      
      // If too many connections would be created, reduce the number for performance
      const currentSample = Math.min(currentLayerPositions.length, 20);
      const nextSample = Math.min(nextLayerPositions.length, 20);
      
      // Create connections from sampled neurons in current layer to sampled neurons in next layer
      for (let i = 0; i < currentSample; i++) {
        const currentNeuron = currentLayerPositions[Math.floor(i * currentLayerPositions.length / currentSample)];
        
        for (let j = 0; j < nextSample; j++) {
          const nextNeuron = nextLayerPositions[Math.floor(j * nextLayerPositions.length / nextSample)];
          
          // Create connection
          const points = [
            new THREE.Vector3(currentNeuron.x, currentNeuron.y, currentNeuron.z),
            new THREE.Vector3(nextNeuron.x, nextNeuron.y, nextNeuron.z)
          ];
          
          const geometry = new THREE.BufferGeometry().setFromPoints(points);
          
          const material = new THREE.LineBasicMaterial({ 
            color: connectionColor,
            transparent: true,
            opacity: 0.1
          });
          
          const line = new THREE.Line(geometry, material);
          this.scene.add(line);
          this.connections.push(line);
        }
      }
    }
    
    // Add visual indicator for data direction (encoder → bottleneck → decoder)
    const addDirectionArrow = (start: number, end: number, color: number) => {
      // Position arrow in center between layers
      const midLayerIdx = (start + end) / 2;
      const arrowX = midLayerIdx * layerSpacing - networkWidth / 2;
      
      // Create arrow shape
      const arrowDir = new THREE.Vector3(1, 0, 0);
      const length = 1;
      const headWidth = 0.3;
      const headLength = 0.5;
      
      const arrowHelper = new THREE.ArrowHelper(
        arrowDir, 
        new THREE.Vector3(arrowX - length/2, -2.5, 0), 
        length, 
        color,
        headLength,
        headWidth
      );
      
      this.scene.add(arrowHelper);
    };
    
    // Add encoder and decoder direction indicators
    const encoderMidpoint = Math.floor(this.encoderLayers.length / 2);
    const decoderMidpoint = this.encoderLayers.length + Math.floor((this.decoderLayers.length - 1) / 2);
    
    addDirectionArrow(0, encoderMidpoint, 0x4285f4); // Encoder flow
    addDirectionArrow(this.encoderLayers.length, decoderMidpoint, 0x7c4dff); // Decoder flow
  }
  
  private initLatentSpaceViz(): void {
    // Set up SVG
    const width = this.latentSpaceCanvasRef.nativeElement.clientWidth;
    const height = this.latentSpaceCanvasRef.nativeElement.clientHeight;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    
    this.latentSpaceSvg = d3.select(this.latentSpaceCanvasRef.nativeElement)
      .append('svg')
      .attr('width', width)
      .attr('height', height) as D3SVGSelection;
    
    if (!this.latentSpaceSvg) {
      console.error('Failed to create SVG element');
      return;
    }
    
    const chartArea = this.latentSpaceSvg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Set up scales
    this.latentSpaceXScale = d3.scaleLinear()
      .domain([-3, 3])
      .range([0, width - margin.left - margin.right]);
    
    this.latentSpaceYScale = d3.scaleLinear()
      .domain([-3, 3])
      .range([height - margin.top - margin.bottom, 0]);
    
    // Add axes
    chartArea.append('g')
      .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
      .call(d3.axisBottom(this.latentSpaceXScale))
      .attr('color', '#8a9ab0');
    
    chartArea.append('g')
      .call(d3.axisLeft(this.latentSpaceYScale))
      .attr('color', '#8a9ab0');
    
    // Add labels
    chartArea.append('text')
      .attr('x', width / 2 - margin.left)
      .attr('y', height - margin.top)
      .attr('text-anchor', 'middle')
      .text('z₁')
      .attr('fill', '#e1e7f5');
    
    chartArea.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2 + margin.top)
      .attr('y', -30)
      .attr('text-anchor', 'middle')
      .text('z₂')
      .attr('fill', '#e1e7f5');
    
    // Create points for each digit example
    const points = chartArea.selectAll<SVGCircleElement, DigitExample>('circle')
      .data(this.digitExamples)
      .enter()
      .append('circle')
      .attr('cx', (d: DigitExample) => this.latentSpaceXScale ? this.latentSpaceXScale(d.encodedValue[0]) : 0)
      .attr('cy', (d: DigitExample) => this.latentSpaceYScale ? this.latentSpaceYScale(d.encodedValue[1]) : 0)
      .attr('r', 5)
      .attr('fill', (d: DigitExample) => {
        // Color by digit label
        const colors = [
          '#4285f4', '#ea4335', '#fbbc05', '#34a853', 
          '#ff6d01', '#46bdc6', '#7c4dff', '#ff7043',
          '#795548', '#9e9e9e'
        ];
        return colors[d.label % colors.length];
      })
      .attr('opacity', 0.7)
      .attr('stroke', '#ffffff')
      .attr('stroke-width', (d: DigitExample) => d.label === this.digitExamples[this.selectedExample].label ? 2 : 0);
    
    // Add hover interaction
    points
      .on('mouseover', function(this: SVGCircleElement, event: MouseEvent, d: DigitExample) {
        d3.select(this)
          .attr('r', 8)
          .attr('opacity', 1);
      })
      .on('mouseout', function(this: SVGCircleElement, event: MouseEvent, d: DigitExample) {
        d3.select(this)
          .attr('r', 5)
          .attr('opacity', 0.7);
      });
    
    // Add selected point indicator
    const selectedPoint = chartArea.append('circle')
      .attr('cx', this.latentSpaceXScale ? this.latentSpaceXScale(this.digitExamples[this.selectedExample].encodedValue[0]) : 0)
      .attr('cy', this.latentSpaceYScale ? this.latentSpaceYScale(this.digitExamples[this.selectedExample].encodedValue[1]) : 0)
      .attr('r', 10)
      .attr('fill', 'none')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '3,3');
    
    this.latentSpacePoints = { points, selectedPoint };
  }
  
  private initCanvases(): void {
    // Draw initial examples
    this.drawDigitOnCanvas(this.inputCanvasRef.nativeElement, this.digitExamples[this.selectedExample].pixels);
    this.drawDigitOnCanvas(this.outputCanvasRef.nativeElement, this.digitExamples[this.selectedExample].reconstructedPixels);
    
    // Create noisy version
    const noisyPixels = this.addNoise(this.digitExamples[this.selectedExample].pixels, this.noiseLevel);
    this.drawDigitOnCanvas(this.noisyCanvasRef.nativeElement, noisyPixels);
    this.drawDigitOnCanvas(this.denoisedCanvasRef.nativeElement, this.digitExamples[this.selectedExample].reconstructedPixels);
  }
  
  private drawDigitOnCanvas(canvas: HTMLCanvasElement, pixels: number[]): void {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.fillStyle = '#1e3a66';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw digit
    const size = Math.sqrt(pixels.length);
    const cellSize = canvas.width / size;
    
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const pixelValue = pixels[i * size + j];
        const intensity = Math.round(pixelValue * 255);
        ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
      }
    }
  }
  
  private addNoise(pixels: number[], level: number): number[] {
    return pixels.map(pixel => {
      // Add random noise to each pixel
      let noisy = pixel + (Math.random() * 2 - 1) * level;
      // Clamp values between 0 and 1
      return Math.max(0, Math.min(1, noisy));
    });
  }
  
  private generateExampleData(): void {
    // Generate synthetic MNIST-like data for 10 digits (0-9)
    const generateDigit = (label: number): DigitExample => {
      // Create a 28x28 blank image
      const pixels = new Array(28 * 28).fill(0);
      
      // Draw a simple digit-like pattern based on label
      switch (label) {
        case 0: // Draw a circle
          for (let i = 0; i < 28; i++) {
            for (let j = 0; j < 28; j++) {
              const distToCenter = Math.sqrt((i - 14) ** 2 + (j - 14) ** 2);
              if (distToCenter > 6 && distToCenter < 10) {
                pixels[i * 28 + j] = 0.9;
              }
            }
          }
          break;
        case 1: // Draw a vertical line
          for (let i = 5; i < 23; i++) {
            pixels[i * 28 + 14] = 0.9;
            pixels[i * 28 + 15] = 0.9;
          }
          break;
        case 2: // Draw a rough "2" shape
          for (let j = 8; j < 20; j++) {
            pixels[7 * 28 + j] = 0.9; // Top horizontal
          }
          for (let i = 7; i < 14; i++) {
            pixels[i * 28 + 19] = 0.9; // Top-right vertical
          }
          for (let j = 8; j < 20; j++) {
            pixels[14 * 28 + j] = 0.9; // Middle horizontal
          }
          for (let i = 14; i < 21; i++) {
            pixels[i * 28 + 8] = 0.9; // Bottom-left vertical
          }
          for (let j = 8; j < 20; j++) {
            pixels[21 * 28 + j] = 0.9; // Bottom horizontal
          }
          break;
        case 3: // Draw a "3" shape
          for (let j = 8; j < 20; j++) {
            pixels[7 * 28 + j] = 0.9; // Top horizontal
            pixels[14 * 28 + j] = 0.9; // Middle horizontal
            pixels[21 * 28 + j] = 0.9; // Bottom horizontal
          }
          for (let i = 7; i < 22; i++) {
            if (i < 14 || i > 14) {
              pixels[i * 28 + 19] = 0.9; // Right vertical
            }
          }
          break;
        case 4: // Draw a "4" shape
          for (let i = 7; i < 22; i++) {
            pixels[i * 28 + 18] = 0.9; // Right vertical
          }
          for (let i = 7; i < 15; i++) {
            pixels[i * 28 + 10] = 0.9; // Left vertical
          }
          for (let j = 10; j < 19; j++) {
            pixels[14 * 28 + j] = 0.9; // Horizontal
          }
          break;
        case 5: // Draw a "5" shape
          for (let j = 8; j < 20; j++) {
            pixels[7 * 28 + j] = 0.9; // Top horizontal
            pixels[14 * 28 + j] = 0.9; // Middle horizontal
            pixels[21 * 28 + j] = 0.9; // Bottom horizontal
          }
          for (let i = 7; i < 14; i++) {
            pixels[i * 28 + 8] = 0.9; // Top-left vertical
          }
          for (let i = 14; i < 22; i++) {
            pixels[i * 28 + 19] = 0.9; // Bottom-right vertical
          }
          break;
        default:
          // For remaining digits (6-9), create representative patterns
          for (let i = 7; i < 21; i++) {
            for (let j = 7; j < 21; j++) {
              // Create a pattern based on digit
              if ((i + j + label * 2) % 5 === 0) {
                pixels[i * 28 + j] = 0.9;
              }
            }
          }
      }
      
      // Generate an encoded value (simplified 2D latent space)
      // In a real autoencoder, this would be the output of the encoder
      const angle = (label / 10) * Math.PI * 2;
      const radius = 1 + Math.random() * 0.5;
      const encodedValue: [number, number] = [
        Math.cos(angle) * radius,
        Math.sin(angle) * radius
      ];
      
      // Generate a reconstructed output (simplified)
      // In a real autoencoder, this would be the output of the decoder
      const reconstructedPixels = pixels.map(p => {
        // Add some imperfection to the reconstruction
        const noise = Math.random() * 0.2 - 0.1;
        return Math.max(0, Math.min(1, p + noise));
      });
      
      return {
        pixels,
        label,
        encodedValue,
        reconstructedPixels
      };
    };
    
    // Generate examples for digits 0-9
    for (let i = 0; i < 10; i++) {
      this.digitExamples.push(generateDigit(i));
    }
    
    // Set initial latent space position to the first example
    this.latentSpaceDim1 = this.digitExamples[0].encodedValue[0];
    this.latentSpaceDim2 = this.digitExamples[0].encodedValue[1];
  }
  
  private render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
  
  private animate(): void {
    this.ngZone.runOutsideAngular(() => {
      this.animationFrameId = requestAnimationFrame(() => this.animate());
      
      if (this.isPlaying) {
        this.currentStep = (this.currentStep + 1) % this.totalSteps;
        const progress = this.currentStep / this.totalSteps;
        this.updateVisualization(progress);
      }
      
      this.render();
    });
  }
  
  private updateVisualization(progress: number): void {
    // Animate data flow through the network
    const allLayers = [...this.encoderLayers, ...this.decoderLayers.slice(1)];
    
    // Calculate which layer the data particle should be in
    const totalLayers = allLayers.length - 1; // Transitions between layers
    const layerPosition = progress * totalLayers;
    const currentLayerIndex = Math.min(Math.floor(layerPosition), totalLayers - 1);
    const nextLayerIndex = Math.min(currentLayerIndex + 1, totalLayers);
    const layerProgress = layerPosition - currentLayerIndex;
    
    // Get neurons in current and next layers
    const currentLayer = this.neuronPositions.filter(np => np.layerIndex === currentLayerIndex);
    const nextLayer = this.neuronPositions.filter(np => np.layerIndex === nextLayerIndex);
    
    if (currentLayer.length === 0 || nextLayer.length === 0) return;
    
    // Reset active neurons
    this.activeNeurons.clear();
    
    // The bottleneck layer always has active neurons
    const bottleneckIndex = this.encoderLayers.length - 1;
    if (currentLayerIndex >= bottleneckIndex) {
      const bottleneckNeurons = this.neuronPositions.filter(np => np.layerIndex === bottleneckIndex);
      bottleneckNeurons.forEach(neuron => {
        this.activeNeurons.add(this.neuronPositions.indexOf(neuron));
      });
    }
    
    // Activate a pattern of neurons that represents the current digit
    const digitExample = this.digitExamples[this.selectedExample];
    
    // For input and output layers, activate neurons based on the digits pattern
    if (currentLayerIndex === 0) { // Input layer
      // Activate neurons corresponding to pixels with value > 0.5
      const inputPixels = digitExample.pixels;
      for (let i = 0; i < Math.min(currentLayer.length, inputPixels.length); i++) {
        if (inputPixels[i] > 0.5) {
          const neuronIndex = this.neuronPositions.findIndex(
            np => np.layerIndex === 0 && np.neuronIndex === i
          );
          if (neuronIndex >= 0) {
            this.activeNeurons.add(neuronIndex);
          }
        }
      }
    } else if (nextLayerIndex === allLayers.length - 1) { // Output layer
      // Activate neurons based on reconstructed pixels
      const outputPixels = digitExample.reconstructedPixels;
      for (let i = 0; i < Math.min(nextLayer.length, outputPixels.length); i++) {
        if (outputPixels[i] > 0.5) {
          const neuronIndex = this.neuronPositions.findIndex(
            np => np.layerIndex === nextLayerIndex && np.neuronIndex === i
          );
          if (neuronIndex >= 0) {
            this.activeNeurons.add(neuronIndex);
          }
        }
      }
    }
    
    // For intermediate layers, activate some neurons randomly based on the pattern
    else {
      // As we move towards bottleneck, fewer neurons should be active (compression)
      const activationRatio = currentLayerIndex < bottleneckIndex ? 
        0.8 * (1 - currentLayerIndex / bottleneckIndex) : // Encoder (decreasing)
        0.05 + 0.5 * ((currentLayerIndex - bottleneckIndex) / (allLayers.length - bottleneckIndex)); // Decoder (increasing)
      
      const neuronsToActivate = Math.max(1, Math.floor(currentLayer.length * activationRatio));
      
      // Activate random neurons in current layer, but consistently for same digit
      const seed = digitExample.label * 100; // Use digit as seed for pseudo-random selection
      for (let i = 0; i < neuronsToActivate; i++) {
        const index = (seed + i * 17) % currentLayer.length; // Pseudo-random but consistent
        const neuronIndex = this.neuronPositions.indexOf(currentLayer[index]);
        if (neuronIndex >= 0) {
          this.activeNeurons.add(neuronIndex);
        }
      }
    }
    
    // Get random active neurons for data flow visualization
    const activeCurrentNeurons = currentLayer.filter(np => 
      this.activeNeurons.has(this.neuronPositions.indexOf(np))
    );
    
    const activeNextNeurons = nextLayer.filter(np => 
      this.activeNeurons.has(this.neuronPositions.indexOf(np))
    );
    
    // If we have active neurons, animate between them
    if (activeCurrentNeurons.length > 0 && activeNextNeurons.length > 0) {
      const currentNeuron = activeCurrentNeurons[Math.floor(Math.random() * activeCurrentNeurons.length)];
      const nextNeuron = activeNextNeurons[Math.floor(Math.random() * activeNextNeurons.length)];
      
      // Interpolate position
      const x = currentNeuron.x + (nextNeuron.x - currentNeuron.x) * layerProgress;
      const y = currentNeuron.y + (nextNeuron.y - currentNeuron.y) * layerProgress;
      const z = currentNeuron.z + (nextNeuron.z - currentNeuron.z) * layerProgress;
      
      // Update data particle position
      this.dataParticle.position.set(x, y, z);
      this.dataParticle.visible = true;
      
      // Add a trail effect
      if (this.dataTrail.length < 10 && Math.random() > 0.7) {
        const trailParticle = this.dataParticle.clone();
        trailParticle.material = new THREE.MeshBasicMaterial({
          color: currentLayerIndex < bottleneckIndex ? 0x4285f4 : 0x7c4dff,
          transparent: true,
          opacity: 0.5
        });
        this.scene.add(trailParticle);
        this.dataTrail.push(trailParticle);
        
        // Fade out and remove trail particles
        setTimeout(() => {
          const particle = this.dataTrail.shift();
          if (particle) {
            this.scene.remove(particle);
            particle.geometry.dispose();
            (particle.material as THREE.Material).dispose();
          }
        }, 500);
      }
    } else {
      // If no active neurons, hide data particle
      this.dataParticle.visible = false;
    }
    
    // Reset connections
    this.connections.forEach(connection => {
      (connection.material as THREE.LineBasicMaterial).opacity = 0.1;
    });
    
    // Update neurons based on activation state
    this.neurons.forEach((neuron, idx) => {
      const position = this.neuronPositions[idx];
      const isActive = this.activeNeurons.has(idx);
      const isBottleneckNeuron = position.layerIndex === bottleneckIndex;
      
      // Reset neuron appearance
      (neuron.material as THREE.MeshPhongMaterial).opacity = isActive ? 0.9 : 0.4;
      (neuron.material as THREE.MeshPhongMaterial).emissive = new THREE.Color(0x000000);
      
      // Make active neurons glow
      if (isActive) {
        let emissiveColor;
        if (isBottleneckNeuron) {
          emissiveColor = 0x00c9ff; // Cyan for bottleneck
        } else if (position.layerIndex < this.encoderLayers.length) {
          emissiveColor = 0x4285f4; // Blue for encoder
        } else {
          emissiveColor = 0x7c4dff; // Purple for decoder
        }
        
        const pulseIntensity = Math.sin(Date.now() * 0.005) * 0.3 + 0.5;
        (neuron.material as THREE.MeshPhongMaterial).emissive = new THREE.Color(emissiveColor);
        (neuron.material as THREE.MeshPhongMaterial).emissiveIntensity = pulseIntensity;
        
        // Scale active neurons slightly
        const scale = 1 + 0.2 * pulseIntensity;
        neuron.scale.set(scale, scale, scale);
      } else {
        // Reset scale for inactive neurons
        neuron.scale.set(1, 1, 1);
      }
      
      // Special bottleneck neurons always pulse slightly
      if (isBottleneckNeuron && !isActive) {
        const pulseIntensity = Math.sin(Date.now() * 0.003) * 0.1 + 0.1;
        (neuron.material as THREE.MeshPhongMaterial).emissive = new THREE.Color(0x00c9ff);
        (neuron.material as THREE.MeshPhongMaterial).emissiveIntensity = pulseIntensity;
      }
      
      // Highlight neurons near the data particle
      if (this.dataParticle.visible) {
        const distance = neuron.position.distanceTo(this.dataParticle.position);
        if (distance < 0.5) {
          (neuron.material as THREE.MeshPhongMaterial).emissive = new THREE.Color(0x00c9ff);
          (neuron.material as THREE.MeshPhongMaterial).emissiveIntensity = 0.7;
          
          // Make connections from/to this neuron more visible
          this.connections.forEach((connection, connIdx) => {
            const line = connection as THREE.Line;
            const geo = line.geometry as THREE.BufferGeometry;
            const positions = geo.getAttribute('position').array;
            
            // Check if this connection involves the neuron
            const startPoint = new THREE.Vector3(positions[0], positions[1], positions[2]);
            const endPoint = new THREE.Vector3(positions[3], positions[4], positions[5]);
            
            if (startPoint.distanceTo(neuron.position) < 0.2 || endPoint.distanceTo(neuron.position) < 0.2) {
              (connection.material as THREE.LineBasicMaterial).opacity = 0.6;
            }
          });
        }
      }
    });
  }
  
  private onWindowResize(): void {
    if (!this.threeCanvasRef?.nativeElement) return;
    
    const width = this.threeCanvasRef.nativeElement.clientWidth;
    const height = this.threeCanvasRef.nativeElement.clientHeight;
    
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    
    this.renderer.setSize(width, height);
  }
  
  // UI Handlers
  public togglePlayPause(): void {
    this.isPlaying = !this.isPlaying;
    
    if (this.isPlaying) {
      this.startAnimation();
    }
  }
  
  public startAnimation(): void {
    if (!this.animationFrameId) {
      this.animate();
    }
    this.isPlaying = true;
  }
  
  public pauseAnimation(): void {
    this.isPlaying = false;
  }
  
  public resetAnimation(): void {
    this.currentStep = 0;
    this.startAnimation();
  }
  
  public selectExample(index: number): void {
    this.selectedExample = index;
    
    // Update latent space dimensions
    this.latentSpaceDim1 = this.digitExamples[index].encodedValue[0];
    this.latentSpaceDim2 = this.digitExamples[index].encodedValue[1];
    
    // Update visualizations
    this.drawDigitOnCanvas(this.inputCanvasRef.nativeElement, this.digitExamples[index].pixels);
    this.drawDigitOnCanvas(this.outputCanvasRef.nativeElement, this.digitExamples[index].reconstructedPixels);
    
    // Update noisy version
    const noisyPixels = this.addNoise(this.digitExamples[index].pixels, this.noiseLevel);
    this.drawDigitOnCanvas(this.noisyCanvasRef.nativeElement, noisyPixels);
    this.drawDigitOnCanvas(this.denoisedCanvasRef.nativeElement, this.digitExamples[index].reconstructedPixels);
    
    // Update selected point in latent space
    if (this.latentSpacePoints && this.latentSpacePoints.selectedPoint) {
      if (this.latentSpaceXScale && this.latentSpaceYScale) {
        this.latentSpacePoints.selectedPoint
          .attr('cx', this.latentSpaceXScale(this.latentSpaceDim1))
          .attr('cy', this.latentSpaceYScale(this.latentSpaceDim2));
      }
    }
  }
  
  public updateNoiseLevel(event: Event): void {
    const inputElement = event.target as HTMLInputElement;
    const level = parseFloat(inputElement.value);
    this.noiseLevel = level;
    
    // Update noisy version
    const noisyPixels = this.addNoise(this.digitExamples[this.selectedExample].pixels, this.noiseLevel);
    this.drawDigitOnCanvas(this.noisyCanvasRef.nativeElement, noisyPixels);
    
    // Also update the denoised output with a slight delay to simulate processing
    setTimeout(() => {
      // In a real autoencoder, this would be the denoised result
      // For this simulation, we'll just use the clean reconstruction
      this.drawDigitOnCanvas(this.denoisedCanvasRef.nativeElement, this.digitExamples[this.selectedExample].reconstructedPixels);
    }, 300);
  }
  
  public updateLatentSpace(event: Event, dimension: 'dim1' | 'dim2'): void {
    const inputElement = event.target as HTMLInputElement;
    const value = parseFloat(inputElement.value);
    
    if (dimension === 'dim1') {
      this.latentSpaceDim1 = value;
    } else {
      this.latentSpaceDim2 = value;
    }
    
    // Update selected point in latent space
    if (this.latentSpacePoints && this.latentSpacePoints.selectedPoint) {
      if (this.latentSpaceXScale && this.latentSpaceYScale) {
        this.latentSpacePoints.selectedPoint
          .attr('cx', this.latentSpaceXScale(this.latentSpaceDim1))
          .attr('cy', this.latentSpaceYScale(this.latentSpaceDim2));
      }
    }
    
    // In a real system, we would generate a new output from this latent point
    // For this simulation, we'll interpolate between existing examples
    
    // Find the closest examples in latent space and interpolate
    let closestDistance = Infinity;
    let closestIndex = 0;
    let secondClosestDistance = Infinity;
    let secondClosestIndex = 0;
    
    for (let i = 0; i < this.digitExamples.length; i++) {
      const example = this.digitExamples[i];
      const distance = Math.sqrt(
        Math.pow(example.encodedValue[0] - this.latentSpaceDim1, 2) +
        Math.pow(example.encodedValue[1] - this.latentSpaceDim2, 2)
      );
      
      if (distance < closestDistance) {
        secondClosestDistance = closestDistance;
        secondClosestIndex = closestIndex;
        closestDistance = distance;
        closestIndex = i;
      } else if (distance < secondClosestDistance) {
        secondClosestDistance = distance;
        secondClosestIndex = i;
      }
    }
    
    // Calculate weights for interpolation
    const totalDistance = closestDistance + secondClosestDistance;
    const weight1 = totalDistance > 0 ? 1 - (closestDistance / totalDistance) : 1;
    const weight2 = totalDistance > 0 ? 1 - (secondClosestDistance / totalDistance) : 0;
    
    // Normalize weights
    const sumWeights = weight1 + weight2;
    const normalizedWeight1 = weight1 / sumWeights;
    const normalizedWeight2 = weight2 / sumWeights;
    
    // Interpolate reconstructed images
    const interpolatedPixels = this.digitExamples[closestIndex].reconstructedPixels.map((pixel, idx) => {
      return pixel * normalizedWeight1 + this.digitExamples[secondClosestIndex].reconstructedPixels[idx] * normalizedWeight2;
    });
    
    // Update the output visualization with a slight delay to simulate processing
    setTimeout(() => {
      this.drawDigitOnCanvas(this.outputCanvasRef.nativeElement, interpolatedPixels);
    }, 100);
  }
  
  public toggleAccordion(section: string): void {
    if (this.openAccordion === section) {
      this.openAccordion = null;
    } else {
      this.openAccordion = section;
    }
  }
  
  ngOnDestroy(): void {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    
    // Clean up THREE.js resources
    if (this.scene) {
      this.scene.clear();
    }
    
    if (this.renderer) {
      this.renderer.dispose();
    }
    
    // Remove resize listener
    window.removeEventListener('resize', this.onWindowResize);
  }
}