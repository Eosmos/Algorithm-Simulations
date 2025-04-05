import { Component, ElementRef, OnInit, ViewChild, AfterViewInit, OnDestroy, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';
import * as THREE from 'three';

interface DataPoint {
  x: number;
  y: number;
  class: string;
}

interface TreeNode {
  id: string;
  feature?: string;
  threshold?: number;
  gain?: number;
  gini?: number;
  entropy?: number;
  samples: number;
  value: number[];
  class?: string;
  children?: TreeNode[];
  depth: number;
  x?: number;
  y?: number;
}

interface Dataset {
  name: string;
  description: string;
  data: DataPoint[];
  features: string[];
  classes: string[];
}

interface SimulationMode {
  id: string;
  name: string;
  description: string;
}

interface ResearchPaper {
  title: string;
  authors: string;
  year: number;
  publication: string;
  description: string;
}

interface AnimationStep {
  nodes: TreeNode[];
  links: { source: TreeNode, target: TreeNode }[];
  partitions: { x1: number, y1: number, x2: number, y2: number, class: string | null }[];
  currentNode: TreeNode;
  description: string;
}

@Component({
  selector: 'app-decision-tree-simulation',
  templateUrl: './decision-tree-simulation.component.html',
  styleUrls: ['./decision-tree-simulation.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class DecisionTreeSimulationComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('treeCanvas') treeCanvasRef!: ElementRef<HTMLDivElement>;
  @ViewChild('featureSpaceCanvas') featureSpaceCanvasRef!: ElementRef<HTMLDivElement>;
  @ViewChild('impurityCanvas') impurityCanvasRef!: ElementRef<HTMLDivElement>;
  @ViewChild('animationContainer') animationContainerRef!: ElementRef<HTMLDivElement>;
  
  // State variables
  activeMode: string = 'treeBuilding';
  activeInfoTab: string = 'algorithm';
  isPlaying: boolean = false;
  animationSpeed: number = 1;
  currentStep: number = 0;
  maxSteps: number = 0;
  selectedDataset: string = 'default';
  selectedSplitCriterion: string = 'gini';
  maxDepth: number = 4;
  autoplayInterval: ReturnType<typeof setTimeout> | null = null;
  private resizeTimeout: ReturnType<typeof setTimeout> | null = null;
  
  // ThreeJS properties
  private scene: THREE.Scene | null = null;
  private camera: THREE.PerspectiveCamera | null = null;
  private renderer: THREE.WebGLRenderer | null = null;
  private animationFrameId: number | null = null;
  private particles: THREE.Points | null = null;
  private lines: THREE.LineSegments | null = null;
  
  // Visualization objects
  private treeVis: d3.Selection<HTMLDivElement, unknown, null, undefined> | null = null;
  private featureSpaceVis: d3.Selection<HTMLDivElement, unknown, null, undefined> | null = null;
  private impurityVis: d3.Selection<HTMLDivElement, unknown, null, undefined> | null = null;
  private animationSteps: AnimationStep[] = [];
  
  // Simulation modes
  simulationModes: SimulationMode[] = [
    {
      id: 'treeBuilding',
      name: 'Tree Building',
      description: 'Visualize how a decision tree is constructed step by step through recursive partitioning.'
    },
    {
      id: 'featureSpace',
      name: 'Feature Space Partitioning',
      description: 'See how the feature space is divided into regions as the tree grows.'
    },
    {
      id: 'impurityMeasures',
      name: 'Impurity Calculation',
      description: 'Understand how Gini impurity, entropy, and information gain are calculated.'
    },
    {
      id: 'prediction',
      name: 'Prediction Path',
      description: 'Follow the path of a data point through the tree to arrive at a prediction.'
    },
    {
      id: 'overfitting',
      name: 'Overfitting & Pruning',
      description: 'Compare overfitted trees with pruned trees to see the difference in generalization.'
    }
  ];
  
  // Datasets
  datasets: Dataset[] = [
    {
      name: 'default',
      description: 'A simple classification dataset with two features and two classes for demonstrating decision tree partitioning',
      data: this.generateDefaultDataset(),
      features: ['x', 'y'],
      classes: ['A', 'B']
    },
    {
      name: 'circles',
      description: 'Circular class boundaries demonstrate the limitations of decision trees with axis-parallel splits',
      data: this.generateCirclesDataset(),
      features: ['x', 'y'],
      classes: ['A', 'B']
    },
    {
      name: 'xor',
      description: 'XOR pattern showing how decision trees handle non-linear relationships by partitioning the space',
      data: this.generateXORDataset(),
      features: ['x', 'y'],
      classes: ['A', 'B']
    }
  ];
  
  // Research papers
  researchPapers: ResearchPaper[] = [
    {
      title: 'Induction of Decision Trees',
      authors: 'J. R. Quinlan',
      year: 1986,
      publication: 'Machine Learning, 1(1): 81–106',
      description: 'Introduced the ID3 algorithm, using Entropy and Information Gain for categorical attributes.'
    },
    {
      title: 'C4.5: Programs for Machine Learning',
      authors: 'J. R. Quinlan',
      year: 1993,
      publication: 'Morgan Kaufmann Publishers',
      description: 'Improved ID3 by handling continuous attributes, missing values, and pruning techniques.'
    },
    {
      title: 'Classification and Regression Trees',
      authors: 'L. Breiman, J. H. Friedman, R. A. Olshen, & C. J. Stone',
      year: 1984,
      publication: 'Wadsworth & Brooks/Cole Advanced Books & Software',
      description: 'Introduced CART, using Gini Impurity for classification and variance reduction for regression.'
    }
  ];
  
  // Decision tree data
  rootNode: TreeNode | null = null;
  treeNodes: TreeNode[] = [];
  treeLinks: {source: TreeNode, target: TreeNode}[] = [];
  
  constructor() {}
  
  ngOnInit(): void {
    this.initializeDecisionTree();
    this.prepareAnimationSteps();
  }
  
  ngAfterViewInit(): void {
    // Schedule initialization after the change detection cycle to avoid ExpressionChangedAfterItHasBeenCheckedError
    setTimeout(() => {
      this.initializeVisualizations();
      this.initThreeJsAnimation();
      this.renderCurrentStep();
      
      // Force a re-render after a short delay to ensure visualizations appear
      // This helps when dimensions weren't ready in the first render
      setTimeout(() => {
        if (this.treeVis && this.impurityVis) {
          console.log('Performing second render to ensure visibility');
          this.renderCurrentStep();
        }
      }, 500);
    });
  }
  
  ngOnDestroy(): void {
    // Clear all intervals and timeouts
    if (this.autoplayInterval) {
      clearTimeout(this.autoplayInterval);
    }
    
    if (this.resizeTimeout) {
      clearTimeout(this.resizeTimeout);
    }
    
    // Stop animation frame loop
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    // Clean up ThreeJS resources
    if (this.renderer) {
      this.renderer.dispose();
      
      // Remove the canvas from the DOM
      if (this.renderer.domElement && this.renderer.domElement.parentNode) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
    }
    
    // Dispose of ThreeJS geometries and materials
    if (this.particles) {
      const geometry = this.particles.geometry;
      const material = this.particles.material;
      
      if (geometry) {
        geometry.dispose();
      }
      
      if (material && !Array.isArray(material)) {
        material.dispose();
      }
      
      if (this.scene) {
        this.scene.remove(this.particles);
      }
      this.particles = null;
    }
    
    if (this.lines) {
      const geometry = this.lines.geometry;
      const material = this.lines.material;
      
      if (geometry) {
        geometry.dispose();
      }
      
      if (material && !Array.isArray(material)) {
        material.dispose();
      }
      
      if (this.scene) {
        this.scene.remove(this.lines);
      }
      this.lines = null;
    }
    
    // Clear references
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    
    // Clean up D3 selections
    if (this.treeVis) {
      this.treeVis.selectAll("*").interrupt(); // Stop any ongoing transitions
    }
    
    if (this.featureSpaceVis) {
      this.featureSpaceVis.selectAll("*").interrupt();
    }
    
    if (this.impurityVis) {
      this.impurityVis.selectAll("*").interrupt();
    }
    
    this.treeVis = null;
    this.featureSpaceVis = null;
    this.impurityVis = null;
  }
  
  // Helper method to check if all required view child elements are ready
  private checkViewsReady(): boolean {
    return !!this.treeCanvasRef?.nativeElement &&
           !!this.featureSpaceCanvasRef?.nativeElement &&
           !!this.impurityCanvasRef?.nativeElement &&
           !!this.animationContainerRef?.nativeElement;
  }
  
  private initThreeJsAnimation(): void {
    try {
      if (!this.animationContainerRef?.nativeElement) return;
      
      // Get container dimensions
      const container = this.animationContainerRef.nativeElement;
      const width = container.clientWidth || 400;
      const height = container.clientHeight || 300;
      
      // Initialize THREE.js scene
      this.scene = new THREE.Scene();
      
      // Initialize camera
      this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
      this.camera.position.z = 5;
      
      // Initialize renderer
      this.renderer = new THREE.WebGLRenderer({ 
        alpha: true,
        antialias: true  // Add antialiasing for smoother rendering
      });
      this.renderer.setSize(width, height);
      this.renderer.setClearColor(0x000000, 0); // Transparent background
      
      // Check if container already has a child (renderer canvas)
      if (container.childElementCount > 0) {
        container.innerHTML = ''; // Clear container first
      }
      
      // Add renderer to DOM
      container.appendChild(this.renderer.domElement);
      
      // Add particle system to represent data points
      const particlesGeometry = new THREE.BufferGeometry();
      const particleCount = 100;
      
      // Create particle positions
      const positions = new Float32Array(particleCount * 3);
      const colors = new Float32Array(particleCount * 3);
      
      for (let i = 0; i < particleCount; i++) {
        // Position particles in a sphere
        const i3 = i * 3;
        positions[i3] = (Math.random() - 0.5) * 5;
        positions[i3 + 1] = (Math.random() - 0.5) * 5;
        positions[i3 + 2] = (Math.random() - 0.5) * 5;
        
        // Set random colors
        colors[i3] = Math.random();
        colors[i3 + 1] = Math.random();
        colors[i3 + 2] = Math.random();
      }
      
      particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      particlesGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      
      // Create particle material
      const particlesMaterial = new THREE.PointsMaterial({
        size: 0.1,
        vertexColors: true,
        transparent: true,
        opacity: 0.7
      });
      
      // Create particle system
      this.particles = new THREE.Points(particlesGeometry, particlesMaterial);
      this.scene.add(this.particles);
      
      // Add lines to represent decision boundaries
      const linesGeometry = new THREE.BufferGeometry();
      const linePositions = new Float32Array([
        -2, 0, 0,   // line 1 start
        2, 0, 0,    // line 1 end
        0, -2, 0,   // line 2 start
        0, 2, 0     // line 2 end
      ]);
      
      linesGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
      
      const linesMaterial = new THREE.LineBasicMaterial({
        color: 0x4285f4,
        transparent: true,
        opacity: 0.5
      });
      
      this.lines = new THREE.LineSegments(linesGeometry, linesMaterial);
      this.scene.add(this.lines);
      
      // Animation function
      const animate = () => {
        if (!this.renderer || !this.scene || !this.camera || !this.particles || !this.lines) {
          return; // Safety check
        }
        
        this.animationFrameId = requestAnimationFrame(animate);
        
        // Rotate particle system
        this.particles.rotation.x += 0.002;
        this.particles.rotation.y += 0.003;
        
        // Rotate lines
        this.lines.rotation.x += 0.001;
        this.lines.rotation.y += 0.002;
        
        this.renderer.render(this.scene, this.camera);
      };
      
      // Start animation
      animate();
    } catch (error) {
      console.error('Error initializing ThreeJS animation:', error);
    }
  }
  
  private generateDefaultDataset(): DataPoint[] {
    const data: DataPoint[] = [];
    // Generate random points for two classes
    for (let i = 0; i < 100; i++) {
      if (i < 50) {
        // Class A points tend to be in the upper left
        data.push({
          x: Math.random() * 0.5,
          y: 0.5 + Math.random() * 0.5,
          class: 'A'
        });
      } else {
        // Class B points tend to be in the lower right
        data.push({
          x: 0.5 + Math.random() * 0.5,
          y: Math.random() * 0.5,
          class: 'B'
        });
      }
    }
    return data;
  }
  
  private generateCirclesDataset(): DataPoint[] {
    const data: DataPoint[] = [];
    const centerX = 0.5;
    const centerY = 0.5;
    
    for (let i = 0; i < 100; i++) {
      // Generate a point at a random angle
      const angle = Math.random() * Math.PI * 2;
      
      if (i < 50) {
        // Class A points form inner circle
        const radius = 0.1 + Math.random() * 0.1;
        data.push({
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          class: 'A'
        });
      } else {
        // Class B points form outer ring
        const radius = 0.3 + Math.random() * 0.15;
        data.push({
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          class: 'B'
        });
      }
    }
    return data;
  }
  
  private generateXORDataset(): DataPoint[] {
    const data: DataPoint[] = [];
    
    for (let i = 0; i < 100; i++) {
      const x = Math.random();
      const y = Math.random();
      
      // XOR pattern: points in top-right and bottom-left are class A
      // points in top-left and bottom-right are class B
      if ((x < 0.5 && y < 0.5) || (x >= 0.5 && y >= 0.5)) {
        data.push({ x, y, class: 'A' });
      } else {
        data.push({ x, y, class: 'B' });
      }
    }
    return data;
  }
  
  private initializeDecisionTree(): void {
    // Create a sample decision tree structure
    this.rootNode = {
      id: 'root',
      feature: 'x',
      threshold: 0.5,
      gain: 0.4,
      gini: 0.5,
      entropy: 1.0,
      samples: 100,
      value: [50, 50],
      depth: 0,
      children: [
        {
          id: 'node1',
          feature: 'y',
          threshold: 0.7,
          gain: 0.3,
          gini: 0.32,
          entropy: 0.8,
          samples: 40,
          value: [35, 5],
          depth: 1,
          children: [
            {
              id: 'leaf1',
              samples: 30,
              value: [29, 1],
              class: 'A',
              depth: 2
            },
            {
              id: 'leaf2',
              samples: 10,
              value: [6, 4],
              class: 'A',
              depth: 2
            }
          ]
        },
        {
          id: 'node2',
          feature: 'y',
          threshold: 0.3,
          gain: 0.25,
          gini: 0.28,
          entropy: 0.75,
          samples: 60,
          value: [15, 45],
          depth: 1,
          children: [
            {
              id: 'leaf3',
              samples: 15,
              value: [10, 5],
              class: 'A',
              depth: 2
            },
            {
              id: 'leaf4',
              samples: 45,
              value: [5, 40],
              class: 'B',
              depth: 2
            }
          ]
        }
      ]
    };
    
    // Flat list of nodes and links for D3
    this.treeNodes = [];
    this.treeLinks = [];
    this.flattenTree(this.rootNode);
  }
  
  private flattenTree(node: TreeNode | null, parentNode: TreeNode | null = null): void {
    if (!node) return;
    
    this.treeNodes.push(node);
    
    if (parentNode) {
      this.treeLinks.push({
        source: parentNode,
        target: node
      });
    }
    
    if (node.children) {
      for (const child of node.children) {
        this.flattenTree(child, node);
      }
    }
  }
  
  private prepareAnimationSteps(): void {
    // Prepare steps for the animation
    this.animationSteps = [];
    
    // Create proper tree nodes with default positions
    const rootNode: TreeNode = {
      id: 'root',
      feature: 'x',
      threshold: 0.5,
      gain: 0.4,
      gini: 0.5,
      entropy: 1.0,
      samples: 100,
      value: [50, 50],
      depth: 0,
      x: 0,
      y: 0,
      children: [
        {
          id: 'node1',
          feature: 'y',
          threshold: 0.7,
          gain: 0.3,
          gini: 0.32,
          entropy: 0.8,
          samples: 40,
          value: [35, 5],
          depth: 1,
          x: -100,
          y: 100,
          children: [
            {
              id: 'leaf1',
              samples: 30,
              value: [29, 1],
              class: 'A',
              depth: 2,
              x: -150,
              y: 200
            },
            {
              id: 'leaf2',
              samples: 10,
              value: [6, 4],
              class: 'A',
              depth: 2,
              x: -50,
              y: 200
            }
          ]
        },
        {
          id: 'node2',
          feature: 'y',
          threshold: 0.3,
          gain: 0.25,
          gini: 0.28,
          entropy: 0.75,
          samples: 60,
          value: [15, 45],
          depth: 1,
          x: 100,
          y: 100,
          children: [
            {
              id: 'leaf3',
              samples: 15,
              value: [10, 5],
              class: 'A',
              depth: 2,
              x: 50,
              y: 200
            },
            {
              id: 'leaf4',
              samples: 45,
              value: [5, 40],
              class: 'B',
              depth: 2,
              x: 150,
              y: 200
            }
          ]
        }
      ]
    };
    
    // Step 1: Root node only
    this.animationSteps.push({
      nodes: [{ ...rootNode }],
      links: [],
      partitions: [],
      currentNode: rootNode,
      description: 'Starting with the entire dataset at the root node.'
    });
    
    // Step 2: First split evaluation
    this.animationSteps.push({
      nodes: [{ ...rootNode }],
      links: [],
      partitions: [
        { x1: 0, y1: 0, x2: 0.5, y2: 1, class: null },
        { x1: 0.5, y1: 0, x2: 1, y2: 1, class: null }
      ],
      currentNode: rootNode,
      description: 'Evaluating the best feature and split point. Here, feature "x" with threshold 0.5 provides the highest information gain.'
    });
    
    // Step 3: First split complete
    const node1 = rootNode.children?.[0] as TreeNode;
    const node2 = rootNode.children?.[1] as TreeNode;
    
    this.animationSteps.push({
      nodes: [{ ...rootNode }, { ...node1 }, { ...node2 }],
      links: [
        { source: { ...rootNode }, target: { ...node1 } },
        { source: { ...rootNode }, target: { ...node2 } }
      ],
      partitions: [
        { x1: 0, y1: 0, x2: 0.5, y2: 1, class: null },
        { x1: 0.5, y1: 0, x2: 1, y2: 1, class: null }
      ],
      currentNode: node1,
      description: 'Split complete. The left node has 40 samples (35 class A, 5 class B). The right node has 60 samples (15 class A, 45 class B).'
    });
    
    // Step 4: Second split evaluation (left branch)
    this.animationSteps.push({
      nodes: [{ ...rootNode }, { ...node1 }, { ...node2 }],
      links: [
        { source: { ...rootNode }, target: { ...node1 } },
        { source: { ...rootNode }, target: { ...node2 } }
      ],
      partitions: [
        { x1: 0, y1: 0, x2: 0.5, y2: 0.7, class: null },
        { x1: 0, y1: 0.7, x2: 0.5, y2: 1, class: null },
        { x1: 0.5, y1: 0, x2: 1, y2: 1, class: null }
      ],
      currentNode: node1,
      description: 'Evaluating best split for the left node. Feature "y" with threshold 0.7 provides the highest gain.'
    });
    
    // Step 5: Second split complete (left branch)
    const leaf1 = node1.children?.[0] as TreeNode;
    const leaf2 = node1.children?.[1] as TreeNode;
    
    this.animationSteps.push({
      nodes: [{ ...rootNode }, { ...node1 }, { ...node2 }, { ...leaf1 }, { ...leaf2 }],
      links: [
        { source: { ...rootNode }, target: { ...node1 } },
        { source: { ...rootNode }, target: { ...node2 } },
        { source: { ...node1 }, target: { ...leaf1 } },
        { source: { ...node1 }, target: { ...leaf2 } }
      ],
      partitions: [
        { x1: 0, y1: 0, x2: 0.5, y2: 0.7, class: 'A' },
        { x1: 0, y1: 0.7, x2: 0.5, y2: 1, class: 'A' },
        { x1: 0.5, y1: 0, x2: 1, y2: 1, class: null }
      ],
      currentNode: leaf1,
      description: 'Left branch split complete. Both leaf nodes are majority class A, but with different purities.'
    });
    
    // Step 6: Third split evaluation (right branch)
    this.animationSteps.push({
      nodes: [{ ...rootNode }, { ...node1 }, { ...node2 }, { ...leaf1 }, { ...leaf2 }],
      links: [
        { source: { ...rootNode }, target: { ...node1 } },
        { source: { ...rootNode }, target: { ...node2 } },
        { source: { ...node1 }, target: { ...leaf1 } },
        { source: { ...node1 }, target: { ...leaf2 } }
      ],
      partitions: [
        { x1: 0, y1: 0, x2: 0.5, y2: 0.7, class: 'A' },
        { x1: 0, y1: 0.7, x2: 0.5, y2: 1, class: 'A' },
        { x1: 0.5, y1: 0, x2: 1, y2: 0.3, class: null },
        { x1: 0.5, y1: 0.3, x2: 1, y2: 1, class: null }
      ],
      currentNode: node2,
      description: 'Evaluating best split for the right node. Feature "y" with threshold 0.3 provides the highest gain.'
    });
    
    // Step 7: Complete tree
    const leaf3 = node2.children?.[0] as TreeNode;
    const leaf4 = node2.children?.[1] as TreeNode;
    
    this.animationSteps.push({
      nodes: [
        { ...rootNode }, 
        { ...node1 }, 
        { ...node2 }, 
        { ...leaf1 }, 
        { ...leaf2 },
        { ...leaf3 },
        { ...leaf4 }
      ],
      links: [
        { source: { ...rootNode }, target: { ...node1 } },
        { source: { ...rootNode }, target: { ...node2 } },
        { source: { ...node1 }, target: { ...leaf1 } },
        { source: { ...node1 }, target: { ...leaf2 } },
        { source: { ...node2 }, target: { ...leaf3 } },
        { source: { ...node2 }, target: { ...leaf4 } }
      ],
      partitions: [
        { x1: 0, y1: 0, x2: 0.5, y2: 0.7, class: 'A' },
        { x1: 0, y1: 0.7, x2: 0.5, y2: 1, class: 'A' },
        { x1: 0.5, y1: 0, x2: 1, y2: 0.3, class: 'A' },
        { x1: 0.5, y1: 0.3, x2: 1, y2: 1, class: 'B' }
      ],
      currentNode: leaf4,
      description: 'Decision tree complete! The feature space has been partitioned into regions that predict different classes.'
    });
    
    // Set the maximum number of steps
    this.maxSteps = this.animationSteps.length;
  }
  
  private initializeVisualizations(): void {
    try {
      // Initialize the tree visualization
      this.initializeTreeVisualization();
      
      // Initialize the feature space visualization
      this.initializeFeatureSpaceVisualization();
      
      // Initialize the impurity visualization
      this.initializeImpurityVisualization();
    } catch (error) {
      console.error('Error initializing visualizations:', error);
    }
  }
  
  private initializeTreeVisualization(): void {
    if (!this.treeCanvasRef?.nativeElement) {
      console.warn('Tree canvas not available');
      return;
    }
    
    try {
      const width = this.treeCanvasRef.nativeElement.clientWidth || 400;
      const height = this.treeCanvasRef.nativeElement.clientHeight || 300;
      
      console.log('Tree canvas dimensions:', width, height);
      
      // Create D3 tree layout
      this.treeVis = d3.select(this.treeCanvasRef.nativeElement);
      
      // Clear existing SVG if any
      this.treeVis.selectAll("*").remove();
      
      // Create new SVG with placeholder text
      const svg = this.treeVis
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", `translate(${width / 2}, 30)`);
      
      // Add a background rect to ensure visibility
      svg.append("rect")
        .attr("x", -width/2)
        .attr("y", -30)
        .attr("width", width)
        .attr("height", height)
        .attr("fill", "none")
        .attr("stroke", "rgba(255, 255, 255, 0.1)");
      
      // Add initial placeholder for the tree
      svg.append("text")
        .attr("class", "placeholder-text")
        .attr("text-anchor", "middle")
        .attr("dy", height/3)
        .text("Tree visualization will appear here")
        .style("opacity", 0.5)
        .style("font-size", "14px")
        .style("fill", "var(--text-muted)");
        
      // Force immediate rendering of tree
      if (this.animationSteps.length > 0 && this.currentStep >= 0) {
        this.updateTreeVisualization(this.animationSteps[this.currentStep]);
      }
    } catch (error) {
      console.error('Error initializing tree visualization:', error);
    }
  }
  
  private initializeFeatureSpaceVisualization(): void {
    if (!this.featureSpaceCanvasRef?.nativeElement) {
      console.warn('Feature space canvas not available');
      return;
    }
    
    const width = this.featureSpaceCanvasRef.nativeElement.clientWidth || 400;
    const height = this.featureSpaceCanvasRef.nativeElement.clientHeight || 300;
    
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Create D3 scatter plot
    this.featureSpaceVis = d3.select(this.featureSpaceCanvasRef.nativeElement);
    
    // Clear existing SVG if any
    this.featureSpaceVis.selectAll("*").remove();
    
    // Create new SVG
    const svg = this.featureSpaceVis
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
    
    // Add a background rect to ensure visibility
    svg.append("rect")
      .attr("width", innerWidth)
      .attr("height", innerHeight)
      .attr("fill", "none")
      .attr("stroke", "rgba(255, 255, 255, 0.1)");
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0]);
    
    // Create axes
    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale);
    
    // Add X axis
    svg.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis);
    
    // Add Y axis
    svg.append("g")
      .attr("class", "y-axis")
      .call(yAxis);
    
    // Add axis labels
    svg.append("text")
      .attr("class", "x-label")
      .attr("text-anchor", "middle")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight + margin.bottom - 5)
      .text("Feature X");
    
    svg.append("text")
      .attr("class", "y-label")
      .attr("text-anchor", "middle")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -margin.left + 10)
      .text("Feature Y");
    
    // Get the dataset
    const dataset = this.datasets.find(d => d.name === this.selectedDataset)?.data || [];
    
    // Add data points
    svg.selectAll(".data-point")
      .data(dataset)
      .enter()
      .append("circle")
      .attr("class", "data-point")
      .attr("cx", (d: DataPoint) => xScale(d.x))
      .attr("cy", (d: DataPoint) => yScale(d.y))
      .attr("r", 5)
      .attr("fill", (d: DataPoint) => d.class === 'A' ? "#4285f4" : "#ff9d45")
      .attr("opacity", 0.7);
  }
  
  private initializeImpurityVisualization(): void {
    if (!this.impurityCanvasRef?.nativeElement) {
      console.warn('Impurity canvas not available');
      return;
    }
    
    try {
      const width = this.impurityCanvasRef.nativeElement.clientWidth || 400;
      const height = this.impurityCanvasRef.nativeElement.clientHeight || 300;
      
      // Create D3 bar chart for impurity
      this.impurityVis = d3.select(this.impurityCanvasRef.nativeElement);
      
      // Clear existing SVG if any
      this.impurityVis.selectAll("*").remove();
      
      // Create new SVG with a proper viewBox for better scaling
      const svg = this.impurityVis
        .append("svg")
        .attr("width", "100%")
        .attr("height", "100%")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .append("g")
        .attr("transform", `translate(40,20)`);
      
      // Add a background rect for better visibility
      svg.append("rect")
        .attr("width", width - 40)
        .attr("height", height - 20)
        .attr("fill", "none")
        .attr("stroke", "rgba(255, 255, 255, 0.1)");
        
      // Add placeholder text
      svg.append("text")
        .attr("class", "placeholder-text")
        .attr("text-anchor", "middle")
        .attr("x", (width - 40) / 2)
        .attr("y", height / 2)
        .attr("dy", ".35em")
        .text("Impurity measures will appear here")
        .style("opacity", 0.5)
        .style("font-size", "14px")
        .style("fill", "var(--text-muted)");
      
      // Force immediate rendering of impurity visualization
      if (this.animationSteps.length > 0 && this.currentStep >= 0) {
        this.updateImpurityVisualization(this.animationSteps[this.currentStep]);
      }
    } catch (error) {
      console.error('Error initializing impurity visualization:', error);
    }
  }
  
  private renderCurrentStep(): void {
    try {
      if (this.currentStep < 0 || this.currentStep >= this.maxSteps) return;
      
      const step = this.animationSteps[this.currentStep];
      if (!step) return;
      
      console.log('Rendering step:', this.currentStep, step);
      
      // Update the visualizations based on the current step
      this.updateTreeVisualization(step);
      this.updateFeatureSpaceVisualization(step);
      this.updateImpurityVisualization(step);
    } catch (error) {
      console.error('Error rendering current step:', error);
    }
  }
  
  private updateTreeVisualization(step: AnimationStep): void {
    // Update the tree visualization based on the current step
    if (!this.treeVis || !step) return;
    
    const width = this.treeCanvasRef?.nativeElement?.clientWidth || 400;
    const height = this.treeCanvasRef?.nativeElement?.clientHeight || 300;
    
    console.log('Tree dimensions:', width, height);
    
    // Check if SVG already exists, otherwise create it
    const svgElement = this.treeVis.select("svg").node();
    if (!svgElement) {
      this.initializeTreeVisualization();
      return;
    }
    
    const svg = d3.select(svgElement);
    let mainGroup = svg.select("g");
    
    // Update SVG dimensions
    svg.attr("width", width)
       .attr("height", height);
       
    // Update background rect dimensions
    mainGroup.select("rect")
      .attr("x", -width/2)
      .attr("y", -30)
      .attr("width", width)
      .attr("height", height);
    
    // Remove placeholder text if it exists
    mainGroup.select(".placeholder-text").remove();
    
    // Draw graph if there are nodes
    if (step.nodes && step.nodes.length > 0) {
      // Handle links
      const links = mainGroup.selectAll<SVGPathElement, {source: TreeNode, target: TreeNode}>("path.link")
        .data(step.links, (d) => `${d.source.id}-${d.target.id}`);
      
      // Remove old links
      links.exit()
        .transition()
        .duration(500)
        .style("opacity", 0)
        .remove();
      
      // Add new links
      links.enter()
        .append("path")
        .attr("class", "link")
        .attr("d", (d) => {
          const sourceX = d.source.x ?? 0;
          const sourceY = d.source.y ?? 0;
          return `M${sourceX},${sourceY} L${sourceX},${sourceY}`; // Start at source for animation
        })
        .attr("stroke", "rgba(138, 154, 176, 0.6)")
        .attr("stroke-width", 1.5)
        .attr("fill", "none")
        .style("opacity", 0)
        .transition()
        .duration(800)
        .style("opacity", 1)
        .attr("d", (d) => {
          const sourceX = d.source.x ?? 0;
          const sourceY = d.source.y ?? 0;
          const targetX = d.target.x ?? 0;
          const targetY = d.target.y ?? 0;
          return `M${sourceX},${sourceY} L${targetX},${targetY}`;
        });
      
      // Update existing links
      links.transition()
        .duration(800)
        .attr("d", (d) => {
          const sourceX = d.source.x ?? 0;
          const sourceY = d.source.y ?? 0;
          const targetX = d.target.x ?? 0;
          const targetY = d.target.y ?? 0;
          return `M${sourceX},${sourceY} L${targetX},${targetY}`;
        });
      
      // Handle nodes
      const nodes = mainGroup.selectAll<SVGGElement, TreeNode>("g.node")
        .data(step.nodes, (d) => d.id);
      
      // Remove old nodes
      nodes.exit()
        .transition()
        .duration(500)
        .style("opacity", 0)
        .remove();
      
      // Add new nodes
      const enterNodes = nodes.enter()
        .append("g")
        .attr("class", (d) => `node ${d.children ? "node--internal" : "node--leaf"}`)
        .attr("transform", (d) => {
          const nodeX = d.x ?? 0;
          const nodeY = d.y ?? 0;
          return `translate(${nodeX},${nodeY})`;
        })
        .style("opacity", 0);
      
      // Add circles to new nodes
      enterNodes.append("circle")
        .attr("r", 0) // Start small for animation
        .attr("class", (d) => step.currentNode && d.id === step.currentNode.id ? "current-node" : "")
        .attr("fill", (d) => d.class ? "#24b47e" : "#2c5cbd")
        .attr("stroke", "#8bb4fa")
        .attr("stroke-width", 2);
      
      // Add text to new nodes
      enterNodes.append("text")
        .attr("dy", ".35em")
        .attr("y", (d) => d.children ? -20 : 20)
        .attr("text-anchor", "middle")
        .attr("fill", "#e1e7f5")
        .text((d) => {
          return d.class ? 
            `Class: ${d.class}` : 
            (d.feature ? `${d.feature} <= ${d.threshold?.toFixed(1)}` : '');
        });
      
      // Animate new nodes appearing
      enterNodes.transition()
        .duration(800)
        .style("opacity", 1)
        .select("circle")
        .attr("r", 10);
      
      // Update existing nodes
      nodes.transition()
        .duration(800)
        .attr("transform", (d) => {
          const nodeX = d.x ?? 0;
          const nodeY = d.y ?? 0;
          return `translate(${nodeX},${nodeY})`;
        });
      
      // Update circles in existing nodes
      nodes.select("circle")
        .attr("class", (d) => step.currentNode && d.id === step.currentNode.id ? "current-node" : "")
        .attr("fill", (d) => d.class ? "#24b47e" : "#2c5cbd");
      
      // Update text on existing nodes
      nodes.select("text")
        .text((d) => {
          return d.class ? 
            `Class: ${d.class}` : 
            (d.feature ? `${d.feature} <= ${d.threshold?.toFixed(1)}` : '');
        });
    }
  }
  
  private updateFeatureSpaceVisualization(step: AnimationStep): void {
    // Update the feature space visualization based on the current step
    if (!this.featureSpaceVis || !step) return;
    
    const svg = this.featureSpaceVis.select("svg g");
    if (svg.empty()) {
      console.error("Feature space SVG not initialized properly");
      return;
    }
    
    // Get container dimensions for calculations
    const width = this.featureSpaceCanvasRef?.nativeElement?.clientWidth || 400;
    const height = this.featureSpaceCanvasRef?.nativeElement?.clientHeight || 300;
    
    // Calculate usable area
    const usableWidth = width - 80; // accounting for margins
    const usableHeight = height - 80;
    
    // Create a key function to uniquely identify partitions
    const partitionKey = (d: any, i: number) => {
      return `${d.x1}-${d.y1}-${d.x2}-${d.y2}-${d.class || 'none'}-${i}`;
    };
    
    // Update partitions
    const partitions = svg.selectAll<SVGRectElement, {x1: number, y1: number, x2: number, y2: number, class: string | null}>("rect.partition")
      .data(step.partitions || [], partitionKey);
    
    // Remove exiting partitions with animation
    partitions.exit()
      .transition()
      .duration(500)
      .style("opacity", 0)
      .remove();
    
    // Update existing partitions with animation
    partitions.transition()
      .duration(800)
      .attr("x", (d) => d.x1 * usableWidth)
      .attr("y", (d) => (1 - d.y2) * usableHeight)
      .attr("width", (d) => (d.x2 - d.x1) * usableWidth)
      .attr("height", (d) => (d.y2 - d.y1) * usableHeight)
      .attr("fill", (d) => d.class === 'A' ? "rgba(66, 133, 244, 0.2)" : 
                      d.class === 'B' ? "rgba(255, 157, 69, 0.2)" : "rgba(255, 255, 255, 0.05)")
      .attr("stroke", (d) => d.class === 'A' ? "rgba(66, 133, 244, 0.8)" : 
                        d.class === 'B' ? "rgba(255, 157, 69, 0.8)" : "rgba(255, 255, 255, 0.3)");
    
    // Add entering partitions with animation
    partitions.enter()
      .append("rect")
      .attr("class", "partition")
      .attr("x", (d) => d.x1 * usableWidth)
      .attr("y", (d) => (1 - d.y2) * usableHeight)
      .attr("width", (d) => (d.x2 - d.x1) * usableWidth)
      .attr("height", (d) => (d.y2 - d.y1) * usableHeight)
      .attr("fill", (d) => d.class === 'A' ? "rgba(66, 133, 244, 0.2)" : 
                      d.class === 'B' ? "rgba(255, 157, 69, 0.2)" : "rgba(255, 255, 255, 0.05)")
      .attr("stroke", (d) => d.class === 'A' ? "rgba(66, 133, 244, 0.8)" : 
                        d.class === 'B' ? "rgba(255, 157, 69, 0.8)" : "rgba(255, 255, 255, 0.3)")
      .attr("stroke-width", 2)
      .style("opacity", 0)
      .transition()
      .duration(800)
      .style("opacity", 1);
      
    // Highlight data points that fall within the current node region
    const dataPoints = svg.selectAll<SVGCircleElement, DataPoint>("circle.data-point");
    
    dataPoints.transition()
      .duration(500)
      .attr("r", (d) => {
        // Check if this point is within the current node's partition
        const isHighlighted = step.partitions.some(p => {
          return d.x >= p.x1 && d.x <= p.x2 && d.y >= p.y1 && d.y <= p.y2 && 
                 (p.class === null || p.class === d.class);
        });
        return isHighlighted ? 6 : 4;
      })
      .attr("stroke-width", (d) => {
        const isHighlighted = step.partitions.some(p => {
          return d.x >= p.x1 && d.x <= p.x2 && d.y >= p.y1 && d.y <= p.y2 && 
                 (p.class === null || p.class === d.class);
        });
        return isHighlighted ? 1.5 : 0;
      })
      .attr("stroke", "#ffffff")
      .style("opacity", (d) => {
        const isHighlighted = step.partitions.some(p => {
          return d.x >= p.x1 && d.x <= p.x2 && d.y >= p.y1 && d.y <= p.y2;
        });
        return isHighlighted ? 1 : 0.5;
      });
  }
  
  private updateImpurityVisualization(step: AnimationStep): void {
    // Update the impurity visualization based on the current step
    if (!this.impurityVis || !step || !step.currentNode) return;
    
    const width = this.impurityCanvasRef?.nativeElement?.clientWidth || 400;
    const height = this.impurityCanvasRef?.nativeElement?.clientHeight || 300;
    
    // Check if SVG already exists
    const svgElement = this.impurityVis.select("svg").node();
    if (!svgElement) {
      this.initializeImpurityVisualization();
      return;
    }
    
    const svg = d3.select(svgElement);
    let mainGroup = svg.select("g");
    
    // Update SVG dimensions
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    
    // Update background rect
    mainGroup.select("rect")
      .attr("width", width - 40)
      .attr("height", height - 30);
    
    // Remove placeholder text if it exists
    mainGroup.select(".placeholder-text").remove();
    
    // Prepare data based on current node
    const currentNode = step.currentNode;
    let impurityData: {
      label: string;
      gini: number;
      entropy: number;
      samples: number;
      values: number[];
    }[] = [];
    
    if (currentNode.children && currentNode.children.length > 0) {
      impurityData = [
        { 
          label: "Parent", 
          gini: currentNode.gini || 0, 
          entropy: currentNode.entropy || 0,
          samples: currentNode.samples || 0,
          values: currentNode.value || [0, 0]
        },
        { 
          label: "Left Child", 
          gini: currentNode.children[0].gini || 0, 
          entropy: currentNode.children[0].entropy || 0,
          samples: currentNode.children[0].samples || 0,
          values: currentNode.children[0].value || [0, 0]
        }
      ];
      
      if (currentNode.children.length > 1) {
        impurityData.push({ 
          label: "Right Child", 
          gini: currentNode.children[1].gini || 0, 
          entropy: currentNode.children[1].entropy || 0,
          samples: currentNode.children[1].samples || 0,
          values: currentNode.children[1].value || [0, 0]
        });
      }
    } else {
      impurityData = [
        { 
          label: "Current Node", 
          gini: currentNode.gini || 0, 
          entropy: currentNode.entropy || 0,
          samples: currentNode.samples || 0,
          values: currentNode.value || [0, 0]
        }
      ];
    }
    
    // Create scales with reduced width to fit better in the container
    const chartWidth = width - 80;
    const chartHeight = height - 100; // Reduce height to make room for pie charts
    
    const xScale = d3.scaleBand()
      .domain(impurityData.map(d => d.label))
      .range([0, chartWidth])
      .padding(0.2);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([chartHeight, 0]);
    
    // Create or update axes
    let xAxisGroup = mainGroup.select<SVGGElement>(".x-axis");
    let yAxisGroup = mainGroup.select<SVGGElement>(".y-axis");
    
    // Create axes functions
    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale).ticks(5);
    
    if (xAxisGroup.empty()) {
      // Create X axis if it doesn't exist
      xAxisGroup = mainGroup.append<SVGGElement>("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0,${chartHeight})`)
        .call(xAxis);
    } else {
      // Update existing X axis
      xAxisGroup.transition()
        .duration(500)
        .attr("transform", `translate(0,${chartHeight})`)
        .call(xAxis);
    }
    
    if (yAxisGroup.empty()) {
      // Create Y axis if it doesn't exist
      yAxisGroup = mainGroup.append<SVGGElement>("g")
        .attr("class", "y-axis")
        .call(yAxis);
    } else {
      // Update existing Y axis
      yAxisGroup.transition()
        .duration(500)
        .call(yAxis);
    }
    
    // Create or update chart title
    let chartTitle = mainGroup.select<SVGTextElement>(".chart-title");
    
    if (chartTitle.empty()) {
      // Create title if it doesn't exist
      chartTitle = mainGroup.append<SVGTextElement>("text")
        .attr("class", "chart-title")
        .attr("x", chartWidth / 2)
        .attr("y", -5)
        .attr("text-anchor", "middle")
        .text(`${this.selectedSplitCriterion === 'gini' ? 'Gini Impurity' : 'Entropy'} Values`);
    } else {
      // Update existing title
      chartTitle.text(`${this.selectedSplitCriterion === 'gini' ? 'Gini Impurity' : 'Entropy'} Values`);
    }
    
    // Update bars with proper enter/update/exit pattern
    const bars = mainGroup.selectAll<SVGRectElement, any>(".bar")
      .data(impurityData, (d) => d.label);
    
    // Remove old bars
    bars.exit()
      .transition()
      .duration(300)
      .attr("y", chartHeight)
      .attr("height", 0)
      .remove();
    
    // Update existing bars
    bars.transition()
      .duration(800)
      .attr("x", (d) => {
        const bandPos = xScale(d.label);
        return bandPos !== undefined ? bandPos : 0;
      })
      .attr("y", (d) => yScale(this.selectedSplitCriterion === 'gini' ? d.gini : d.entropy))
      .attr("width", xScale.bandwidth())
      .attr("height", (d) => chartHeight - yScale(this.selectedSplitCriterion === 'gini' ? d.gini : d.entropy))
      .attr("fill", "#7c4dff");
    
    // Add new bars
    bars.enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", (d) => {
        const bandPos = xScale(d.label);
        return bandPos !== undefined ? bandPos : 0;
      })
      .attr("y", chartHeight) // Start at bottom for animation
      .attr("width", xScale.bandwidth())
      .attr("height", 0) // Start with no height for animation
      .attr("fill", "#7c4dff")
      .transition()
      .duration(800)
      .attr("y", (d) => yScale(this.selectedSplitCriterion === 'gini' ? d.gini : d.entropy))
      .attr("height", (d) => chartHeight - yScale(this.selectedSplitCriterion === 'gini' ? d.gini : d.entropy));
    
    // Update bar labels
    const barLabels = mainGroup.selectAll<SVGTextElement, any>(".bar-label")
      .data(impurityData, (d) => d.label);
    
    // Remove old labels
    barLabels.exit()
      .transition()
      .duration(300)
      .style("opacity", 0)
      .remove();
    
    // Update existing labels
    barLabels.transition()
      .duration(800)
      .attr("x", (d) => {
        const bandPos = xScale(d.label);
        return (bandPos !== undefined ? bandPos : 0) + xScale.bandwidth() / 2;
      })
      .attr("y", (d) => yScale(this.selectedSplitCriterion === 'gini' ? d.gini : d.entropy) - 5)
      .text((d) => (this.selectedSplitCriterion === 'gini' ? d.gini : d.entropy).toFixed(2));
    
    // Add new labels
    barLabels.enter()
      .append("text")
      .attr("class", "bar-label")
      .attr("x", (d) => {
        const bandPos = xScale(d.label);
        return (bandPos !== undefined ? bandPos : 0) + xScale.bandwidth() / 2;
      })
      .attr("y", (d) => yScale(this.selectedSplitCriterion === 'gini' ? d.gini : d.entropy) - 5)
      .attr("text-anchor", "middle")
      .style("opacity", 0)
      .text((d) => (this.selectedSplitCriterion === 'gini' ? d.gini : d.entropy).toFixed(2))
      .transition()
      .duration(800)
      .style("opacity", 1);
    
    // Handle pie chart title
    let pieTitle = mainGroup.select<SVGTextElement>(".pie-title");
    const pieY = chartHeight + 50; // Position below the x-axis
    
    if (pieTitle.empty()) {
      pieTitle = mainGroup.append<SVGTextElement>("text")
        .attr("class", "pie-title")
        .attr("x", chartWidth / 2)
        .attr("y", chartHeight + 25)
        .attr("text-anchor", "middle")
        .attr("fill", "var(--text-light)")
        .attr("font-size", "12px")
        .text("Class Distribution");
    }
    
    // Update pie charts
    const pieRadius = Math.min(15, xScale.bandwidth() / 2.5); // Reduce pie chart radius
    const pie = d3.pie<number>().sort(null);
    const arc = d3.arc<d3.PieArcDatum<number>>().innerRadius(0).outerRadius(pieRadius);
    
    // Update pie groups with proper enter/update/exit pattern
    const pieGroups = mainGroup.selectAll<SVGGElement, any>(".pie-group")
      .data(impurityData, (d) => d.label);
    
    // Remove old pie groups
    pieGroups.exit()
      .transition()
      .duration(300)
      .style("opacity", 0)
      .remove();
    
    // Update existing pie groups
    pieGroups.transition()
      .duration(800)
      .attr("transform", (d) => {
        const bandPos = xScale(d.label);
        return `translate(${(bandPos !== undefined ? bandPos : 0) + xScale.bandwidth() / 2}, ${pieY})`;
      });
    
    // Add new pie groups
    const enterPieGroups = pieGroups.enter()
      .append("g")
      .attr("class", "pie-group")
      .attr("transform", (d) => {
        const bandPos = xScale(d.label);
        return `translate(${(bandPos !== undefined ? bandPos : 0) + xScale.bandwidth() / 2}, ${pieY})`;
      })
      .style("opacity", 0);
    
    // Add animation to new pie groups
    enterPieGroups.transition()
      .duration(800)
      .style("opacity", 1);
    
    // Handle the pie chart arcs for new groups
    enterPieGroups.each(function(d) {
      const total = d.values[0] + d.values[1];
      // If total is 0, use [1, 1] to still show a 50/50 pie
      const data = total > 0 ? d.values.map((v) => v / total) : [0.5, 0.5];
      
      // Start with small radius for animation
      const enterArc = d3.arc<d3.PieArcDatum<number>>().innerRadius(0).outerRadius(0);
      
      const paths = d3.select(this).selectAll<SVGPathElement, d3.PieArcDatum<number>>("path")
        .data(pie(data))
        .enter()
        .append<SVGPathElement>("path")
        .attr("d", enterArc as any)
        .attr("fill", (_d, i) => i === 0 ? "#4285f4" : "#ff9d45")
        .attr("stroke", "rgba(255, 255, 255, 0.3)")
        .attr("stroke-width", 0.5);
      
      // Animate to full size
      paths.transition()
        .duration(800)
        .attrTween("d", function(this: SVGPathElement, d: d3.PieArcDatum<number>) {
          // Create interpolator for just the angle
          const angleInterpolate = d3.interpolate(
            d.startAngle,
            d.endAngle
          );
          const radiusInterpolate = d3.interpolate(0, pieRadius);
          
          return function(t: number): string {
            const arcGen = d3.arc<d3.PieArcDatum<number>>()
              .innerRadius(0)
              .outerRadius(radiusInterpolate(t));
              
            // Create a complete datum by copying the original but modifying the endAngle
            const interpolatedDatum = {
              ...d,
              endAngle: angleInterpolate(t)
            };
            
            // Make sure we always return a string
            return arcGen(interpolatedDatum) || '';
          };
        });
      
      // Add small legends below pie charts
      d3.select(this).append("text")
        .attr("y", pieRadius + 15)
        .attr("text-anchor", "middle")
        .attr("fill", "var(--text-light)")
        .attr("font-size", "10px")
        .style("opacity", 0)
        .text(() => {
          const classA = d.values[0];
          const classB = d.values[1];
          return `A: ${classA}, B: ${classB}`;
        })
        .transition()
        .duration(800)
        .style("opacity", 1);
    });
    
    // Update existing pie paths
    pieGroups.each(function(d) {
      const total = d.values[0] + d.values[1];
      const data = total > 0 ? d.values.map((v) => v / total) : [0.5, 0.5];
      
      const paths = d3.select(this).selectAll<SVGPathElement, d3.PieArcDatum<number>>("path")
        .data(pie(data));
      
      paths.transition()
        .duration(800)
        .attrTween("d", function(d) {
          const arcGen = d3.arc<d3.PieArcDatum<number>>().innerRadius(0).outerRadius(pieRadius);
          return function(t) {
            return arcGen(d) as string;
          };
        });
        
      // Update legend text
      d3.select(this).select("text")
        .text(() => {
          const classA = d.values[0];
          const classB = d.values[1];
          return `A: ${classA}, B: ${classB}`;
        });
    });
  }
  
  @HostListener('window:resize')
  private setupResizeListener(): void {
    // Properly debounce the resize event using class variable
    if (this.resizeTimeout) {
      clearTimeout(this.resizeTimeout);
    }
    
    this.resizeTimeout = setTimeout(() => {
      console.log('Window resized, reinitializing visualizations');
      this.initializeVisualizations();
      this.renderCurrentStep();
      this.resizeTimeout = null;
    }, 250);
  }
  
  // UI control methods
  
  playSimulation(): void {
    if (this.isPlaying) {
      this.pauseSimulation();
      return;
    }
    
    this.isPlaying = true;
    
    // Enhanced animation playback with step transitions
    const playNextStep = () => {
      if (this.currentStep < this.maxSteps - 1) {
        this.nextStep();
        
        // Schedule next step with adjusted timing based on animation speed
        this.autoplayInterval = setTimeout(
          playNextStep, 
          2000 / this.animationSpeed
        );
      } else {
        this.pauseSimulation();
      }
    };
    
    // Start the first step immediately
    playNextStep();
  }
  
  pauseSimulation(): void {
    this.isPlaying = false;
    if (this.autoplayInterval) {
      clearTimeout(this.autoplayInterval);
      this.autoplayInterval = null;
    }
  }
  
  resetSimulation(): void {
    this.pauseSimulation();
    this.currentStep = 0;
    this.renderCurrentStep();
  }
  
  previousStep(): void {
    if (this.currentStep > 0) {
      this.currentStep--;
      this.renderCurrentStep();
    }
  }
  
  nextStep(): void {
    if (this.currentStep < this.maxSteps - 1) {
      this.currentStep++;
      this.renderCurrentStep();
    }
  }
  
  changeMode(mode: string): void {
    this.activeMode = mode;
    this.resetSimulation();
    
    // Need to re-initialize visualizations when changing modes
    // as different modes use different canvas elements
    setTimeout(() => {
      this.initializeVisualizations();
      this.renderCurrentStep();
    }, 50);
  }
  
  getDatasetDescription(): string {
    const dataset = this.datasets.find(d => d.name === this.selectedDataset);
    return dataset ? dataset.description : '';
  }
  
  changeDataset(dataset: string): void {
    this.selectedDataset = dataset;
    this.initializeFeatureSpaceVisualization();
    this.resetSimulation();
  }
  
  changeSplitCriterion(criterion: string): void {
    this.selectedSplitCriterion = criterion;
    this.initializeImpurityVisualization();
    this.renderCurrentStep();
  }
  
  onMaxDepthChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.changeMaxDepth(input.value);
  }
  
  changeMaxDepth(depth: string | number): void {
    this.maxDepth = typeof depth === 'string' ? parseInt(depth, 10) : depth;
    // Re-initialize the decision tree with the new max depth
    // This would involve rebuilding the tree with the specified depth limit
    this.resetSimulation();
  }
  
  onAnimationSpeedChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.changeAnimationSpeed(input.value);
  }
  
  changeAnimationSpeed(speed: string | number): void {
    this.animationSpeed = typeof speed === 'string' ? parseFloat(speed) : speed;
    
    // If currently playing, restart with new speed
    if (this.isPlaying) {
      this.pauseSimulation();
      this.playSimulation();
    }
  }
  
  getCurrentStepDescription(): string {
    if (this.currentStep < 0 || this.currentStep >= this.maxSteps) {
      return '';
    }
    
    return this.animationSteps[this.currentStep].description || '';
  }
}