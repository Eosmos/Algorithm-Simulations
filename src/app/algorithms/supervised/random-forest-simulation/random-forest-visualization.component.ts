import { Component, ElementRef, OnInit, ViewChild, AfterViewInit, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface TreeNode {
  id: string;
  feature?: string;
  threshold?: number;
  value?: any;
  left?: TreeNode;
  right?: TreeNode;
  samples: number;
  depth: number;
  impurity: number;
  x?: number;
  y?: number;
  z?: number;
}

interface Sample {
  id: number;
  x1: number;
  x2: number;
  class: string;
  selected: boolean[];
}

interface Tree {
  id: number;
  root: TreeNode;
  samples: number[];
  features: string[];
}

interface SimulationStep {
  title: string;
  description: string;
  action: () => void;
}

@Component({
  selector: 'app-random-forest-visualization',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './random-forest-visualization.component.html',
  styleUrls: ['./random-forest-visualization.component.scss']
})
export class RandomForestVisualizationComponent implements OnInit, AfterViewInit {
  @ViewChild('simulationContainer') simulationContainer!: ElementRef;
  @ViewChild('datasetContainer') datasetContainer!: ElementRef;
  @ViewChild('treeContainer') treeContainer!: ElementRef;
  @ViewChild('forestContainer') forestContainer!: ElementRef;
  @ViewChild('predictionContainer') predictionContainer!: ElementRef;
  @ViewChild('canvas3d') canvas3d!: ElementRef;

  // Simulation parameters
  public numTrees = 5;
  public maxDepth = 3;
  public numSamples = 100;
  public featureRandomness = true;
  public showTestSample = true;
  
  // Design parameters from design system
  private colors = {
    primary: '#4285f4',
    lightPrimary: '#8bb4fa',
    darkPrimary: '#2c5cbd',
    secondary: '#7c4dff',
    lightSecondary: '#ae94ff',
    darkSecondary: '#5c35cc',
    accent: '#00c9ff',
    lightAccent: '#6edfff',
    darkAccent: '#0099cc',
    background: '#0c1428',
    cardBackground: '#162a4a',
    elementBackground: '#1e3a66',
    hoverBackground: '#2a4980',
    text: '#e1e7f5',
    mutedText: '#8a9ab0',
    success: '#24b47e',
    warning: '#ff9d45',
    error: '#ff6b6b',
    white: '#ffffff'
  };

  // Simulation state
  public simulationState: 'stopped' | 'playing' | 'paused' = 'stopped';
  public currentStepIndex = 0;
  private simulationInterval: any;
  public simulationSpeed = 2000; // ms
  
  // Data
  private samples: Sample[] = [];
  private trees: Tree[] = [];
  private testSample: Sample | null = null;
  private predictionResults: any[] = [];

  // SVG elements
  private datasetSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private treeSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private forestSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private predictionSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  
  // 3D elements
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private forest3D: THREE.Object3D[] = [];
  
  // Current step explanation
  public currentExplanation = 'Welcome to the Random Forest Simulation. Press Play to start or use the Step buttons to manually progress.';
  
  // Simulation steps
  private simulationSteps: SimulationStep[] = [];
  
  constructor(private ngZone: NgZone) {}

  ngOnInit(): void {
    this.generateSampleData();
    this.buildForest();
    this.initializeSimulationSteps();
  }

  ngAfterViewInit(): void {
    this.initializeVisualization();
    this.init3DForest();
    this.animate();
  }

  private init3DForest(): void {
    const canvas = this.canvas3d.nativeElement;
    
    // Initialize scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.colors.background);
    
    // Initialize camera
    this.camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
    this.camera.position.z = 15;
    this.camera.position.y = 5;
    
    // Initialize renderer
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    
    // Add controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(10, 10, 10);
    this.scene.add(directionalLight);
    
    // Add grid
    const gridHelper = new THREE.GridHelper(20, 20, 0x555555, 0x333333);
    this.scene.add(gridHelper);
    
    // Create forest
    this.create3DForest();
    
    // Handle window resize
    window.addEventListener('resize', () => {
      this.camera.aspect = canvas.clientWidth / canvas.clientHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    });
  }

  private create3DForest(): void {
    // Clear existing forest
    this.forest3D.forEach(object => this.scene.remove(object));
    this.forest3D = [];
    
    // Create trees in a circular pattern
    const radius = 8;
    const angleStep = (2 * Math.PI) / this.numTrees;
    
    for (let i = 0; i < this.numTrees; i++) {
      const treeGroup = new THREE.Group();
      treeGroup.name = `tree-${i}`;
      
      // Position the tree in a circle
      const angle = i * angleStep;
      const x = radius * Math.cos(angle);
      const z = radius * Math.sin(angle);
      
      // Start trees from below ground
      treeGroup.position.set(x, -2, z);
      
      // Create trunk
      const trunkGeometry = new THREE.CylinderGeometry(0.2, 0.3, 2, 8);
      const trunkMaterial = new THREE.MeshStandardMaterial({ 
        color: 0x8B4513,
        roughness: 0.8
      });
      const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
      trunk.position.y = 1;
      treeGroup.add(trunk);
      
      // Create foliage (representing nodes)
      const createLevel = (y: number, radius: number, nodes: number) => {
        const levelGroup = new THREE.Group();
        levelGroup.position.y = y;
        
        const foliageGeometry = new THREE.SphereGeometry(radius, 16, 16);
        
        for (let j = 0; j < nodes; j++) {
          const nodeAngle = j * (2 * Math.PI / nodes);
          const nx = radius * 0.8 * Math.cos(nodeAngle);
          const nz = radius * 0.8 * Math.sin(nodeAngle);
          
          // Alternate node colors based on classes
          const nodeMaterial = new THREE.MeshStandardMaterial({ 
            color: j % 2 === 0 ? this.hexToRgb(this.colors.primary) : this.hexToRgb(this.colors.secondary),
            roughness: 0.5,
            metalness: 0.2
          });
          
          const node = new THREE.Mesh(foliageGeometry, nodeMaterial);
          node.position.set(nx, 0, nz);
          
          // Start with scale 0 and we'll animate it
          node.scale.set(0, 0, 0);
          levelGroup.add(node);
          
          // Store initial scale to animate
          setTimeout(() => {
            // Create animation for node scale
            const animateScale = () => {
              if (node.scale.x < 0.5) {
                node.scale.x += 0.05;
                node.scale.y += 0.05;
                node.scale.z += 0.05;
                requestAnimationFrame(animateScale);
              }
            };
            
            animateScale();
          }, i * 200 + j * 100);
        }
        
        treeGroup.add(levelGroup);
      };
      
      // Create 3 levels of nodes
      createLevel(2, 1, 1);  // Root
      createLevel(3, 0.8, 2); // Level 1
      createLevel(4, 0.6, 4); // Level 2
      
      this.scene.add(treeGroup);
      this.forest3D.push(treeGroup);
      
      // Animate the tree position with a bounce effect
      const animatePosition = () => {
        if (treeGroup.position.y < 0) {
          // Moving up
          treeGroup.position.y += 0.2;
          requestAnimationFrame(animatePosition);
        } else if (treeGroup.position.y < 0.2) {
          // Small bounce down
          treeGroup.position.y = -0.2;
          setTimeout(() => {
            const finalBounce = () => {
              if (treeGroup.position.y < 0) {
                treeGroup.position.y += 0.05;
                requestAnimationFrame(finalBounce);
              } else {
                treeGroup.position.y = 0;
              }
            };
            
            finalBounce();
          }, 100);
        }
      };
      
      setTimeout(() => {
        animatePosition();
      }, i * 200);
    }
    
    // Add test sample if enabled
    if (this.showTestSample && this.testSample) {
      const sampleGeometry = new THREE.SphereGeometry(0.5, 16, 16);
      const sampleMaterial = new THREE.MeshStandardMaterial({ 
        color: this.hexToRgb(this.colors.accent),
        roughness: 0.3,
        metalness: 0.7,
        emissive: this.hexToRgb(this.colors.lightAccent),
        emissiveIntensity: 0.3
      });
      
      const sampleMesh = new THREE.Mesh(sampleGeometry, sampleMaterial);
      sampleMesh.position.set(0, 0.5, 0);
      sampleMesh.name = 'test-sample';
      
      this.scene.add(sampleMesh);
      this.forest3D.push(sampleMesh);
      
      // Animate the sample with a floating effect
      let animationTime = 0;
      const animateSample = () => {
        animationTime += 0.01;
        sampleMesh.position.y = 0.5 + Math.sin(animationTime * 2) * 0.5;
        requestAnimationFrame(animateSample);
      };
      
      animateSample();
    }
  }

  private animate(): void {
    this.ngZone.runOutsideAngular(() => {
      const animate = () => {
        requestAnimationFrame(animate);
        
        this.controls.update();
        
        // Rotate trees slowly
        this.forest3D.forEach((object, index) => {
          if (object.name.startsWith('tree-')) {
            object.rotation.y += 0.002 * ((index % 2) ? 1 : -1);
          }
        });
        
        this.renderer.render(this.scene, this.camera);
      };
      
      animate();
    });
  }

  private hexToRgb(hex: string): number {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return 0x000000;
    
    const r = parseInt(result[1], 16);
    const g = parseInt(result[2], 16);
    const b = parseInt(result[3], 16);
    
    return (r << 16) | (g << 8) | b;
  }

  private generateSampleData(): void {
    // Generate 2D data with two classes
    this.samples = [];
    
    // Create class A: cluster around (25, 25)
    for (let i = 0; i < this.numSamples / 2; i++) {
      this.samples.push({
        id: i,
        x1: 25 + Math.random() * 15 - 7.5,
        x2: 25 + Math.random() * 15 - 7.5,
        class: 'A',
        selected: Array(this.numTrees).fill(false)
      });
    }
    
    // Create class B: cluster around (75, 75)
    for (let i = this.numSamples / 2; i < this.numSamples; i++) {
      this.samples.push({
        id: i,
        x1: 75 + Math.random() * 15 - 7.5,
        x2: 75 + Math.random() * 15 - 7.5,
        class: 'B',
        selected: Array(this.numTrees).fill(false)
      });
    }
    
    // Add some noise by mixing a few points
    for (let i = 0; i < this.numSamples * 0.1; i++) {
      const idx = Math.floor(Math.random() * this.numSamples);
      this.samples[idx].x1 = Math.random() * 100;
      this.samples[idx].x2 = Math.random() * 100;
    }
    
    // Create a test sample
    this.testSample = {
      id: this.numSamples,
      x1: 50,
      x2: 50,
      class: '?',
      selected: Array(this.numTrees).fill(false)
    };
  }

  private bootstrapSamples(treeIndex: number): number[] {
    // Bootstrap: Sample with replacement
    const sampledIndices = [];
    
    // Reset selection state
    this.samples.forEach(sample => {
      sample.selected[treeIndex] = false;
    });
    
    for (let i = 0; i < this.numSamples; i++) {
      const idx = Math.floor(Math.random() * this.numSamples);
      sampledIndices.push(idx);
      this.samples[idx].selected[treeIndex] = true;
    }
    
    return sampledIndices;
  }

  private selectRandomFeatures(): string[] {
    const allFeatures = ['x1', 'x2'];
    
    if (!this.featureRandomness) {
      return allFeatures;
    }
    
    // For demonstrative purposes with only 2 features:
    // Randomly select 1 or 2 features
    const numFeaturesToSelect = Math.max(1, Math.floor(Math.random() * allFeatures.length) + 1);
    
    // Shuffle and select
    return [...allFeatures]
      .sort(() => Math.random() - 0.5)
      .slice(0, numFeaturesToSelect);
  }

  private calculateImpurity(samples: Sample[]): number {
    // Calculate Gini impurity
    if (samples.length === 0) return 0;
    
    const classCounts: Record<string, number> = {};
    for (const sample of samples) {
      if (!classCounts[sample.class]) {
        classCounts[sample.class] = 0;
      }
      classCounts[sample.class]++;
    }
    
    let impurity = 1;
    for (const cls in classCounts) {
      const p = classCounts[cls] / samples.length;
      impurity -= p * p;
    }
    
    return impurity;
  }

  private findBestSplit(samples: Sample[], features: string[]): { feature: string; threshold: number; impurityDecrease: number } {
    let bestFeature = '';
    let bestThreshold = 0;
    let bestImpurityDecrease = -Infinity;
    
    const parentImpurity = this.calculateImpurity(samples);
    
    for (const feature of features) {
      // Sort samples by feature value
      const sortedSamples = [...samples].sort((a, b) => (a as any)[feature] - (b as any)[feature]);
      
      // Try different thresholds (simplified for demonstration)
      for (let i = 0; i < sortedSamples.length - 1; i += Math.max(1, Math.floor(sortedSamples.length / 10))) {
        const threshold = ((sortedSamples[i] as any)[feature] + (sortedSamples[i + 1] as any)[feature]) / 2;
        
        const leftSamples = sortedSamples.filter(s => (s as any)[feature] <= threshold);
        const rightSamples = sortedSamples.filter(s => (s as any)[feature] > threshold);
        
        if (leftSamples.length === 0 || rightSamples.length === 0) continue;
        
        const leftImpurity = this.calculateImpurity(leftSamples);
        const rightImpurity = this.calculateImpurity(rightSamples);
        
        const leftWeight = leftSamples.length / samples.length;
        const rightWeight = rightSamples.length / samples.length;
        
        const impurityDecrease = parentImpurity - (leftWeight * leftImpurity + rightWeight * rightImpurity);
        
        if (impurityDecrease > bestImpurityDecrease) {
          bestImpurityDecrease = impurityDecrease;
          bestFeature = feature;
          bestThreshold = threshold;
        }
      }
    }
    
    return { feature: bestFeature, threshold: bestThreshold, impurityDecrease: bestImpurityDecrease };
  }

  private buildTree(samples: Sample[], features: string[], depth: number, maxDepth: number, nodeId: string): TreeNode {
    // Base case: max depth reached or pure node
    const impurity = this.calculateImpurity(samples);
    
    if (depth >= maxDepth || impurity < 0.01 || samples.length < 2) {
      // Create leaf node
      const classCounts: Record<string, number> = {};
      for (const sample of samples) {
        if (!classCounts[sample.class]) {
          classCounts[sample.class] = 0;
        }
        classCounts[sample.class]++;
      }
      
      let maxCount = 0;
      let majorityClass = '';
      for (const cls in classCounts) {
        if (classCounts[cls] > maxCount) {
          maxCount = classCounts[cls];
          majorityClass = cls;
        }
      }
      
      return {
        id: nodeId,
        value: majorityClass,
        samples: samples.length,
        depth,
        impurity
      };
    }
    
    // Find best split
    const { feature, threshold, impurityDecrease } = this.findBestSplit(samples, features);
    
    if (impurityDecrease <= 0 || !feature) {
      // No good split found, create leaf node
      const classCounts: Record<string, number> = {};
      for (const sample of samples) {
        if (!classCounts[sample.class]) {
          classCounts[sample.class] = 0;
        }
        classCounts[sample.class]++;
      }
      
      let maxCount = 0;
      let majorityClass = '';
      for (const cls in classCounts) {
        if (classCounts[cls] > maxCount) {
          maxCount = classCounts[cls];
          majorityClass = cls;
        }
      }
      
      return {
        id: nodeId,
        value: majorityClass,
        samples: samples.length,
        depth,
        impurity
      };
    }
    
    // Split samples
    const leftSamples = samples.filter(s => (s as any)[feature] <= threshold);
    const rightSamples = samples.filter(s => (s as any)[feature] > threshold);
    
    // Create internal node
    const node: TreeNode = {
      id: nodeId,
      feature,
      threshold,
      samples: samples.length,
      depth,
      impurity
    };
    
    // Recursively build left and right subtrees
    node.left = this.buildTree(leftSamples, features, depth + 1, maxDepth, nodeId + 'L');
    node.right = this.buildTree(rightSamples, features, depth + 1, maxDepth, nodeId + 'R');
    
    return node;
  }

  private buildForest(): void {
    this.trees = [];
    
    for (let i = 0; i < this.numTrees; i++) {
      // Bootstrap samples
      const sampledIndices = this.bootstrapSamples(i);
      const sampledData = sampledIndices.map(idx => this.samples[idx]);
      
      // Select random features
      const selectedFeatures = this.selectRandomFeatures();
      
      // Build tree
      const root = this.buildTree(sampledData, selectedFeatures, 0, this.maxDepth, `tree${i}_`);
      
      this.trees.push({
        id: i,
        root,
        samples: sampledIndices,
        features: selectedFeatures
      });
    }
  }

  private predict(sample: Sample, tree: Tree): string {
    let node = tree.root;
    
    while (node.left && node.right) {
      const feature = node.feature!;
      const threshold = node.threshold!;
      
      if ((sample as any)[feature] <= threshold) {
        node = node.left;
      } else {
        node = node.right;
      }
    }
    
    return node.value;
  }

  private makeForestPrediction(sample: Sample): string {
    const predictions = this.trees.map(tree => this.predict(sample, tree));
    
    // Count class frequencies
    const classCounts: Record<string, number> = {};
    for (const prediction of predictions) {
      if (!classCounts[prediction]) {
        classCounts[prediction] = 0;
      }
      classCounts[prediction]++;
    }
    
    // Find majority class
    let maxCount = 0;
    let majorityClass = '';
    for (const cls in classCounts) {
      if (classCounts[cls] > maxCount) {
        maxCount = classCounts[cls];
        majorityClass = cls;
      }
    }
    
    this.predictionResults = predictions.map((p, i) => ({ treeId: i, prediction: p }));
    
    return majorityClass;
  }

  private initializeVisualization(): void {
    // Initialize all SVG containers
    this.initializeDatasetViz();
    this.initializeTreeViz();
    this.initializeForestViz();
    this.initializePredictionViz();
  }

  private initializeDatasetViz(): void {
    const container = this.datasetContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    this.datasetSvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', '0 0 100 100')
      .attr('preserveAspectRatio', 'xMidYMid meet');
    
    // Add background
    this.datasetSvg.append('rect')
      .attr('width', 100)
      .attr('height', 100)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);
    
    // Draw data points
    this.datasetSvg.selectAll<SVGCircleElement, Sample>('circle.sample')
      .data(this.samples)
      .enter()
      .append('circle')
      .attr('class', 'sample')
      .attr('cx', function(d) { return d.x1; })
      .attr('cy', function(d) { return d.x2; })
      .attr('r', 2)
      .attr('fill', (d) => d.class === 'A' ? this.colors.primary : this.colors.secondary)
      .attr('opacity', 0)
      .transition()
      .duration(1000)
      .delay((d, i) => i * 10)
      .attr('opacity', 1);
    
    // Draw test sample
    if (this.testSample) {
      this.datasetSvg.append('circle')
        .attr('class', 'test-sample')
        .attr('cx', this.testSample.x1)
        .attr('cy', this.testSample.x2)
        .attr('r', 3)
        .attr('fill', this.colors.accent)
        .attr('stroke', this.colors.white)
        .attr('stroke-width', 1)
        .attr('opacity', 0)
        .transition()
        .duration(1000)
        .delay(this.samples.length * 10 + 500)
        .attr('opacity', 1);
    }
    
    // Add axes
    this.datasetSvg.append('line')
      .attr('x1', 0)
      .attr('y1', 100)
      .attr('x2', 100)
      .attr('y2', 100)
      .attr('stroke', this.colors.mutedText)
      .attr('stroke-width', 0.5);
    
    this.datasetSvg.append('line')
      .attr('x1', 0)
      .attr('y1', 0)
      .attr('x2', 0)
      .attr('y2', 100)
      .attr('stroke', this.colors.mutedText)
      .attr('stroke-width', 0.5);
    
    // Add labels
    this.datasetSvg.append('text')
      .attr('x', 50)
      .attr('y', 98)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', 4)
      .text('Feature 1');
    
    this.datasetSvg.append('text')
      .attr('x', 2)
      .attr('y', 50)
      .attr('text-anchor', 'start')
      .attr('dominant-baseline', 'middle')
      .attr('transform', 'rotate(-90, 2, 50)')
      .attr('fill', this.colors.text)
      .attr('font-size', 4)
      .text('Feature 2');
    
    // Add legend
    this.datasetSvg.append('circle')
      .attr('cx', 80)
      .attr('cy', 10)
      .attr('r', 2)
      .attr('fill', this.colors.primary);
    
    this.datasetSvg.append('text')
      .attr('x', 84)
      .attr('y', 10)
      .attr('dominant-baseline', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', 3)
      .text('Class A');
    
    this.datasetSvg.append('circle')
      .attr('cx', 80)
      .attr('cy', 16)
      .attr('r', 2)
      .attr('fill', this.colors.secondary);
    
    this.datasetSvg.append('text')
      .attr('x', 84)
      .attr('y', 16)
      .attr('dominant-baseline', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', 3)
      .text('Class B');
    
    this.datasetSvg.append('circle')
      .attr('cx', 80)
      .attr('cy', 22)
      .attr('r', 2)
      .attr('fill', this.colors.accent)
      .attr('stroke', this.colors.white)
      .attr('stroke-width', 0.5);
    
    this.datasetSvg.append('text')
      .attr('x', 84)
      .attr('y', 22)
      .attr('dominant-baseline', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', 3)
      .text('Test Sample');
  }

  private initializeTreeViz(): void {
    const container = this.treeContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    this.treeSvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
    
    // Add background
    this.treeSvg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);
  }

  private initializeForestViz(): void {
    const container = this.forestContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    this.forestSvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
    
    // Add background
    this.forestSvg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);
  }

  private initializePredictionViz(): void {
    const container = this.predictionContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    this.predictionSvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
    
    // Add background
    this.predictionSvg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);
  }

  private drawTree(tree: Tree, svgContainer: d3.Selection<SVGSVGElement, unknown, null, undefined>, width: number, height: number): void {
    svgContainer.selectAll('*').remove();
    
    // Add background
    svgContainer.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);
    
    // We need to properly handle the hierarchy with explicit types
    const treeHierarchy = d3.hierarchy<TreeNode>(tree.root, (d) => {
      const children: TreeNode[] = [];
      if (d.left) children.push(d.left);
      if (d.right) children.push(d.right);
      return children.length > 0 ? children : null;
    });
    
    const treeLayout = d3.tree<TreeNode>().size([width - 60, height - 60]);
    const treeData = treeLayout(treeHierarchy);
    
    // Create a proper D3 link generator with type safety
    const linkGenerator = d3.linkVertical<d3.HierarchyPointLink<TreeNode>, d3.HierarchyPointNode<TreeNode>>()
      .x((d) => d.x)
      .y((d) => d.y);
    
    // Draw links with animation
    const links = svgContainer.append('g')
      .attr('transform', `translate(30, 40)`)
      .selectAll<SVGPathElement, d3.HierarchyPointLink<TreeNode>>('path')
      .data(treeData.links())
      .enter()
      .append('path')
      .attr('d', (d) => {
        // Create a straight line from source to source as starting point
        const source = { x: d.source.x, y: d.source.y };
        const target = { x: d.source.x, y: d.source.y }; // Start at same position
        return linkGenerator({ source, target } as any);
      })
      .attr('stroke', this.colors.mutedText)
      .attr('stroke-width', 1.5)
      .attr('fill', 'none');
    
    // Animate links
    links.transition()
      .duration(800)
      .delay((_: any, i: number) => i * 100)
      .attr('d', linkGenerator);
    
    // Draw nodes
    const nodes = svgContainer.append('g')
      .attr('transform', `translate(30, 40)`)
      .selectAll<SVGGElement, d3.HierarchyPointNode<TreeNode>>('g')
      .data(treeData.descendants())
      .enter()
      .append('g')
      .attr('transform', (d) => `translate(${d.x}, ${d.y})`)
      .attr('opacity', 0)
      .attr('class', 'node')
      .transition()
      .duration(500)
      .delay((_: any, i: number) => i * 100 + 300)
      .attr('opacity', 1);
    
    // Store the reference to this for nested functions
    const self = this;
    
    // Add node circles
    svgContainer.selectAll<SVGGElement, d3.HierarchyPointNode<TreeNode>>('g.node')
      .append('circle')
      .attr('r', (d: d3.HierarchyPointNode<TreeNode>) => d.data.value ? 18 : 15)
      .attr('fill', function(d: d3.HierarchyPointNode<TreeNode>) {
        if (d.data.value) {
          return d.data.value === 'A' ? self.colors.primary : self.colors.secondary;
        }
        return self.colors.elementBackground;
      })
      .attr('stroke', this.colors.white)
      .attr('stroke-width', 1.5);
    
    // Add node labels
    svgContainer.selectAll<SVGGElement, d3.HierarchyPointNode<TreeNode>>('g.node')
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '.3em')
      .attr('fill', this.colors.white)
      .attr('font-size', '12px')
      .text((d: d3.HierarchyPointNode<TreeNode>) => {
        if (d.data.value) {
          return d.data.value; // Leaf node shows class
        } else {
          return `${d.data.feature!.toUpperCase()}`; // Internal node shows feature
        }
      });
    
    // Add threshold labels to internal nodes
    svgContainer.selectAll<SVGGElement, d3.HierarchyPointNode<TreeNode>>('g.node')
      .filter((d: d3.HierarchyPointNode<TreeNode>) => !d.data.value)
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '2em')
      .attr('fill', this.colors.lightAccent)
      .attr('font-size', '10px')
      .text((d: d3.HierarchyPointNode<TreeNode>) => `≤ ${d.data.threshold!.toFixed(1)}`);
    
    // Add impurity info
    svgContainer.selectAll<SVGGElement, d3.HierarchyPointNode<TreeNode>>('g.node')
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', (d: d3.HierarchyPointNode<TreeNode>) => d.data.value ? '3.5em' : '3.5em')
      .attr('fill', this.colors.mutedText)
      .attr('font-size', '8px')
      .text((d: d3.HierarchyPointNode<TreeNode>) => `Gini=${d.data.impurity.toFixed(2)}`);
    
    // Add samples info
    svgContainer.selectAll<SVGGElement, d3.HierarchyPointNode<TreeNode>>('g.node')
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', (d: d3.HierarchyPointNode<TreeNode>) => d.data.value ? '4.7em' : '4.7em')
      .attr('fill', this.colors.mutedText)
      .attr('font-size', '8px')
      .text((d: d3.HierarchyPointNode<TreeNode>) => `n=${d.data.samples}`);
    
    // Add tree title
    svgContainer.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', '16px')
      .text(`Tree ${tree.id + 1}`);
  }

  private visualizeTreePrediction(tree: Tree, testSample: Sample, svgContainer: d3.Selection<SVGSVGElement, unknown, null, undefined>, width: number, height: number): void {
    // Clear and redraw tree
    this.drawTree(tree, svgContainer, width, height);
    
    // Trace prediction path
    let node = tree.root;
    let pathNodes = [node];
    
    while (node.left && node.right) {
      const feature = node.feature!;
      const threshold = node.threshold!;
      
      if ((testSample as any)[feature] <= threshold) {
        node = node.left;
      } else {
        node = node.right;
      }
      
      pathNodes.push(node);
    }
    
    // We need to properly handle the hierarchy with explicit types
    const treeHierarchy = d3.hierarchy<TreeNode>(tree.root, (d) => {
      const children: TreeNode[] = [];
      if (d.left) children.push(d.left);
      if (d.right) children.push(d.right);
      return children.length > 0 ? children : null;
    });
    
    const treeLayout = d3.tree<TreeNode>().size([width - 60, height - 60]);
    const treeData = treeLayout(treeHierarchy);
    
    // Store the reference to this for nested functions
    const self = this;
    
    // Find nodes in the path
    const nodeElements = svgContainer.selectAll<SVGGElement, d3.HierarchyPointNode<TreeNode>>('g.node');
    
    // Highlight path
    nodeElements.select('circle')
      .transition()
      .duration(300)
      .attr('stroke', function(d: d3.HierarchyPointNode<TreeNode>) {
        const nodeId = d.data.id;
        const isInPath = pathNodes.some(n => n.id === nodeId);
        return isInPath ? self.colors.accent : self.colors.white;
      })
      .attr('stroke-width', function(d: d3.HierarchyPointNode<TreeNode>) {
        const nodeId = d.data.id;
        const isInPath = pathNodes.some(n => n.id === nodeId);
        return isInPath ? 3 : 1.5;
      });
    
    // Find links in the path
    const links = svgContainer.selectAll<SVGPathElement, d3.HierarchyPointLink<TreeNode>>('path');
    
    // Highlight links
    links.transition()
      .duration(300)
      .attr('stroke', function(d: d3.HierarchyPointLink<TreeNode>) {
        const sourceId = d.source.data.id;
        const targetId = d.target.data.id;
        const isInPath = pathNodes.some(n => n.id === sourceId) && 
                         pathNodes.some(n => n.id === targetId);
        return isInPath ? self.colors.accent : self.colors.mutedText;
      })
      .attr('stroke-width', function(d: d3.HierarchyPointLink<TreeNode>) {
        const sourceId = d.source.data.id;
        const targetId = d.target.data.id;
        const isInPath = pathNodes.some(n => n.id === sourceId) && 
                         pathNodes.some(n => n.id === targetId);
        return isInPath ? 2.5 : 1.5;
      });
    
    // Add title showing the prediction
    svgContainer.append('text')
      .attr('x', width / 2)
      .attr('y', 40)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.accent)
      .attr('font-size', '14px')
      .text(`Predicts: Class ${node.value}`)
      .attr('opacity', 0)
      .transition()
      .duration(500)
      .attr('opacity', 1);
  }

  private drawForest(): void {
    const container = this.forestContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    this.forestSvg.selectAll('*').remove();
    
    // Add background
    this.forestSvg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);
    
    // Draw mini trees
    const treeWidth = width / this.numTrees;
    const treeHeight = height * 0.6;
    const treeY = height * 0.3;
    
    for (let i = 0; i < this.numTrees; i++) {
      const tree = this.trees[i];
      const x = treeWidth * i + treeWidth / 2;
      
      // Draw tree icon (simplified tree)
      const treeGroup = this.forestSvg.append('g')
        .attr('transform', `translate(${x}, ${treeY})`)
        .attr('opacity', 0)
        .transition()
        .duration(500)
        .delay(i * 200)
        .attr('opacity', 1);
      
      // Draw trunk
      this.forestSvg.append('rect')
        .attr('x', x - 5)
        .attr('y', treeY + 30)
        .attr('width', 10)
        .attr('height', 40)
        .attr('fill', '#8B4513')
        .attr('rx', 2)
        .attr('ry', 2);
      
      // Draw foliage levels
      const colors = [
        this.colors.primary,
        this.colors.secondary,
        tree.features.includes('x1') && tree.features.includes('x2') ? 
          this.colors.success : 
          this.colors.warning
      ];
      
      for (let j = 0; j < 3; j++) {
        const radius = 25 - j * 5;
        this.forestSvg.append('circle')
          .attr('cx', x)
          .attr('cy', treeY - j * 20)
          .attr('r', radius)
          .attr('fill', colors[j])
          .attr('opacity', 0.8);
      }
      
      // Add tree label
      this.forestSvg.append('text')
        .attr('x', x)
        .attr('y', treeY + 90)
        .attr('text-anchor', 'middle')
        .attr('fill', this.colors.text)
        .attr('font-size', '14px')
        .text(`Tree ${i + 1}`);
      
      // Add features used
      this.forestSvg.append('text')
        .attr('x', x)
        .attr('y', treeY + 110)
        .attr('text-anchor', 'middle')
        .attr('fill', this.colors.mutedText)
        .attr('font-size', '12px')
        .text(`Features: ${tree.features.join(', ')}`);
    }
    
    // Add forest title
    this.forestSvg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', '18px')
      .text(`Random Forest (${this.numTrees} Trees)`);
  }

  private drawPrediction(): void {
    const container = this.predictionContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    this.predictionSvg.selectAll('*').remove();
    
    // Add background
    this.predictionSvg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);
    
    if (!this.testSample || this.predictionResults.length === 0) {
      // No prediction yet
      this.predictionSvg.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', this.colors.mutedText)
        .attr('font-size', '16px')
        .text('Prediction will appear here');
      return;
    }
    
    // Count predictions by class
    const classCounts: Record<string, number> = {};
    for (const result of this.predictionResults) {
      if (!classCounts[result.prediction]) {
        classCounts[result.prediction] = 0;
      }
      classCounts[result.prediction]++;
    }
    
    // Find majority class
    let maxCount = 0;
    let majorityClass = '';
    for (const cls in classCounts) {
      if (classCounts[cls] > maxCount) {
        maxCount = classCounts[cls];
        majorityClass = cls;
      }
    }
    
    // Draw vote bars
    const barHeight = 40;
    const barMargin = 20;
    const barY = height / 2 - barHeight / 2;
    
    let totalVotes = 0;
    for (const cls in classCounts) {
      totalVotes += classCounts[cls];
    }
    
    const classes = Object.keys(classCounts).sort();
    const barWidth = (width - 100) / classes.length;
    
    for (let i = 0; i < classes.length; i++) {
      const cls = classes[i];
      const x = 50 + i * barWidth;
      const votePercentage = classCounts[cls] / totalVotes;
      const voteHeight = barHeight + (height / 2 - barHeight) * votePercentage;
      
      // Draw bar background
      this.predictionSvg.append('rect')
        .attr('x', x)
        .attr('y', barY)
        .attr('width', barWidth - barMargin)
        .attr('height', barHeight)
        .attr('fill', this.colors.elementBackground)
        .attr('rx', 6)
        .attr('ry', 6);
      
      // Draw vote bar with animation
      this.predictionSvg.append('rect')
        .attr('x', x)
        .attr('y', barY + barHeight)
        .attr('width', barWidth - barMargin)
        .attr('height', 0)
        .attr('fill', cls === 'A' ? this.colors.primary : this.colors.secondary)
        .attr('rx', 6)
        .attr('ry', 6)
        .transition()
        .duration(1000)
        .attr('y', barY + barHeight - voteHeight)
        .attr('height', voteHeight);
      
      // Add class label
      this.predictionSvg.append('text')
        .attr('x', x + (barWidth - barMargin) / 2)
        .attr('y', barY + barHeight + 20)
        .attr('text-anchor', 'middle')
        .attr('fill', this.colors.text)
        .attr('font-size', '14px')
        .text(`Class ${cls}`);
      
      // Add vote count
      this.predictionSvg.append('text')
        .attr('x', x + (barWidth - barMargin) / 2)
        .attr('y', barY + barHeight + 40)
        .attr('text-anchor', 'middle')
        .attr('fill', this.colors.mutedText)
        .attr('font-size', '12px')
        .text(`${classCounts[cls]} votes (${(votePercentage * 100).toFixed(0)}%)`);
      
      // Highlight majority class
      if (cls === majorityClass) {
        this.predictionSvg.append('text')
          .attr('x', x + (barWidth - barMargin) / 2)
          .attr('y', barY - 15)
          .attr('text-anchor', 'middle')
          .attr('fill', this.colors.accent)
          .attr('font-size', '12px')
          .text('MAJORITY');
        
        this.predictionSvg.append('polygon')
          .attr('points', `${x + (barWidth - barMargin) / 2 - 10},${barY - 10} ${x + (barWidth - barMargin) / 2 + 10},${barY - 10} ${x + (barWidth - barMargin) / 2},${barY}`)
          .attr('fill', this.colors.accent);
      }
    }
    
    // Add prediction title
    this.predictionSvg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', '18px')
      .text('Forest Prediction');
    
    // Add final prediction
    this.predictionSvg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.white)
      .attr('font-size', '20px')
      .attr('font-weight', 'bold')
      .text(`Final Prediction: Class ${majorityClass}`)
      .attr('opacity', 0)
      .transition()
      .duration(800)
      .delay(1000)
      .attr('opacity', 1);
  }

  private updateVisualization(treeIndex?: number): void {
    // Update dataset visualization
    this.updateDatasetViz(treeIndex);
    
    // Update tree visualization if a specific tree is selected
    if (treeIndex !== undefined && this.trees[treeIndex]) {
      const container = this.treeContainer.nativeElement;
      this.drawTree(this.trees[treeIndex], this.treeSvg, container.clientWidth, container.clientHeight);
    }
    
    // Update forest visualization
    this.drawForest();
    
    // Update prediction if we have a test sample
    if (this.testSample && this.predictionResults.length > 0) {
      this.drawPrediction();
    }
  }

  private updateDatasetViz(treeIndex?: number): void {
    if (treeIndex === undefined) {
      // Reset all selections
      this.datasetSvg.selectAll<SVGCircleElement, Sample>('circle.sample')
        .transition()
        .duration(300)
        .attr('r', 2)
        .attr('fill-opacity', 1)
        .attr('stroke', 'none');
      return;
    }
    
    // Update points to show bootstrap samples
    this.datasetSvg.selectAll<SVGCircleElement, Sample>('circle.sample')
      .transition()
      .duration(300)
      .attr('r', function(d) { return d.selected[treeIndex] ? 3 : 2; })
      .attr('fill-opacity', function(d) { return d.selected[treeIndex] ? 1 : 0.3; })
      .attr('stroke', (d) => d.selected[treeIndex] ? this.colors.white : 'none')
      .attr('stroke-width', 0.5);
  }

  private initializeSimulationSteps(): void {
    this.simulationSteps = [
      {
        title: 'Introduction',
        description: 'Random Forests combine multiple decision trees trained on different data subsets with random feature selection to improve accuracy and reduce overfitting.',
        action: () => {
          // Reset visualizations
          this.updateVisualization();
        }
      },
      {
        title: 'Dataset',
        description: 'We start with a dataset containing samples from two classes (A and B) with two features (x1 and x2).',
        action: () => {
          // Show all data points
          this.updateVisualization();
        }
      },
      {
        title: 'Bootstrap Sampling',
        description: 'For each tree, we create a bootstrap sample by randomly selecting samples with replacement from the original dataset.',
        action: () => {
          // Highlight samples for the first tree
          this.updateVisualization(0);
        }
      },
      {
        title: 'Feature Randomness',
        description: 'At each node, only a random subset of features is considered for splitting (typically sqrt(num_features) for classification).',
        action: () => {
          // Show the first tree
          const container = this.treeContainer.nativeElement;
          this.drawTree(this.trees[0], this.treeSvg, container.clientWidth, container.clientHeight);
        }
      },
      {
        title: 'Growing Trees',
        description: 'Each tree is grown independently using its bootstrap sample and random feature selection at each node.',
        action: () => {
          // Show a different tree
          const container = this.treeContainer.nativeElement;
          this.drawTree(this.trees[1], this.treeSvg, container.clientWidth, container.clientHeight);
        }
      },
      {
        title: 'Multiple Trees',
        description: 'A Random Forest contains multiple trees, each trained on different data and with different feature subsets.',
        action: () => {
          // Show the forest
          this.drawForest();
        }
      },
      {
        title: 'Making Predictions',
        description: 'To predict a new sample, we run it through all trees and take a majority vote (for classification) or average (for regression).',
        action: () => {
          // Make prediction for test sample
          if (this.testSample) {
            this.makeForestPrediction(this.testSample);
            this.drawPrediction();
          }
        }
      },
      {
        title: 'Tree Prediction',
        description: 'Let\'s see how a single tree makes its prediction by tracing the path through the tree.',
        action: () => {
          // Show prediction path for the first tree
          if (this.testSample) {
            const container = this.treeContainer.nativeElement;
            this.visualizeTreePrediction(
              this.trees[0], 
              this.testSample, 
              this.treeSvg, 
              container.clientWidth, 
              container.clientHeight
            );
          }
        }
      },
      {
        title: 'Different Tree, Different Path',
        description: 'Different trees may have different structures and predictions due to bootstrap sampling and feature randomness.',
        action: () => {
          // Show prediction path for another tree
          if (this.testSample) {
            const container = this.treeContainer.nativeElement;
            this.visualizeTreePrediction(
              this.trees[1], 
              this.testSample, 
              this.treeSvg, 
              container.clientWidth, 
              container.clientHeight
            );
          }
        }
      },
      {
        title: 'Aggregating Predictions',
        description: 'The final prediction is determined by aggregating predictions from all trees (majority voting for classification).',
        action: () => {
          // Show the final prediction
          this.drawPrediction();
        }
      },
      {
        title: 'Benefits of Random Forests',
        description: 'Random Forests reduce overfitting, handle high-dimensional data well, and provide feature importance metrics.',
        action: () => {
          // Reset visualizations to show the complete picture
          this.updateVisualization();
          this.drawForest();
          this.drawPrediction();
        }
      }
    ];
  }

  // Simulation control methods
  public startSimulation(): void {
    if (this.simulationState === 'playing') return;
    
    this.simulationState = 'playing';
    this.playNextStep();
    
    this.simulationInterval = setInterval(() => {
      this.playNextStep();
    }, this.simulationSpeed);
  }

  public pauseSimulation(): void {
    this.simulationState = 'paused';
    if (this.simulationInterval) {
      clearInterval(this.simulationInterval);
    }
  }

  public stopSimulation(): void {
    this.simulationState = 'stopped';
    this.currentStepIndex = 0;
    if (this.simulationInterval) {
      clearInterval(this.simulationInterval);
    }
    
    // Reset visualizations
    this.currentExplanation = 'Welcome to the Random Forest Simulation. Press Play to start or use the Step buttons to manually progress.';
    this.updateVisualization();
  }

  public nextStep(): void {
    if (this.simulationState === 'playing') {
      this.pauseSimulation();
    }
    
    this.playNextStep();
  }

  public previousStep(): void {
    if (this.simulationState === 'playing') {
      this.pauseSimulation();
    }
    
    if (this.currentStepIndex > 0) {
      this.currentStepIndex--;
      this.playCurrentStep();
    }
  }

  private playNextStep(): void {
    if (this.currentStepIndex < this.simulationSteps.length) {
      this.playCurrentStep();
      this.currentStepIndex++;
    } else {
      this.pauseSimulation();
      this.currentStepIndex = 0;
    }
  }

  private playCurrentStep(): void {
    const step = this.simulationSteps[this.currentStepIndex];
    this.currentExplanation = `${step.title}: ${step.description}`;
    step.action();
  }

  // UI event handlers
  public onNumTreesChange(event: Event): void {
    this.numTrees = parseInt((event.target as HTMLInputElement).value);
    this.regenerateSimulation();
  }

  public onMaxDepthChange(event: Event): void {
    this.maxDepth = parseInt((event.target as HTMLInputElement).value);
    this.regenerateSimulation();
  }

  public onFeatureRandomnessChange(event: Event): void {
    this.featureRandomness = (event.target as HTMLInputElement).checked;
    this.regenerateSimulation();
  }

  public onShowTestSampleChange(event: Event): void {
    this.showTestSample = (event.target as HTMLInputElement).checked;
    this.regenerateSimulation();
  }

  public onSimulationSpeedChange(event: Event): void {
    this.simulationSpeed = parseInt((event.target as HTMLInputElement).value);
    
    if (this.simulationState === 'playing') {
      // Restart with new speed
      this.pauseSimulation();
      this.startSimulation();
    }
  }

  private regenerateSimulation(): void {
    // Stop any ongoing simulation
    this.stopSimulation();
    
    // Regenerate data and forest
    this.generateSampleData();
    this.buildForest();
    
    // Update visualizations
    this.initializeVisualization();
    
    // Update 3D forest
    this.create3DForest();
  }
}