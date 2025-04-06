import { Component, ElementRef, OnInit, ViewChild, AfterViewInit, NgZone, OnDestroy, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface TreeNode {
  id: string;
  feature?: string;
  threshold?: number;
  value?: string;
  left?: TreeNode;
  right?: TreeNode;
  samples: number;
  depth: number;
  impurity: number;
  featureImportance?: number;
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
  oobForTree: boolean[];
}

interface Tree {
  id: number;
  root: TreeNode;
  samples: number[];
  features: string[];
  oobError?: number;
  oobIndices?: number[];
}

interface SimulationStep {
  title: string;
  description: string;
  action: () => void;
}

interface FeatureImportance {
  feature: string;
  importance: number;
  permutationImportance?: number;
}

@Component({
  selector: 'app-random-forest-visualization',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './random-forest-visualization.component.html',
  styleUrls: ['./random-forest-visualization.component.scss']
})
export class RandomForestVisualizationComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('simulationContainer') simulationContainer!: ElementRef;
  @ViewChild('datasetContainer') datasetContainer!: ElementRef;
  @ViewChild('treeContainer') treeContainer!: ElementRef;
  @ViewChild('forestContainer') forestContainer!: ElementRef;
  @ViewChild('predictionContainer') predictionContainer!: ElementRef;
  @ViewChild('featureImportanceContainer') featureImportanceContainer!: ElementRef;
  @ViewChild('oobErrorContainer') oobErrorContainer!: ElementRef;
  @ViewChild('comparisonContainer') comparisonContainer!: ElementRef;
  @ViewChild('canvas3d') canvas3d!: ElementRef;

  // Simulation parameters
  public numTrees = 5;
  public maxDepth = 3;
  public numSamples = 100;
  public featureRandomness = true;
  public showTestSample = true;
  public viewMode: 'basic' | 'advanced' = 'basic';

  // Research paper references
  public researchReferences = [
    {
      title: "Random Forests",
      author: "Leo Breiman",
      year: 2001,
      journal: "Machine Learning",
      volume: "45(1)",
      pages: "5-32",
      doi: "10.1023/A:1010933404324",
      link: "https://link.springer.com/article/10.1023/A:1010933404324"
    },
    {
      title: "Bagging Predictors",
      author: "Leo Breiman",
      year: 1996,
      journal: "Machine Learning",
      volume: "24(2)",
      pages: "123-140",
      doi: "10.1007/BF00058655",
      link: "https://link.springer.com/article/10.1007/BF00058655"
    }
  ];

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
  public selectedTreeIndex = 0;
  public showOobSamples = false;
  public showSingleTree = false;

  // Data
  private samples: Sample[] = [];
  private trees: Tree[] = [];
  private featureImportance: FeatureImportance[] = [];
  private singleTree: Tree | null = null;
  private testSample: Sample | null = null;
  private predictionResults: any[] = [];
  private oobErrorRates: number[] = [];
  private oobCurve: { numTrees: number, error: number }[] = [];

  // SVG elements
  private datasetSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private treeSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private forestSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private predictionSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private featureImportanceSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private oobErrorSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private comparisonSvg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;

  // 3D elements
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private forest3D: THREE.Object3D[] = [];
  private animationFrameId?: number;

  // Current step explanation
  public currentExplanation = 'Welcome to the Random Forest Simulation. Press Play to start or use the Step buttons to manually progress. Toggle between Basic and Advanced view modes to see different levels of detail.';

  // Simulation steps
  // This is intentionally private to avoid direct access from template
  private simulationSteps: SimulationStep[] = [];

  constructor(private ngZone: NgZone) { }

  ngOnInit(): void {
    this.generateSampleData();
    this.buildForest();
    this.buildSingleDecisionTree();
    this.calculateFeatureImportance();
    this.calculateOobError();
    this.initializeSimulationSteps();
  }

  ngAfterViewInit(): void {
    this.initializeVisualization();
    this.init3DForest();
    this.animate();
  }

  ngOnDestroy(): void {
    if (this.simulationInterval) {
      clearInterval(this.simulationInterval);
    }
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    // Clean up THREE.js resources
    if (this.renderer) {
      this.renderer.dispose();
    }
    this.forest3D.forEach(object => {
      if (object instanceof THREE.Mesh) {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach(material => material.dispose());
          } else {
            object.material.dispose();
          }
        }
      }
    });
  }

  @HostListener('window:resize')
  onWindowResize(): void {
    if (this.camera && this.renderer && this.canvas3d) {
      const canvas = this.canvas3d.nativeElement;
      this.camera.aspect = canvas.clientWidth / canvas.clientHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    }
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

      // Create levels based on maxDepth
      createLevel(2, 1, 1);  // Root
      if (this.maxDepth >= 1) createLevel(3, 0.8, 2); // Level 1
      if (this.maxDepth >= 2) createLevel(4, 0.6, 4); // Level 2
      if (this.maxDepth >= 3) createLevel(5, 0.5, 8); // Level 3
      if (this.maxDepth >= 4) createLevel(6, 0.4, 16); // Level 4

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

    // Add feature importance indicators
    if (this.featureImportance.length > 0) {
      // Sort features by importance
      const sortedFeatures = [...this.featureImportance].sort((a, b) => b.importance - a.importance);

      // Create pillars to represent feature importance
      const pillarGroup = new THREE.Group();
      pillarGroup.position.set(-10, 0, -5);

      sortedFeatures.forEach((feature, index) => {
        const height = feature.importance * 5; // Scale for visualization
        const pillarGeometry = new THREE.BoxGeometry(1, height, 1);
        const pillarMaterial = new THREE.MeshStandardMaterial({
          color: index === 0 ? this.hexToRgb(this.colors.success) : this.hexToRgb(this.colors.primary),
          roughness: 0.5,
          metalness: 0.2
        });

        const pillar = new THREE.Mesh(pillarGeometry, pillarMaterial);
        pillar.position.set(index * 2, height / 2, 0);

        // Add label
        const textGeometry = new THREE.PlaneGeometry(1.5, 0.5);
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 30px Arial';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(feature.feature, 64, 32);

          const texture = new THREE.Texture(canvas);
          texture.needsUpdate = true;

          const textMaterial = new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true,
            side: THREE.DoubleSide
          });

          const text = new THREE.Mesh(textGeometry, textMaterial);
          text.position.set(index * 2, height + 0.5, 0);
          text.rotation.x = -Math.PI / 2;

          pillarGroup.add(text);
        }

        pillarGroup.add(pillar);
      });

      this.scene.add(pillarGroup);
      this.forest3D.push(pillarGroup);
    }

    // Add a decorative label for the visualization
    const createLabel = (text: string, position: THREE.Vector3) => {
      const textGeometry = new THREE.PlaneGeometry(10, 2);
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 36px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 256, 64);

        const texture = new THREE.Texture(canvas);
        texture.needsUpdate = true;

        const textMaterial = new THREE.MeshBasicMaterial({
          map: texture,
          transparent: true,
          side: THREE.DoubleSide
        });

        const textMesh = new THREE.Mesh(textGeometry, textMaterial);
        textMesh.position.copy(position);

        this.scene.add(textMesh);
        this.forest3D.push(textMesh);
      }
    };

    createLabel('Random Forest 3D Visualization', new THREE.Vector3(0, 10, -10));
  }

  private animate(): void {
    this.ngZone.runOutsideAngular(() => {
      const animate = () => {
        this.animationFrameId = requestAnimationFrame(animate);

        this.controls.update();

        // Rotate trees slowly
        this.forest3D.forEach((object, index) => {
          if (object.name && object.name.startsWith('tree-')) {
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
        selected: Array(this.numTrees).fill(false),
        oobForTree: Array(this.numTrees).fill(false)
      });
    }

    // Create class B: cluster around (75, 75)
    for (let i = this.numSamples / 2; i < this.numSamples; i++) {
      this.samples.push({
        id: i,
        x1: 75 + Math.random() * 15 - 7.5,
        x2: 75 + Math.random() * 15 - 7.5,
        class: 'B',
        selected: Array(this.numTrees).fill(false),
        oobForTree: Array(this.numTrees).fill(false)
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
      selected: Array(this.numTrees).fill(false),
      oobForTree: Array(this.numTrees).fill(false)
    };
  }

  private bootstrapSamples(treeIndex: number): { sampledIndices: number[], oobIndices: number[] } {
    // Bootstrap: Sample with replacement
    const sampledIndices: number[] = [];
    const selectedSet = new Set<number>();

    // Reset selection state
    this.samples.forEach(sample => {
      sample.selected[treeIndex] = false;
      sample.oobForTree[treeIndex] = true; // Initially mark all as OOB
    });

    // Select samples with replacement
    for (let i = 0; i < this.numSamples; i++) {
      const idx = Math.floor(Math.random() * this.numSamples);
      sampledIndices.push(idx);
      selectedSet.add(idx);
      this.samples[idx].selected[treeIndex] = true;
      this.samples[idx].oobForTree[treeIndex] = false; // This sample is in the bootstrap sample
    }

    // Find OOB samples (not selected)
    const oobIndices: number[] = [];
    for (let i = 0; i < this.numSamples; i++) {
      if (!selectedSet.has(i)) {
        oobIndices.push(i);
      }
    }

    return { sampledIndices, oobIndices };
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

  private findBestSplit(samples: Sample[], features: string[]): { feature: string; threshold: number; impurityDecrease: number; leftSamples: Sample[]; rightSamples: Sample[] } {
    let bestFeature = '';
    let bestThreshold = 0;
    let bestImpurityDecrease = -Infinity;
    let bestLeftSamples: Sample[] = [];
    let bestRightSamples: Sample[] = [];

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
          bestLeftSamples = leftSamples;
          bestRightSamples = rightSamples;
        }
      }
    }

    return {
      feature: bestFeature,
      threshold: bestThreshold,
      impurityDecrease: bestImpurityDecrease,
      leftSamples: bestLeftSamples,
      rightSamples: bestRightSamples
    };
  }

  private buildTree(
    samples: Sample[],
    features: string[],
    depth: number,
    maxDepth: number,
    nodeId: string,
    featureImportanceMap: Map<string, number> = new Map()
  ): TreeNode {
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
    const { feature, threshold, impurityDecrease, leftSamples, rightSamples } = this.findBestSplit(samples, features);

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

    // Update feature importance
    const currentImportance = featureImportanceMap.get(feature) || 0;
    featureImportanceMap.set(feature, currentImportance + impurityDecrease * samples.length);

    // Create internal node
    const node: TreeNode = {
      id: nodeId,
      feature,
      threshold,
      samples: samples.length,
      depth,
      impurity,
      featureImportance: impurityDecrease * samples.length
    };

    // Recursively build left and right subtrees
    node.left = this.buildTree(leftSamples, features, depth + 1, maxDepth, nodeId + 'L', featureImportanceMap);
    node.right = this.buildTree(rightSamples, features, depth + 1, maxDepth, nodeId + 'R', featureImportanceMap);

    return node;
  }

  private buildForest(): void {
    this.trees = [];
    const forestFeatureImportance = new Map<string, number>();

    for (let i = 0; i < this.numTrees; i++) {
      // Bootstrap samples
      const { sampledIndices, oobIndices } = this.bootstrapSamples(i);
      const sampledData = sampledIndices.map(idx => this.samples[idx]);

      // Select random features
      const selectedFeatures = this.selectRandomFeatures();

      // Map to track feature importance for this tree
      const treeFeatureImportance = new Map<string, number>();

      // Build tree
      const root = this.buildTree(sampledData, selectedFeatures, 0, this.maxDepth, `tree${i}_`, treeFeatureImportance);

      // Update forest-level feature importance
      for (const [feature, importance] of treeFeatureImportance.entries()) {
        const currentImportance = forestFeatureImportance.get(feature) || 0;
        forestFeatureImportance.set(feature, currentImportance + importance / this.numTrees);
      }

      this.trees.push({
        id: i,
        root,
        samples: sampledIndices,
        features: selectedFeatures,
        oobIndices
      });
    }

    // Convert Map to array for visualization
    this.featureImportance = [];
    for (const [feature, importance] of forestFeatureImportance.entries()) {
      this.featureImportance.push({
        feature,
        importance: importance / this.numSamples // Normalize by number of samples
      });
    }

    // Sort by importance
    this.featureImportance.sort((a, b) => b.importance - a.importance);
  }

  private buildSingleDecisionTree(): void {
    // Build a single decision tree with all samples and all features
    const allFeatures = ['x1', 'x2'];
    const featureImportanceMap = new Map<string, number>();

    const root = this.buildTree(this.samples, allFeatures, 0, this.maxDepth, 'singleTree_', featureImportanceMap);

    this.singleTree = {
      id: -1, // Special ID for single tree
      root,
      samples: this.samples.map((_, i) => i),
      features: allFeatures
    };
  }

  private calculateFeatureImportance(): void {
    // Calculate permutation importance
    for (const featureInfo of this.featureImportance) {
      const feature = featureInfo.feature;

      // Store original values
      const originalValues = this.samples.map(s => (s as any)[feature]);

      // Predict with original values
      const originalPredictions = this.samples.map(s => this.makeForestPredictionForSample(s));
      const originalAccuracy = originalPredictions.filter((pred, i) => pred === this.samples[i].class).length / this.samples.length;

      // Shuffle feature values
      const shuffledValues = [...originalValues].sort(() => Math.random() - 0.5);

      // Replace with shuffled values
      this.samples.forEach((s, i) => (s as any)[feature] = shuffledValues[i]);

      // Predict with shuffled values
      const shuffledPredictions = this.samples.map(s => this.makeForestPredictionForSample(s));
      const shuffledAccuracy = shuffledPredictions.filter((pred, i) => pred === this.samples[i].class).length / this.samples.length;

      // Calculate permutation importance as decrease in accuracy
      featureInfo.permutationImportance = originalAccuracy - shuffledAccuracy;

      // Restore original values
      this.samples.forEach((s, i) => (s as any)[feature] = originalValues[i]);
    }

    // Re-sort by permutation importance
    this.featureImportance.sort((a, b) =>
      (b.permutationImportance || 0) - (a.permutationImportance || 0)
    );
  }

  private calculateOobError(): void {
    // Calculate OOB error for each tree and for increasing numbers of trees
    this.oobErrorRates = Array(this.numTrees).fill(0);
    this.oobCurve = [];

    for (let t = 0; t < this.numTrees; t++) {
      const tree = this.trees[t];
      let correctPredictions = 0;
      let totalOobSamples = 0;

      // Get OOB samples for this tree
      const oobSamples = tree.oobIndices ? tree.oobIndices.map(idx => this.samples[idx]) : [];
      totalOobSamples = oobSamples.length;

      // Predict each OOB sample
      for (const sample of oobSamples) {
        const prediction = this.predict(sample, tree);
        if (prediction === sample.class) {
          correctPredictions++;
        }
      }

      // Calculate error rate
      const errorRate = totalOobSamples > 0 ? 1 - (correctPredictions / totalOobSamples) : 0;
      this.oobErrorRates[t] = errorRate;
      this.trees[t].oobError = errorRate;

      // Calculate cumulative OOB error for forest of size 1 to t+1
      const forestSize = t + 1;

      // For each sample, get majority vote from trees where it's OOB
      let forestCorrectPredictions = 0;

      for (const sample of this.samples) {
        // Get indices of trees where this sample is OOB
        const oobTreeIndices = Array.from({ length: forestSize }, (_, i) => i)
          .filter(treeIdx => sample.oobForTree[treeIdx]);

        if (oobTreeIndices.length === 0) continue; // Skip if not OOB for any tree

        // Get votes from OOB trees
        const votes: Record<string, number> = {};
        for (const treeIdx of oobTreeIndices) {
          const prediction = this.predict(sample, this.trees[treeIdx]);
          votes[prediction] = (votes[prediction] || 0) + 1;
        }

        // Find majority vote
        let maxVotes = 0;
        let majorityClass = '';
        for (const cls in votes) {
          if (votes[cls] > maxVotes) {
            maxVotes = votes[cls];
            majorityClass = cls;
          }
        }

        if (majorityClass === sample.class) {
          forestCorrectPredictions++;
        }
      }

      const forestErrorRate = 1 - (forestCorrectPredictions / this.samples.length);

      this.oobCurve.push({
        numTrees: forestSize,
        error: forestErrorRate
      });
    }
  }

  private predict(sample: Sample, tree: Tree): string {
    let node = tree.root;

    while (node.left && node.right && node.feature) {
      const feature = node.feature;
      const threshold = node.threshold!;

      if ((sample as any)[feature] <= threshold) {
        node = node.left;
      } else {
        node = node.right;
      }
    }

    return node.value || '';
  }

  private makeForestPredictionForSample(sample: Sample): string {
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

    return majorityClass;
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

    // Initialize feature importance and OOB error visualizations for advanced mode
    if (this.featureImportanceContainer) {
      this.initializeFeatureImportanceViz();
    }

    if (this.oobErrorContainer) {
      this.initializeOobErrorViz();
    }

    if (this.comparisonContainer) {
      this.initializeComparisonViz();
    }
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
      .attr('cx', function (d) { return d.x1; })
      .attr('cy', function (d) { return d.x2; })
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

    // Add OOB legend
    this.datasetSvg.append('circle')
      .attr('cx', 80)
      .attr('cy', 28)
      .attr('r', 2)
      .attr('fill', this.colors.white)
      .attr('stroke', this.colors.error)
      .attr('stroke-width', 0.5);

    this.datasetSvg.append('text')
      .attr('x', 84)
      .attr('y', 28)
      .attr('dominant-baseline', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', 3)
      .text('OOB Sample');
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

  private initializeFeatureImportanceViz(): void {
    const container = this.featureImportanceContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;

    this.featureImportanceSvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    // Add background
    this.featureImportanceSvg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);
  }

  private initializeOobErrorViz(): void {
    const container = this.oobErrorContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;

    this.oobErrorSvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    // Add background
    this.oobErrorSvg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);
  }

  private initializeComparisonViz(): void {
    const container = this.comparisonContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;

    this.comparisonSvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    // Add background
    this.comparisonSvg.append('rect')
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

    // We need to properly handle the hierarchy
    const treeHierarchy = d3.hierarchy<TreeNode>(tree.root, (d) => {
      const children: TreeNode[] = [];
      if (d.left) children.push(d.left);
      if (d.right) children.push(d.right);
      return children.length > 0 ? children : null;
    });

    const treeLayout = d3.tree<TreeNode>().size([width - 60, height - 60]);
    const treeData = treeLayout(treeHierarchy);

    // Create a proper D3 link generator
    const linkGenerator = d3.linkVertical<any, any>()
      .x((d) => d.x)
      .y((d) => d.y);

    // Draw links with animation
    const links = svgContainer.append('g')
      .attr('transform', `translate(30, 40)`)
      .selectAll('path')
      .data(treeData.links())
      .enter()
      .append('path')
      .attr('d', (d) => {
        // Create a straight line from source to source as starting point
        const source = { x: d.source.x, y: d.source.y };
        const target = { x: d.source.x, y: d.source.y };
        return linkGenerator({ source, target });
      })
      .attr('stroke', this.colors.mutedText)
      .attr('stroke-width', 1.5)
      .attr('fill', 'none');

    // Animate links
    links.transition()
      .duration(800)
      .delay((_, i) => i * 100)
      .attr('d', linkGenerator);

    // Draw nodes
    const nodes = svgContainer.append('g')
      .attr('transform', `translate(30, 40)`)
      .selectAll('g')
      .data(treeData.descendants())
      .enter()
      .append('g')
      .attr('transform', (d) => `translate(${d.x}, ${d.y})`)
      .attr('opacity', 0)
      .attr('class', 'node')
      .transition()
      .duration(500)
      .delay((_, i) => i * 100 + 300)
      .attr('opacity', 1);

    // Store the reference to this for nested functions
    const self = this;

    // Add node circles
    svgContainer.selectAll('g.node')
      .append('circle')
      .attr('r', (d: any) => d.data.value ? 18 : 15)
      .attr('fill', function (d: any) {
        if (d.data.value) {
          return d.data.value === 'A' ? self.colors.primary : self.colors.secondary;
        }
        return self.colors.elementBackground;
      })
      .attr('stroke', this.colors.white)
      .attr('stroke-width', 1.5);

    // Add node labels
    svgContainer.selectAll('g.node')
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '.3em')
      .attr('fill', this.colors.white)
      .attr('font-size', '12px')
      .text((d: any) => {
        if (d.data.value) {
          return d.data.value; // Leaf node shows class
        } else {
          return `${d.data.feature!.toUpperCase()}`; // Internal node shows feature
        }
      });

    // Add threshold labels to internal nodes
    svgContainer.selectAll('g.node')
      .filter((d: any) => !d.data.value)
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '2em')
      .attr('fill', this.colors.lightAccent)
      .attr('font-size', '10px')
      .text((d: any) => `≤ ${d.data.threshold!.toFixed(1)}`);

    // Add impurity info
    svgContainer.selectAll('g.node')
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '3.5em')
      .attr('fill', this.colors.mutedText)
      .attr('font-size', '8px')
      .text((d: any) => `Gini=${d.data.impurity.toFixed(2)}`);

    // Add samples info
    svgContainer.selectAll('g.node')
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '4.7em')
      .attr('fill', this.colors.mutedText)
      .attr('font-size', '8px')
      .text((d: any) => `n=${d.data.samples}`);

    // Add feature importance if available
    svgContainer.selectAll('g.node')
      .filter((d: any) => d.data.featureImportance !== undefined)
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '5.9em')
      .attr('fill', this.colors.success)
      .attr('font-size', '8px')
      .text((d: any) => `Imp=${d.data.featureImportance!.toFixed(3)}`);

    // Add tree title
    svgContainer.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', '16px')
      .text(`${tree.id === -1 ? 'Single Decision Tree' : `Tree ${tree.id + 1}`}`);

    // Add OOB error if available
    if (tree.oobError !== undefined) {
      svgContainer.append('text')
        .attr('x', width / 2)
        .attr('y', height - 20)
        .attr('text-anchor', 'middle')
        .attr('fill', this.colors.error)
        .attr('font-size', '14px')
        .text(`OOB Error: ${(tree.oobError * 100).toFixed(1)}%`);
    }

    // Add selected features
    svgContainer.append('text')
      .attr('x', width / 2)
      .attr('y', height - 40)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.lightAccent)
      .attr('font-size', '12px')
      .text(`Features: ${tree.features.join(', ')}`);
  }

  private visualizeTreePrediction(tree: Tree, testSample: Sample, svgContainer: d3.Selection<SVGSVGElement, unknown, null, undefined>, width: number, height: number): void {
    // Clear and redraw tree
    this.drawTree(tree, svgContainer, width, height);

    // Trace prediction path
    let node = tree.root;
    let pathNodes: TreeNode[] = [node];

    while (node.left && node.right && node.feature) {
      const feature = node.feature;
      const threshold = node.threshold!;

      if ((testSample as any)[feature] <= threshold) {
        node = node.left;
      } else {
        node = node.right;
      }

      pathNodes.push(node);
    }

    // We need to properly handle the hierarchy
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
    const nodeElements = svgContainer.selectAll('g.node');

    // Highlight path
    nodeElements.select('circle')
      .transition()
      .duration(300)
      .attr('stroke', function (d: any) {
        const nodeId = d.data.id;
        const isInPath = pathNodes.some(n => n.id === nodeId);
        return isInPath ? self.colors.accent : self.colors.white;
      })
      .attr('stroke-width', function (d: any) {
        const nodeId = d.data.id;
        const isInPath = pathNodes.some(n => n.id === nodeId);
        return isInPath ? 3 : 1.5;
      });

    // Find links in the path
    const links = svgContainer.selectAll('path');

    // Highlight links
    links.transition()
      .duration(300)
      .attr('stroke', function (d: any) {
        const sourceId = d.source.data.id;
        const targetId = d.target.data.id;
        const isInPath = pathNodes.some(n => n.id === sourceId) &&
          pathNodes.some(n => n.id === targetId);
        return isInPath ? self.colors.accent : self.colors.mutedText;
      })
      .attr('stroke-width', function (d: any) {
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

      // Create a clickable group
      const treeGroup = this.forestSvg.append('g')
        .attr('class', 'tree-group')
        .attr('data-tree-id', i)
        .style('cursor', 'pointer')
        .on('click', () => {
          this.selectedTreeIndex = i;
          this.updateVisualization(i);
        });

      treeGroup.append('rect')
        .attr('x', x - treeWidth / 2 + 5)
        .attr('y', 50)
        .attr('width', treeWidth - 10)
        .attr('height', treeHeight)
        .attr('fill', 'transparent')
        .attr('stroke', this.selectedTreeIndex === i ? this.colors.accent : 'transparent')
        .attr('stroke-width', 2)
        .attr('rx', 8)
        .attr('ry', 8);

      // Draw tree icon (simplified tree)
      treeGroup.attr('opacity', 0)
        .transition()
        .duration(500)
        .delay(i * 200)
        .attr('opacity', 1);

      // Draw trunk
      treeGroup.append('rect')
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
        treeGroup.append('circle')
          .attr('cx', x)
          .attr('cy', treeY - j * 20)
          .attr('r', radius)
          .attr('fill', colors[j])
          .attr('opacity', 0.8);
      }

      // Add tree label
      treeGroup.append('text')
        .attr('x', x)
        .attr('y', treeY + 90)
        .attr('text-anchor', 'middle')
        .attr('fill', this.colors.text)
        .attr('font-size', '14px')
        .text(`Tree ${i + 1}`);

      // Add features used
      treeGroup.append('text')
        .attr('x', x)
        .attr('y', treeY + 110)
        .attr('text-anchor', 'middle')
        .attr('fill', this.colors.mutedText)
        .attr('font-size', '12px')
        .text(`Features: ${tree.features.join(', ')}`);

      // Add OOB error if available
      if (tree.oobError !== undefined) {
        treeGroup.append('text')
          .attr('x', x)
          .attr('y', treeY + 130)
          .attr('text-anchor', 'middle')
          .attr('fill', this.colors.error)
          .attr('font-size', '12px')
          .text(`OOB: ${(tree.oobError * 100).toFixed(1)}%`);
      }
    }

    // Add forest title
    this.forestSvg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', '18px')
      .text(`Random Forest (${this.numTrees} Trees)`);

    // Add instruction to click
    this.forestSvg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 20)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.mutedText)
      .attr('font-size', '14px')
      .text('Click on a tree to see its details');
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

    // Add majority voting explanation
    this.predictionSvg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 20)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.mutedText)
      .attr('font-size', '14px')
      .text('Ensemble uses majority voting for classification');
  }

  private drawFeatureImportance(): void {
    if (!this.featureImportanceContainer) return;

    const container = this.featureImportanceContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;

    this.featureImportanceSvg.selectAll('*').remove();

    // Add background
    this.featureImportanceSvg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);

    if (this.featureImportance.length === 0) {
      this.featureImportanceSvg.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', this.colors.mutedText)
        .attr('font-size', '16px')
        .text('Feature Importance will appear here');
      return;
    }

    // Add title
    this.featureImportanceSvg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', '18px')
      .text('Feature Importance');

    // Prepare data for visualization
    const margin = { top: 50, right: 30, bottom: 50, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create scales
    const yScale = d3.scaleBand()
      .domain(this.featureImportance.map(d => d.feature))
      .range([0, innerHeight])
      .padding(0.1);

    // Find max importance value for scale
    const maxImportance = Math.max(
      ...this.featureImportance.map(d => Math.max(d.importance, d.permutationImportance || 0))
    );

    const xScale = d3.scaleLinear()
      .domain([0, maxImportance * 1.1]) // Add 10% for padding
      .range([0, innerWidth]);

    // Create axes
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale);

    // Create chart group
    const chartGroup = this.featureImportanceSvg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add axes
    chartGroup.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis as any)
      .selectAll('text')
      .attr('fill', this.colors.text);

    chartGroup.append('g')
      .call(yAxis as any)
      .selectAll('text')
      .attr('fill', this.colors.text);

    // Add x-axis label
    chartGroup.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 40)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .text('Importance Score');

    // Draw MDI bars
    chartGroup.selectAll('.mdi-bar')
      .data(this.featureImportance)
      .enter()
      .append('rect')
      .attr('class', 'mdi-bar')
      .attr('y', d => yScale(d.feature)!)
      .attr('x', 0)
      .attr('height', yScale.bandwidth() / 2)
      .attr('width', 0)
      .attr('fill', this.colors.primary)
      .transition()
      .duration(1000)
      .attr('width', d => xScale(d.importance));

    // Draw permutation importance bars if available
    if (this.featureImportance[0].permutationImportance !== undefined) {
      chartGroup.selectAll('.perm-bar')
        .data(this.featureImportance)
        .enter()
        .append('rect')
        .attr('class', 'perm-bar')
        .attr('y', d => (yScale(d.feature)! + yScale.bandwidth() / 2))
        .attr('x', 0)
        .attr('height', yScale.bandwidth() / 2)
        .attr('width', 0)
        .attr('fill', this.colors.secondary)
        .transition()
        .duration(1000)
        .delay(500)
        .attr('width', d => xScale(d.permutationImportance || 0));

      // Add legend
      const legendGroup = this.featureImportanceSvg.append('g')
        .attr('transform', `translate(${width - 150}, 60)`);

      legendGroup.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 15)
        .attr('height', 15)
        .attr('fill', this.colors.primary);

      legendGroup.append('text')
        .attr('x', 20)
        .attr('y', 12)
        .attr('fill', this.colors.text)
        .text('MDI');

      legendGroup.append('rect')
        .attr('x', 0)
        .attr('y', 25)
        .attr('width', 15)
        .attr('height', 15)
        .attr('fill', this.colors.secondary);

      legendGroup.append('text')
        .attr('x', 20)
        .attr('y', 37)
        .attr('fill', this.colors.text)
        .text('Permutation');
    }

    // Add explanation
    this.featureImportanceSvg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 20)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.mutedText)
      .attr('font-size', '14px')
      .text('Higher values indicate more important features');
  }

  private drawOobError(): void {
    if (!this.oobErrorContainer) return;

    const container = this.oobErrorContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;

    this.oobErrorSvg.selectAll('*').remove();

    // Add background
    this.oobErrorSvg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);

    if (this.oobCurve.length === 0) {
      this.oobErrorSvg.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', this.colors.mutedText)
        .attr('font-size', '16px')
        .text('OOB Error will appear here');
      return;
    }

    // Add title
    this.oobErrorSvg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', '18px')
      .text('Out-of-Bag Error Rate');

    // Prepare data for visualization
    const margin = { top: 50, right: 30, bottom: 50, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([1, this.numTrees])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, Math.max(0.5, d3.max(this.oobCurve, d => d.error) || 0.5)])
      .range([innerHeight, 0]);

    // Create axes
    const xAxis = d3.axisBottom(xScale).ticks(Math.min(this.numTrees, 10)).tickFormat(d3.format('d'));
    //const yAxis = d3.axisLeft(yScale).tickFormat((d: number) => `${(d * 100).toFixed(0)}%`);
    const yAxis = d3.axisLeft(yScale).tickFormat((d: d3.NumberValue) => `${(Number(d) * 100).toFixed(0)}%`);

    // Create chart group
    const chartGroup = this.oobErrorSvg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add axes
    chartGroup.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .selectAll('text')
      .attr('fill', this.colors.text);

    chartGroup.append('g')
      .call(yAxis)
      .selectAll('text')
      .attr('fill', this.colors.text);

    // Add axes labels
    chartGroup.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 40)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .text('Number of Trees');

    chartGroup.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .text('OOB Error Rate');

    // Create line generator
    const lineGenerator = d3.line<{ numTrees: number, error: number }>()
      .x(d => xScale(d.numTrees))
      .y(d => yScale(d.error))
      .curve(d3.curveMonotoneX);

    // Draw path
    const path = chartGroup.append('path')
      .datum(this.oobCurve)
      .attr('fill', 'none')
      .attr('stroke', this.colors.accent)
      .attr('stroke-width', 3)
      .attr('d', lineGenerator);

    // Animate path
    const pathLength = path.node()!.getTotalLength();

    path.attr('stroke-dasharray', pathLength)
      .attr('stroke-dashoffset', pathLength)
      .transition()
      .duration(2000)
      .attr('stroke-dashoffset', 0);

    // Add dots
    chartGroup.selectAll('.dot')
      .data(this.oobCurve)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(d.numTrees))
      .attr('cy', d => yScale(d.error))
      .attr('r', 0)
      .attr('fill', this.colors.white)
      .attr('stroke', this.colors.accent)
      .attr('stroke-width', 2)
      .transition()
      .duration(1000)
      .delay((_, i) => 2000 + i * 100)
      .attr('r', 4);

    // Add explanation
    this.oobErrorSvg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 20)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.mutedText)
      .attr('font-size', '14px')
      .text('Error typically decreases as more trees are added, then stabilizes');
  }

  private drawTreeComparison(): void {
    if (!this.comparisonContainer) return;
    if (!this.singleTree) return;

    const container = this.comparisonContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;

    this.comparisonSvg.selectAll('*').remove();

    // Add background
    this.comparisonSvg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', this.colors.cardBackground)
      .attr('rx', 12)
      .attr('ry', 12);

    // Add title
    this.comparisonSvg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', '18px')
      .text('Single Tree vs Random Forest Comparison');

    // Draw decision boundaries
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, 100])
      .range([50, width - 50]);

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([height - 50, 50]);

    // Generate grid for visualization
    const gridSize = 50;
    const gridStep = 100 / gridSize;
    const grid: { x: number, y: number, singlePred: string, forestPred: string, match: boolean }[] = [];

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x = i * gridStep + gridStep / 2;
        const y = j * gridStep + gridStep / 2;

        const gridSample: Sample = {
          id: -1,
          x1: x,
          x2: y,
          class: '?',
          selected: [],
          oobForTree: []
        };

        const singlePred = this.predict(gridSample, this.singleTree);
        const forestPred = this.makeForestPredictionForSample(gridSample);

        grid.push({
          x,
          y,
          singlePred,
          forestPred,
          match: singlePred === forestPred
        });
      }
    }

    // Draw single tree predictions
    this.comparisonSvg.append('g')
      .attr('transform', `translate(${width / 4}, ${height / 2})`)
      .selectAll('.single-pred')
      .data(grid)
      .enter()
      .append('rect')
      .attr('class', 'single-pred')
      .attr('x', d => xScale(d.x) - width / 2 - gridStep / 2)
      .attr('y', d => yScale(d.y) - height / 4 - gridStep / 2)
      .attr('width', xScale(gridStep) - xScale(0))
      .attr('height', yScale(0) - yScale(gridStep))
      .attr('fill', d => d.singlePred === 'A' ? this.colors.primary : this.colors.secondary)
      .attr('opacity', 0.7);

    // Add single tree title
    this.comparisonSvg.append('text')
      .attr('x', width / 4)
      .attr('y', 70)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', '16px')
      .text('Single Decision Tree');

    // Draw forest predictions
    this.comparisonSvg.append('g')
      .attr('transform', `translate(${width * 3 / 4}, ${height / 2})`)
      .selectAll('.forest-pred')
      .data(grid)
      .enter()
      .append('rect')
      .attr('class', 'forest-pred')
      .attr('x', d => xScale(d.x) - width / 2 - gridStep / 2)
      .attr('y', d => yScale(d.y) - height / 4 - gridStep / 2)
      .attr('width', xScale(gridStep) - xScale(0))
      .attr('height', yScale(0) - yScale(gridStep))
      .attr('fill', d => d.forestPred === 'A' ? this.colors.primary : this.colors.secondary)
      .attr('opacity', 0.7);

    // Add forest title
    this.comparisonSvg.append('text')
      .attr('x', width * 3 / 4)
      .attr('y', 70)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.text)
      .attr('font-size', '16px')
      .text('Random Forest');

    // Draw reference axes for both
    [width / 4, width * 3 / 4].forEach(x => {
      // X axis
      this.comparisonSvg.append('line')
        .attr('x1', x - width / 4 + 50)
        .attr('y1', height / 2 + height / 4 - 50)
        .attr('x2', x + width / 4 - 50)
        .attr('y2', height / 2 + height / 4 - 50)
        .attr('stroke', this.colors.mutedText)
        .attr('stroke-width', 1);

      // Y axis
      this.comparisonSvg.append('line')
        .attr('x1', x - width / 4 + 50)
        .attr('y1', height / 2 - height / 4 + 50)
        .attr('x2', x - width / 4 + 50)
        .attr('y2', height / 2 + height / 4 - 50)
        .attr('stroke', this.colors.mutedText)
        .attr('stroke-width', 1);

      // X label
      this.comparisonSvg.append('text')
        .attr('x', x)
        .attr('y', height / 2 + height / 4 - 30)
        .attr('text-anchor', 'middle')
        .attr('fill', this.colors.text)
        .attr('font-size', '12px')
        .text('Feature 1');

      // Y label
      this.comparisonSvg.append('text')
        .attr('transform', `rotate(-90)`)
        .attr('x', -(height / 2))
        .attr('y', x - width / 4 + 30)
        .attr('text-anchor', 'middle')
        .attr('fill', this.colors.text)
        .attr('font-size', '12px')
        .text('Feature 2');
    });

    // Add samples
    this.comparisonSvg.selectAll('.sample-points')
      .data([width / 4, width * 3 / 4])
      .enter()
      .append('g')
      .attr('class', 'sample-points')
      .attr('transform', d => `translate(${d}, ${height / 2})`)
      .each((xPos, i, nodes) => {
        d3.select(nodes[i])
          .selectAll('.sample')
          .data(this.samples)
          .enter()
          .append('circle')
          .attr('class', 'sample')
          .attr('cx', d => xScale(d.x1) - width / 2)
          .attr('cy', d => yScale(d.x2) - height / 4)
          .attr('r', 2)
          .attr('fill', d => d.class === 'A' ? this.colors.primary : this.colors.secondary)
          .attr('stroke', this.colors.white)
          .attr('stroke-width', 0.5);
      });

    // Add explanation
    this.comparisonSvg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 20)
      .attr('text-anchor', 'middle')
      .attr('fill', this.colors.mutedText)
      .attr('font-size', '14px')
      .text('Random Forest creates smoother decision boundaries than a single tree');
  }

  private updateVisualization(treeIndex?: number): void {
    // Update dataset visualization
    this.updateDatasetViz(treeIndex);

    // Update tree visualization if a specific tree is selected
    if (treeIndex !== undefined && this.trees[treeIndex]) {
      const container = this.treeContainer.nativeElement;
      this.drawTree(this.trees[treeIndex], this.treeSvg, container.clientWidth, container.clientHeight);
    } else if (this.showSingleTree && this.singleTree) {
      // Show single tree if selected
      const container = this.treeContainer.nativeElement;
      this.drawTree(this.singleTree, this.treeSvg, container.clientWidth, container.clientHeight);
    }

    // Update forest visualization
    this.drawForest();

    // Update prediction if we have a test sample
    if (this.testSample && this.predictionResults.length > 0) {
      this.drawPrediction();
    }

    // Update advanced visualizations if in advanced mode
    if (this.viewMode === 'advanced') {
      this.drawFeatureImportance();
      this.drawOobError();
      this.drawTreeComparison();
    }
  }

  private updateDatasetViz(treeIndex?: number): void {
    // Base update - reset all selections
    this.datasetSvg.selectAll<SVGCircleElement, Sample>('circle.sample')
      .transition()
      .duration(300)
      .attr('r', 2)
      .attr('fill-opacity', 1)
      .attr('stroke', 'none');

    if (treeIndex === undefined) {
      return;
    }

    // Update points to show bootstrap samples
    // Store reference to component for use in callback
    const self = this;
    this.datasetSvg.selectAll<SVGCircleElement, Sample>('circle.sample')
      .transition()
      .duration(300)
      .attr('r', function (d) {
        return d.selected[treeIndex] ? 3 : 2;
      })
      .attr('fill-opacity', function (d) {
        if (d.selected[treeIndex]) return 1;
        if (d.oobForTree[treeIndex] && self.showOobSamples) return 0.7;
        return 0.3;
      })
      .attr('stroke', function (d) {
        if (d.selected[treeIndex]) return self.colors.white;
        if (d.oobForTree[treeIndex] && self.showOobSamples) return self.colors.error;
        return 'none';
      })
      .attr('stroke-width', function (d) {
        if (d.selected[treeIndex]) return 0.5;
        if (d.oobForTree[treeIndex] && self.showOobSamples) return 1;
        return 0;
      });
  }

  private initializeSimulationSteps(): void {
    // Initialize steps for simulation
    this.simulationSteps = [
      {
        title: 'Introduction to Random Forests',
        description: 'Random Forests combine multiple decision trees trained on different data subsets with random feature selection to improve accuracy and reduce overfitting. Introduced by Leo Breiman in 2001, they have become one of the most popular and effective machine learning algorithms.',
        action: () => {
          // Reset visualizations
          this.updateVisualization();
        }
      },
      {
        title: 'Dataset Exploration',
        description: 'We start with a dataset containing samples from two classes (A and B) with two features (x1 and x2). In a real-world scenario, Random Forests can handle many more features and classes.',
        action: () => {
          // Show all data points
          this.updateVisualization();
        }
      },
      {
        title: 'Bootstrap Sampling (Bagging)',
        description: 'For each tree, we create a bootstrap sample by randomly selecting samples with replacement from the original dataset. This means some samples may be selected multiple times while others may not be selected at all.',
        action: () => {
          // Highlight samples for the first tree
          this.updateVisualization(0);
          this.showOobSamples = true;
        }
      },
      {
        title: 'Out-of-Bag (OOB) Samples',
        description: 'Approximately 37% of samples are not selected in each bootstrap sample. These "out-of-bag" samples can be used for unbiased error estimation without needing a separate validation set.',
        action: () => {
          // Highlight OOB samples for the first tree
          this.updateVisualization(0);
          this.showOobSamples = true;
        }
      },
      {
        title: 'Feature Randomness',
        description: 'At each node, only a random subset of features is considered for splitting. For classification, typically sqrt(num_features) features are used. This creates diverse trees that make different kinds of mistakes.',
        action: () => {
          // Show the first tree
          const container = this.treeContainer.nativeElement;
          this.drawTree(this.trees[0], this.treeSvg, container.clientWidth, container.clientHeight);
        }
      },
      {
        title: 'Growing Trees',
        description: 'Each tree is grown independently using its bootstrap sample and random feature selection at each node. Trees are usually grown deep without pruning, as the ensemble nature handles overfitting.',
        action: () => {
          // Show a different tree
          const container = this.treeContainer.nativeElement;
          this.drawTree(this.trees[1], this.treeSvg, container.clientWidth, container.clientHeight);
        }
      },
      {
        title: 'The Random Forest',
        description: 'A Random Forest contains multiple trees, each trained on different data and with different feature subsets. This diversity is key to its performance, as it reduces the correlation between individual tree predictions.',
        action: () => {
          // Show the forest
          this.drawForest();
        }
      },
      {
        title: 'Making Predictions',
        description: 'To predict a new sample, we run it through all trees and take a majority vote (for classification) or average (for regression). This aggregation of predictions leads to more stable and accurate results.',
        action: () => {
          // Make prediction for test sample
          if (this.testSample) {
            this.makeForestPrediction(this.testSample);
            this.drawPrediction();
          }
        }
      },
      {
        title: 'Tree Prediction Path',
        description: 'Let\'s see how a single tree makes its prediction by tracing the path through the tree. Each internal node tests a feature against a threshold to determine which branch to take.',
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
        title: 'Different Trees, Different Paths',
        description: 'Different trees may have different structures and predictions due to bootstrap sampling and feature randomness. This diversity is what makes Random Forests work so well.',
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
        title: 'Feature Importance',
        description: 'Random Forests provide a measure of feature importance based on how much each feature contributes to reducing impurity across all trees. This helps identify which features are most predictive.',
        action: () => {
          // Show feature importance
          this.viewMode = 'advanced';
          this.drawFeatureImportance();
        }
      },
      {
        title: 'OOB Error Estimation',
        description: 'The out-of-bag error is calculated using samples not included in each tree\'s training set. It provides an unbiased estimate of the test error and shows how performance improves with more trees.',
        action: () => {
          // Show OOB error curve
          this.viewMode = 'advanced';
          this.drawOobError();
        }
      },
      {
        title: 'Single Tree vs. Random Forest',
        description: 'A Random Forest creates smoother decision boundaries compared to a single decision tree, leading to better generalization and less overfitting. The ensemble averages out the errors of individual trees.',
        action: () => {
          // Show comparison
          this.viewMode = 'advanced';
          this.showSingleTree = true;
          this.drawTreeComparison();
        }
      },
      {
        title: 'Benefits of Random Forests',
        description: 'Random Forests reduce overfitting, handle high-dimensional data well, provide feature importance metrics, are robust to outliers and noise, and work well for both classification and regression tasks.',
        action: () => {
          // Reset visualizations to show the complete picture
          this.updateVisualization();
          this.drawForest();
          this.drawPrediction();
          this.viewMode = 'advanced';
          this.drawFeatureImportance();
          this.drawOobError();
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
    this.currentExplanation = 'Welcome to the Random Forest Simulation. Press Play to start or use the Step buttons to manually progress. Toggle between Basic and Advanced view modes to see different levels of detail.';
    this.updateVisualization();
    this.viewMode = 'basic';
    this.showOobSamples = false;
    this.showSingleTree = false;
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

  public onNumSamplesChange(event: Event): void {
    this.numSamples = parseInt((event.target as HTMLInputElement).value);
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

  public onViewModeChange(mode: 'basic' | 'advanced'): void {
    this.viewMode = mode;
    // Reinitialize visualizations for new mode
    setTimeout(() => {
      this.initializeVisualization();
      this.updateVisualization(this.selectedTreeIndex);
    }, 0);
  }

  public onShowOobSamplesChange(event: Event): void {
    this.showOobSamples = (event.target as HTMLInputElement).checked;
    this.updateDatasetViz(this.selectedTreeIndex);
  }

  public onShowSingleTreeChange(event: Event): void {
    this.showSingleTree = (event.target as HTMLInputElement).checked;
    this.updateVisualization(this.selectedTreeIndex);
  }

  public onSimulationSpeedChange(event: Event): void {
    this.simulationSpeed = parseInt((event.target as HTMLInputElement).value);

    if (this.simulationState === 'playing') {
      // Restart with new speed
      this.pauseSimulation();
      this.startSimulation();
    }
  }

  public setTestSamplePosition(event: MouseEvent): void {
    if (!this.testSample) return;

    const container = this.datasetContainer.nativeElement;
    const rect = container.getBoundingClientRect();

    // Convert click position to dataset coordinates
    const x = ((event.clientX - rect.left) / container.clientWidth) * 100;
    const y = ((event.clientY - rect.top) / container.clientHeight) * 100;

    this.testSample.x1 = x;
    this.testSample.x2 = y;

    // Update visualization
    this.datasetSvg.select('circle.test-sample')
      .attr('cx', x)
      .attr('cy', y);

    // Update prediction if already predicted
    if (this.predictionResults.length > 0) {
      this.makeForestPrediction(this.testSample);
      this.drawPrediction();
    }
  }

  private regenerateSimulation(): void {
    // Stop any ongoing simulation
    this.stopSimulation();

    // Regenerate data and forest
    this.generateSampleData();
    this.buildForest();
    this.buildSingleDecisionTree();
    this.calculateFeatureImportance();
    this.calculateOobError();

    // Update visualizations
    this.initializeVisualization();

    // Update 3D forest
    this.create3DForest();
  }

  public toggleTreeDetails(treeIndex: number): void {
    this.selectedTreeIndex = treeIndex;
    this.updateVisualization(treeIndex);
  }
  // Helper methods for template
  public getStepsLength(): number {
    if (!this.simulationSteps) return 1; // Prevent division by zero
    return this.simulationSteps.length;
  }
}