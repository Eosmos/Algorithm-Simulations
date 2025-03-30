import { Component, ElementRef, OnInit, ViewChild, AfterViewInit, OnDestroy, HostListener } from '@angular/core';
import * as d3 from 'd3';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

interface DataPoint {
  [key: string]: any;
  feature0: number;
  feature1: number;
  feature2?: number;
  feature3?: number;
  class: string;
}

interface TreeNode {
  type: 'leaf' | 'internal';
  class?: string;
  feature?: string;
  splitValue?: number;
  left?: TreeNode;
  right?: TreeNode;
  depth: number;
}

interface ForestTree {
  id: number;
  sampleData: DataPoint[];
  tree: TreeNode | null;
}

interface HierarchyNode {
  name: string;
  type: 'leaf' | 'internal';
  feature?: string;
  class?: string;
  children?: HierarchyNode[];
}

interface GridPoint {
  x: number;
  y: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface PredictionPoint {
  x: number;
  y: number;
  class: string;
  confidence?: number;
}

@Component({
  selector: 'app-random-forest-simulation',
  templateUrl: './random-forest-simulation.component.html',
  styleUrls: ['./random-forest-simulation.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class RandomForestSimulationComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('simulationContainer') simulationContainer!: ElementRef;
  @ViewChild('treeContainer') treeContainer!: ElementRef;
  @ViewChild('featureImportanceContainer') featureImportanceContainer!: ElementRef;
  @ViewChild('decisionBoundaryContainer') decisionBoundaryContainer!: ElementRef;
  
  // Simulation state - changed to public for template access
  public currentStep = 0;
  private isPlaying = false;
  private animationIntervalId: any;
  
  // Simulation configuration
  private numTrees = 5;
  private maxDepth = 3;
  private featureCount = 4;
  private sampleSize = 100;
  
  // Visualization objects
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  
  // Simulated dataset
  private dataset: DataPoint[] = [];
  private features: string[] = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'];
  private forest: ForestTree[] = [];
  
  // UI state
  public currentStepLabel = 'Initialization';
  public currentStepDescription = 'Preparing the Random Forest simulation...';
  public isAutoPlayEnabled = false;
  
  constructor() { }

  ngOnInit(): void {
    this.generateSimulatedData();
  }

  ngAfterViewInit(): void {
    this.initializeVisualization();
    this.startSimulation();
  }

  @HostListener('window:resize')
  onResize() {
    if (this.renderer && this.camera) {
      const container = this.simulationContainer.nativeElement;
      const width = container.clientWidth;
      const height = container.clientHeight;
      
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    }
  }

  private generateSimulatedData(): void {
    // Generate simulated data for the Random Forest
    this.dataset = [];
    
    // Create a dataset with 2 classes
    for (let i = 0; i < this.sampleSize; i++) {
      const dataPoint: DataPoint = {
        feature0: 0,
        feature1: 0,
        class: ''
      };
      
      // Generate random feature values
      for (let j = 0; j < this.featureCount; j++) {
        dataPoint[`feature${j}`] = Math.random();
      }
      
      // Assign class based on some arbitrary rule to create separable clusters
      if ((dataPoint.feature0 > 0.7 && dataPoint.feature1 > 0.7) || 
          (dataPoint.feature0 < 0.3 && dataPoint.feature1 < 0.3)) {
        dataPoint.class = 'A';
      } else {
        dataPoint.class = 'B';
      }
      
      this.dataset.push(dataPoint);
    }
  }

  private initializeVisualization(): void {
    this.initializeThreeJS();
    this.initializeD3Visualizations();
  }

  private initializeThreeJS(): void {
    // Set up Three.js scene for 3D forest visualization
    const container = this.simulationContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf0f0f0);
    
    // Create camera
    this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    this.camera.position.set(0, 10, 20);
    
    // Create renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);
    
    // Add controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    
    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    directionalLight.castShadow = true;
    this.scene.add(directionalLight);
    
    // Start animation loop
    this.animate();
  }

  private animate(): void {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private initializeD3Visualizations(): void {
    this.initializeDecisionTreeViz();
    this.initializeFeatureImportanceViz();
    this.initializeDecisionBoundaryViz();
  }

  private initializeDecisionTreeViz(): void {
    const container = this.treeContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Clear previous SVG
    d3.select(container).selectAll("*").remove();
    
    // Create SVG
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${width / 2}, 50)`);
    
    // Initial message
    svg.append("text")
      .attr("text-anchor", "middle")
      .text("Decision Trees will appear here");
  }

  private initializeFeatureImportanceViz(): void {
    const container = this.featureImportanceContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Clear previous SVG
    d3.select(container).selectAll("*").remove();
    
    // Create SVG
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height);
    
    // Initial message
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .text("Feature Importance will appear here");
  }

  private initializeDecisionBoundaryViz(): void {
    const container = this.decisionBoundaryContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Clear previous SVG
    d3.select(container).selectAll("*").remove();
    
    // Create SVG
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height);
    
    // Initial message
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .text("Decision Boundaries will appear here");
  }

  private startSimulation(): void {
    this.currentStep = 0;
    this.updateSimulationStep();
  }

  private updateSimulationStep(): void {
    switch (this.currentStep) {
      case 0:
        this.showIntroduction();
        break;
      case 1:
        this.showDatasetGeneration();
        break;
      case 2:
        this.showBootstrapping();
        break;
      case 3:
        this.showTreeConstruction();
        break;
      case 4:
        this.showFeatureImportance();
        break;
      case 5:
        this.showPredictionAggregation();
        break;
      case 6:
        this.showDecisionBoundaries();
        break;
      case 7:
        this.showConclusion();
        break;
      default:
        this.resetSimulation();
        break;
    }
  }

  // Simulation step implementations
  private showIntroduction(): void {
    this.currentStepLabel = 'Introduction to Random Forests';
    this.currentStepDescription = 'Random Forests are ensemble learning methods that combine multiple decision trees to improve prediction accuracy. They were introduced by Leo Breiman in 2001 and have become one of the most powerful and widely used machine learning algorithms.';
    
    // Clear the 3D scene except for the ground
    this.clearForestScene();
    this.addGroundToScene();
  }

  private showDatasetGeneration(): void {
    this.currentStepLabel = 'Dataset Generation';
    this.currentStepDescription = 'We start with a dataset containing features and target values. Each data point has multiple features and belongs to a class. Random Forests handle both categorical and numerical data, and can work with high-dimensional datasets.';
    
    // Visualize the dataset in 2D scatter plot
    this.visualizeDataset();
  }

  private showBootstrapping(): void {
    this.currentStepLabel = 'Bootstrapping (Bagging)';
    this.currentStepDescription = 'Random Forests create multiple subsets of the training data by sampling with replacement (bagging). This introduces randomness and diversity among the trees, as each tree is trained on a slightly different dataset.';
    
    // Visualize bootstrapping process
    this.visualizeBootstrapping();
  }

  private showTreeConstruction(): void {
    this.currentStepLabel = 'Tree Construction with Feature Randomness';
    this.currentStepDescription = 'Each tree is built independently using its bootstrapped dataset. At each split, only a random subset of features is considered (typically sqrt(n) for classification). This feature randomness further diversifies the trees.';
    
    // Visualize tree construction
    this.visualizeTreeConstruction();
  }

  private showFeatureImportance(): void {
    this.currentStepLabel = 'Feature Importance Analysis';
    this.currentStepDescription = 'Random Forests provide insights into which features are most important for making predictions. This is calculated based on how much each feature reduces impurity (e.g., Gini impurity for classification) across all trees.';
    
    // Visualize feature importance
    this.visualizeFeatureImportance();
  }

  private showPredictionAggregation(): void {
    this.currentStepLabel = 'Prediction Aggregation (Voting)';
    this.currentStepDescription = 'For classification tasks, each tree votes for a class, and the class with the most votes wins. For regression tasks, the predictions from all trees are averaged. This aggregation reduces variance and improves generalization.';
    
    // Visualize prediction aggregation
    this.visualizePredictionAggregation();
  }

  private showDecisionBoundaries(): void {
    this.currentStepLabel = 'Decision Boundaries Comparison';
    this.currentStepDescription = 'Random Forests create smoother and more accurate decision boundaries compared to individual trees. This helps prevent overfitting and improves performance on unseen data.';
    
    // Visualize decision boundaries
    this.visualizeDecisionBoundaries();
  }

  private showConclusion(): void {
    this.currentStepLabel = 'Strengths and Applications';
    this.currentStepDescription = 'Random Forests are powerful models that offer high accuracy, handle overfitting well, and provide feature importance insights. They are used in various fields including finance (fraud detection), healthcare (disease prediction), ecology, and computer vision.';
    
    // Show the final forest and conclusions
    this.visualizeFullForest();
  }

  // Helper visualization methods
  private clearForestScene(): void {
    // Remove all objects except camera and lights
    while (this.scene.children.length > 0) {
      this.scene.remove(this.scene.children[0]);
    }
    
    // Re-add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    directionalLight.castShadow = true;
    this.scene.add(directionalLight);
  }

  private addGroundToScene(): void {
    const groundGeometry = new THREE.PlaneGeometry(50, 50);
    const groundMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x7cbb78,
      roughness: 0.8
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2; // Rotate to be horizontal
    ground.position.y = -0.5; // Slightly below the origin
    ground.receiveShadow = true;
    this.scene.add(ground);
  }

  private createTree3D(position: THREE.Vector3, size: number, complexity: number): void {
    // Create a 3D tree representation
    const trunkGeometry = new THREE.CylinderGeometry(0.2 * size, 0.3 * size, size, 8);
    const trunkMaterial = new THREE.MeshStandardMaterial({ color: 0x8B4513 });
    const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
    trunk.position.copy(position);
    trunk.position.y += size / 2;
    trunk.castShadow = true;
    trunk.receiveShadow = true;
    this.scene.add(trunk);
    
    // Create tree crown (leaves)
    const crownGeometry = new THREE.ConeGeometry(size * 0.8, size * 1.5, 8);
    const crownMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x228B22,
      roughness: 0.7
    });
    const crown = new THREE.Mesh(crownGeometry, crownMaterial);
    crown.position.copy(position);
    crown.position.y += size + (size * 0.75);
    crown.castShadow = true;
    crown.receiveShadow = true;
    this.scene.add(crown);
    
    // Add some randomness to the tree shape based on complexity
    crown.scale.set(
      1 + (Math.random() * 0.2 - 0.1) * complexity,
      1 + (Math.random() * 0.3) * complexity,
      1 + (Math.random() * 0.2 - 0.1) * complexity
    );
  }

  private visualizeDataset(): void {
    const container = this.decisionBoundaryContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Clear previous SVG
    d3.select(container).selectAll("*").remove();
    
    // Create SVG
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height);
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, 1])
      .range([50, width - 50]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height - 50, 50]);
    
    // Draw axes
    const xAxis = d3.axisBottom(xScale);
    svg.append("g")
      .attr("transform", `translate(0, ${height - 50})`)
      .call(xAxis);
    
    const yAxis = d3.axisLeft(yScale);
    svg.append("g")
      .attr("transform", `translate(50, 0)`)
      .call(yAxis);
    
    // Add axis labels
    svg.append("text")
      .attr("text-anchor", "middle")
      .attr("x", width / 2)
      .attr("y", height - 10)
      .text("Feature 1");
    
    svg.append("text")
      .attr("text-anchor", "middle")
      .attr("transform", `translate(15, ${height / 2}) rotate(-90)`)
      .text("Feature 2");
    
    // Plot data points
    svg.selectAll("circle")
      .data(this.dataset)
      .enter()
      .append("circle")
      .attr("cx", (d: DataPoint) => xScale(d.feature0))
      .attr("cy", (d: DataPoint) => yScale(d.feature1))
      .attr("r", 5)
      .attr("fill", (d: DataPoint) => d.class === 'A' ? "#ff6b6b" : "#4ecdc4")
      .attr("opacity", 0.7)
      .attr("stroke", "#fff")
      .attr("stroke-width", 1);
    
    // Add legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width - 120}, 30)`);
    
    legend.append("circle")
      .attr("cx", 0)
      .attr("cy", 0)
      .attr("r", 5)
      .attr("fill", "#ff6b6b");
    
    legend.append("text")
      .attr("x", 10)
      .attr("y", 5)
      .text("Class A");
    
    legend.append("circle")
      .attr("cx", 0)
      .attr("cy", 20)
      .attr("r", 5)
      .attr("fill", "#4ecdc4");
    
    legend.append("text")
      .attr("x", 10)
      .attr("y", 25)
      .text("Class B");
  }

  private visualizeBootstrapping(): void {
    // Create bootstrapped datasets
    this.forest = [];
    
    for (let i = 0; i < this.numTrees; i++) {
      const bootstrapSample: DataPoint[] = [];
      
      // Sample with replacement
      for (let j = 0; j < this.dataset.length; j++) {
        const randomIndex = Math.floor(Math.random() * this.dataset.length);
        bootstrapSample.push(this.dataset[randomIndex]);
      }
      
      this.forest.push({
        id: i,
        sampleData: bootstrapSample,
        tree: null
      });
    }
    
    // Visualize bootstrapping in the 3D scene
    this.clearForestScene();
    this.addGroundToScene();
    
    // Place trees in a grid formation
    const spacing = 8;
    const rows = Math.ceil(Math.sqrt(this.numTrees));
    const cols = Math.ceil(this.numTrees / rows);
    
    let treeIndex = 0;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        if (treeIndex < this.numTrees) {
          const x = (col - Math.floor(cols / 2)) * spacing;
          const z = (row - Math.floor(rows / 2)) * spacing;
          
          // Create tree 3D visualization
          const position = new THREE.Vector3(x, 0, z);
          
          // Create a simple placeholder for now
          const treeGeometry = new THREE.BoxGeometry(1, 1, 1);
          const treeMaterial = new THREE.MeshStandardMaterial({ color: 0x228B22 });
          const treePlaceholder = new THREE.Mesh(treeGeometry, treeMaterial);
          treePlaceholder.position.copy(position);
          treePlaceholder.position.y += 0.5;
          treePlaceholder.castShadow = true;
          treePlaceholder.receiveShadow = true;
          this.scene.add(treePlaceholder);
          
          treeIndex++;
        }
      }
    }
    
    // Also visualize bootstrapping in the 2D chart
    this.visualizeBootstrappingSamples();
  }

  private visualizeBootstrappingSamples(): void {
    const container = this.decisionBoundaryContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Get the existing SVG
    const svg = d3.select(container).select("svg");
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, 1])
      .range([50, width - 50]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height - 50, 50]);
    
    // Update explanation
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 30)
      .attr("text-anchor", "middle")
      .attr("font-weight", "bold")
      .text("Bootstrapped Samples (Sampling with Replacement)");
    
    // Highlight one of the bootstrap samples
    if (this.forest.length > 0) {
      const sampleData = this.forest[0].sampleData;
      
      // Add highlight circles to show bootstrap samples
      svg.selectAll(".bootstrap-highlight")
        .data(sampleData)
        .enter()
        .append("circle")
        .attr("class", "bootstrap-highlight")
        .attr("cx", (d: DataPoint) => xScale(d.feature0))
        .attr("cy", (d: DataPoint) => yScale(d.feature1))
        .attr("r", 8)
        .attr("fill", "none")
        .attr("stroke", "#FFD700")
        .attr("stroke-width", 2)
        .attr("opacity", 0.7);
    }
  }

  private visualizeTreeConstruction(): void {
    // Simulate tree construction
    this.forest.forEach((treeData, index) => {
      // Simulate a simple decision tree structure
      const tree = this.simulateDecisionTree(treeData.sampleData, 0, this.maxDepth);
      treeData.tree = tree;
    });
    
    // Visualize trees in 3D scene
    this.clearForestScene();
    this.addGroundToScene();
    
    // Place trees in a grid formation
    const spacing = 8;
    const rows = Math.ceil(Math.sqrt(this.numTrees));
    const cols = Math.ceil(this.numTrees / rows);
    
    let treeIndex = 0;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        if (treeIndex < this.numTrees) {
          const x = (col - Math.floor(cols / 2)) * spacing;
          const z = (row - Math.floor(rows / 2)) * spacing;
          
          const position = new THREE.Vector3(x, 0, z);
          const treeSize = 2 + Math.random() * 1; // Random size variation
          const complexity = this.forest[treeIndex].tree?.depth || 0 / this.maxDepth;
          
          this.createTree3D(position, treeSize, complexity);
          treeIndex++;
        }
      }
    }
    
    // Visualize a selected tree in 2D
    if (this.forest[0].tree) {
      this.visualizeDecisionTree(this.forest[0].tree);
    }
  }

  private simulateDecisionTree(data: DataPoint[], currentDepth: number, maxDepth: number): TreeNode {
    // This is a simplified simulation, not an actual decision tree algorithm
    if (currentDepth >= maxDepth || data.length < 5) {
      // Leaf node - determine the majority class
      const classCount: {[key: string]: number} = {};
      data.forEach(d => {
        if (!classCount[d.class]) classCount[d.class] = 0;
        classCount[d.class]++;
      });
      
      let majorityClass = '';
      let maxCount = 0;
      for (const cls in classCount) {
        if (classCount[cls] > maxCount) {
          maxCount = classCount[cls];
          majorityClass = cls;
        }
      }
      
      return {
        type: 'leaf',
        class: majorityClass,
        depth: currentDepth
      };
    }
    
    // Internal node - make a random split
    const featureIndex = Math.floor(Math.random() * this.featureCount);
    const featureName = `feature${featureIndex}`;
    const splitValue = Math.random(); // Random split point
    
    const leftData = data.filter(d => d[featureName] < splitValue);
    const rightData = data.filter(d => d[featureName] >= splitValue);
    
    return {
      type: 'internal',
      feature: featureName,
      splitValue: splitValue,
      left: this.simulateDecisionTree(leftData, currentDepth + 1, maxDepth),
      right: this.simulateDecisionTree(rightData, currentDepth + 1, maxDepth),
      depth: currentDepth
    };
  }

  private visualizeDecisionTree(tree: TreeNode): void {
    const container = this.treeContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Clear previous SVG
    d3.select(container).selectAll("*").remove();
    
    // Convert the tree data to a hierarchical structure for d3
    const hierarchyData = this.convertTreeToHierarchy(tree);
    // Use type assertion to handle the hierarchy type correctly
    const root = d3.hierarchy(hierarchyData) as d3.HierarchyNode<unknown>;
    
    // Create a tree layout
    const treeLayout = d3.tree<unknown>().size([width - 100, height - 100]);
    
    // Apply the layout to the root node
    treeLayout(root);
    
    // Create SVG
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(50, 50)`);
    
    // Add links between nodes
    svg.selectAll("path.link")
      .data(root.links())
      .enter()
      .append("path")
      .attr("class", "link")
      .attr("d", d => {
        // Ensure all properties are non-null with proper checks
        const sourceY = d.source.y ?? 0;
        const targetY = d.target.y ?? 0;
        const sourceX = d.source.x ?? 0;
        const targetX = d.target.x ?? 0;
        const midY = (sourceY + targetY) / 2;
        
        return `M${sourceX},${sourceY}C${sourceX},${midY} ${targetX},${midY} ${targetX},${targetY}`;
      })
      .attr("fill", "none")
      .attr("stroke", "#999")
      .attr("stroke-width", 1.5);
    
    // Add nodes
    const nodes = svg.selectAll("g.node")
      .data(root.descendants())
      .enter()
      .append("g")
      .attr("class", "node")
      .attr("transform", d => `translate(${d.x ?? 0}, ${d.y ?? 0})`);
    
    // Add circles for internal nodes, rectangles for leaf nodes
    nodes.filter(d => (d.data as HierarchyNode).type === 'internal')
      .append("circle")
      .attr("r", 20)
      .attr("fill", "#69b3a2");
    
    nodes.filter(d => (d.data as HierarchyNode).type === 'leaf')
      .append("rect")
      .attr("x", -20)
      .attr("y", -20)
      .attr("width", 40)
      .attr("height", 40)
      .attr("fill", d => (d.data as HierarchyNode).class === 'A' ? "#ff6b6b" : "#4ecdc4");
    
    // Add labels to nodes
    nodes.filter(d => (d.data as HierarchyNode).type === 'internal')
      .append("text")
      .attr("dy", 5)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("fill", "white")
      .text(d => {
        const feature = (d.data as HierarchyNode).feature;
        return feature ? feature.replace('feature', 'F') : '';
      });
    
    nodes.filter(d => (d.data as HierarchyNode).type === 'leaf')
      .append("text")
      .attr("dy", 5)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .attr("fill", "white")
      .text(d => (d.data as HierarchyNode).class || '');
    
    // Add title
    svg.append("text")
      .attr("x", width / 2 - 50)
      .attr("y", 0)
      .attr("font-size", "16px")
      .attr("font-weight", "bold")
      .text("Decision Tree Visualization");
  }

  private convertTreeToHierarchy(tree: TreeNode): HierarchyNode {
    if (tree.type === 'leaf') {
      return {
        name: `Class ${tree.class}`,
        type: 'leaf',
        class: tree.class
      };
    } else {
      return {
        name: `${tree.feature} < ${tree.splitValue?.toFixed(2)}`,
        type: 'internal',
        feature: tree.feature,
        children: [
          this.convertTreeToHierarchy(tree.left!),
          this.convertTreeToHierarchy(tree.right!)
        ]
      };
    }
  }

  private visualizeFeatureImportance(): void {
    // Calculate simulated feature importance for each feature
    const featureImportance: FeatureImportance[] = [];
    
    for (let i = 0; i < this.featureCount; i++) {
      const featureName = `Feature ${i + 1}`;
      // Simulate importance with random values, in a real scenario this would be calculated from the model
      const importance = Math.random();
      featureImportance.push({ feature: featureName, importance });
    }
    
    // Sort features by importance
    featureImportance.sort((a, b) => b.importance - a.importance);
    
    // Visualize feature importance as a bar chart
    const container = this.featureImportanceContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Clear previous SVG
    d3.select(container).selectAll("*").remove();
    
    // Create SVG
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height);
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(featureImportance, d => d.importance) as number])
      .range([0, width - 150]);
    
    const yScale = d3.scaleBand()
      .domain(featureImportance.map(d => d.feature))
      .range([50, height - 50])
      .padding(0.3);
    
    // Add title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 25)
      .attr("text-anchor", "middle")
      .attr("font-size", "16px")
      .attr("font-weight", "bold")
      .text("Feature Importance");
    
    // Add bars
    svg.selectAll(".bar")
      .data(featureImportance)
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", 100)
      .attr("y", d => yScale(d.feature) as number)
      .attr("width", d => xScale(d.importance))
      .attr("height", yScale.bandwidth())
      .attr("fill", "#69b3a2");
    
    // Add feature labels
    svg.selectAll(".feature-label")
      .data(featureImportance)
      .enter()
      .append("text")
      .attr("class", "feature-label")
      .attr("x", 95)
      .attr("y", d => (yScale(d.feature) as number) + yScale.bandwidth() / 2)
      .attr("text-anchor", "end")
      .attr("dominant-baseline", "middle")
      .text(d => d.feature);
    
    // Add importance value labels
    svg.selectAll(".importance-label")
      .data(featureImportance)
      .enter()
      .append("text")
      .attr("class", "importance-label")
      .attr("x", d => 105 + xScale(d.importance))
      .attr("y", d => (yScale(d.feature) as number) + yScale.bandwidth() / 2)
      .attr("dominant-baseline", "middle")
      .text(d => d.importance.toFixed(2));
    
    // Update 3D visualization - make trees with more important features taller
    this.clearForestScene();
    this.addGroundToScene();
    
    // Place trees in a grid formation
    const spacing = 8;
    const rows = Math.ceil(Math.sqrt(this.numTrees));
    const cols = Math.ceil(this.numTrees / rows);
    
    let treeIndex = 0;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        if (treeIndex < this.numTrees) {
          const x = (col - Math.floor(cols / 2)) * spacing;
          const z = (row - Math.floor(rows / 2)) * spacing;
          
          const position = new THREE.Vector3(x, 0, z);
          
          // Use feature importance to determine tree size
          // Make trees with a higher weight for more important features taller
          const treeSize = 2 + featureImportance[0].importance * 2;
          const complexity = Math.random() * 0.5 + 0.5; // Random complexity
          
          this.createTree3D(position, treeSize, complexity);
          treeIndex++;
        }
      }
    }
  }

  private visualizePredictionAggregation(): void {
    // Simulate predictions from each tree
    const predictions: { tree: number; predictions: PredictionPoint[] }[] = [];
    
    // Create a grid of test points
    const testPoints: GridPoint[] = [];
    for (let x = 0.1; x <= 0.9; x += 0.2) {
      for (let y = 0.1; y <= 0.9; y += 0.2) {
        testPoints.push({ x, y });
      }
    }
    
    // Generate predictions for each tree
    this.forest.forEach((treeData, treeIndex) => {
      const treePredictions: PredictionPoint[] = [];
      
      testPoints.forEach(point => {
        // Simulate class prediction - in a real scenario this would use the actual tree
        // Here we're making it slightly random but biased by the point location
        let predictedClass;
        if ((point.x > 0.7 && point.y > 0.7) || (point.x < 0.3 && point.y < 0.3)) {
          // Higher probability of class A
          predictedClass = Math.random() < 0.8 ? 'A' : 'B';
        } else {
          // Higher probability of class B
          predictedClass = Math.random() < 0.2 ? 'A' : 'B';
        }
        
        treePredictions.push({
          x: point.x,
          y: point.y,
          class: predictedClass
        });
      });
      
      predictions.push({
        tree: treeIndex,
        predictions: treePredictions
      });
    });
    
    // Calculate aggregate predictions (majority vote)
    const aggregatePredictions: PredictionPoint[] = [];
    
    testPoints.forEach((point, pointIndex) => {
      // Count votes for each class
      const votes: { [key: string]: number } = { 'A': 0, 'B': 0 };
      
      predictions.forEach(treePrediction => {
        const prediction = treePrediction.predictions[pointIndex];
        votes[prediction.class]++;
      });
      
      // Determine the majority class and confidence
      let majorityClass: string;
      let confidence: number;
      
      if (votes['A'] > votes['B']) {
        majorityClass = 'A';
        confidence = votes['A'] / this.numTrees;
      } else {
        majorityClass = 'B';
        confidence = votes['B'] / this.numTrees;
      }
      
      aggregatePredictions.push({
        x: point.x,
        y: point.y,
        class: majorityClass,
        confidence: confidence
      });
    });
    
    // Visualize predictions in the decision boundary container
    const container = this.decisionBoundaryContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Clear previous SVG
    d3.select(container).selectAll("*").remove();
    
    // Create SVG
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height);
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, 1])
      .range([50, width - 50]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height - 50, 50]);
    
    // Draw axes
    const xAxis = d3.axisBottom(xScale);
    svg.append("g")
      .attr("transform", `translate(0, ${height - 50})`)
      .call(xAxis);
    
    const yAxis = d3.axisLeft(yScale);
    svg.append("g")
      .attr("transform", `translate(50, 0)`)
      .call(yAxis);
    
    // Add title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 25)
      .attr("text-anchor", "middle")
      .attr("font-size", "16px")
      .attr("font-weight", "bold")
      .text("Prediction Aggregation (Voting)");
    
    // Plot predictions
    svg.selectAll(".prediction")
      .data(aggregatePredictions)
      .enter()
      .append("circle")
      .attr("class", "prediction")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", d => 5 + (d.confidence || 0) * 15) // Size based on confidence
      .attr("fill", d => d.class === 'A' ? "#ff6b6b" : "#4ecdc4")
      .attr("fill-opacity", d => 0.3 + (d.confidence || 0) * 0.7) // Opacity based on confidence
      .attr("stroke", d => d.class === 'A' ? "#ff6b6b" : "#4ecdc4")
      .attr("stroke-width", 2);
    
    // Add legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width - 120}, 30)`);
    
    legend.append("circle")
      .attr("cx", 0)
      .attr("cy", 0)
      .attr("r", 5)
      .attr("fill", "#ff6b6b");
    
    legend.append("text")
      .attr("x", 10)
      .attr("y", 5)
      .text("Class A");
    
    legend.append("circle")
      .attr("cx", 0)
      .attr("cy", 20)
      .attr("r", 5)
      .attr("fill", "#4ecdc4");
    
    legend.append("text")
      .attr("x", 10)
      .attr("y", 25)
      .text("Class B");
    
    legend.append("circle")
      .attr("cx", 0)
      .attr("cy", 50)
      .attr("r", 10)
      .attr("fill", "#999")
      .attr("fill-opacity", 0.5);
    
    legend.append("text")
      .attr("x", 15)
      .attr("y", 55)
      .text("High Confidence");
    
    legend.append("circle")
      .attr("cx", 0)
      .attr("cy", 70)
      .attr("r", 5)
      .attr("fill", "#999")
      .attr("fill-opacity", 0.3);
    
    legend.append("text")
      .attr("x", 15)
      .attr("y", 75)
      .text("Low Confidence");
  }

  private visualizeDecisionBoundaries(): void {
    // Generate a grid of points to visualize decision boundaries
    const gridSize = 40;
    const gridPoints: GridPoint[] = [];
    
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        gridPoints.push({
          x: i / (gridSize - 1),
          y: j / (gridSize - 1)
        });
      }
    }
    
    // Predict classes for each grid point for each tree and for the ensemble
    const treePredictions: { [key: string]: string }[] = [];
    for (let i = 0; i < this.numTrees; i++) {
      treePredictions.push({});
    }
    
    const forestPredictions: { [key: string]: string } = {};
    
    gridPoints.forEach(point => {
      // Generate key for the point
      const key = `${point.x},${point.y}`;
      
      // Simulate predictions for each tree
      const votes: { [key: string]: number } = { 'A': 0, 'B': 0 };
      
      for (let i = 0; i < this.numTrees; i++) {
        // Simulate individual tree prediction
        let treePrediction;
        
        // Roughly simulate a decision boundary
        if (i % 2 === 0) {
          // Even-indexed trees
          treePrediction = point.x > point.y ? 'A' : 'B';
        } else {
          // Odd-indexed trees
          treePrediction = point.x + point.y > 1 ? 'A' : 'B';
        }
        
        // Add some randomness
        if (Math.random() < 0.2) {
          treePrediction = treePrediction === 'A' ? 'B' : 'A';
        }
        
        treePredictions[i][key] = treePrediction;
        votes[treePrediction]++;
      }
      
      // Forest prediction (majority vote)
      forestPredictions[key] = votes['A'] > votes['B'] ? 'A' : 'B';
    });
    
    // Visualize decision boundaries
    const container = this.decisionBoundaryContainer.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Clear previous SVG
    d3.select(container).selectAll("*").remove();
    
    // Create SVG
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height);
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, 1])
      .range([50, width - 50]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height - 50, 50]);
    
    // Draw axes
    const xAxis = d3.axisBottom(xScale);
    svg.append("g")
      .attr("transform", `translate(0, ${height - 50})`)
      .call(xAxis);
    
    const yAxis = d3.axisLeft(yScale);
    svg.append("g")
      .attr("transform", `translate(50, 0)`)
      .call(yAxis);
    
    // Add title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 25)
      .attr("text-anchor", "middle")
      .attr("font-size", "16px")
      .attr("font-weight", "bold")
      .text("Random Forest Decision Boundary");
    
    // Plot forest predictions
    svg.selectAll(".grid-point")
      .data(gridPoints)
      .enter()
      .append("rect")
      .attr("class", "grid-point")
      .attr("x", d => xScale(d.x) - (width - 100) / (2 * gridSize))
      .attr("y", d => yScale(d.y) - (height - 100) / (2 * gridSize))
      .attr("width", (width - 100) / gridSize)
      .attr("height", (height - 100) / gridSize)
      .attr("fill", d => {
        const key = `${d.x},${d.y}`;
        return forestPredictions[key] === 'A' ? "#ff6b6b" : "#4ecdc4";
      })
      .attr("opacity", 0.7);
    
    // Plot original data points on top
    svg.selectAll(".data-point")
      .data(this.dataset)
      .enter()
      .append("circle")
      .attr("class", "data-point")
      .attr("cx", d => xScale(d.feature0))
      .attr("cy", d => yScale(d.feature1))
      .attr("r", 5)
      .attr("fill", d => d.class === 'A' ? "#ff6b6b" : "#4ecdc4")
      .attr("stroke", "#333")
      .attr("stroke-width", 1);
    
    // Add legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width - 120}, 30)`);
    
    legend.append("rect")
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "#ff6b6b")
      .attr("opacity", 0.7);
    
    legend.append("text")
      .attr("x", 20)
      .attr("y", 12)
      .text("Class A Region");
    
    legend.append("rect")
      .attr("y", 20)
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "#4ecdc4")
      .attr("opacity", 0.7);
    
    legend.append("text")
      .attr("x", 20)
      .attr("y", 32)
      .text("Class B Region");
    
    legend.append("circle")
      .attr("cx", 7)
      .attr("cy", 55)
      .attr("r", 5)
      .attr("fill", "#ff6b6b")
      .attr("stroke", "#333");
    
    legend.append("text")
      .attr("x", 20)
      .attr("y", 58)
      .text("Class A Point");
    
    legend.append("circle")
      .attr("cx", 7)
      .attr("cy", 75)
      .attr("r", 5)
      .attr("fill", "#4ecdc4")
      .attr("stroke", "#333");
    
    legend.append("text")
      .attr("x", 20)
      .attr("y", 78)
      .text("Class B Point");
  }

  private visualizeFullForest(): void {
    // Create a full forest visualization in 3D
    this.clearForestScene();
    this.addGroundToScene();
    
    // Add more trees in a circular arrangement
    const numForestTrees = 20;
    const radius = 15;
    
    for (let i = 0; i < numForestTrees; i++) {
      const angle = (i / numForestTrees) * Math.PI * 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      
      const position = new THREE.Vector3(x, 0, z);
      const size = 2 + Math.random() * 2;
      const complexity = Math.random();
      
      this.createTree3D(position, size, complexity);
    }
    
    // Add a visual representation of the forest's prediction power
    const sphereGeometry = new THREE.SphereGeometry(5, 32, 32);
    const sphereMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x69b3a2,
      transparent: true,
      opacity: 0.6
    });
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.set(0, 5, 0);
    this.scene.add(sphere);
    
    // Add a light source in the center
    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(0, 10, 0);
    this.scene.add(pointLight);
    
    // Visualize a summary of feature importance
    this.visualizeFeatureImportance();
    
    // Focus the camera on the forest
    this.camera.position.set(0, 15, 30);
    this.camera.lookAt(0, 5, 0);
    this.controls.update();
  }

  // UI Control Methods
  public nextStep(): void {
    this.currentStep++;
    if (this.currentStep > 7) {
      this.currentStep = 0;
    }
    this.updateSimulationStep();
  }

  public previousStep(): void {
    this.currentStep--;
    if (this.currentStep < 0) {
      this.currentStep = 7;
    }
    this.updateSimulationStep();
  }

  public resetSimulation(): void {
    this.currentStep = 0;
    this.updateSimulationStep();
    this.stopAutoPlay();
  }

  public toggleAutoPlay(): void {
    if (this.isPlaying) {
      this.stopAutoPlay();
    } else {
      this.startAutoPlay();
    }
  }

  private startAutoPlay(): void {
    this.isPlaying = true;
    this.isAutoPlayEnabled = true;
    this.animationIntervalId = setInterval(() => {
      this.nextStep();
    }, 5000); // Change step every 5 seconds
  }

  private stopAutoPlay(): void {
    this.isPlaying = false;
    this.isAutoPlayEnabled = false;
    if (this.animationIntervalId) {
      clearInterval(this.animationIntervalId);
    }
  }

  ngOnDestroy(): void {
    this.stopAutoPlay();
    
    // Clean up Three.js resources
    if (this.renderer) {
      this.renderer.dispose();
    }
  }
}