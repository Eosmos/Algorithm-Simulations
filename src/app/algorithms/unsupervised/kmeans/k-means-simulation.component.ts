import { Component, ElementRef, OnInit, ViewChild, AfterViewInit, OnDestroy } from '@angular/core';
import { NgFor, NgIf } from '@angular/common';
import * as d3 from 'd3';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface DataPoint {
  id: number;
  x: number;
  y: number;
  cluster: number;
  distance?: number;
  zPos?: number; // Add zPos as optional property for 3D visualization
}

interface Centroid {
  x: number;
  y: number;
  oldX?: number;
  oldY?: number;
}

interface Dataset {
  name: string;
  points: DataPoint[];
  description: string;
}

@Component({
  selector: 'app-kmeans-simulation',
  templateUrl: './k-means-simulation.component.html',
  styleUrls: ['./k-means-simulation.component.scss'],
  standalone: true,
  imports: [NgFor, NgIf]
})
export class KMeansSimulationComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('visualizationContainer') visualizationContainer!: ElementRef;
  @ViewChild('elbowChart') elbowChartRef!: ElementRef;
  @ViewChild('canvas3d') canvas3dRef!: ElementRef;

  // Simulation parameters
  k: number = 3;
  maxK: number = 10;
  iterations: number = 0;
  maxIterations: number = 50;
  wcss: number = 0;
  converged: boolean = false;
  animationSpeed: number = 1000; // milliseconds per step
  
  // UI state
  isPlaying: boolean = false;
  currentStep: 'initialize' | 'assign' | 'update' | 'complete' = 'initialize';
  showElbowMethod: boolean = false;
  show3D: boolean = false;
  initMethod: 'random' | 'kmeansplusplus' = 'kmeansplusplus';
  helpSectionOpen: boolean = false;
  animationInProgress: boolean = false;
  selectedDatasetIndex: number = 0;
  
  // Data
  datasets: Dataset[] = [];
  points: DataPoint[] = [];
  centroids: Centroid[] = [];
  elbowData: {k: number, wcss: number}[] = [];
  
  // D3 elements
  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private width: number = 0;
  private height: number = 0;
  private xScale!: d3.ScaleLinear<number, number>;
  private yScale!: d3.ScaleLinear<number, number>;
  
  // THREE.js elements
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private pointMeshes: THREE.Mesh[] = [];
  private centroidMeshes: THREE.Mesh[] = [];
  private lineMeshes: THREE.Line[] = [];
  
  // Animation timers
  private animationTimer: any;
  private renderTimer: any;

  // Color schemes based on design guide
  readonly COLORS = {
    primary: '#00c9ff', // Cyan for Unsupervised Learning
    primaryLight: '#6edfff',
    primaryDark: '#0099cc',
    background: '#0c1428',
    cardBackground: '#162a4a',
    elementBackground: '#1e3a66',
    hoverBackground: '#2a4980',
    textPrimary: '#e1e7f5',
    textSecondary: '#8a9ab0',
    textEmphasis: '#ffffff',
    success: '#24b47e',
    warning: '#ff9d45',
    error: '#ff6b6b',
    info: '#64b5f6',
    clusterColors: [
      '#00c9ff', // Primary cyan
      '#7c4dff', // Purple
      '#ff9d45', // Orange
      '#24b47e', // Green
      '#ff6b6b', // Red
      '#4285f4', // Blue
      '#f9a825', // Yellow
      '#9c27b0', // Deep Purple
      '#e91e63', // Pink
      '#009688', // Teal
    ]
  };

  // Algorithm explanation
  readonly STEP_EXPLANATIONS = {
    initialize: 'Initialization: Select initial positions for K centroids. Using k-means++ initialization helps spread the starting centroids for better convergence.',
    assign: 'Assignment Step: Each data point is assigned to the nearest centroid based on Euclidean distance.',
    update: 'Update Step: Each centroid is moved to the mean position of all points assigned to its cluster.',
    complete: 'Convergence: The algorithm has converged when centroids no longer move significantly or max iterations are reached.'
  };

  constructor() { }

  ngOnInit(): void {
    this.initializeDatasets();
    this.resetSimulation();
  }

  ngAfterViewInit(): void {
    // Use a small timeout to ensure elements are rendered
    setTimeout(() => {
      if (this.visualizationContainer) {
        this.initializeVisualization();
      }
      
      if (this.elbowChartRef) {
        this.initializeElbowChart();
      }
      
      if (this.canvas3dRef) {
        this.initialize3DVisualization();
      }
      
      this.generateDataPoints();
      this.resetSimulation();
    }, 100);
  }

  ngOnDestroy(): void {
    this.stopAnimation();
    if (this.renderTimer) {
      cancelAnimationFrame(this.renderTimer);
    }
    // Clean up THREE.js resources
    if (this.renderer) {
      this.renderer.dispose();
    }
    this.pointMeshes.forEach(mesh => {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    });
    this.centroidMeshes.forEach(mesh => {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    });
    this.lineMeshes.forEach(line => {
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
    });
  }

  // Initialization methods
  private initializeDatasets(): void {
    // Create different datasets for demonstration
    this.datasets = [
      {
        name: 'Well-separated clusters',
        description: 'Distinct, spherical clusters that are ideal for K-means',
        points: []
      },
      {
        name: 'Overlapping clusters',
        description: 'Clusters with some overlap, challenging but manageable for K-means',
        points: []
      },
      {
        name: 'Uneven clusters',
        description: 'Clusters of different sizes and densities - a challenge for K-means',
        points: []
      },
      {
        name: 'Non-spherical clusters',
        description: 'Elongated or irregular shapes that violate K-means assumptions',
        points: []
      }
    ];
  }

  private initializeVisualization(): void {
    const container = this.visualizationContainer.nativeElement;
    this.width = container.clientWidth;
    this.height = container.clientHeight;

    // Remove existing SVG if present
    d3.select(container).select('svg').remove();

    // Create SVG
    this.svg = d3.select(container)
      .append('svg')
      .attr('width', this.width)
      .attr('height', this.height)
      .attr('viewBox', [0, 0, this.width, this.height])
      .style('background-color', this.COLORS.cardBackground);

    // Create scales
    this.xScale = d3.scaleLinear()
      .domain([0, 100])
      .range([50, this.width - 50]);

    this.yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([this.height - 50, 50]);

    // Add axes
    const xAxis = d3.axisBottom(this.xScale);
    const yAxis = d3.axisLeft(this.yScale);

    this.svg.append('g')
      .attr('transform', `translate(0, ${this.height - 50})`)
      .attr('color', this.COLORS.textSecondary)
      .call(xAxis as any);

    this.svg.append('g')
      .attr('transform', `translate(50, 0)`)
      .attr('color', this.COLORS.textSecondary)
      .call(yAxis as any);
  }

  private initializeElbowChart(): void {
    if (!this.elbowChartRef) return;
    
    const container = this.elbowChartRef.nativeElement;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Remove existing SVG if present
    d3.select(container).select('svg').remove();

    // Create SVG
    const elbowSvg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])
      .style('background-color', this.COLORS.cardBackground);

    // Add title
    elbowSvg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('fill', this.COLORS.textEmphasis)
      .text('Elbow Method: WCSS vs. K');

    // Add axes placeholders (to be populated later)
    elbowSvg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${height - 50})`)
      .attr('color', this.COLORS.textSecondary);

    elbowSvg.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(50, 0)`)
      .attr('color', this.COLORS.textSecondary);

    // Add axis labels
    elbowSvg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 10)
      .attr('text-anchor', 'middle')
      .attr('fill', this.COLORS.textPrimary)
      .text('Number of Clusters (k)');

    elbowSvg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('fill', this.COLORS.textPrimary)
      .text('WCSS (Within-Cluster Sum of Squares)');
  }

  private initialize3DVisualization(): void {
    if (!this.canvas3dRef || !this.canvas3dRef.nativeElement) {
      console.error('3D canvas reference not available');
      return;
    }
    
    try {
      const canvas = this.canvas3dRef.nativeElement;
      // Get computed dimensions from the element's parent container
      const parentElement = canvas.parentElement;
      const width = parentElement.clientWidth || 600;
      const height = parentElement.clientHeight || 500;
      
      console.log('Initializing 3D visualization with dimensions:', width, height);

      // Clean up existing Three.js resources if they exist
      if (this.renderer) {
        this.renderer.dispose();
        this.pointMeshes.forEach(mesh => {
          if (mesh.geometry) mesh.geometry.dispose();
          if (mesh.material) (mesh.material as THREE.Material).dispose();
        });
        this.centroidMeshes.forEach(mesh => {
          if (mesh.geometry) mesh.geometry.dispose();
          if (mesh.material) (mesh.material as THREE.Material).dispose();
        });
        this.lineMeshes.forEach(line => {
          if (line.geometry) line.geometry.dispose();
          if (line.material) (line.material as THREE.Material).dispose();
        });
      }

      // Scene setup
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(this.COLORS.cardBackground);

      // Camera setup with appropriate aspect ratio
      this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
      this.camera.position.set(5, 5, 10); // Position camera for 3D view
      this.camera.lookAt(0, 0, 0);
      
      // Renderer setup
      this.renderer = new THREE.WebGLRenderer({ 
        canvas,
        antialias: true,
        alpha: true
      });
      this.renderer.setSize(width, height);
      this.renderer.setPixelRatio(window.devicePixelRatio);

      // Controls
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.enableZoom = true;
      this.controls.enablePan = true;

      // Lights
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
      this.scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
      directionalLight.position.set(1, 1, 1);
      this.scene.add(directionalLight);

      // Grid helper in XY plane
      const gridHelper = new THREE.GridHelper(10, 10, 0x888888, 0x444444);
      this.scene.add(gridHelper);
      
      // Add coordinate axes for reference
      const axesHelper = new THREE.AxesHelper(5);
      this.scene.add(axesHelper);
      
      // Add a second grid on the XZ plane for better 3D orientation
      const gridHelperXZ = new THREE.GridHelper(10, 10, 0x888888, 0x444444);
      gridHelperXZ.rotation.x = Math.PI / 2; // Rotate to XZ plane
      this.scene.add(gridHelperXZ);

      // Start animation loop
      if (this.renderTimer) {
        cancelAnimationFrame(this.renderTimer);
      }
      this.animate3D();
      
      console.log('3D visualization initialized successfully');
    } catch (error) {
      console.error('Error initializing 3D visualization:', error);
    }
  }

  // Data generation methods
  private generateDataPoints(): void {
    // Generate data for each dataset
    this.generateWellSeparatedClusters();
    this.generateOverlappingClusters();
    this.generateUnevenClusters();
    this.generateNonSphericalClusters();
    
    // Initially set points to the first dataset
    this.points = [...this.datasets[this.selectedDatasetIndex].points];
  }

  private generateWellSeparatedClusters(): void {
    const points: DataPoint[] = [];
    let id = 0;

    // Generate 3 well-separated clusters
    const centers = [
      { x: 25, y: 25 },
      { x: 75, y: 25 },
      { x: 50, y: 75 }
    ];

    centers.forEach((center, i) => {
      // Generate 50 points around each center
      for (let j = 0; j < 50; j++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * 10;
        points.push({
          id: id++,
          x: center.x + Math.cos(angle) * radius,
          y: center.y + Math.sin(angle) * radius,
          cluster: -1 // Initially unassigned
        });
      }
    });

    this.datasets[0].points = points;
  }

  private generateOverlappingClusters(): void {
    const points: DataPoint[] = [];
    let id = 0;

    // Generate 3 clusters with some overlap
    const centers = [
      { x: 40, y: 40 },
      { x: 60, y: 60 },
      { x: 30, y: 70 }
    ];

    centers.forEach((center, i) => {
      // Generate 50 points around each center
      for (let j = 0; j < 50; j++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * 15; // Larger radius for overlap
        points.push({
          id: id++,
          x: center.x + Math.cos(angle) * radius,
          y: center.y + Math.sin(angle) * radius,
          cluster: -1
        });
      }
    });

    this.datasets[1].points = points;
  }

  private generateUnevenClusters(): void {
    const points: DataPoint[] = [];
    let id = 0;

    // Generate clusters with different sizes and densities
    const clusters = [
      { center: { x: 30, y: 30 }, count: 100, radius: 15 },
      { center: { x: 70, y: 40 }, count: 25, radius: 10 },
      { center: { x: 50, y: 70 }, count: 50, radius: 5 }
    ];

    clusters.forEach(cluster => {
      for (let i = 0; i < cluster.count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * cluster.radius;
        points.push({
          id: id++,
          x: cluster.center.x + Math.cos(angle) * radius,
          y: cluster.center.y + Math.sin(angle) * radius,
          cluster: -1
        });
      }
    });

    this.datasets[2].points = points;
  }

  private generateNonSphericalClusters(): void {
    const points: DataPoint[] = [];
    let id = 0;

    // Crescent shape
    for (let i = 0; i < 75; i++) {
      const angle = Math.random() * Math.PI;
      const radius = 25 + Math.random() * 5;
      points.push({
        id: id++,
        x: 50 + Math.cos(angle) * radius,
        y: 50 + Math.sin(angle) * radius,
        cluster: -1
      });
    }

    // Elongated shape
    for (let i = 0; i < 75; i++) {
      points.push({
        id: id++,
        x: 20 + Math.random() * 60,
        y: 20 + Math.random() * 10,
        cluster: -1
      });
    }

    this.datasets[3].points = points;
  }

  // K-means algorithm methods
  private initializeCentroids(): void {
    this.centroids = [];
    if (this.initMethod === 'random') {
      this.initializeRandomCentroids();
    } else {
      this.initializeKMeansPlusPlusCentroids();
    }
    this.updateVisualization();
    this.update3DVisualization();
  }

  private initializeRandomCentroids(): void {
    // For random initialization, we'll use strategies appropriate for the dataset size and k
    
    // Strategy 1: If k is small relative to dataset, sample actual data points
    if (this.k <= this.points.length / 2) {
      // Create a copy of points to sample from without repetition
      const availablePoints = [...this.points];
      
      for (let i = 0; i < this.k && availablePoints.length > 0; i++) {
        // Pick a random point from available points
        const randomIndex = Math.floor(Math.random() * availablePoints.length);
        const point = availablePoints[randomIndex];
        
        // Add as centroid and remove from available points
        this.centroids.push({
          x: point.x,
          y: point.y
        });
        
        availablePoints.splice(randomIndex, 1);
      }
    } 
    // Strategy 2: If k is large or close to dataset size, use random positions within data bounds
    else {
      // Calculate bounds of data
      let minX = 100, maxX = 0, minY = 100, maxY = 0;
      this.points.forEach(point => {
        minX = Math.min(minX, point.x);
        maxX = Math.max(maxX, point.x);
        minY = Math.min(minY, point.y);
        maxY = Math.max(maxY, point.y);
      });
      
      // Add some padding to bounds
      const padX = (maxX - minX) * 0.1;
      const padY = (maxY - minY) * 0.1;
      minX = Math.max(0, minX - padX);
      maxX = Math.min(100, maxX + padX);
      minY = Math.max(0, minY - padY);
      maxY = Math.min(100, maxY + padY);
      
      // Create k centroids within these bounds
      for (let i = 0; i < this.k; i++) {
        this.centroids.push({
          x: minX + Math.random() * (maxX - minX),
          y: minY + Math.random() * (maxY - minY)
        });
      }
    }
    
    // If we didn't get enough centroids (should not happen), fill with random points
    while (this.centroids.length < this.k) {
      this.centroids.push({
        x: Math.random() * 100,
        y: Math.random() * 100
      });
    }
  }

  private initializeKMeansPlusPlusCentroids(): void {
    // Sanity check - don't try to create more clusters than we have points
    const effectiveK = Math.min(this.k, this.points.length);
    
    // Choose first centroid randomly
    const firstIndex = Math.floor(Math.random() * this.points.length);
    this.centroids.push({
      x: this.points[firstIndex].x,
      y: this.points[firstIndex].y
    });

    // Choose remaining centroids with probability proportional to squared distance
    for (let i = 1; i < effectiveK; i++) {
      let distances: number[] = [];
      let sum = 0;

      // Calculate squared distance to nearest existing centroid for each point
      this.points.forEach(point => {
        let minDist = Number.MAX_VALUE;
        this.centroids.forEach(centroid => {
          const dist = this.calculateSquaredDistance(point, centroid);
          minDist = Math.min(minDist, dist);
        });
        distances.push(minDist);
        sum += minDist;
      });

      // Handle the case where all points might be very close to existing centroids
      if (sum < 0.001) {
        // If sum is very small, place remaining centroids randomly
        for (let j = i; j < effectiveK; j++) {
          const randomIndex = Math.floor(Math.random() * this.points.length);
          this.centroids.push({
            x: this.points[randomIndex].x + (Math.random() - 0.5) * 5, // Add some jitter
            y: this.points[randomIndex].y + (Math.random() - 0.5) * 5
          });
        }
        return;
      }

      // Normalize distances to get probabilities
      const probabilities = distances.map(d => d / sum);
      
      // Choose next centroid using weighted probability
      let rand = Math.random();
      let cumulativeProb = 0;
      let nextCentroidIndex = -1;
      
      for (let j = 0; j < probabilities.length; j++) {
        cumulativeProb += probabilities[j];
        if (rand <= cumulativeProb) {
          nextCentroidIndex = j;
          break;
        }
      }

      if (nextCentroidIndex === -1) {
        nextCentroidIndex = probabilities.length - 1;
      }

      // Add the new centroid
      this.centroids.push({
        x: this.points[nextCentroidIndex].x,
        y: this.points[nextCentroidIndex].y
      });
    }
    
    // If k is greater than the number of points, add random centroids with jitter
    if (this.k > this.points.length) {
      for (let i = this.points.length; i < this.k; i++) {
        // Get a random existing point and add some jitter
        const basePoint = this.points[Math.floor(Math.random() * this.points.length)];
        this.centroids.push({
          x: basePoint.x + (Math.random() - 0.5) * 20, // Add significant jitter
          y: basePoint.y + (Math.random() - 0.5) * 20
        });
      }
    }
  }

  private assignPointsToClusters(): void {
    // Store old positions for animation
    this.centroids.forEach(centroid => {
      centroid.oldX = centroid.x;
      centroid.oldY = centroid.y;
    });

    // Assign each point to nearest centroid
    this.points.forEach(point => {
      let minDist = Number.MAX_VALUE;
      let closestCluster = -1;

      this.centroids.forEach((centroid, i) => {
        const dist = this.calculateSquaredDistance(point, centroid);
        if (dist < minDist) {
          minDist = dist;
          closestCluster = i;
        }
      });

      point.cluster = closestCluster;
      point.distance = Math.sqrt(minDist); // Store distance for visualization
    });

    this.calculateWCSS();
  }

  private updateCentroids(): void {
    let moved = false;
    const newCentroids: Centroid[] = [];

    // Calculate new centroid positions
    for (let i = 0; i < this.k; i++) {
      const clusterPoints = this.points.filter(p => p.cluster === i);
      
      if (clusterPoints.length > 0) {
        // Calculate mean position
        const sumX = clusterPoints.reduce((sum, p) => sum + p.x, 0);
        const sumY = clusterPoints.reduce((sum, p) => sum + p.y, 0);
        const newX = sumX / clusterPoints.length;
        const newY = sumY / clusterPoints.length;

        // Check if centroid moved
        const oldX = this.centroids[i].x;
        const oldY = this.centroids[i].y;
        if (Math.abs(newX - oldX) > 0.001 || Math.abs(newY - oldY) > 0.001) {
          moved = true;
        }

        newCentroids.push({
          x: newX,
          y: newY,
          oldX: oldX,
          oldY: oldY
        });
      } else {
        // If no points in cluster, reinitialize this centroid to a better position
        // Find the point furthest from any existing centroid
        let maxMinDist = -1;
        let furthestPoint = null;
        
        for (const point of this.points) {
          // Calculate minimum distance to any centroid in newCentroids
          let minDist = Number.MAX_VALUE;
          for (const cent of newCentroids) {
            const dist = this.calculateSquaredDistance(point, cent);
            minDist = Math.min(minDist, dist);
          }
          
          // If this point is further than any we've seen, select it
          if (minDist > maxMinDist) {
            maxMinDist = minDist;
            furthestPoint = point;
          }
        }
        
        if (furthestPoint) {
          // Use the furthest point as new centroid position
          newCentroids.push({
            x: furthestPoint.x,
            y: furthestPoint.y,
            oldX: this.centroids[i].x,
            oldY: this.centroids[i].y
          });
          moved = true;
        } else {
          // Fallback: keep old position
          newCentroids.push({...this.centroids[i]});
        }
      }
    }

    this.centroids = newCentroids;
    this.converged = !moved;
  }

  private calculateWCSS(): void {
    let sum = 0;
    this.points.forEach(point => {
      if (point.cluster >= 0 && point.distance !== undefined) {
        sum += point.distance * point.distance;
      }
    });
    this.wcss = sum;
  }

  private calculateSquaredDistance(point: {x: number, y: number}, centroid: {x: number, y: number}): number {
    const dx = point.x - centroid.x;
    const dy = point.y - centroid.y;
    return dx * dx + dy * dy;
  }

  // Visualization methods
  private updateVisualization(): void {
    // Clear previous elements
    this.svg.selectAll('.data-point').remove();
    this.svg.selectAll('.centroid').remove();
    this.svg.selectAll('.assignment-line').remove();

    // Draw assignment lines if in assign step
    if (this.currentStep === 'assign') {
      this.svg.selectAll('.assignment-line')
        .data(this.points)
        .enter()
        .append('line')
        .attr('class', 'assignment-line')
        .attr('x1', d => this.xScale(d.x))
        .attr('y1', d => this.yScale(d.y))
        .attr('x2', d => {
          const centroid = this.centroids[d.cluster];
          return centroid ? this.xScale(centroid.x) : this.xScale(d.x);
        })
        .attr('y2', d => {
          const centroid = this.centroids[d.cluster];
          return centroid ? this.yScale(centroid.y) : this.yScale(d.y);
        })
        .attr('stroke', d => d.cluster >= 0 ? this.COLORS.clusterColors[d.cluster % this.COLORS.clusterColors.length] : '#ffffff')
        .attr('stroke-width', 0.5)
        .attr('stroke-opacity', 0.3);
    }

    // Draw data points
    this.svg.selectAll('.data-point')
      .data(this.points)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', d => this.xScale(d.x))
      .attr('cy', d => this.yScale(d.y))
      .attr('r', 5)
      .attr('fill', d => d.cluster >= 0 ? this.COLORS.clusterColors[d.cluster % this.COLORS.clusterColors.length] : this.COLORS.textSecondary)
      .attr('stroke', this.COLORS.textPrimary)
      .attr('stroke-width', 1);

    // Draw centroids
    this.svg.selectAll('.centroid')
      .data(this.centroids)
      .enter()
      .append('circle')
      .attr('class', 'centroid')
      .attr('cx', d => this.xScale(d.x))
      .attr('cy', d => this.yScale(d.y))
      .attr('r', 8)
      .attr('fill', (d, i) => this.COLORS.clusterColors[i % this.COLORS.clusterColors.length])
      .attr('stroke', this.COLORS.textEmphasis)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '3,3');
  }

  private updateElbowChart(): void {
    if (!this.elbowChartRef || !this.elbowData.length) return;
    
    const container = this.elbowChartRef.nativeElement;
    const width = container.clientWidth || 600;
    const height = container.clientHeight || 350;
    
    console.log('Updating elbow chart with data:', this.elbowData);
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([1, Math.max(this.maxK, d3.max(this.elbowData, d => d.k) || 10)])
      .range([50, width - 50]);
    
    const maxWcss = d3.max(this.elbowData, d => d.wcss) || 100;
    const yScale = d3.scaleLinear()
      .domain([0, maxWcss * 1.1])
      .range([height - 50, 50]);
    
    // Clear previous chart
    d3.select(container).select('svg').remove();
    
    // Create new SVG
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])
      .style('background-color', this.COLORS.cardBackground);
    
    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('fill', this.COLORS.textEmphasis)
      .text('Elbow Method: WCSS vs. K');
    
    // Add axes
    const xAxis = d3.axisBottom(xScale).ticks(Math.min(this.elbowData.length, 10)).tickFormat(d3.format('d'));
    const yAxis = d3.axisLeft(yScale);
    
    svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${height - 50})`)
      .attr('color', this.COLORS.textSecondary)
      .call(xAxis as any);
    
    svg.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(50, 0)`)
      .attr('color', this.COLORS.textSecondary)
      .call(yAxis as any);
    
    // Add axis labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 10)
      .attr('text-anchor', 'middle')
      .attr('fill', this.COLORS.textPrimary)
      .text('Number of Clusters (k)');
    
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('fill', this.COLORS.textPrimary)
      .text('WCSS (Within-Cluster Sum of Squares)');
    
    // Draw line
    const line = d3.line<{k: number, wcss: number}>()
      .x(d => xScale(d.k))
      .y(d => yScale(d.wcss))
      .curve(d3.curveMonotoneX);
    
    svg.append('path')
      .datum(this.elbowData)
      .attr('class', 'elbow-line')
      .attr('fill', 'none')
      .attr('stroke', this.COLORS.primary)
      .attr('stroke-width', 2)
      .attr('d', line);
    
    // Draw points
    svg.selectAll('.elbow-point')
      .data(this.elbowData)
      .enter()
      .append('circle')
      .attr('class', 'elbow-point')
      .attr('cx', d => xScale(d.k))
      .attr('cy', d => yScale(d.wcss))
      .attr('r', d => d.k === this.k ? 8 : 5)
      .attr('fill', d => d.k === this.k ? this.COLORS.warning : this.COLORS.primary)
      .attr('stroke', this.COLORS.textEmphasis)
      .attr('stroke-width', d => d.k === this.k ? 2 : 1);
    
    // Find the elbow point (largest second derivative)
    if (this.elbowData.length >= 3) {
      const secondDerivatives: {k: number, derivative: number}[] = [];
      
      for (let i = 1; i < this.elbowData.length - 1; i++) {
        const prevK = this.elbowData[i-1];
        const currK = this.elbowData[i];
        const nextK = this.elbowData[i+1];
        
        // Calculate approximate second derivative
        const firstDeriv1 = (prevK.wcss - currK.wcss) / (currK.k - prevK.k);
        const firstDeriv2 = (currK.wcss - nextK.wcss) / (nextK.k - currK.k);
        const secondDeriv = Math.abs(firstDeriv2 - firstDeriv1);
        
        secondDerivatives.push({k: currK.k, derivative: secondDeriv});
      }
      
      // Find k with max second derivative
      secondDerivatives.sort((a, b) => b.derivative - a.derivative);
      const suggestedK = secondDerivatives[0]?.k;
      
      if (suggestedK) {
        // Mark suggested k
        const suggestedPoint = this.elbowData.find(d => d.k === suggestedK);
        if (suggestedPoint) {
          svg.append('circle')
            .attr('cx', xScale(suggestedPoint.k))
            .attr('cy', yScale(suggestedPoint.wcss))
            .attr('r', 12)
            .attr('fill', 'none')
            .attr('stroke', this.COLORS.success)
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '4,4');
          
          svg.append('text')
            .attr('x', xScale(suggestedPoint.k))
            .attr('y', yScale(suggestedPoint.wcss) - 15)
            .attr('text-anchor', 'middle')
            .attr('fill', this.COLORS.success)
            .text('Suggested K: ' + suggestedK);
        }
      }
    }
    
    // Add labels for each point
    svg.selectAll('.elbow-label')
      .data(this.elbowData)
      .enter()
      .append('text')
      .attr('class', 'elbow-label')
      .attr('x', d => xScale(d.k))
      .attr('y', d => yScale(d.wcss) - 10)
      .attr('text-anchor', 'middle')
      .attr('fill', this.COLORS.textPrimary)
      .text(d => d.wcss.toFixed(0));
  }

  private update3DVisualization(): void {
    if (!this.scene || !this.show3D) return;
    
    console.log('Updating 3D visualization with points:', this.points.length);

    // Clear previous objects
    this.pointMeshes.forEach(mesh => this.scene.remove(mesh));
    this.centroidMeshes.forEach(mesh => this.scene.remove(mesh));
    this.lineMeshes.forEach(line => this.scene.remove(line));
    
    this.pointMeshes = [];
    this.centroidMeshes = [];
    this.lineMeshes = [];

    // Scale factors to fit in 3D space
    const scaleX = 10 / 100;
    const scaleY = 10 / 100;
    const scaleZ = 10 / 100;
    const centerX = 50;
    const centerY = 50;

    // Create a true 3D distribution - we'll assign a persistent Z coordinate for each point
    // This ensures points keep their Z position throughout the simulation for consistency
    
    // If we haven't assigned 3D positions yet, create them
    if (!this.points[0].hasOwnProperty('zPos')) {
      console.log('Generating 3D positions for points');
      
      // Create a new property on each point for its z position
      this.points.forEach((point, index) => {
        // Generate a z position based on multiple factors to create a truly 3D distribution:
        
        // 1. Use point's x, y coordinates to influence z (creates a curved surface)
        const xComponent = Math.sin(point.x / 20) * 3;
        const yComponent = Math.cos(point.y / 20) * 3;
        
        // 2. Add point-specific randomness, but use id to make it deterministic
        const randomComponent = (Math.sin(point.id * 5.731) + Math.cos(point.id * 2.457)) * 3;
        
        // 3. For points in the same region, add a slight vertical spread
        const spreadComponent = (index % 5) * 0.5; 
        
        // Combine all components
        const zPos = xComponent + yComponent + randomComponent + spreadComponent;
        
        // Store the z position with the point
        Object.defineProperty(point, 'zPos', { 
          value: zPos, 
          writable: false,
          configurable: true
        });
      });
    }
    
    // Create centroids with their own 3D positions
    const centroidGeometry = new THREE.SphereGeometry(0.3, 24, 24);
    const centroidPositions: THREE.Vector3[] = [];
    
    this.centroids.forEach((centroid, i) => {
      try {
        // Assign a z-position for each centroid that's the average of its points' z-positions
        let zPos = 0;
        let count = 0;
        
        // Find all points in this cluster and average their z positions
        this.points.forEach(point => {
          if (point.cluster === i) {
            zPos += point.zPos ?? 0; // Use nullish coalescing to handle undefined
            count++;
          }
        });
        
        // If no points in cluster yet, give it a random position
        if (count === 0) {
          zPos = (Math.random() - 0.5) * 10;
        } else {
          zPos /= count;
        }
        
        const material = new THREE.MeshStandardMaterial({
          color: this.COLORS.clusterColors[i % this.COLORS.clusterColors.length],
          emissive: this.COLORS.clusterColors[i % this.COLORS.clusterColors.length],
          emissiveIntensity: 0.5,
          metalness: 0.8,
          roughness: 0.2
        });
        
        const mesh = new THREE.Mesh(centroidGeometry, material);
        mesh.position.set(
          (centroid.x - centerX) * scaleX,
          (centroid.y - centerY) * scaleY,
          zPos * scaleZ
        );
        
        centroidPositions[i] = mesh.position.clone();
        
        this.scene.add(mesh);
        this.centroidMeshes.push(mesh);
      } catch (error) {
        console.error('Error creating 3D centroid:', error);
      }
    });
    
    // Create points using their persistent z positions
    const pointGeometry = new THREE.SphereGeometry(0.15, 16, 16);
    
    this.points.forEach((point, index) => {
      try {
        const material = new THREE.MeshStandardMaterial({
          color: point.cluster >= 0 
            ? this.COLORS.clusterColors[point.cluster % this.COLORS.clusterColors.length] 
            : this.COLORS.textSecondary,
          emissive: point.cluster >= 0 
            ? this.COLORS.clusterColors[point.cluster % this.COLORS.clusterColors.length] 
            : this.COLORS.textSecondary,
          emissiveIntensity: 0.3
        });
        
        const mesh = new THREE.Mesh(pointGeometry, material);
        mesh.position.set(
          (point.x - centerX) * scaleX,
          (point.y - centerY) * scaleY,
          (point.zPos ?? 0) * scaleZ // Use nullish coalescing to handle undefined
        );
        
        this.scene.add(mesh);
        this.pointMeshes.push(mesh);
        
        // Add assignment lines in assign step
        if (this.currentStep === 'assign' && point.cluster >= 0) {
          const centroidPos = centroidPositions[point.cluster];
          
          if (centroidPos) {
            const lineGeometry = new THREE.BufferGeometry();
            lineGeometry.setAttribute(
              'position', 
              new THREE.Float32BufferAttribute([
                mesh.position.x, mesh.position.y, mesh.position.z,
                centroidPos.x, centroidPos.y, centroidPos.z
              ], 3)
            );
            
            const lineMaterial = new THREE.LineBasicMaterial({
              color: this.COLORS.clusterColors[point.cluster % this.COLORS.clusterColors.length],
              transparent: true,
              opacity: 0.3,
              linewidth: 1
            });
            
            const line = new THREE.Line(lineGeometry, lineMaterial);
            this.scene.add(line);
            this.lineMeshes.push(line);
          }
        }
      } catch (error) {
        console.error('Error creating 3D point:', error);
      }
    });
    
    // Position camera to better view the 3D volume
    if (this.camera) {
      this.camera.position.set(8, 8, 8);
      this.camera.lookAt(0, 0, 0);
    }
    
    if (this.controls) {
      this.controls.update();
    }
    
    console.log('3D visualization updated with points:', this.pointMeshes.length, 'centroids:', this.centroidMeshes.length);
  }

  private animate3D(): void {
    try {
      this.renderTimer = requestAnimationFrame(() => this.animate3D());
      
      if (this.controls) {
        this.controls.update();
      }
      
      if (this.renderer && this.scene && this.camera) {
        this.renderer.render(this.scene, this.camera);
      }
    } catch (error) {
      console.error('Error in 3D animation loop:', error);
      // Cancel animation frame to stop the loop if there's an error
      if (this.renderTimer) {
        cancelAnimationFrame(this.renderTimer);
      }
    }
  }

  // Simulation control methods
  resetSimulation(): void {
    this.stopAnimation();
    this.iterations = 0;
    this.wcss = 0;
    this.converged = false;
    this.currentStep = 'initialize';
    
    // Reset point clusters but preserve any 3D positions
    this.points.forEach(point => {
      point.cluster = -1;
      point.distance = undefined;
    });
    
    this.centroids = [];
    
    if (this.svg) {
      this.updateVisualization();
    }
    
    if (this.scene) {
      this.update3DVisualization();
    }
  }

  nextStep(): void {
    if (this.animationInProgress) return;
    this.animationInProgress = true;
    
    switch (this.currentStep) {
      case 'initialize':
        this.initializeCentroids();
        this.currentStep = 'assign';
        break;
        
      case 'assign':
        this.assignPointsToClusters();
        this.updateVisualization();
        if (this.show3D) {
          this.update3DVisualization();
        }
        this.currentStep = 'update';
        break;
        
      case 'update':
        this.updateCentroids();
        this.iterations++;
        this.updateVisualization();
        if (this.show3D) {
          // Regenerate all z-positions if in 3D view to ensure proper visualization
          if (this.iterations % 5 === 0) { // Regenerate occasionally for visual interest
            this.points.forEach(point => {
              point.zPos = undefined;
            });
          }
          this.update3DVisualization();
        }
        
        if (this.converged || this.iterations >= this.maxIterations) {
          this.currentStep = 'complete';
          // Add current K and WCSS to elbow data
          const existingDataPoint = this.elbowData.find(d => d.k === this.k);
          if (existingDataPoint) {
            existingDataPoint.wcss = this.wcss;
          } else {
            this.elbowData.push({ k: this.k, wcss: this.wcss });
            // Sort by k
            this.elbowData.sort((a, b) => a.k - b.k);
          }
          this.updateElbowChart();
        } else {
          this.currentStep = 'assign';
        }
        break;
        
      case 'complete':
        this.resetSimulation();
        break;
    }
    
    setTimeout(() => {
      this.animationInProgress = false;
    }, 500);
  }

  startAnimation(): void {
    if (this.isPlaying) return;
    
    this.isPlaying = true;
    this.runAnimationStep();
  }

  stopAnimation(): void {
    this.isPlaying = false;
    if (this.animationTimer) {
      clearTimeout(this.animationTimer);
    }
  }

  private runAnimationStep(): void {
    if (!this.isPlaying) return;
    
    this.nextStep();
    
    this.animationTimer = setTimeout(() => {
      if (this.isPlaying) {
        this.runAnimationStep();
      }
    }, this.animationSpeed);
  }

  // UI event handlers
  togglePlayPause(): void {
    if (this.isPlaying) {
      this.stopAnimation();
    } else {
      this.startAnimation();
    }
  }

  changeK(newK: number): void {
    // Ensure k is a number and within reasonable bounds
    newK = Math.max(1, Math.min(parseInt(newK as any) || 2, Math.min(10, Math.floor(this.points.length / 5))));
    
    // Only update if the value actually changed
    if (this.k !== newK) {
      console.log(`Changing k from ${this.k} to ${newK}`);
      this.k = newK;
      
      // Reset 3D positions when changing k if in 3D view
      if (this.show3D) {
        this.points.forEach(point => {
          point.zPos = undefined;
        });
      }
      
      this.resetSimulation();
    }
  }

  changeDataset(index: number): void {
    this.selectedDatasetIndex = index;
    this.points = [...this.datasets[index].points];
    
    // When changing datasets, clear any existing 3D positions 
    // to regenerate them for the new dataset
    this.points.forEach(point => {
      point.zPos = undefined;
    });
    
    this.resetSimulation();
  }

  changeInitMethod(method: 'random' | 'kmeansplusplus'): void {
    this.initMethod = method;
    
    // Reset 3D positions when changing initialization method if in 3D view
    if (this.show3D) {
      this.points.forEach(point => {
        point.zPos = undefined;
      });
    }
    
    this.resetSimulation();
  }

  toggleElbowMethod(): void {
    this.showElbowMethod = !this.showElbowMethod;
    
    if (this.showElbowMethod) {
      // Generate elbow data if it's empty
      if (this.elbowData.length < 2) {
        this.generateElbowData();
      }
      
      setTimeout(() => {
        this.updateElbowChart();
      }, 100);
    }
  }
  
  // Generate elbow data for different k values
  private generateElbowData(): void {
    // Store original k value to restore it after
    const originalK = this.k;
    const originalCentroids = [...this.centroids];
    const originalPoints = this.points.map(p => ({...p}));
    
    console.log('Generating elbow data');
    this.elbowData = [];
    
    // Run K-means for different values of k to get WCSS
    for (let k = 1; k <= Math.min(9, Math.floor(this.points.length / 10)); k++) {
      this.k = k;
      this.centroids = [];
      
      // Reset points to original state
      this.points.forEach((point, i) => {
        point.cluster = -1;
        point.distance = undefined;
      });
      
      // Initialize centroids
      if (this.initMethod === 'random') {
        this.initializeRandomCentroids();
      } else {
        this.initializeKMeansPlusPlusCentroids();
      }
      
      // Run algorithm for a fixed number of iterations
      let tempConverged = false;
      let tempIterations = 0;
      
      while (!tempConverged && tempIterations < 10) {
        this.assignPointsToClusters();
        tempIterations++;
        
        // Update centroids and check convergence
        let moved = false;
        const newCentroids: Centroid[] = [];
        
        for (let i = 0; i < this.k; i++) {
          const clusterPoints = this.points.filter(p => p.cluster === i);
          
          if (clusterPoints.length > 0) {
            const sumX = clusterPoints.reduce((sum, p) => sum + p.x, 0);
            const sumY = clusterPoints.reduce((sum, p) => sum + p.y, 0);
            const newX = sumX / clusterPoints.length;
            const newY = sumY / clusterPoints.length;
            
            const oldX = this.centroids[i].x;
            const oldY = this.centroids[i].y;
            if (Math.abs(newX - oldX) > 0.001 || Math.abs(newY - oldY) > 0.001) {
              moved = true;
            }
            
            newCentroids.push({
              x: newX,
              y: newY,
              oldX: oldX,
              oldY: oldY
            });
          } else {
            newCentroids.push({...this.centroids[i]});
          }
        }
        
        this.centroids = newCentroids;
        tempConverged = !moved;
      }
      
      // Calculate WCSS for this k
      this.calculateWCSS();
      this.elbowData.push({ k: k, wcss: this.wcss });
    }
    
    // Restore original state
    this.k = originalK;
    this.centroids = originalCentroids;
    
    // Reset points to original state
    this.points.forEach((point, i) => {
      Object.assign(point, originalPoints[i]);
    });
    
    // Sort by k
    this.elbowData.sort((a, b) => a.k - b.k);
    console.log('Elbow data generated:', this.elbowData);
  }

  toggle3DView(): void {
    const wasIn3D = this.show3D;
    this.show3D = !this.show3D;
    
    // Allow time for DOM to update before initializing/updating 3D view
    setTimeout(() => {
      if (this.show3D) {
        console.log('Initializing 3D view');
        
        // Clear existing 3D coordinates to force regeneration
        this.points.forEach(point => {
          point.zPos = undefined;
        });
        
        // Re-initialize the 3D view when toggling to ensure proper setup
        this.initialize3DVisualization();
        this.update3DVisualization();
      }
    }, 100);
  }

  toggleHelpSection(): void {
    this.helpSectionOpen = !this.helpSectionOpen;
  }

  changeAnimationSpeed(speed: number): void {
    this.animationSpeed = speed;
  }
}