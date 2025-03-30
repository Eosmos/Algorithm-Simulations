import { Component, OnInit, ElementRef, ViewChild, AfterViewInit, NgZone } from '@angular/core';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';
import { CommonModule } from '@angular/common';

interface TreeNode {
  name: string;
  feature?: string;
  threshold?: number;
  gini?: number;
  entropy?: number;
  mse?: number;
  samples: number;
  value: number[];
  children?: TreeNode[];
  x?: number;
  y?: number;
  x0?: number;
  y0?: number;
  id?: string;
  depth?: number;
  class?: string;
  description?: string;
}

interface DataPoint {
  id: number;
  features: {[key: string]: number};
  label: string;
  x?: number;
  y?: number;
  currentNode?: string;
}

@Component({
  selector: 'app-decision-tree',
  templateUrl: './decision-tree.component.html',
  styleUrls: ['./decision-tree.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class DecisionTreeComponent implements OnInit, AfterViewInit {
  @ViewChild('treeContainer') private treeContainer!: ElementRef;
  @ViewChild('dataContainer') private dataContainer!: ElementRef;
  
  // Simulation state
  private width = 900;
  private height = 700;
  private margin = { top: 20, right: 90, bottom: 30, left: 90 };
  private svg: any;
  private treeSvg: any;
  private dataSvg: any;
  private tree: any;
  private root: any;
  private treeData!: TreeNode;
  private dataPoints: DataPoint[] = [];
  public animationSpeed = 1000;
  private isPlaying = false;
  private currentStep = 0;
  public totalSteps = 6;
  private animationTimer: any;
  private diagonal: any;
  private nodeRadius = 30;
  public metricType: 'gini' | 'entropy' | 'mse' = 'gini';
  private tooltipDiv: any;

  // UI control variables
  stepDescription = 'Welcome to Decision Tree Simulation';
  currentStepNumber = 0;
  isAutoPlaying = false;
  selectedFeature = 'age';
  availableFeatures = ['age', 'income', 'education', 'credit_score'];
  buildProgress = 0;
  
  constructor(private zone: NgZone) {}

  ngOnInit(): void {
    this.initializeDataPoints();
    this.initializeTreeData();
    
    // Make sure selectedFeature is set to a valid value from availableFeatures
    if (!this.availableFeatures.includes(this.selectedFeature)) {
      this.selectedFeature = this.availableFeatures[0];
    }
  }

  ngAfterViewInit(): void {
    this.initializeVisualization();
  }

  private initializeDataPoints(): void {
    // Create sample data for visualization
    const features = ['age', 'income', 'education', 'credit_score'];
    const labels = ['yes', 'no'];
    
    for (let i = 0; i < 50; i++) {
      const dataPoint: DataPoint = {
        id: i,
        features: {},
        label: labels[Math.floor(Math.random() * labels.length)]
      };
      
      // Generate random feature values
      features.forEach(feature => {
        dataPoint.features[feature] = Math.random() * 100;
      });
      
      this.dataPoints.push(dataPoint);
    }
  }

  private initializeTreeData(): void {
    // Initial empty tree with just the root node
    this.treeData = {
      name: 'Root',
      samples: this.dataPoints.length,
      value: [30, 20], // Example values for binary classification
      gini: 0.48,
      entropy: 0.971,
      mse: 0.24,
      description: 'This is the root node where all data starts.',
      children: []
    };
  }

  private initializeVisualization(): void {
    // Create main SVG container
    this.svg = d3.select(this.treeContainer.nativeElement)
      .append('svg')
      .attr('width', this.width)
      .attr('height', this.height)
      .append('g')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

    // Create tooltip div
    this.tooltipDiv = d3.select('body').append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0);

    // Initialize tree layout with horizontal orientation (better for decision trees)
    this.tree = d3.tree<TreeNode>()
      .size([this.height - this.margin.top - this.margin.bottom, this.width - this.margin.left - this.margin.right - 100]);

    // Create path generator for links between nodes
    this.diagonal = d3.linkHorizontal<any, any>()
      .x(d => d.y)
      .y(d => d.x);

    // Initialize data visualization
    this.dataSvg = d3.select(this.dataContainer.nativeElement)
      .append('svg')
      .attr('width', this.width / 3)
      .attr('height', this.height / 3)
      .append('g')
      .attr('transform', `translate(${this.width / 6},${this.height / 6})`);

    // Render initial state
    this.renderTreeData();
    this.renderDataPoints();
  }

  private renderTreeData(): void {
    // Convert hierarchical data to d3 hierarchy
    this.root = d3.hierarchy(this.treeData, (d: TreeNode) => d.children);
    this.root.x0 = this.height / 2;
    this.root.y0 = 0;

    // Compute the tree layout
    this.tree(this.root);

    // Clear previous nodes
    this.svg.selectAll('*').remove();

    // Add links between nodes with smoother curves
    const links = this.svg.append('g')
      .attr('class', 'links')
      .selectAll('path')
      .data(this.root.links())
      .enter()
      .append('path')
      .attr('d', (d: any) => {
        return `M${d.source.y},${d.source.x}
                C${(d.source.y + d.target.y) / 2 - 50},${d.source.x}
                 ${(d.source.y + d.target.y) / 2 + 50},${d.target.x}
                 ${d.target.y},${d.target.x}`;
      })
      .attr('fill', 'none')
      .attr('stroke', '#88A0A8')
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.8)
      .attr('stroke-linecap', 'round');

    // Add nodes
    const nodes = this.svg.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(this.root.descendants())
      .enter()
      .append('g')
      .attr('class', (d: any) => `node ${d.children ? 'node--internal' : 'node--leaf'}`)
      .attr('transform', (d: any) => `translate(${d.y},${d.x})`)
      .on('mouseover', (event: any, d: any) => {
        this.showNodeTooltip(event, d);
      })
      .on('mouseout', () => {
        this.hideTooltip();
      })
      .on('click', (event: any, d: any) => {
        this.handleNodeClick(event, d);
      });

    // Add drop shadow filter for node highlighting
    const defs = this.svg.append('defs');
    const filter = defs.append('filter')
      .attr('id', 'drop-shadow')
      .attr('height', '130%');
    
    filter.append('feGaussianBlur')
      .attr('in', 'SourceAlpha')
      .attr('stdDeviation', 3)
      .attr('result', 'blur');
    
    filter.append('feOffset')
      .attr('in', 'blur')
      .attr('dx', 2)
      .attr('dy', 2)
      .attr('result', 'offsetBlur');
    
    const feComponentTransfer = filter.append('feComponentTransfer')
      .attr('in', 'offsetBlur')
      .attr('result', 'endBlur');
    
    feComponentTransfer.append('feFuncA')
      .attr('type', 'linear')
      .attr('slope', 0.3);
    
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode')
      .attr('in', 'endBlur');
    feMerge.append('feMergeNode')
      .attr('in', 'SourceGraphic');

    // Add circles to nodes with improved styling
    nodes.append('circle')
      .attr('r', this.nodeRadius)
      .attr('fill', (d: any) => {
        if (d.children) {
          return '#69b3a2'; // Internal nodes
        } else {
          // For leaf nodes, use a gradient based on class distribution
          const yesRatio = d.data.value[0] / (d.data.value[0] + d.data.value[1]);
          return d.data.class === 'yes' ? 
            d3.interpolateRgb('#28a745', '#50c878')(yesRatio) : 
            d3.interpolateRgb('#dc3545', '#ff6b6b')(1 - yesRatio);
        }
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('filter', 'url(#drop-shadow)')
      .style('cursor', 'pointer')
      .style('transition', 'all 0.3s ease');

    // Add labels to nodes
    nodes.append('text')
      .attr('dy', '.35em')
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .text((d: any) => d.data.feature ? d.data.feature : (d.children ? 'Root' : d.data.class));

    // Add split criterion for internal nodes
    nodes.filter((d: any) => d.data.feature)
      .append('text')
      .attr('dy', '1.75em')
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', 'white')
      .text((d: any) => d.data.threshold ? `≤ ${d.data.threshold.toFixed(1)}` : '');

    // Add impurity measure with cleaner look
    nodes.append('text')
      .attr('dy', '-1.5em')
      .attr('text-anchor', 'middle')
      .attr('font-size', '9px')
      .attr('fill', '#555')
      .attr('font-weight', 'normal')
      .text((d: any) => {
        if (this.metricType === 'gini' && d.data.gini !== undefined) {
          return `Gini: ${d.data.gini.toFixed(2)}`;
        } else if (this.metricType === 'entropy' && d.data.entropy !== undefined) {
          return `Entropy: ${d.data.entropy.toFixed(2)}`;
        } else if (this.metricType === 'mse' && d.data.mse !== undefined) {
          return `MSE: ${d.data.mse.toFixed(2)}`;
        }
        return '';
      });

    // Add samples count with better positioning
    nodes.append('text')
      .attr('dy', '-2.5em')
      .attr('text-anchor', 'middle')
      .attr('font-size', '9px')
      .attr('fill', '#555')
      .text((d: any) => `Samples: ${d.data.samples}`);
  }

  private renderDataPoints(): void {
    // Clear previous data points
    this.dataSvg.selectAll('*').remove();

    // Create a color scale
    const colorScale = d3.scaleOrdinal<string>()
      .domain(['yes', 'no'])
      .range(['#28a745', '#dc3545']);

    // Create x and y scales for data points
    const xScale = d3.scaleLinear()
      .domain([0, 100])
      .range([-this.width / 7, this.width / 7]);

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([-this.height / 7, this.height / 7]);

    // Add a subtle grid for better visual reference
    const gridSize = 20;
    const grid = this.dataSvg.append('g')
      .attr('class', 'grid');
    
    // Add horizontal grid lines
    for (let i = -this.height / 7; i <= this.height / 7; i += gridSize) {
      grid.append('line')
        .attr('x1', -this.width / 7)
        .attr('y1', i)
        .attr('x2', this.width / 7)
        .attr('y2', i)
        .attr('stroke', '#e0e0e0')
        .attr('stroke-width', 0.5)
        .attr('stroke-dasharray', '2,2');
    }
    
    // Add vertical grid lines
    for (let i = -this.width / 7; i <= this.width / 7; i += gridSize) {
      grid.append('line')
        .attr('x1', i)
        .attr('y1', -this.height / 7)
        .attr('x2', i)
        .attr('y2', this.height / 7)
        .attr('stroke', '#e0e0e0')
        .attr('stroke-width', 0.5)
        .attr('stroke-dasharray', '2,2');
    }

    // Add data points with subtle animation on initial render
    this.dataSvg.selectAll('circle')
      .data(this.dataPoints)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', (d: DataPoint) => {
        // Use the current feature for x position
        d.x = xScale(d.features[this.selectedFeature]);
        return d.x;
      })
      .attr('cy', (d: DataPoint) => {
        // Use a different feature for y position
        const yFeature = this.availableFeatures.find(f => f !== this.selectedFeature) || 'income';
        d.y = yScale(d.features[yFeature]);
        return d.y;
      })
      .attr('r', 0) // Start with radius 0 for animation
      .attr('fill', (d: DataPoint) => colorScale(d.label))
      .attr('stroke', 'white')
      .attr('stroke-width', 1)
      .attr('opacity', 0.8)
      .style('filter', 'drop-shadow(0 1px 2px rgba(0,0,0,0.1))')
      .on('mouseover', (event: any, d: any) => {
        this.showDataTooltip(event, d);
      })
      .on('mouseout', () => {
        this.hideTooltip();
      })
      // Animate the data points appearing
      .transition()
      .duration(800)
      .delay((d: DataPoint, i: number) => i * 10) // Stagger the animation
      .attr('r', 5)
      .attr('opacity', 0.8);

    // Add x-axis with better styling
    this.dataSvg.append('g')
      .attr('transform', `translate(0,${this.height / 6})`)
      .attr('class', 'x-axis')
      .call(d3.axisBottom(xScale).ticks(5).tickSize(-this.height/3).tickFormat(d3.format("d")))
      .call((g: any) => g.select(".domain").attr("stroke", "#888"))
      .call((g: any) => g.selectAll(".tick line").attr("stroke", "#e0e0e0"))
      .call((g: any) => g.selectAll(".tick text").attr("fill", "#666").attr("font-size", "9px"))
      .append('text')
      .attr('x', this.width / 14)
      .attr('y', 30)
      .attr('fill', '#333')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .attr('text-anchor', 'middle')
      .text(this.selectedFeature.toUpperCase());

    // Add y-axis with better styling
    const yFeature = this.availableFeatures.find(f => f !== this.selectedFeature) || 'income';
    this.dataSvg.append('g')
      .attr('transform', `translate(${-this.width / 7},0)`)
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale).ticks(5).tickSize(-this.width/3.5).tickFormat(d3.format("d")))
      .call((g: any) => g.select(".domain").attr("stroke", "#888"))
      .call((g: any) => g.selectAll(".tick line").attr("stroke", "#e0e0e0"))
      .call((g: any) => g.selectAll(".tick text").attr("fill", "#666").attr("font-size", "9px"))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -this.height / 14)
      .attr('y', -30)
      .attr('fill', '#333')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .attr('text-anchor', 'middle')
      .text(yFeature.toUpperCase());
  }

  private showNodeTooltip(event: any, d: any): void {
    const nodeData = d.data;
    let tooltipContent = `<div class="tooltip-header">${nodeData.feature || 'Leaf Node'}</div>`;
    
    if (nodeData.feature) {
      tooltipContent += `<div class="tooltip-row"><span class="tooltip-label">Threshold:</span> <span class="tooltip-value">${nodeData.threshold?.toFixed(1) || 'N/A'}</span></div>`;
    }
    
    tooltipContent += `<div class="tooltip-row"><span class="tooltip-label">Samples:</span> <span class="tooltip-value">${nodeData.samples}</span></div>`;
    tooltipContent += `<div class="tooltip-row"><span class="tooltip-label">Distribution:</span> <span class="tooltip-value">[${nodeData.value.join(', ')}]</span></div>`;
    
    if (nodeData.gini !== undefined) {
      tooltipContent += `<div class="tooltip-row"><span class="tooltip-label">Gini:</span> <span class="tooltip-value">${nodeData.gini.toFixed(3)}</span></div>`;
    }
    
    if (nodeData.entropy !== undefined) {
      tooltipContent += `<div class="tooltip-row"><span class="tooltip-label">Entropy:</span> <span class="tooltip-value">${nodeData.entropy.toFixed(3)}</span></div>`;
    }
    
    if (nodeData.mse !== undefined) {
      tooltipContent += `<div class="tooltip-row"><span class="tooltip-label">MSE:</span> <span class="tooltip-value">${nodeData.mse.toFixed(3)}</span></div>`;
    }
    
    if (nodeData.description) {
      tooltipContent += `<div class="tooltip-description">${nodeData.description}</div>`;
    }

    this.tooltipDiv.transition()
      .duration(200)
      .style('opacity', .95);
    
    this.tooltipDiv.html(tooltipContent)
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY - 28) + 'px');
      
    // Highlight the node
    d3.select(event.currentTarget).select('circle')
      .transition()
      .duration(200)
      .attr('r', this.nodeRadius * 1.1)
      .style('filter', 'drop-shadow(0 0 5px rgba(0,0,0,0.3))');
  }

  private showDataTooltip(event: any, d: DataPoint): void {
    let tooltipContent = `<strong>Data Point #${d.id}</strong><br>`;
    tooltipContent += `Class: ${d.label}<br><br>`;
    
    // Show all features
    for (const [feature, value] of Object.entries(d.features)) {
      tooltipContent += `${feature}: ${value.toFixed(1)}<br>`;
    }

    this.tooltipDiv.transition()
      .duration(200)
      .style('opacity', .9);
    
    this.tooltipDiv.html(tooltipContent)
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY - 28) + 'px');
  }

  private hideTooltip(): void {
    this.tooltipDiv.transition()
      .duration(500)
      .style('opacity', 0);
  }

  private handleNodeClick(event: any, d: any): void {
    // Add visual feedback for the click
    const circle = d3.select(event.currentTarget).select('circle');
    circle.transition()
      .duration(200)
      .attr('r', this.nodeRadius * 1.2)
      .transition()
      .duration(200)
      .attr('r', this.nodeRadius);
    
    // Handle node click interactions
    console.log('Node clicked:', d.data);
    
    // If it's a leaf node, no expansion
    if (!d.children && !d._children) {
      return;
    }
    
    // Toggle children
    if (d.children) {
      d._children = d.children;
      d.children = null;
    } else {
      d.children = d._children;
      d._children = null;
    }
    
    // Re-render the tree with animation
    this.updateTreeWithAnimation(d);
  }
  
  private updateTreeWithAnimation(source: any): void {
    // Compute the new tree layout
    this.tree(this.root);
    
    // Get all nodes and links
    const nodes = this.root.descendants();
    const links = this.root.links();
    
    // Normalize for fixed-depth
    nodes.forEach((d: any) => {
      d.y = d.depth * 180; // Fixed distance between levels
    });
    
    // Update the nodes
    const node = this.svg.selectAll('g.node')
      .data(nodes, (d: any) => d.id || (d.id = ++this.currentStep));
    
    // Enter new nodes at the parent's previous position
    const nodeEnter = node.enter().append('g')
      .attr('class', (d: any) => `node ${d.children ? 'node--internal' : 'node--leaf'}`)
      .attr('transform', (d: any) => `translate(${source.y0 || source.y},${source.x0 || source.x})`)
      .on('click', (event: any, d: any) => {
        this.handleNodeClick(event, d);
      })
      .on('mouseover', (event: any, d: any) => {
        this.showNodeTooltip(event, d);
      })
      .on('mouseout', () => {
        this.hideTooltip();
      });
    
    // Add Circle for the nodes
    nodeEnter.append('circle')
      .attr('r', 0)
      .attr('fill', (d: any) => {
        if (d.children) {
          return '#69b3a2'; // Internal nodes
        } else {
          // For leaf nodes, use a gradient based on class distribution
          const yesRatio = d.data.value[0] / (d.data.value[0] + d.data.value[1]);
          return d.data.class === 'yes' ? 
            d3.interpolateRgb('#28a745', '#50c878')(yesRatio) : 
            d3.interpolateRgb('#dc3545', '#ff6b6b')(1 - yesRatio);
        }
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('filter', 'url(#drop-shadow)')
      .style('cursor', 'pointer');
    
    // Add text labels
    nodeEnter.append('text')
      .attr('dy', '.35em')
      .attr('x', 0)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .style('fill-opacity', 0)
      .text((d: any) => d.data.feature ? d.data.feature : (d.children ? 'Root' : d.data.class));
    
    // Add other text elements as needed
    
    // UPDATE
    const nodeUpdate = nodeEnter.merge(node);
    
    // Transition to the proper position for the node
    nodeUpdate.transition()
      .duration(this.animationSpeed)
      .attr('transform', (d: any) => `translate(${d.y},${d.x})`);
    
    // Update the node attributes and style
    nodeUpdate.select('circle')
      .transition()
      .duration(this.animationSpeed)
      .attr('r', this.nodeRadius)
      .attr('fill', (d: any) => {
        if (d.children) {
          return '#69b3a2'; // Internal nodes
        } else {
          // For leaf nodes, use a gradient based on class distribution
          const yesRatio = d.data.value[0] / (d.data.value[0] + d.data.value[1]);
          return d.data.class === 'yes' ? 
            d3.interpolateRgb('#28a745', '#50c878')(yesRatio) : 
            d3.interpolateRgb('#dc3545', '#ff6b6b')(1 - yesRatio);
        }
      });
    
    nodeUpdate.select('text')
      .transition()
      .duration(this.animationSpeed)
      .style('fill-opacity', 1);
    
    // Remove any exiting nodes
    const nodeExit = node.exit().transition()
      .duration(this.animationSpeed)
      .attr('transform', (d: any) => `translate(${source.y},${source.x})`)
      .remove();
    
    nodeExit.select('circle')
      .attr('r', 0);
    
    nodeExit.select('text')
      .style('fill-opacity', 0);
    
    // Update the links
    const link = this.svg.selectAll('path.link')
      .data(links, (d: any) => d.target.id);
    
    // Enter any new links at the parent's previous position
    const linkEnter = link.enter().insert('path', 'g')
      .attr('class', 'link')
      .attr('d', (d: any) => {
        const o = {x: source.x0 || source.x, y: source.y0 || source.y};
        return this.diagonal({source: o, target: o});
      })
      .attr('fill', 'none')
      .attr('stroke', '#88A0A8')
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0);
    
    // UPDATE
    const linkUpdate = linkEnter.merge(link);
    
    // Transition back to the parent element position
    linkUpdate.transition()
      .duration(this.animationSpeed)
      .attr('d', this.diagonal)
      .attr('stroke-opacity', 0.8);
    
    // Remove any exiting links
    link.exit().transition()
      .duration(this.animationSpeed)
      .attr('d', (d: any) => {
        const o = {x: source.x, y: source.y};
        return this.diagonal({source: o, target: o});
      })
      .attr('stroke-opacity', 0)
      .remove();
    
    // Store the old positions for transition
    nodes.forEach((d: any) => {
      d.x0 = d.x;
      d.y0 = d.y;
    });
  }

  // Simulation control methods
  playSimulation(): void {
    this.isAutoPlaying = true;
    this.currentStepNumber = 0;
    this.clearAnimationTimer();
    
    // Run the simulation steps in sequence
    this.animationTimer = setInterval(() => {
      this.zone.run(() => {
        this.nextStep();
        if (this.currentStepNumber >= this.totalSteps) {
          this.stopSimulation();
        }
      });
    }, this.animationSpeed);
  }

  stopSimulation(): void {
    this.isAutoPlaying = false;
    this.clearAnimationTimer();
  }

  prevStep(): void {
    if (this.currentStepNumber > 0) {
      this.currentStepNumber--;
      this.buildProgress = (this.currentStepNumber / this.totalSteps) * 100;
      this.updateSimulationStep();
    }
  }

  nextStep(): void {
    if (this.currentStepNumber < this.totalSteps) {
      this.currentStepNumber++;
      this.buildProgress = (this.currentStepNumber / this.totalSteps) * 100;
      this.updateSimulationStep();
    }
  }

  private clearAnimationTimer(): void {
    if (this.animationTimer) {
      clearInterval(this.animationTimer);
      this.animationTimer = null;
    }
  }

  private updateSimulationStep(): void {
    switch (this.currentStepNumber) {
      case 0:
        this.resetSimulation();
        this.stepDescription = 'Welcome to Decision Tree Simulation. Press Play or Next to begin.';
        break;
      case 1:
        this.prepareData();
        this.stepDescription = 'Step 1: Preparing Data - We gather features and labels for our dataset.';
        break;
      case 2:
        this.calculateBestSplit();
        this.stepDescription = 'Step 2: Finding Best Split - We evaluate each feature to find the optimal split.';
        break;
      case 3:
        this.performFirstSplit();
        this.stepDescription = 'Step 3: First Split - We divide the data based on the best feature and threshold.';
        break;
      case 4:
        this.recursiveSplitting();
        this.stepDescription = 'Step 4: Recursive Splitting - We continue splitting until stopping criteria are met.';
        break;
      case 5:
        this.applyStoppingCriteria();
        this.stepDescription = 'Step 5: Stopping Criteria - We stop when nodes are pure or other criteria are met.';
        break;
      case 6:
        this.demonstratePrediction();
        this.stepDescription = 'Step 6: Making Predictions - We can now use the tree to classify new instances.';
        break;
      default:
        break;
    }
  }

  // Simulation step implementations
  private resetSimulation(): void {
    this.initializeTreeData();
    this.renderTreeData();
    this.renderDataPoints();
  }

  private prepareData(): void {
    // Animate data points gathering
    d3.selectAll('.data-point')
      .transition()
      .duration(this.animationSpeed / 2)
      .attr('r', 7)
      .attr('stroke-width', 1)
      .transition()
      .duration(this.animationSpeed / 2)
      .attr('r', 5)
      .attr('stroke-width', 0.5);
  }

  private calculateBestSplit(): void {
    // Show calculation for finding best feature to split on
    const bestFeature = 'age';
    const bestThreshold = 35;
    
    // Update metadata to show we're evaluating splits
    this.treeData.feature = bestFeature;
    this.treeData.threshold = bestThreshold;
    this.treeData.description = `Evaluating splits: '${bestFeature}' with threshold ${bestThreshold} gives the best information gain.`;
    
    // Re-render with updated information
    this.renderTreeData();
    
    // Highlight the selected feature in the data visualization
    this.selectedFeature = bestFeature;
    this.renderDataPoints();
  }

  private performFirstSplit(): void {
    // Create first split in the tree
    const leftNode: TreeNode = {
      name: 'Left',
      feature: 'income',
      threshold: 50,
      samples: 30,
      value: [22, 8],
      gini: 0.35,
      entropy: 0.82,
      mse: 0.18,
      description: 'Node for data points with age ≤ 35.',
      children: []
    };
    
    const rightNode: TreeNode = {
      name: 'Right',
      feature: 'credit_score',
      threshold: 70,
      samples: 20,
      value: [8, 12],
      gini: 0.42,
      entropy: 0.88,
      mse: 0.21,
      description: 'Node for data points with age > 35.',
      children: []
    };
    
    this.treeData.children = [leftNode, rightNode];
    this.treeData.description = 'Root node split on age with threshold 35.';
    
    // Re-render the tree with the new structure
    this.renderTreeData();
    
    // Animate data points moving to their respective branches
    this.animateDataSplit('age', 35);
  }

  private recursiveSplitting(): void {
    // Add more splits to demonstrate recursive nature
    if (this.treeData.children && this.treeData.children.length > 0) {
      const leftNode = this.treeData.children[0];
      const rightNode = this.treeData.children[1];
      
      // Add children to left node
      leftNode.children = [
        {
          name: 'Left-Left',
          samples: 20,
          value: [18, 2],
          gini: 0.18,
          entropy: 0.42,
          mse: 0.09,
          description: 'Node for data with age ≤ 35 and income ≤ 50.',
          class: 'yes'
        },
        {
          name: 'Left-Right',
          samples: 10,
          value: [4, 6],
          gini: 0.48,
          entropy: 0.92,
          mse: 0.24,
          description: 'Node for data with age ≤ 35 and income > 50.',
          feature: 'education',
          threshold: 60,
          children: []
        }
      ];
      
      // Add children to right node
      rightNode.children = [
        {
          name: 'Right-Left',
          samples: 12,
          value: [3, 9],
          gini: 0.375,
          entropy: 0.78,
          mse: 0.1875,
          description: 'Node for data with age > 35 and credit_score ≤ 70.',
          class: 'no'
        },
        {
          name: 'Right-Right',
          samples: 8,
          value: [5, 3],
          gini: 0.46,
          entropy: 0.94,
          mse: 0.23,
          description: 'Node for data with age > 35 and credit_score > 70.',
          feature: 'education',
          threshold: 80,
          children: []
        }
      ];
      
      // Re-render the tree with the new structure
      this.renderTreeData();
    }
  }

  private applyStoppingCriteria(): void {
    // Complete the tree by adding final leaves
    if (this.treeData.children && this.treeData.children.length > 0) {
      // Left branch completion
      if (this.treeData.children[0].children && this.treeData.children[0].children.length > 0) {
        const leftRightNode = this.treeData.children[0].children[1];
        if (leftRightNode.children) {
          leftRightNode.children = [
            {
              name: 'Leaf',
              samples: 6,
              value: [1, 5],
              gini: 0.28,
              entropy: 0.65,
              mse: 0.14,
              class: 'no',
              description: 'Leaf node for data with age ≤ 35, income > 50, and education ≤ 60.'
            },
            {
              name: 'Leaf',
              samples: 4,
              value: [3, 1],
              gini: 0.375,
              entropy: 0.81,
              mse: 0.19,
              class: 'yes',
              description: 'Leaf node for data with age ≤ 35, income > 50, and education > 60.'
            }
          ];
        }
      }
      
      // Right branch completion
      if (this.treeData.children[1].children && this.treeData.children[1].children.length > 0) {
        const rightRightNode = this.treeData.children[1].children[1];
        if (rightRightNode.children !== undefined) {
          rightRightNode.children = [
            {
              name: 'Leaf',
              samples: 3,
              value: [1, 2],
              gini: 0.44,
              entropy: 0.92,
              mse: 0.22,
              class: 'no',
              description: 'Leaf node for data with age > 35, credit_score > 70, and education ≤ 80.'
            },
            {
              name: 'Leaf',
              samples: 5,
              value: [4, 1],
              gini: 0.32,
              entropy: 0.72,
              mse: 0.16,
              class: 'yes',
              description: 'Leaf node for data with age > 35, credit_score > 70, and education > 80.'
            }
          ];
        }
      }
      
      // Re-render the tree with the complete structure
      this.renderTreeData();
    }
  }

  private demonstratePrediction(): void {
    // Animate a new data point flowing through the tree to show prediction
    const newDataPoint: DataPoint = {
      id: 999,
      features: {
        age: 28,
        income: 75,
        education: 85,
        credit_score: 90
      },
      label: '?',
      x: 0,
      y: 0,
      currentNode: 'root'
    };
    
    // Add the new data point with distinctive styling
    this.dataSvg.append('circle')
      .attr('class', 'prediction-point')
      .attr('cx', () => {
        const xScale = d3.scaleLinear().domain([0, 100]).range([-this.width / 7, this.width / 7]);
        return xScale(newDataPoint.features[this.selectedFeature]);
      })
      .attr('cy', () => {
        const yFeature = this.availableFeatures.find(f => f !== this.selectedFeature) || 'income';
        const yScale = d3.scaleLinear().domain([0, 100]).range([-this.height / 7, this.height / 7]);
        return yScale(newDataPoint.features[yFeature]);
      })
      .attr('r', 10)
      .attr('fill', 'blue')
      .attr('stroke', 'black')
      .attr('stroke-width', 2)
      .attr('opacity', 0.7);
    
    // Animate the prediction path through the tree
    this.animatePrediction(newDataPoint);
  }

  private animateDataSplit(feature: string, threshold: number): void {
    // Update data points to show which side of the split they belong to
    d3.selectAll('.data-point')
      .transition()
      .duration(this.animationSpeed)
      .attr('cx', (d: any) => {
        const xScale = d3.scaleLinear().domain([0, 100]).range([-this.width / 7, this.width / 7]);
        const featureKey = this.selectedFeature;
        // Shift points slightly left or right based on the threshold
        const offset = (d.features[featureKey] <= threshold) ? -10 : 10;
        return xScale(d.features[featureKey]) + offset;
      });
  }

  private animatePrediction(dataPoint: DataPoint): void {
    // This would be a multi-step animation showing the data point
    // traversing the tree based on its feature values
    // For simplicity, we're using a placeholder here
    console.log('Animating prediction for', dataPoint);
    
    // Highlight path in the tree (simplified version)
    // In a full implementation, this would trace the actual path
    this.svg.selectAll('.link')
      .transition()
      .duration(this.animationSpeed / 3)
      .attr('stroke', (d: any) => {
        // Determine if this link is in the path for our prediction
        // This is a simplified version - a real implementation would trace the actual path
        return d.target.data.feature === 'age' || d.target.data.feature === 'income' ? 'orange' : '#ccc';
      })
      .attr('stroke-width', (d: any) => {
        return d.target.data.feature === 'age' || d.target.data.feature === 'income' ? 4 : 2;
      });
    
    // Update description to show the prediction result
    this.stepDescription = `Prediction: Following the path through the tree based on features (age: ${dataPoint.features['age']}, income: ${dataPoint.features['income']}, etc.), this new instance would be classified as 'yes'.`;
  }

  // UI Event handlers
  onFeatureChange(feature: string): void {
    this.selectedFeature = feature;
    this.renderDataPoints();
  }

  onMetricChange(metric: 'gini' | 'entropy' | 'mse'): void {
    this.metricType = metric;
    this.renderTreeData();
  }

  onSpeedChange(speed: number): void {
    this.animationSpeed = speed;
  }
  
  // Accordion toggle functionality
  accordionToggle(event: Event): void {
    const header = event.currentTarget as HTMLElement;
    const content = header.nextElementSibling as HTMLElement;
    
    // Toggle active class on header
    header.classList.toggle('active');
    
    // Toggle show class on content
    if (content.classList.contains('show')) {
      content.classList.remove('show');
    } else {
      content.classList.add('show');
    }
  }
}