import { Component, OnInit, ElementRef, ViewChild, AfterViewInit, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as d3 from 'd3';

@Component({
  selector: 'app-svm-visualization',
  templateUrl: './svm-visualization.component.html',
  styleUrls: ['./svm-visualization.component.scss'],
  standalone: true,
  imports: [CommonModule]
})
export class SvmVisualizationComponent implements OnInit, AfterViewInit {
  @ViewChild('marginCanvas') marginCanvasRef!: ElementRef;
  @ViewChild('kernelCanvas') kernelCanvasRef!: ElementRef;
  @ViewChild('cParamCanvas') cParamCanvasRef!: ElementRef;

  // Active tab
  activeTab: string = 'tab1';

  // Animation control
  isPlaying: boolean = false;
  animationFrameId: number | null = null;
  animationSpeed: number = 1000; // milliseconds between frames
  currentStep: number = 0;
  maxSteps: number = 50;
  
  // Visualization parameters
  marginWidth: number = 800;
  marginHeight: number = 500;
  kernelWidth: number = 800;
  kernelHeight: number = 500;
  cParamWidth: number = 800;
  cParamHeight: number = 500;
  
  // SVM parameters that users can adjust
  cValue: number = 1; // Regularization parameter
  kernelType: string = 'rbf'; // 'linear', 'rbf', 'poly'
  gammaValue: number = 0.1; // RBF kernel parameter

  // Data sets
  linearData: Array<[number, number, number]> = []; // [x, y, class]
  nonLinearData: Array<[number, number, number]> = []; // [x, y, class]
  overlappingData: Array<[number, number, number]> = []; // [x, y, class]

  // SVG elements
  marginSvg: any;
  kernelSvg: any;
  cParamSvg: any;

  // Scales for each visualization
  marginXScale: any;
  marginYScale: any;
  kernelXScale: any;
  kernelYScale: any;
  cParamXScale: any;
  cParamYScale: any;

  // Current SVM parameters
  currentMargin: number = 0.1;
  currentHyperplane: { w: [number, number], b: number } = { w: [1, 1], b: 0 };
  
  // Animation timers
  marginTimer: any;
  kernelTimer: any;
  cParamTimer: any;

  // Tooltip data
  tooltipText: string = '';
  showTooltip: boolean = false;
  tooltipX: number = 0;
  tooltipY: number = 0;

  constructor() { }

  ngOnInit(): void {
    this.generateData();
  }

  ngAfterViewInit(): void {
    // Ensure we run initialization after the view is fully rendered
    setTimeout(() => {
      // Check if references are available before initializing
      if (this.marginCanvasRef?.nativeElement) {
        this.initMarginVisualization();
      } else {
        console.warn('marginCanvasRef not available');
      }
      
      if (this.kernelCanvasRef?.nativeElement) {
        this.initKernelVisualization();
      } else {
        console.warn('kernelCanvasRef not available');
      }
      
      if (this.cParamCanvasRef?.nativeElement) {
        this.initCParamVisualization();
      } else {
        console.warn('cParamCanvasRef not available');
      }
    }, 0);
  }

  @HostListener('window:resize')
  onResize() {
    // Clear previous visualizations
    this.clearVisualizations();
    
    // Reinitialize visualizations
    this.initMarginVisualization();
    this.initKernelVisualization();
    this.initCParamVisualization();
  }

  clearVisualizations() {
    if (this.marginSvg) d3.select(this.marginCanvasRef.nativeElement).selectAll('*').remove();
    if (this.kernelSvg) d3.select(this.kernelCanvasRef.nativeElement).selectAll('*').remove();
    if (this.cParamSvg) d3.select(this.cParamCanvasRef.nativeElement).selectAll('*').remove();
  }

  generateData(): void {
    // Generate linearly separable data
    this.linearData = [];
    for (let i = 0; i < 50; i++) {
      const x = Math.random() * 10;
      const y = Math.random() * 10;
      const cls = (y > x) ? 1 : -1;
      this.linearData.push([x, y, cls]);
    }

    // Generate non-linearly separable data (circular pattern)
    this.nonLinearData = [];
    for (let i = 0; i < 100; i++) {
      const r = Math.random() * 2 * Math.PI;
      const rad = Math.random() * 5;
      const x = 5 + rad * Math.cos(r);
      const y = 5 + rad * Math.sin(r);
      const cls = rad < 2.5 ? 1 : -1;
      this.nonLinearData.push([x, y, cls]);
    }

    // Generate overlapping data for C parameter visualization
    this.overlappingData = [];
    for (let i = 0; i < 80; i++) {
      const x = Math.random() * 10;
      const y = Math.random() * 10;
      // Add some overlap between classes
      let cls;
      if (y > x + 1) cls = 1;
      else if (y < x - 1) cls = -1;
      else cls = Math.random() > 0.5 ? 1 : -1;
      this.overlappingData.push([x, y, cls]);
    }
  }

  // Tab handling
  setActiveTab(tab: string): void {
    this.activeTab = tab;
    this.resetSimulations();
    
    // Reinitialize the currently visible visualization after a short delay
    // to ensure the DOM elements are ready
    setTimeout(() => {
      if (tab === 'tab1' && this.marginCanvasRef?.nativeElement) {
        this.initMarginVisualization();
      } else if (tab === 'tab2' && this.kernelCanvasRef?.nativeElement) {
        this.initKernelVisualization();
      } else if (tab === 'tab3' && this.cParamCanvasRef?.nativeElement) {
        this.initCParamVisualization();
      }
    }, 100);
  }

  // Animation control
  playSimulation(): void {
    this.isPlaying = true;
    this.currentStep = 0;
    
    // Clear any existing intervals
    this.resetSimulations();
    
    // Start the appropriate animation
    if (this.activeTab === 'tab1' && this.marginCanvasRef?.nativeElement) {
      this.animateMarginMaximization();
    } else if (this.activeTab === 'tab2' && this.kernelCanvasRef?.nativeElement) {
      this.animateKernelTransformation();
    } else if (this.activeTab === 'tab3' && this.cParamCanvasRef?.nativeElement) {
      this.animateCParameterEffect();
    }
  }

  stopSimulation(): void {
    this.isPlaying = false;
    this.resetSimulations();
  }

  resetSimulations(): void {
    // Clear all timers and animation frames
    if (this.marginTimer) clearInterval(this.marginTimer);
    if (this.kernelTimer) clearInterval(this.kernelTimer);
    if (this.cParamTimer) clearInterval(this.cParamTimer);
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
    this.currentStep = 0;
  }

  // Speed control
  setSpeed(speed: number): void {
    this.animationSpeed = speed;
    if (this.isPlaying) {
      this.stopSimulation();
      this.playSimulation();
    }
  }

  // Parameter adjustment
  setCValue(value: number): void {
    this.cValue = value;
    if (this.activeTab === 'tab3') {
      this.initCParamVisualization();
    }
  }

  setKernelType(type: string): void {
    this.kernelType = type;
    if (this.activeTab === 'tab2') {
      this.initKernelVisualization();
    }
  }

  setGammaValue(value: number): void {
    this.gammaValue = value;
    if (this.activeTab === 'tab2' && this.kernelType === 'rbf') {
      this.initKernelVisualization();
    }
  }

  // Visualizations initialization
  initMarginVisualization(): void {
    if (!this.marginCanvasRef) {
      console.error('marginCanvasRef is not defined');
      return;
    }
    
    const element = this.marginCanvasRef.nativeElement;
    if (!element) {
      console.error('marginCanvasRef.nativeElement is not defined');
      return;
    }
    
    const width = element.clientWidth || this.marginWidth;
    const height = element.clientHeight || this.marginHeight;
    
    // Clear previous SVG
    d3.select(element).selectAll('*').remove();
    
    // Create SVG
    this.marginSvg = d3.select(element)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
      
    // Create scales
    this.marginXScale = d3.scaleLinear()
      .domain([0, 10])
      .range([50, width - 50]);
      
    this.marginYScale = d3.scaleLinear()
      .domain([0, 10])
      .range([height - 50, 50]);
      
    // Draw axes
    this.drawAxes(this.marginSvg, this.marginXScale, this.marginYScale, width, height);
    
    // Draw initial data points
    this.drawDataPoints(this.marginSvg, this.linearData, this.marginXScale, this.marginYScale);
    
    // Initial hyperplane (random)
    this.currentHyperplane = { w: [Math.random() * 2 - 1, Math.random() * 2 - 1], b: Math.random() * 2 - 1 };
    this.currentMargin = 0.1;
    
    // Draw initial hyperplane
    this.drawHyperplane(
      this.marginSvg, 
      this.currentHyperplane, 
      this.marginXScale, 
      this.marginYScale, 
      this.currentMargin
    );
  }

  initKernelVisualization(): void {
    if (!this.kernelCanvasRef) {
      console.error('kernelCanvasRef is not defined');
      return;
    }
    
    const element = this.kernelCanvasRef.nativeElement;
    if (!element) {
      console.error('kernelCanvasRef.nativeElement is not defined');
      return;
    }
    
    const width = element.clientWidth || this.kernelWidth;
    const height = element.clientHeight || this.kernelHeight;
    
    // Clear previous SVG
    d3.select(element).selectAll('*').remove();
    
    // Create SVG with two views side by side
    this.kernelSvg = d3.select(element)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
      
    // Create scales for original space (left)
    const originalWidth = width / 2 - 20;
    this.kernelXScale = d3.scaleLinear()
      .domain([0, 10])
      .range([20, originalWidth - 20]);
      
    this.kernelYScale = d3.scaleLinear()
      .domain([0, 10])
      .range([height - 50, 50]);
      
    // Create scales for transformed space (right)
    const transformedWidth = width / 2;
    const transformedXScale = d3.scaleLinear()
      .domain([0, 10])
      .range([width / 2 + 20, width - 40]);
      
    const transformedYScale = d3.scaleLinear()
      .domain([0, 10])
      .range([height - 50, 50]);
    
    // Draw original data (left side)
    this.kernelSvg.append('g')
      .attr('class', 'original-data')
      .selectAll('circle')
      .data(this.nonLinearData)
      .enter()
      .append('circle')
      .attr('cx', (d: [number, number, number]) => this.kernelXScale(d[0]))
      .attr('cy', (d: [number, number, number]) => this.kernelYScale(d[1]))
      .attr('r', 6)
      .attr('fill', (d: [number, number, number]) => d[2] === 1 ? '#4285F4' : '#EA4335')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5);
      
    // Draw axes for original space
    this.drawAxes(
      this.kernelSvg.append('g').attr('class', 'original-axes'), 
      this.kernelXScale, 
      this.kernelYScale, 
      originalWidth, 
      height
    );
    
    // Draw axes for transformed space
    this.drawAxes(
      this.kernelSvg.append('g').attr('class', 'transformed-axes'), 
      transformedXScale, 
      transformedYScale, 
      width, 
      height
    );
    
    // Draw separating line if linear kernel
    if (this.kernelType === 'linear') {
      // Draw non-linear boundary (not possible for truly non-linear data)
      this.drawNonLinearBoundary(
        this.kernelSvg.append('g').attr('class', 'original-boundary'),
        this.kernelXScale,
        this.kernelYScale,
        originalWidth,
        height,
        false
      );
    } else {
      // Draw non-linear boundary
      this.drawNonLinearBoundary(
        this.kernelSvg.append('g').attr('class', 'original-boundary'),
        this.kernelXScale,
        this.kernelYScale,
        originalWidth,
        height,
        true
      );
    }
    
    // Draw transformed data points (right side)
    // This is a simplified visualization of the kernel transformation
    this.drawTransformedData(
      this.kernelSvg.append('g').attr('class', 'transformed-data'),
      this.nonLinearData,
      transformedXScale,
      transformedYScale,
      this.kernelType,
      this.gammaValue
    );
    
    // Draw linear hyperplane in transformed space
    this.drawLinearBoundaryInTransformedSpace(
      this.kernelSvg.append('g').attr('class', 'transformed-boundary'),
      transformedXScale,
      transformedYScale,
      width,
      height
    );
    
    // Add labels
    this.kernelSvg.append('text')
      .attr('x', originalWidth / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text('Original Space');
      
    this.kernelSvg.append('text')
      .attr('x', width / 2 + originalWidth / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text('Transformed Space');
  }

  initCParamVisualization(): void {
    if (!this.cParamCanvasRef) {
      console.error('cParamCanvasRef is not defined');
      return;
    }
    
    const element = this.cParamCanvasRef.nativeElement;
    if (!element) {
      console.error('cParamCanvasRef.nativeElement is not defined');
      return;
    }
    
    const width = element.clientWidth || this.cParamWidth;
    const height = element.clientHeight || this.cParamHeight;
    
    // Clear previous SVG
    d3.select(element).selectAll('*').remove();
    
    // Create SVG for four different C values
    this.cParamSvg = d3.select(element)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
      
    // Define the four C values to display
    const cValues = [0.1, 1, 10, 100];
    const gridWidth = width / 2;
    const gridHeight = height / 2;
    
    // Create SVG groups for each C value
    for (let i = 0; i < 4; i++) {
      const row = Math.floor(i / 2);
      const col = i % 2;
      const x = col * gridWidth;
      const y = row * gridHeight;
      
      const cValue = cValues[i];
      
      // Create group for this C value
      const group = this.cParamSvg.append('g')
        .attr('class', `c-param-${cValue}`)
        .attr('transform', `translate(${x}, ${y})`);
        
      // Add label
      group.append('text')
        .attr('x', gridWidth / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .style('font-weight', 'bold')
        .text(`C = ${cValue}`);
        
      // Create scales for this grid
      const xScale = d3.scaleLinear()
        .domain([0, 10])
        .range([50, gridWidth - 20]);
        
      const yScale = d3.scaleLinear()
        .domain([0, 10])
        .range([gridHeight - 50, 50]);
        
      // Draw axes
      this.drawAxes(group, xScale, yScale, gridWidth, gridHeight);
      
      // Draw data points
      this.drawDataPoints(group, this.overlappingData, xScale, yScale);
      
      // Draw SVM hyperplane for this C value
      const hyperplane = this.computeSVMHyperplane(this.overlappingData, cValue);
      this.drawHyperplane(group, hyperplane, xScale, yScale, 1 / Math.sqrt(cValue));
    }
  }

  // Drawing utilities
  drawAxes(svg: any, xScale: any, yScale: any, width: number, height: number): void {
    // X axis
    svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${height - 50})`)
      .call(d3.axisBottom(xScale));
      
    // Y axis
    svg.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(50, 0)`)
      .call(d3.axisLeft(yScale));
  }

  drawDataPoints(svg: any, data: Array<[number, number, number]>, xScale: any, yScale: any): void {
    svg.selectAll('circle.data-point')
      .data(data)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', (d: [number, number, number]) => xScale(d[0]))
      .attr('cy', (d: [number, number, number]) => yScale(d[1]))
      .attr('r', 6)
      .attr('fill', (d: [number, number, number]) => d[2] === 1 ? '#4285F4' : '#EA4335')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .on('mouseover', (event: any, d: [number, number, number]) => {
        this.showTooltip = true;
        this.tooltipX = event.pageX;
        this.tooltipY = event.pageY;
        this.tooltipText = `Point (${d[0].toFixed(2)}, ${d[1].toFixed(2)}), Class: ${d[2] === 1 ? '+1' : '-1'}`;
      })
      .on('mouseout', () => {
        this.showTooltip = false;
      });
  }

  drawHyperplane(
    svg: any, 
    hyperplane: { w: [number, number], b: number }, 
    xScale: any, 
    yScale: any, 
    margin: number
  ): void {
    const { w, b } = hyperplane;
    
    // Calculate two points for drawing the line (across the x-axis)
    const x1 = 0;
    const x2 = 10;
    
    // If hyperplane is w[0]*x + w[1]*y + b = 0, then y = -(w[0]*x + b)/w[1]
    const y1 = -(w[0] * x1 + b) / w[1];
    const y2 = -(w[0] * x2 + b) / w[1];
    
    // Draw decision boundary
    svg.append('line')
      .attr('class', 'hyperplane')
      .attr('x1', xScale(x1))
      .attr('y1', yScale(y1))
      .attr('x2', xScale(x2))
      .attr('y2', yScale(y2))
      .attr('stroke', '#34A853')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '0');
      
    // Draw margin boundaries (w*x + b = +1 and w*x + b = -1)
    // Upper margin
    const y1Upper = -(w[0] * x1 + b - 1) / w[1];
    const y2Upper = -(w[0] * x2 + b - 1) / w[1];
    
    svg.append('line')
      .attr('class', 'margin-boundary upper')
      .attr('x1', xScale(x1))
      .attr('y1', yScale(y1Upper))
      .attr('x2', xScale(x2))
      .attr('y2', yScale(y2Upper))
      .attr('stroke', '#34A853')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,5');
      
    // Lower margin
    const y1Lower = -(w[0] * x1 + b + 1) / w[1];
    const y2Lower = -(w[0] * x2 + b + 1) / w[1];
    
    svg.append('line')
      .attr('class', 'margin-boundary lower')
      .attr('x1', xScale(x1))
      .attr('y1', yScale(y1Lower))
      .attr('x2', xScale(x2))
      .attr('y2', yScale(y2Lower))
      .attr('stroke', '#34A853')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,5');
      
    // Fill the margin area
    const points = [
      [x1, y1Upper], [x2, y2Upper], 
      [x2, y2Lower], [x1, y1Lower]
    ];
    
    svg.append('polygon')
      .attr('class', 'margin-area')
      .attr('points', points.map(p => `${xScale(p[0])},${yScale(p[1])}`).join(' '))
      .attr('fill', '#34A853')
      .attr('fill-opacity', 0.1);
      
    // Identify support vectors (points that lie on or near the margin boundaries)
    // For simplicity, we'll use a threshold to determine if a point is close enough to the margin
    const threshold = 0.1;
    
    // Extract the data from the existing circles
    const data: Array<[number, number, number]> = svg.selectAll('circle.data-point').data();
    
    // For each data point, calculate distance to hyperplane
    data.forEach((d: [number, number, number]) => {
      const x = d[0];
      const y = d[1];
      const distance = Math.abs(w[0] * x + w[1] * y + b) / Math.sqrt(w[0] * w[0] + w[1] * w[1]);
      
      // If distance is close to 1, it's a support vector
      if (Math.abs(distance - 1) < threshold) {
        // Mark as support vector by adding a ring
        svg.append('circle')
          .attr('class', 'support-vector')
          .attr('cx', xScale(x))
          .attr('cy', yScale(y))
          .attr('r', 10)
          .attr('fill', 'none')
          .attr('stroke', '#FBBC05')
          .attr('stroke-width', 2);
      }
    });
  }

  drawNonLinearBoundary(
    svg: any, 
    xScale: any, 
    yScale: any, 
    width: number, 
    height: number, 
    isNonLinear: boolean
  ): void {
    if (isNonLinear) {
      // Draw a circular decision boundary for the simple non-linear case
      svg.append('circle')
        .attr('cx', xScale(5))
        .attr('cy', yScale(5))
        .attr('r', xScale(2.5) - xScale(0))
        .attr('fill', 'none')
        .attr('stroke', '#34A853')
        .attr('stroke-width', 2);
    } else {
      // Draw a straight line for linear case (which won't work well for non-linear data)
      const x1 = 0;
      const x2 = 10;
      const y1 = 5;
      const y2 = 5;
      
      svg.append('line')
        .attr('x1', xScale(x1))
        .attr('y1', yScale(y1))
        .attr('x2', xScale(x2))
        .attr('y2', yScale(y2))
        .attr('stroke', '#34A853')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5');
    }
  }

  drawTransformedData(
    svg: any, 
    data: Array<[number, number, number]>, 
    xScale: any, 
    yScale: any, 
    kernelType: string,
    gammaValue: number
  ): void {
    // Apply a simplified kernel transformation
    const transformedData = data.map((d: [number, number, number]) => {
      const [x, y, cls] = d;
      
      let newX, newY;
      
      if (kernelType === 'linear') {
        // Linear kernel - no transformation
        newX = x;
        newY = y;
      } else if (kernelType === 'rbf') {
        // RBF kernel - transform into a different arrangement
        // This is a simplified visualization - real kernel transformations are more complex
        const distance = Math.sqrt((x - 5) ** 2 + (y - 5) ** 2);
        const angle = Math.atan2(y - 5, x - 5);
        
        // Apply gamma parameter to control the shape of the transformation
        const transformedDist = 5 * (1 - Math.exp(-gammaValue * distance ** 2));
        
        newX = 5 + transformedDist * Math.cos(angle);
        newY = 5 + (cls === 1 ? 2 : -2) + transformedDist * Math.sin(angle) * 0.2;
      } else if (kernelType === 'poly') {
        // Polynomial kernel - raise to power 2 for visualization
        newX = ((x - 5) ** 2) * 0.05 + 5;
        newY = ((y - 5) ** 2) * 0.05 + (cls === 1 ? 7 : 3);
      } else {
        newX = x;
        newY = y;
      }
      
      return [newX, newY, cls];
    });
    
    // Draw transformed data points
    svg.selectAll('circle')
      .data(transformedData)
      .enter()
      .append('circle')
      .attr('cx', (d: [number, number, number]) => xScale(d[0]))
      .attr('cy', (d: [number, number, number]) => yScale(d[1]))
      .attr('r', 6)
      .attr('fill', (d: [number, number, number]) => d[2] === 1 ? '#4285F4' : '#EA4335')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5);
  }

  drawLinearBoundaryInTransformedSpace(
    svg: any, 
    xScale: any, 
    yScale: any, 
    width: number, 
    height: number
  ): void {
    // Draw a line separating the transformed classes
    let x1, x2, y1, y2;
    
    if (this.kernelType === 'linear') {
      // Similar to original space, linear boundary won't work well
      x1 = 0;
      x2 = 10;
      y1 = 5;
      y2 = 5;
    } else if (this.kernelType === 'rbf') {
      // For RBF, the boundary becomes a horizontal line in our simplified visualization
      x1 = 0;
      x2 = 10;
      y1 = 5;
      y2 = 5;
    } else if (this.kernelType === 'poly') {
      // For polynomial, also a horizontal line in our simplified visualization
      x1 = 0;
      x2 = 10;
      y1 = 5;
      y2 = 5;
    } else {
      x1 = 0;
      x2 = 10;
      y1 = 5;
      y2 = 5;
    }
    
    svg.append('line')
      .attr('x1', xScale(x1))
      .attr('y1', yScale(y1))
      .attr('x2', xScale(x2))
      .attr('y2', yScale(y2))
      .attr('stroke', '#34A853')
      .attr('stroke-width', 2);
  }

  // SVM computation (simplified)
  computeSVMHyperplane(data: Array<[number, number, number]>, cValue: number): { w: [number, number], b: number } {
    // This is a very simplified version - in reality, SVM optimization is much more complex
    // For demonstration purposes, we'll use a heuristic based on the data
    
    // Separate positive and negative examples
    const positivePoints = data.filter(d => d[2] === 1);
    const negativePoints = data.filter(d => d[2] === -1);
    
    // Compute centroids
    const centroidPos = this.computeCentroid(positivePoints);
    const centroidNeg = this.computeCentroid(negativePoints);
    
    // Weight vector is the difference between centroids
    const w: [number, number] = [
      centroidPos[0] - centroidNeg[0],
      centroidPos[1] - centroidNeg[1]
    ];
    
    // Normalize weight vector
    const normW = Math.sqrt(w[0] ** 2 + w[1] ** 2);
    w[0] /= normW;
    w[1] /= normW;
    
    // Compute bias to place the hyperplane halfway between centroids
    const midpoint = [(centroidPos[0] + centroidNeg[0]) / 2, (centroidPos[1] + centroidNeg[1]) / 2];
    const b = -(w[0] * midpoint[0] + w[1] * midpoint[1]);
    
    // For C parameter effect: small C = larger margin = less fitted to data
    // large C = smaller margin = more fitted to data
    // Scale w inversely with C to simulate effect (approximately)
    w[0] *= Math.sqrt(cValue);
    w[1] *= Math.sqrt(cValue);
    
    return { w, b };
  }

  computeCentroid(points: Array<[number, number, number]>): [number, number] {
    if (points.length === 0) return [0, 0];
    
    const sumX = points.reduce((sum, p) => sum + p[0], 0);
    const sumY = points.reduce((sum, p) => sum + p[1], 0);
    
    return [sumX / points.length, sumY / points.length];
  }

  // Animations
  animateMarginMaximization(): void {
    // Start with a random hyperplane
    this.currentHyperplane = { 
      w: [Math.random() * 2 - 1, Math.random() * 2 - 1], 
      b: Math.random() * 2 - 1 
    };
    this.currentMargin = 0.1;
    
    let step = 0;
    const totalSteps = 50;
    
    // Compute the optimal hyperplane (simplified)
    const optimalHyperplane = this.computeSVMHyperplane(this.linearData, 1);
    
    // Animate the hyperplane and margin gradually approaching the optimal solution
    this.marginTimer = setInterval(() => {
      step++;
      
      if (step >= totalSteps) {
        clearInterval(this.marginTimer);
        this.isPlaying = false;
        return;
      }
      
      // Interpolate between current and optimal
      const t = step / totalSteps;
      this.currentHyperplane.w[0] = (1 - t) * this.currentHyperplane.w[0] + t * optimalHyperplane.w[0];
      this.currentHyperplane.w[1] = (1 - t) * this.currentHyperplane.w[1] + t * optimalHyperplane.w[1];
      this.currentHyperplane.b = (1 - t) * this.currentHyperplane.b + t * optimalHyperplane.b;
      
      // Margin starts small and grows
      this.currentMargin = 0.1 + 0.9 * t;
      
      // Clear and redraw
      this.marginSvg.selectAll('.hyperplane, .margin-boundary, .margin-area, .support-vector').remove();
      this.drawHyperplane(
        this.marginSvg, 
        this.currentHyperplane, 
        this.marginXScale, 
        this.marginYScale, 
        this.currentMargin
      );
      
      // Update step display
      this.currentStep = step;
    }, this.animationSpeed / totalSteps);
  }

  animateKernelTransformation(): void {
    // This animation will show how data transforms from original to kernel space
    let step = 0;
    const totalSteps = 50;
    
    // Clear existing transformed data and boundary
    this.kernelSvg.selectAll('.transformed-data circle, .transformed-boundary line').remove();
    
    this.kernelTimer = setInterval(() => {
      step++;
      
      if (step >= totalSteps) {
        clearInterval(this.kernelTimer);
        this.isPlaying = false;
        return;
      }
      
      // Interpolation factor
      const t = step / totalSteps;
      
      // Get the width of each half
      const originalWidth = this.kernelWidth / 2 - 20;
      const transformedWidth = this.kernelWidth / 2;
      
      // Create scales for transformed space
      const transformedXScale = d3.scaleLinear()
        .domain([0, 10])
        .range([this.kernelWidth / 2 + 20, this.kernelWidth - 40]);
        
      const transformedYScale = d3.scaleLinear()
        .domain([0, 10])
        .range([this.kernelHeight - 50, 50]);
      
      // Apply gradually increasing transformation
      const transformedData = this.nonLinearData.map((d: [number, number, number]) => {
        const [x, y, cls] = d;
        
        let newX, newY;
        
        if (this.kernelType === 'linear') {
          // Linear kernel - no transformation
          newX = x;
          newY = y;
        } else if (this.kernelType === 'rbf') {
          // RBF kernel - transform into a different arrangement
          const distance = Math.sqrt((x - 5) ** 2 + (y - 5) ** 2);
          const angle = Math.atan2(y - 5, x - 5);
          
          // Apply gamma parameter and interpolation factor
          const transformedDist = 5 * (1 - Math.exp(-this.gammaValue * distance ** 2)) * t;
          
          newX = 5 + transformedDist * Math.cos(angle);
          newY = 5 + (cls === 1 ? 2 : -2) * t + transformedDist * Math.sin(angle) * 0.2;
        } else if (this.kernelType === 'poly') {
          // Polynomial kernel - interpolate the transformation
          newX = ((x - 5) ** 2) * 0.05 * t + 5;
          newY = ((y - 5) ** 2) * 0.05 * t + 5 + (cls === 1 ? 2 : -2) * t;
        } else {
          newX = x;
          newY = y;
        }
        
        // Interpolate between original and transformed
        const interpX = (1 - t) * x + t * newX;
        const interpY = (1 - t) * y + t * newY;
        
        return [interpX, interpY, cls];
      });
      
      // Clear and redraw transformed data
      this.kernelSvg.selectAll('.transformed-data circle').remove();
      
      this.kernelSvg.select('.transformed-data')
        .selectAll('circle')
        .data(transformedData)
        .enter()
        .append('circle')
        .attr('cx', (d: [number, number, number]) => transformedXScale(d[0]))
        .attr('cy', (d: [number, number, number]) => transformedYScale(d[1]))
        .attr('r', 6)
        .attr('fill', (d: [number, number, number]) => d[2] === 1 ? '#4285F4' : '#EA4335')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5);
        
      // Draw linear boundary in transformed space
      this.kernelSvg.selectAll('.transformed-boundary line').remove();
      
      // Draw a line that becomes more defined as the animation progresses
      let x1, x2, y1, y2;
      
      if (this.kernelType === 'linear') {
        x1 = 0;
        x2 = 10;
        y1 = 5;
        y2 = 5;
      } else if (this.kernelType === 'rbf') {
        x1 = 0;
        x2 = 10;
        y1 = 5;
        y2 = 5;
      } else if (this.kernelType === 'poly') {
        x1 = 0;
        x2 = 10;
        y1 = 5;
        y2 = 5;
      } else {
        x1 = 0;
        x2 = 10;
        y1 = 5;
        y2 = 5;
      }
      
      this.kernelSvg.select('.transformed-boundary')
        .append('line')
        .attr('x1', transformedXScale(x1))
        .attr('y1', transformedYScale(y1))
        .attr('x2', transformedXScale(x2))
        .attr('y2', transformedYScale(y2))
        .attr('stroke', '#34A853')
        .attr('stroke-width', 2 * t); // Line gets thicker
      
      // Update step display
      this.currentStep = step;
    }, this.animationSpeed / totalSteps);
  }

  animateCParameterEffect(): void {
    // This animation will transition between different C values
    const cValues = [0.1, 1, 10, 100];
    let step = 0;
    const stepsPerC = 10;
    const totalSteps = cValues.length * stepsPerC;
    
    this.cParamTimer = setInterval(() => {
      step++;
      
      if (step >= totalSteps) {
        clearInterval(this.cParamTimer);
        this.isPlaying = false;
        return;
      }
      
      // Determine which C value to highlight
      const cIndex = Math.floor(step / stepsPerC);
      const cValue = cValues[cIndex];
      
      // Highlight the current C value panel
      this.cParamSvg.selectAll('.highlight-border').remove();
      
      const row = Math.floor(cIndex / 2);
      const col = cIndex % 2;
      const x = col * (this.cParamWidth / 2);
      const y = row * (this.cParamHeight / 2);
      
      this.cParamSvg.append('rect')
        .attr('class', 'highlight-border')
        .attr('x', x + 5)
        .attr('y', y + 5)
        .attr('width', this.cParamWidth / 2 - 10)
        .attr('height', this.cParamHeight / 2 - 10)
        .attr('fill', 'none')
        .attr('stroke', '#FBBC05')
        .attr('stroke-width', 3)
        .attr('rx', 5)
        .attr('ry', 5);
      
      // Update step display
      this.currentStep = step;
    }, this.animationSpeed / totalSteps);
  }
}