import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import * as d3 from 'd3';

@Component({
  selector: 'app-cnn-visualization',
  templateUrl: './cnn-visualization.component.html',
  styleUrls: ['./cnn-visualization.component.scss']
})
export class CnnVisualizationComponent implements OnInit {
  @ViewChild('visualizationContainer', { static: true }) container!: ElementRef;
  @ViewChild('imageCanvas', { static: true }) imageCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('featureMapCanvas', { static: true }) featureMapCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('outputCanvas', { static: true }) outputCanvas!: ElementRef<HTMLCanvasElement>;

  // Animation state
  private animationState: 'playing' | 'paused' = 'paused';
  private animationStep = 0;
  private totalSteps = 100;
  private animationInterval: any;
  private animationSpeed = 50; // ms between steps

  // Current phase of the visualization
  public currentPhase: 'convolution' | 'activation' | 'pooling' | 'hierarchical' | 'full_network' = 'convolution';
  
  // CNN parameters
  private inputImageSize = 28;
  public filterSize = 5;
  public stride = 1;
  public padding = 0;
  
  // Calculated output size based on input, filter, stride, and padding
  private outputSize = Math.floor((this.inputImageSize + 2 * this.padding - this.filterSize) / this.stride) + 1;
  
  // UI controls
  public isPlaying = false;
  public currentOperationIndex = 0;
  public progress = 0;
  public showAdvancedControls = false;
  public sliderValue = 150; // For the speed slider
  public storyModeActive = false;
  private storyModeInterval: any;
  
  // Computed property for speed label
  public get speedLabel(): string {
    const speed = 200 - this.sliderValue;
    if (speed < 50) {
      return 'Fast';
    } else if (speed > 150) {
      return 'Slow';
    } else {
      return 'Normal';
    }
  }
  
  // Operation descriptions for the UI
  public operations = [
    { 
      name: 'Convolution', 
      description: 'Slides a filter across the input image, computing dot products to produce feature maps. Detects patterns like edges, textures, and shapes.',
      phase: 'convolution'
    },
    { 
      name: 'ReLU Activation', 
      description: 'Applies non-linearity by converting all negative values to zero. Allows the network to learn complex, non-linear patterns.',
      phase: 'activation'
    },
    { 
      name: 'Max Pooling', 
      description: 'Downsamples feature maps by selecting maximum values in local regions. Reduces spatial dimensions and provides translation invariance.',
      phase: 'pooling'
    },
    { 
      name: 'Hierarchical Features', 
      description: 'Shows how deeper layers build upon earlier features. Early layers detect edges, middle layers detect shapes, and deeper layers detect complex objects.',
      phase: 'hierarchical'
    },
    { 
      name: 'Full Network Flow', 
      description: 'Visualizes how data flows through the entire CNN: from input image through convolutions, activations, and pooling, to fully connected layers and final classification.',
      phase: 'full_network'
    }
  ];

  // Sample filters
  private filters = [
    // Horizontal edge detector
    [
      [-1, -1, -1],
      [2, 2, 2],
      [-1, -1, -1]
    ],
    // Vertical edge detector
    [
      [-1, 2, -1],
      [-1, 2, -1],
      [-1, 2, -1]
    ],
    // Sobel filter (edge detection)
    [
      [1, 0, -1],
      [2, 0, -2],
      [1, 0, -1]
    ]
  ];

  // Feature maps for different layers
  private featureMaps = {
    layer1: [] as number[][],
    layer2: [] as number[][],
    layer3: [] as number[][]
  };

  // Example input image - simplified MNIST-like digit
  private inputImage: number[][] = [];

  constructor() {}

  ngOnInit(): void {
    this.initializeInputImage();
    this.drawInputImage();
    this.generateFeatureMaps();
  }

  // Initialize a sample input image (simplified digit)
  private initializeInputImage(): void {
    // Create empty 28x28 image filled with zeros
    this.inputImage = Array(this.inputImageSize).fill(0).map(() => Array(this.inputImageSize).fill(0));
    
    // Draw a simplified digit "5"
    // Horizontal lines
    for (let i = 5; i < 20; i++) {
      this.inputImage[5][i] = 1; // Top horizontal
      this.inputImage[12][i] = 1; // Middle horizontal
      this.inputImage[24][i] = 1; // Bottom horizontal
    }
    
    // Vertical lines
    for (let i = 5; i < 12; i++) {
      this.inputImage[i][5] = 1; // Top left vertical
    }
    
    for (let i = 12; i < 24; i++) {
      this.inputImage[i][20] = 1; // Bottom right vertical
    }
  }

  // Generate feature maps by applying convolution with sample filters
  private generateFeatureMaps(): void {
    // Apply first filter to input image
    this.featureMaps.layer1 = this.applyConvolution(this.inputImage, this.filters[0]);
    
    // Apply ReLU to layer1 feature map
    const reluMap = this.applyReLU(this.featureMaps.layer1);
    
    // Apply max pooling to ReLU output
    this.featureMaps.layer2 = this.applyMaxPooling(reluMap, 2, 2);
    
    // Apply second filter to layer2
    this.featureMaps.layer3 = this.applyConvolution(this.featureMaps.layer2, this.filters[1]);
  }

  // Apply convolution with a filter to an input matrix
  private applyConvolution(input: number[][], filter: number[][]): number[][] {
    const inputHeight = input.length;
    const inputWidth = input[0].length;
    const filterHeight = filter.length;
    const filterWidth = filter[0].length;
    
    const outputHeight = Math.floor((inputHeight + 2 * this.padding - filterHeight) / this.stride) + 1;
    const outputWidth = Math.floor((inputWidth + 2 * this.padding - filterWidth) / this.stride) + 1;
    
    const output: number[][] = Array(outputHeight).fill(0).map(() => Array(outputWidth).fill(0));
    
    // Perform convolution
    for (let y = 0; y < outputHeight; y++) {
      for (let x = 0; x < outputWidth; x++) {
        let sum = 0;
        
        for (let fy = 0; fy < filterHeight; fy++) {
          for (let fx = 0; fx < filterWidth; fx++) {
            const inputY = y * this.stride + fy - this.padding;
            const inputX = x * this.stride + fx - this.padding;
            
            // Check if within bounds of input
            if (inputY >= 0 && inputY < inputHeight && inputX >= 0 && inputX < inputWidth) {
              sum += input[inputY][inputX] * filter[fy][fx];
            }
          }
        }
        
        output[y][x] = sum;
      }
    }
    
    return output;
  }

  // Apply ReLU activation function (max(0, x))
  private applyReLU(input: number[][]): number[][] {
    const height = input.length;
    const width = input[0].length;
    const output: number[][] = Array(height).fill(0).map(() => Array(width).fill(0));
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        output[y][x] = Math.max(0, input[y][x]);
      }
    }
    
    return output;
  }

  // Apply max pooling with given pool size and stride
  private applyMaxPooling(input: number[][], poolSize: number, poolStride: number): number[][] {
    const inputHeight = input.length;
    const inputWidth = input[0].length;
    
    const outputHeight = Math.floor((inputHeight - poolSize) / poolStride) + 1;
    const outputWidth = Math.floor((inputWidth - poolSize) / poolStride) + 1;
    
    const output: number[][] = Array(outputHeight).fill(0).map(() => Array(outputWidth).fill(0));
    
    for (let y = 0; y < outputHeight; y++) {
      for (let x = 0; x < outputWidth; x++) {
        let maxVal = -Infinity;
        
        for (let py = 0; py < poolSize; py++) {
          for (let px = 0; px < poolSize; px++) {
            const inputY = y * poolStride + py;
            const inputX = x * poolStride + px;
            
            if (inputY < inputHeight && inputX < inputWidth) {
              maxVal = Math.max(maxVal, input[inputY][inputX]);
            }
          }
        }
        
        output[y][x] = maxVal;
      }
    }
    
    return output;
  }

  // Draw input image on canvas
  private drawInputImage(): void {
    const canvas = this.imageCanvas.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const cellSize = canvas.width / this.inputImageSize;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    for (let y = 0; y < this.inputImageSize; y++) {
      for (let x = 0; x < this.inputImageSize; x++) {
        const value = this.inputImage[y][x];
        
        // Map value from [0,1] to grayscale
        const intensity = Math.floor(value * 255);
        ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
        
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
      }
    }
  }

  // Draw feature map on canvas
  private drawFeatureMap(featureMap: number[][], canvas: HTMLCanvasElement): void {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const height = featureMap.length;
    const width = featureMap[0].length;
    const cellSize = Math.min(canvas.width / width, canvas.height / height);
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Find min and max values for normalization
    let minVal = Infinity;
    let maxVal = -Infinity;
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        minVal = Math.min(minVal, featureMap[y][x]);
        maxVal = Math.max(maxVal, featureMap[y][x]);
      }
    }
    
    // Normalize and draw
    const range = maxVal - minVal;
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const normalizedValue = range === 0 ? 0.5 : (featureMap[y][x] - minVal) / range;
        
        // Use blue color scheme for feature maps
        const blue = Math.floor(normalizedValue * 255);
        ctx.fillStyle = `rgb(${50}, ${100 + blue/2}, ${blue})`;
        
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        
        // Add grid lines
        ctx.strokeStyle = 'rgba(150, 150, 150, 0.3)';
        ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);
      }
    }
  }

  // Visualize the convolution operation at the current step
  private visualizeConvolution(): void {
    const canvas = this.outputCanvas.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const cellSize = canvas.width / this.inputImageSize;
    
    // Draw input image faded
    for (let y = 0; y < this.inputImageSize; y++) {
      for (let x = 0; x < this.inputImageSize; x++) {
        const value = this.inputImage[y][x];
        const intensity = Math.floor(value * 200); // Slightly faded
        ctx.fillStyle = `rgba(${intensity}, ${intensity}, ${intensity}, 0.5)`;
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
      }
    }
    
    // Calculate current filter position based on animation step
    const maxPositionX = this.inputImageSize - this.filterSize;
    const maxPositionY = this.inputImageSize - this.filterSize;
    const totalPositions = (maxPositionX + 1) * (maxPositionY + 1);
    
    const stepProgress = (this.animationStep % this.totalSteps) / this.totalSteps;
    const currentPosition = Math.floor(stepProgress * totalPositions);
    
    const filterY = Math.floor(currentPosition / (maxPositionX + 1));
    const filterX = currentPosition % (maxPositionX + 1);
    
    // Draw filter overlay
    const filter = this.filters[0]; // Use first filter
    for (let y = 0; y < this.filterSize; y++) {
      for (let x = 0; x < this.filterSize; x++) {
        const filterValue = filter[y % filter.length][x % filter[0].length];
        
        let color;
        if (filterValue > 0) {
          color = `rgba(66, 133, 244, 0.6)`; // Blue for positive values
        } else if (filterValue < 0) {
          color = `rgba(234, 67, 53, 0.6)`; // Red for negative values
        } else {
          color = `rgba(255, 255, 255, 0.3)`; // White for zeros
        }
        
        ctx.fillStyle = color;
        ctx.fillRect((filterX + x) * cellSize, (filterY + y) * cellSize, cellSize, cellSize);
        
        // Add filter value text
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = `${cellSize * 0.6}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          filterValue.toString(),
          (filterX + x) * cellSize + cellSize / 2,
          (filterY + y) * cellSize + cellSize / 2
        );
      }
    }
    
    // Draw filter border
    ctx.strokeStyle = '#7c4dff'; // Purple from the design guidelines
    ctx.lineWidth = 2;
    ctx.strokeRect(
      filterX * cellSize,
      filterY * cellSize,
      this.filterSize * cellSize,
      this.filterSize * cellSize
    );
    
    // Calculate and display the output value for current position
    let outputValue = 0;
    for (let y = 0; y < this.filterSize; y++) {
      for (let x = 0; x < this.filterSize; x++) {
        const inputY = filterY + y;
        const inputX = filterX + x;
        
        if (inputY < this.inputImageSize && inputX < this.inputImageSize) {
          outputValue += this.inputImage[inputY][inputX] * filter[y % filter.length][x % filter[0].length];
        }
      }
    }
    
    // Display the computed output value
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(10, canvas.height - 40, 120, 30);
    ctx.fillStyle = 'white';
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Output: ${outputValue.toFixed(2)}`, 20, canvas.height - 20);
    
    // Draw the feature map being generated
    this.drawFeatureMap(this.featureMaps.layer1, this.featureMapCanvas.nativeElement);
  }

  // Visualize the ReLU activation at the current step
  private visualizeReLU(): void {
    const canvas = this.outputCanvas.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Get feature map size
    const featureMap = this.featureMaps.layer1;
    const height = featureMap.length;
    const width = featureMap[0].length;
    const cellSize = Math.min(canvas.width / width, canvas.height / height);
    
    // Calculate current position based on animation progress
    const totalCells = width * height;
    const stepProgress = (this.animationStep % this.totalSteps) / this.totalSteps;
    const currentCellIndex = Math.floor(stepProgress * totalCells);
    const currentY = Math.floor(currentCellIndex / width);
    const currentX = currentCellIndex % width;
    
    // Draw original feature map
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const value = featureMap[y][x];
        
        // Normalize for coloring
        const normalizedValue = (value + 5) / 10; // Assuming values roughly in [-5, 5] range
        const clampedValue = Math.max(0, Math.min(1, normalizedValue));
        
        // Use gradient from blue (negative) to white (zero) to red (positive)
        let color;
        if (value < 0) {
          // Blue for negative
          const intensity = Math.floor((1 - clampedValue) * 255);
          color = `rgb(${intensity}, ${intensity}, 255)`;
        } else {
          // Red for positive
          const intensity = Math.floor(clampedValue * 255);
          color = `rgb(255, ${255 - intensity}, ${255 - intensity})`;
        }
        
        ctx.fillStyle = color;
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        
        // Add grid
        ctx.strokeStyle = 'rgba(50, 50, 50, 0.2)';
        ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);
        
        // Add value text for larger cells
        if (cellSize > 20) {
          ctx.fillStyle = 'black';
          ctx.font = `${cellSize * 0.4}px Arial`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(
            value.toFixed(1),
            x * cellSize + cellSize / 2,
            y * cellSize + cellSize / 2
          );
        }
      }
    }
    
    // Draw ReLU overlay for cells that have been processed
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (y < currentY || (y === currentY && x <= currentX)) {
          const value = featureMap[y][x];
          
          // ReLU: max(0, x)
          const reluValue = Math.max(0, value);
          
          if (value < 0) {
            // Highlight cells that get zeroed out by ReLU
            ctx.fillStyle = 'rgba(80, 80, 80, 0.7)';
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            
            ctx.fillStyle = 'white';
            ctx.font = `${cellSize * 0.4}px Arial`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(
              '0',
              x * cellSize + cellSize / 2,
              y * cellSize + cellSize / 2
            );
          }
        }
      }
    }
    
    // Highlight current cell
    ctx.strokeStyle = '#7c4dff'; // Purple from design guide
    ctx.lineWidth = 3;
    ctx.strokeRect(
      currentX * cellSize,
      currentY * cellSize,
      cellSize,
      cellSize
    );
    
    // Display explanation text
    const currentValue = featureMap[currentY][currentX];
    const reluValue = Math.max(0, currentValue);
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(10, canvas.height - 60, 250, 50);
    ctx.fillStyle = 'white';
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`ReLU: max(0, ${currentValue.toFixed(2)}) = ${reluValue.toFixed(2)}`, 20, canvas.height - 35);
    ctx.fillText(`Zeros out negative values, keeping positives`, 20, canvas.height - 15);
    
    // Draw the feature map after ReLU
    this.drawFeatureMap(this.applyReLU(this.featureMaps.layer1), this.featureMapCanvas.nativeElement);
  }

  // Visualize the max pooling operation at the current step
  private visualizePooling(): void {
    const canvas = this.outputCanvas.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Get feature map (after ReLU)
    const featureMap = this.applyReLU(this.featureMaps.layer1);
    const height = featureMap.length;
    const width = featureMap[0].length;
    
    // Pooling parameters
    const poolSize = 2;
    const poolStride = 2;
    const outputHeight = Math.floor((height - poolSize) / poolStride) + 1;
    const outputWidth = Math.floor((width - poolSize) / poolStride) + 1;
    
    // Display sizes
    const inputCellSize = Math.min(canvas.width / width * 0.8, canvas.height / height * 0.8);
    const inputOffsetX = canvas.width * 0.05;
    const inputOffsetY = canvas.height * 0.05;
    
    const outputCellSize = inputCellSize * 1.5;
    const outputOffsetX = inputOffsetX + width * inputCellSize + 50;
    const outputOffsetY = inputOffsetY;
    
    // Calculate current pooling window based on animation progress
    const totalPoolingWindows = outputHeight * outputWidth;
    const stepProgress = (this.animationStep % this.totalSteps) / this.totalSteps;
    const currentWindowIndex = Math.floor(stepProgress * totalPoolingWindows);
    const currentPoolY = Math.floor(currentWindowIndex / outputWidth);
    const currentPoolX = currentWindowIndex % outputWidth;
    
    // Draw input feature map
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const value = featureMap[y][x];
        
        // Normalize for display
        const normalizedValue = Math.max(0, Math.min(1, value / 5));
        const blue = Math.floor(normalizedValue * 255);
        
        ctx.fillStyle = `rgb(${50}, ${100 + blue/2}, ${blue})`;
        ctx.fillRect(
          inputOffsetX + x * inputCellSize,
          inputOffsetY + y * inputCellSize,
          inputCellSize,
          inputCellSize
        );
        
        // Add grid
        ctx.strokeStyle = 'rgba(50, 50, 50, 0.2)';
        ctx.strokeRect(
          inputOffsetX + x * inputCellSize,
          inputOffsetY + y * inputCellSize,
          inputCellSize,
          inputCellSize
        );
        
        // Add value text for larger cells
        if (inputCellSize > 15) {
          ctx.fillStyle = 'white';
          ctx.font = `${inputCellSize * 0.5}px Arial`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(
            value.toFixed(1),
            inputOffsetX + x * inputCellSize + inputCellSize / 2,
            inputOffsetY + y * inputCellSize + inputCellSize / 2
          );
        }
      }
    }
    
    // Draw current pooling window
    const currentInputY = currentPoolY * poolStride;
    const currentInputX = currentPoolX * poolStride;
    
    ctx.strokeStyle = '#ff9d45'; // Orange from design guide
    ctx.lineWidth = 3;
    ctx.strokeRect(
      inputOffsetX + currentInputX * inputCellSize,
      inputOffsetY + currentInputY * inputCellSize,
      poolSize * inputCellSize,
      poolSize * inputCellSize
    );
    
    // Find max value in current pooling window
    let maxVal = -Infinity;
    let maxX = 0;
    let maxY = 0;
    
    for (let py = 0; py < poolSize; py++) {
      for (let px = 0; px < poolSize; px++) {
        const inputY = currentPoolY * poolStride + py;
        const inputX = currentPoolX * poolStride + px;
        
        if (inputY < height && inputX < width) {
          const val = featureMap[inputY][inputX];
          if (val > maxVal) {
            maxVal = val;
            maxY = py;
            maxX = px;
          }
        }
      }
    }
    
    // Highlight max value in pooling window
    ctx.fillStyle = 'rgba(255, 255, 0, 0.3)';
    ctx.fillRect(
      inputOffsetX + (currentInputX + maxX) * inputCellSize,
      inputOffsetY + (currentInputY + maxY) * inputCellSize,
      inputCellSize,
      inputCellSize
    );
    
    // Draw output feature map (after pooling)
    const pooledMap = this.applyMaxPooling(featureMap, poolSize, poolStride);
    
    for (let y = 0; y < outputHeight; y++) {
      for (let x = 0; x < outputWidth; x++) {
        // Only show pooled values that have been computed so far
        if (y < currentPoolY || (y === currentPoolY && x <= currentPoolX)) {
          const value = pooledMap[y][x];
          
          // Normalize for display
          const normalizedValue = Math.max(0, Math.min(1, value / 5));
          const blue = Math.floor(normalizedValue * 255);
          
          ctx.fillStyle = `rgb(${50}, ${100 + blue/2}, ${blue})`;
          ctx.fillRect(
            outputOffsetX + x * outputCellSize,
            outputOffsetY + y * outputCellSize,
            outputCellSize,
            outputCellSize
          );
          
          // Add grid
          ctx.strokeStyle = 'rgba(50, 50, 50, 0.2)';
          ctx.strokeRect(
            outputOffsetX + x * outputCellSize,
            outputOffsetY + y * outputCellSize,
            outputCellSize,
            outputCellSize
          );
          
          // Add value text
          ctx.fillStyle = 'white';
          ctx.font = `${outputCellSize * 0.5}px Arial`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(
            value.toFixed(1),
            outputOffsetX + x * outputCellSize + outputCellSize / 2,
            outputOffsetY + y * outputCellSize + outputCellSize / 2
          );
        }
      }
    }
    
    // Highlight current output cell
    if (currentPoolY < outputHeight && currentPoolX < outputWidth) {
      ctx.strokeStyle = '#7c4dff'; // Purple from design guide
      ctx.lineWidth = 3;
      ctx.strokeRect(
        outputOffsetX + currentPoolX * outputCellSize,
        outputOffsetY + currentPoolY * outputCellSize,
        outputCellSize,
        outputCellSize
      );
    }
    
    // Draw explanation
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(10, canvas.height - 80, 300, 70);
    ctx.fillStyle = 'white';
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Max Pooling (2×2, stride 2)', 20, canvas.height - 60);
    ctx.fillText(`Takes maximum value from each ${poolSize}×${poolSize} window`, 20, canvas.height - 40);
    ctx.fillText(`Current max: ${maxVal.toFixed(2)} at position (${maxX},${maxY})`, 20, canvas.height - 20);
    
    // Draw the pooled feature map
    this.drawFeatureMap(pooledMap, this.featureMapCanvas.nativeElement);
  }

  // Visualize hierarchical feature learning
  private visualizeHierarchical(): void {
    const canvas = this.outputCanvas.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate dimensions
    const layerWidth = canvas.width * 0.27;
    const layerHeight = canvas.height * 0.7;
    const margin = canvas.width * 0.05;
    
    // Draw three layers of feature maps (early, middle, late)
    const layers = [
      { 
        title: 'Layer 1: Edges & Textures', 
        featureMap: this.featureMaps.layer1,
        description: 'Detects simple features like edges, corners, and textures'
      },
      { 
        title: 'Layer 2: Shapes & Patterns', 
        featureMap: this.featureMaps.layer2,
        description: 'Combines edges to detect shapes, curves, and simple patterns'
      },
      { 
        title: 'Layer 3: Complex Features', 
        featureMap: this.featureMaps.layer3,
        description: 'Combines shapes to detect complex features and object parts'
      }
    ];
    
    // Animation step determines which layer to highlight
    const stepProgress = (this.animationStep % this.totalSteps) / this.totalSteps;
    const highlightLayer = Math.floor(stepProgress * 3);
    
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const x = margin + i * (layerWidth + margin);
      const y = (canvas.height - layerHeight) / 2;
      
      // Draw title
      ctx.fillStyle = i === highlightLayer ? '#4285f4' : '#8a9ab0';
      ctx.font = 'bold 16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(layer.title, x + layerWidth / 2, y - 20);
      
      // Draw feature map
      const featureMap = layer.featureMap;
      const height = featureMap.length;
      const width = featureMap[0].length;
      const cellSize = Math.min(layerWidth / width, layerHeight / height);
      
      for (let fy = 0; fy < height; fy++) {
        for (let fx = 0; fx < width; fx++) {
          const value = featureMap[fy][fx];
          
          // Normalize for display
          const normalizedValue = Math.max(0, Math.min(1, (value + 5) / 10));
          
          // Use deeper blue for deeper layers
          const blue = Math.floor(normalizedValue * 255);
          const layerIntensity = 40 + i * 30;
          ctx.fillStyle = `rgb(${layerIntensity}, ${layerIntensity + blue/3}, ${blue})`;
          
          ctx.fillRect(
            x + fx * cellSize,
            y + fy * cellSize,
            cellSize,
            cellSize
          );
        }
      }
      
      // Draw border
      ctx.strokeStyle = i === highlightLayer ? '#7c4dff' : '#2a4980';
      ctx.lineWidth = i === highlightLayer ? 3 : 1;
      ctx.strokeRect(x, y, width * cellSize, height * cellSize);
      
      // Draw description
      ctx.fillStyle = i === highlightLayer ? '#e1e7f5' : '#8a9ab0';
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(
        layer.description,
        x + layerWidth / 2,
        y + height * cellSize + 25
      );
    }
    
    // Draw connections between layers
    ctx.strokeStyle = 'rgba(124, 77, 255, 0.3)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i < layers.length - 1; i++) {
      const fromLayer = layers[i];
      const toLayer = layers[i + 1];
      
      const fromHeight = fromLayer.featureMap.length;
      const fromWidth = fromLayer.featureMap[0].length;
      const fromCellSize = Math.min(layerWidth / fromWidth, layerHeight / fromHeight);
      
      const toHeight = toLayer.featureMap.length;
      const toWidth = toLayer.featureMap[0].length;
      const toCellSize = Math.min(layerWidth / toWidth, layerHeight / toHeight);
      
      const fromX = margin + i * (layerWidth + margin) + fromWidth * fromCellSize;
      const fromY = (canvas.height - layerHeight) / 2 + fromHeight * fromCellSize / 2;
      
      const toX = margin + (i + 1) * (layerWidth + margin);
      const toY = (canvas.height - layerHeight) / 2 + toHeight * toCellSize / 2;
      
      ctx.beginPath();
      ctx.moveTo(fromX, fromY);
      ctx.lineTo(toX, toY);
      ctx.stroke();
    }
    
    // Draw explanation text for the animation
    let explanation = 'CNNs learn a hierarchy of features';
    switch (highlightLayer) {
      case 0:
        explanation = 'Early layers detect basic features like edges and textures';
        break;
      case 1:
        explanation = 'Middle layers combine edges to form shapes and patterns';
        break;
      case 2:
        explanation = 'Deeper layers detect complex features and object parts';
        break;
    }
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(
      (canvas.width - 400) / 2,
      canvas.height - 50,
      400,
      40
    );
    ctx.fillStyle = 'white';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(
      explanation,
      canvas.width / 2,
      canvas.height - 25
    );
  }

  // Visualize the full network flow
  private visualizeFullNetwork(): void {
    const canvas = this.outputCanvas.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate step progress
    const stepProgress = (this.animationStep % this.totalSteps) / this.totalSteps;
    
    // Define network architecture stages
    const stages = [
      { name: 'Input', width: 28, height: 28, depth: 1 },
      { name: 'Conv1', width: 24, height: 24, depth: 32 },
      { name: 'Pool1', width: 12, height: 12, depth: 32 },
      { name: 'Conv2', width: 8, height: 8, depth: 64 },
      { name: 'Pool2', width: 4, height: 4, depth: 64 },
      { name: 'Flatten', width: 1, height: 1, depth: 1024 },
      { name: 'FC1', width: 1, height: 1, depth: 128 },
      { name: 'Output', width: 1, height: 1, depth: 10 }
    ];
    
    // Calculate which stage is active
    const activeStageIndex = Math.min(
      stages.length - 1,
      Math.floor(stepProgress * stages.length)
    );
    
    // Layout parameters
    const marginX = 20;
    const totalWidth = canvas.width - 2 * marginX;
    const stageWidth = totalWidth / stages.length;
    const centerY = canvas.height / 2;
    
    // Draw stages
    for (let i = 0; i < stages.length; i++) {
      const stage = stages[i];
      const x = marginX + i * stageWidth + stageWidth / 2;
      
      // Calculate size of the block
      const maxBlockHeight = canvas.height * 0.6;
      const maxBlockWidth = stageWidth * 0.8;
      
      // Scale dimensions proportionally
      const maxDim = Math.max(stage.width, stage.height);
      const scale = Math.min(maxBlockHeight / maxDim, maxBlockWidth / maxDim);
      
      const blockWidth = i >= 5 ? maxBlockWidth / 3 : stage.width * scale;
      const blockHeight = i >= 5 ? 
                         (stage.depth / (i === 5 ? 1024 : (i === 6 ? 128 : 10))) * maxBlockHeight * 0.8 :
                         stage.height * scale;
      
      // Calculate depth representation (for 3D effect)
      const depthOffset = i >= 5 ? 0 : Math.min(20, stage.depth * 0.5);
      
      // Check if stage is active
      const isActive = i <= activeStageIndex;
      const isCurrentlyActive = i === activeStageIndex;
      
      // Draw stage block
      if (i < 5) {
        // Draw 3D representation for convolution and pooling layers
        // Back face (3D effect)
        ctx.fillStyle = isActive ? 
                       (isCurrentlyActive ? '#4285f4' : 'rgba(66, 133, 244, 0.5)') : 
                       'rgba(22, 42, 74, 0.7)';
        ctx.fillRect(
          x - blockWidth / 2 + depthOffset,
          centerY - blockHeight / 2 - depthOffset,
          blockWidth,
          blockHeight
        );
        
        // Top face (3D effect)
        ctx.fillStyle = isActive ? 
                       (isCurrentlyActive ? '#5c35cc' : 'rgba(92, 53, 204, 0.5)') : 
                       'rgba(30, 58, 102, 0.7)';
        ctx.beginPath();
        ctx.moveTo(x - blockWidth / 2, centerY - blockHeight / 2);
        ctx.lineTo(x - blockWidth / 2 + depthOffset, centerY - blockHeight / 2 - depthOffset);
        ctx.lineTo(x + blockWidth / 2 + depthOffset, centerY - blockHeight / 2 - depthOffset);
        ctx.lineTo(x + blockWidth / 2, centerY - blockHeight / 2);
        ctx.closePath();
        ctx.fill();
        
        // Right face (3D effect)
        ctx.fillStyle = isActive ? 
                       (isCurrentlyActive ? '#7c4dff' : 'rgba(124, 77, 255, 0.5)') : 
                       'rgba(42, 73, 128, 0.7)';
        ctx.beginPath();
        ctx.moveTo(x + blockWidth / 2, centerY - blockHeight / 2);
        ctx.lineTo(x + blockWidth / 2 + depthOffset, centerY - blockHeight / 2 - depthOffset);
        ctx.lineTo(x + blockWidth / 2 + depthOffset, centerY + blockHeight / 2 - depthOffset);
        ctx.lineTo(x + blockWidth / 2, centerY + blockHeight / 2);
        ctx.closePath();
        ctx.fill();
        
        // Front face
        ctx.fillStyle = isActive ? 
                       (isCurrentlyActive ? '#00c9ff' : 'rgba(0, 201, 255, 0.7)') : 
                       'rgba(26, 50, 96, 0.5)';
        ctx.fillRect(
          x - blockWidth / 2,
          centerY - blockHeight / 2,
          blockWidth,
          blockHeight
        );
      } else {
        // Draw flat representation for FC layers
        ctx.fillStyle = isActive ? 
                       (isCurrentlyActive ? '#00c9ff' : 'rgba(0, 201, 255, 0.7)') : 
                       'rgba(26, 50, 96, 0.5)';
        ctx.fillRect(
          x - blockWidth / 2,
          centerY - blockHeight / 2,
          blockWidth,
          blockHeight
        );
      }
      
      // Draw stage border
      ctx.strokeStyle = isCurrentlyActive ? '#ffffff' : '#2a4980';
      ctx.lineWidth = isCurrentlyActive ? 2 : 1;
      ctx.strokeRect(
        x - blockWidth / 2,
        centerY - blockHeight / 2,
        blockWidth,
        blockHeight
      );
      
      // Draw stage name
      ctx.fillStyle = isActive ? '#ffffff' : '#8a9ab0';
      ctx.font = isCurrentlyActive ? 'bold 14px Arial' : '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(
        stage.name,
        x,
        centerY + blockHeight / 2 + 20
      );
      
      // Draw dimensions
      if (i < 5) {
        ctx.font = '11px Arial';
        ctx.fillText(
          `${stage.width}×${stage.height}×${stage.depth}`,
          x,
          centerY + blockHeight / 2 + 40
        );
      } else {
        ctx.font = '11px Arial';
        ctx.fillText(
          `${stage.depth}`,
          x,
          centerY + blockHeight / 2 + 40
        );
      }
    }
    
    // Draw connections between stages
    for (let i = 0; i < stages.length - 1; i++) {
      const fromX = marginX + i * stageWidth + stageWidth;
      const toX = marginX + (i + 1) * stageWidth;
      
      // Check if this connection is active
      const isActive = i < activeStageIndex;
      
      ctx.strokeStyle = isActive ? 'rgba(0, 201, 255, 0.6)' : 'rgba(26, 50, 96, 0.3)';
      ctx.lineWidth = isActive ? 2 : 1;
      
      // Draw connection line
      ctx.beginPath();
      ctx.moveTo(fromX, centerY);
      ctx.lineTo(toX, centerY);
      ctx.stroke();
      
      // Draw arrow
      if (isActive) {
        ctx.fillStyle = 'rgba(0, 201, 255, 0.6)';
        ctx.beginPath();
        ctx.moveTo(toX, centerY);
        ctx.lineTo(toX - 10, centerY - 5);
        ctx.lineTo(toX - 10, centerY + 5);
        ctx.closePath();
        ctx.fill();
      }
    }
    
    // Draw current stage description
    const activeStage = stages[activeStageIndex];
    let description = '';
    
    switch (activeStage.name) {
      case 'Input':
        description = 'Input Layer: Takes the raw pixel values of the image';
        break;
      case 'Conv1':
        description = 'Convolutional Layer 1: Applies filters to detect low-level features';
        break;
      case 'Pool1':
        description = 'Pooling Layer 1: Reduces spatial dimensions and prevents overfitting';
        break;
      case 'Conv2':
        description = 'Convolutional Layer 2: Detects higher-level features';
        break;
      case 'Pool2':
        description = 'Pooling Layer 2: Further reduces dimensions for final processing';
        break;
      case 'Flatten':
        description = 'Flatten Layer: Converts 2D feature maps to 1D vector';
        break;
      case 'FC1':
        description = 'Fully Connected Layer: Combines features for classification';
        break;
      case 'Output':
        description = 'Output Layer: Produces final classification probabilities';
        break;
    }
    
    // Draw description box
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(
      (canvas.width - 500) / 2,
      canvas.height - 50,
      500,
      40
    );
    ctx.fillStyle = 'white';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(
      description,
      canvas.width / 2,
      canvas.height - 25
    );
    
    // Also update feature map visualization to match current layer
    if (activeStageIndex === 0) {
      this.drawInputImage();
    } else if (activeStageIndex === 1) {
      this.drawFeatureMap(this.featureMaps.layer1, this.featureMapCanvas.nativeElement);
    } else if (activeStageIndex === 2 || activeStageIndex === 3) {
      this.drawFeatureMap(this.featureMaps.layer2, this.featureMapCanvas.nativeElement);
    } else {
      this.drawFeatureMap(this.featureMaps.layer3, this.featureMapCanvas.nativeElement);
    }
  }

  // Main animation/visualization method
  private updateVisualization(): void {
    // Update progress
    this.progress = (this.animationStep % this.totalSteps) / this.totalSteps * 100;
    
    // Visualize based on current phase
    switch (this.currentPhase) {
      case 'convolution':
        this.visualizeConvolution();
        break;
      case 'activation':
        this.visualizeReLU();
        break;
      case 'pooling':
        this.visualizePooling();
        break;
      case 'hierarchical':
        this.visualizeHierarchical();
        break;
      case 'full_network':
        this.visualizeFullNetwork();
        break;
    }
  }

  // Animation control methods
  public play(): void {
    if (!this.isPlaying) {
      this.isPlaying = true;
      this.animationInterval = setInterval(() => {
        this.animationStep++;
        this.updateVisualization();
      }, this.animationSpeed);
    }
  }

  public pause(): void {
    if (this.isPlaying) {
      this.isPlaying = false;
      clearInterval(this.animationInterval);
    }
  }

  public reset(): void {
    this.animationStep = 0;
    this.updateVisualization();
  }

  public step(): void {
    this.animationStep++;
    this.updateVisualization();
  }

  public stepBackward(): void {
    if (this.animationStep > 0) {
      this.animationStep--;
      this.updateVisualization();
    }
  }

  // Set animation speed and update slider value
  public setSpeed(speed: number): void {
    this.animationSpeed = Math.max(10, Math.min(200, speed));
    this.sliderValue = 200 - this.animationSpeed;
    
    if (this.isPlaying) {
      this.pause();
      this.play();
    }
  }
  
  // Handle slider input and update animation speed
  public handleSpeedInput(event: Event): void {
    const inputElement = event.target as HTMLInputElement;
    if (inputElement) {
      const inputValue = Number(inputElement.value);
      this.sliderValue = inputValue;
      this.animationSpeed = 200 - inputValue;
      
      if (this.isPlaying) {
        this.pause();
        this.play();
      }
    }
  }
  
  // Get label for current speed
  public getSpeedLabel(): string {
    const speed = 200 - this.sliderValue;
    if (speed < 50) {
      return 'Fast';
    } else if (speed > 150) {
      return 'Slow';
    } else {
      return 'Normal';
    }
  }

  public selectOperation(index: number): void {
    this.currentOperationIndex = index;
    this.currentPhase = this.operations[index].phase as any;
    this.animationStep = 0;
    this.updateVisualization();
  }

  // Toggle advanced controls visibility
  public toggleAdvancedControls(): void {
    this.showAdvancedControls = !this.showAdvancedControls;
    console.log('Advanced controls toggled:', this.showAdvancedControls);
    
    // Add a slight delay to ensure DOM updates before we check
    setTimeout(() => {
      const advancedControlsElement = document.querySelector('.advanced-controls');
      console.log('Advanced controls element:', advancedControlsElement);
      console.log('Is visible:', this.showAdvancedControls);
      
      // Force repaint if needed
      if (advancedControlsElement) {
        (advancedControlsElement as HTMLElement).style.display = this.showAdvancedControls ? 'block' : 'none';
      }
    }, 100);
  }
  
  // Story mode - automatically walks through all operations
  public startStoryMode(): void {
    if (this.storyModeActive) return;
    
    this.storyModeActive = true;
    this.pause(); // Pause any current animation
    this.currentOperationIndex = 0;
    this.selectOperation(0);
    this.animationStep = 0;
    this.play();
    
    const storyStepDuration = 10000; // 10 seconds per operation
    
    this.storyModeInterval = setInterval(() => {
      this.currentOperationIndex = (this.currentOperationIndex + 1) % this.operations.length;
      this.selectOperation(this.currentOperationIndex);
      this.animationStep = 0;
      this.play();
      
      if (this.currentOperationIndex === 0) {
        this.stopStoryMode();
      }
    }, storyStepDuration);
  }
  
  public stopStoryMode(): void {
    if (!this.storyModeActive) return;
    
    clearInterval(this.storyModeInterval);
    this.storyModeActive = false;
  }
  
  ngOnDestroy(): void {
    if (this.animationInterval) {
      clearInterval(this.animationInterval);
    }
    if (this.storyModeInterval) {
      clearInterval(this.storyModeInterval);
    }
  }
}