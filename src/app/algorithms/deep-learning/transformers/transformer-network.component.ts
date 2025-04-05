import { Component, OnInit, ElementRef, ViewChild, AfterViewInit, HostListener, OnDestroy } from '@angular/core';
import * as THREE from 'three';
import * as d3 from 'd3';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { NgIf, NgFor, NgSwitch, NgSwitchCase, NgSwitchDefault, CommonModule } from '@angular/common';

interface ResearchPaper {
  title: string;
  authors: string;
  year: number;
  publication: string;
  link: string;
  description: string;
}

@Component({
  selector: 'app-transformer-network',
  templateUrl: './transformer-network.component.html',
  styleUrls: ['./transformer-network.component.scss'],
  imports: [NgIf, NgFor, NgSwitch, NgSwitchCase, NgSwitchDefault, CommonModule],
  standalone: true
})
export class TransformerNetworkComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('threeCanvas', { static: true }) private threeCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('attentionCanvas', { static: true }) private attentionCanvas!: ElementRef<HTMLDivElement>;
  @ViewChild('posEncodingCanvas', { static: true }) private posEncodingCanvas!: ElementRef<HTMLDivElement>;
  
  // State management
  currentStep = 0;
  isPlaying = false;
  playInterval: ReturnType<typeof setInterval> | null = null;
  
  // Configuration
  totalSteps = 8;
  autoPlaySpeed = 3500; // ms between steps
  
  // Three.js properties
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private animationFrameId?: number;
  
  // Animation objects
  private encoder: THREE.Group[] = [];
  private decoder: THREE.Group[] = [];
  private connectionLines: THREE.Line[] = [];
  
  // D3 visualizations
  private attentionSvg: any;
  private posEncodingSvg: any;
  
  // Sample data
  sampleSentence = "The transformer model processes input sequences efficiently.";
  tokens: string[] = [];
  
  // Attention weights simulation (sample data)
  attentionWeights: number[][] = [];
  
  // Sections visibility
  activeSection = 'overview';
  
  // Research papers
  researchPapers: ResearchPaper[] = [
    {
      title: "Attention Is All You Need",
      authors: "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.",
      year: 2017,
      publication: "Advances in Neural Information Processing Systems (NIPS)",
      link: "https://arxiv.org/abs/1706.03762",
      description: "The seminal paper that introduced the Transformer architecture."
    },
    {
      title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      authors: "Devlin, J., Chang, M. W., Lee, K., & Toutanova, K.",
      year: 2018,
      publication: "Proceedings of NAACL-HLT 2019",
      link: "https://arxiv.org/abs/1810.04805",
      description: "Introduces BERT, a transformer-based model that revolutionized NLP."
    },
    {
      title: "Language Models are Few-Shot Learners",
      authors: "Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ..., Amodei, D.",
      year: 2020,
      publication: "Advances in Neural Information Processing Systems",
      link: "https://arxiv.org/abs/2005.14165",
      description: "Describes GPT-3, showing the capabilities of large-scale transformer models."
    },
    {
      title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
      authors: "Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ..., Houlsby, N.",
      year: 2020,
      publication: "ICLR 2021",
      link: "https://arxiv.org/abs/2010.11929",
      description: "Introduces Vision Transformer (ViT), adapting the Transformer architecture for computer vision tasks."
    }
  ];
  
  constructor() {
    // Initialize tokens
    this.tokens = this.sampleSentence.split(/\s+/);
    
    // Generate random attention weights for demonstration
    this.initializeAttentionWeights();
  }
  
  ngOnInit(): void {
    // Initialize default state
  }
  
  ngAfterViewInit(): void {
    this.initThreeJs();
    this.initAttentionVisualization();
    this.initPositionalEncodingVisualization();
    
    // Start with first step visualization
    this.updateSimulation();
  }
  
  ngOnDestroy(): void {
    // Clean up
    if (this.isPlaying) {
      this.stopAutoPlay();
    }
    
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    
    // Dispose Three.js resources
    if (this.scene) {
      this.scene.children.forEach(child => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (child.material instanceof THREE.Material) {
            child.material.dispose();
          } else if (Array.isArray(child.material)) {
            child.material.forEach(m => m.dispose());
          }
        }
      });
    }
    
    if (this.renderer) {
      this.renderer.dispose();
    }
  }
  
  @HostListener('window:resize')
  onWindowResize(): void {
    if (this.camera && this.renderer && this.threeCanvas) {
      this.camera.aspect = this.threeCanvas.nativeElement.clientWidth / this.threeCanvas.nativeElement.clientHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(this.threeCanvas.nativeElement.clientWidth, this.threeCanvas.nativeElement.clientHeight);
    }
    
    // Resize D3 visualizations as needed
    this.updateAttentionVisualization();
    this.updatePositionalEncodingVisualization();
  }
  
  // Initialize random attention weights for demonstration
  private initializeAttentionWeights(): void {
    const tokenCount = this.tokens.length;
    this.attentionWeights = [];
    
    for (let i = 0; i < tokenCount; i++) {
      const weights: number[] = [];
      // Generate weights that sum to 1
      let sum = 0;
      for (let j = 0; j < tokenCount; j++) {
        // Create relationships with nearby words and syntactically related words
        let weight = Math.random();
        // Make self-attention stronger
        if (i === j) weight *= 2;
        // Make nearby tokens have stronger attention
        if (Math.abs(i - j) <= 2) weight *= 1.5;
        weights.push(weight);
        sum += weight;
      }
      
      // Normalize weights to sum to 1
      for (let j = 0; j < tokenCount; j++) {
        weights[j] = weights[j] / sum;
      }
      
      this.attentionWeights.push(weights);
    }
  }
  
  // Three.js initialization and rendering
  private initThreeJs(): void {
    if (!this.threeCanvas) return;
    
    // Set up scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0c1428); // Using darkest blue from the design system
    
    // Set up camera
    const width = this.threeCanvas.nativeElement.clientWidth;
    const height = this.threeCanvas.nativeElement.clientHeight;
    this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    this.camera.position.set(0, 8, 20);
    
    // Set up renderer
    this.renderer = new THREE.WebGLRenderer({ 
      canvas: this.threeCanvas.nativeElement,
      antialias: true 
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    
    // Set up controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 10;
    this.controls.maxDistance = 50;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
    directionalLight.position.set(1, 2, 1);
    this.scene.add(directionalLight);
    
    const backLight = new THREE.DirectionalLight(0xffffff, 0.5);
    backLight.position.set(-1, -1, -1);
    this.scene.add(backLight);
    
    // Create transformer architecture
    this.createTransformerArchitecture();
    
    // Animation loop
    this.animate();
  }
  
  private animate(): void {
    this.animationFrameId = requestAnimationFrame(() => this.animate());
    
    // Update controls
    if (this.controls) {
      this.controls.update();
    }
    
    // Render scene
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }
  
  private createTransformerArchitecture(): void {
    if (!this.scene) return;
    
    const layerCount = 6; // Typical transformer has 6 layers
    const layerSpacing = 1.5;
    const blockWidth = 3;
    const blockHeight = 0.6;
    const blockDepth = 0.5;
    
    // Create input/output embeddings
    const inputEmbedding = this.createBlock(blockWidth, blockHeight, blockDepth, 0x64b5f6, 0.9); // Light blue
    inputEmbedding.position.set(-6, -2, 0);
    this.scene.add(inputEmbedding);
    
    const outputEmbedding = this.createBlock(blockWidth, blockHeight, blockDepth, 0x64b5f6, 0.9);
    outputEmbedding.position.set(6, -2, 0);
    this.scene.add(outputEmbedding);
    
    // Create positional encoding blocks
    const inputPosEncoding = this.createBlock(blockWidth, blockHeight, blockDepth, 0xff9d45, 0.9); // Orange
    inputPosEncoding.position.set(-6, -1, 0);
    this.scene.add(inputPosEncoding);
    
    const outputPosEncoding = this.createBlock(blockWidth, blockHeight, blockDepth, 0xff9d45, 0.9);
    outputPosEncoding.position.set(6, -1, 0);
    this.scene.add(outputPosEncoding);
    
    // Create final output layer
    const linearLayer = this.createBlock(blockWidth, blockHeight, blockDepth, 0x24b47e, 0.9); // Green
    linearLayer.position.set(6, layerCount * layerSpacing + 1, 0);
    this.scene.add(linearLayer);
    
    const softmaxLayer = this.createBlock(blockWidth, blockHeight, blockDepth, 0x24b47e, 0.9);
    softmaxLayer.position.set(6, layerCount * layerSpacing + 2, 0);
    this.scene.add(softmaxLayer);
    
    // Add labels for components
    this.addTextLabel("Input Embeddings", -6, -2, 1);
    this.addTextLabel("Positional Encoding", -6, -1, 1);
    this.addTextLabel("Output Embeddings", 6, -2, 1);
    this.addTextLabel("Positional Encoding", 6, -1, 1);
    this.addTextLabel("Linear", 6, layerCount * layerSpacing + 1, 1);
    this.addTextLabel("Softmax", 6, layerCount * layerSpacing + 2, 1);
    
    // Create encoder stack
    for (let i = 0; i < layerCount; i++) {
      const encoderLayer = new THREE.Group();
      
      // Create self-attention block
      const selfAttention = this.createBlock(blockWidth, blockHeight, blockDepth, 0x4285f4, 0.9); // Primary blue
      selfAttention.position.y = 1;
      encoderLayer.add(selfAttention);
      
      // Create FFN block
      const ffn = this.createBlock(blockWidth, blockHeight, blockDepth, 0x7c4dff, 0.9); // Purple
      ffn.position.y = 0;
      encoderLayer.add(ffn);
      
      // Layer norm indicators
      const normIndicator1 = this.createNormIndicator();
      normIndicator1.position.set(1.8, 1, 0);
      encoderLayer.add(normIndicator1);
      
      const normIndicator2 = this.createNormIndicator();
      normIndicator2.position.set(1.8, 0, 0);
      encoderLayer.add(normIndicator2);
      
      // Position the encoder layer
      encoderLayer.position.set(-6, i * layerSpacing + 1, 0);
      this.scene.add(encoderLayer);
      this.encoder.push(encoderLayer);
      
      // Add text to show component names on the first layer
      if (i === 0) {
        this.addTextLabel("Multi-Head Self-Attention", -6, i * layerSpacing + 2, 1);
        this.addTextLabel("Feed Forward Network", -6, i * layerSpacing + 1, 1);
      }
    }
    
    // Create decoder stack
    for (let i = 0; i < layerCount; i++) {
      const decoderLayer = new THREE.Group();
      
      // Create masked self-attention block
      const maskedSelfAttention = this.createBlock(blockWidth, blockHeight, blockDepth, 0x00c9ff, 0.9); // Cyan
      maskedSelfAttention.position.y = 2;
      decoderLayer.add(maskedSelfAttention);
      
      // Create encoder-decoder attention block
      const encoderDecoderAttention = this.createBlock(blockWidth, blockHeight, blockDepth, 0xff9d45, 0.9); // Orange
      encoderDecoderAttention.position.y = 1;
      decoderLayer.add(encoderDecoderAttention);
      
      // Create FFN block
      const ffn = this.createBlock(blockWidth, blockHeight, blockDepth, 0x7c4dff, 0.9); // Purple
      ffn.position.y = 0;
      decoderLayer.add(ffn);
      
      // Layer norm indicators
      const normIndicator1 = this.createNormIndicator();
      normIndicator1.position.set(1.8, 2, 0);
      decoderLayer.add(normIndicator1);
      
      const normIndicator2 = this.createNormIndicator();
      normIndicator2.position.set(1.8, 1, 0);
      decoderLayer.add(normIndicator2);
      
      const normIndicator3 = this.createNormIndicator();
      normIndicator3.position.set(1.8, 0, 0);
      decoderLayer.add(normIndicator3);
      
      // Position the decoder layer
      decoderLayer.position.set(6, i * layerSpacing + 1, 0);
      this.scene.add(decoderLayer);
      this.decoder.push(decoderLayer);
      
      // Add text to show component names on the first layer
      if (i === 0) {
        this.addTextLabel("Masked Self-Attention", 6, i * layerSpacing + 3, 1);
        this.addTextLabel("Encoder-Decoder Attention", 6, i * layerSpacing + 2, 1);
        this.addTextLabel("Feed Forward Network", 6, i * layerSpacing + 1, 1);
      }
      
      // Add connection lines between encoder and decoder
      const lineMaterial = new THREE.LineBasicMaterial({ 
        color: 0x8bb4fa, // Light blue
        opacity: 0.4,
        transparent: true
      });
      
      const points = [];
      points.push(new THREE.Vector3(-4.5, i * layerSpacing + 1.5, 0));
      points.push(new THREE.Vector3(4.5, i * layerSpacing + 1.5, 0));
      
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geometry, lineMaterial);
      this.scene.add(line);
      this.connectionLines.push(line);
    }
    
    // Add vertical connections
    this.addVerticalConnections(-6, 0, layerCount, layerSpacing, 0x8bb4fa);
    this.addVerticalConnections(6, 0, layerCount, layerSpacing, 0x8bb4fa);
    
    // Connect from input embeddings to first encoder layer
    const inputPoints = [];
    inputPoints.push(new THREE.Vector3(-6, -1, 0));
    inputPoints.push(new THREE.Vector3(-6, 0, 0));
    
    const inputGeometry = new THREE.BufferGeometry().setFromPoints(inputPoints);
    const inputLine = new THREE.Line(inputGeometry, new THREE.LineBasicMaterial({ 
      color: 0x8bb4fa,
      opacity: 0.4,
      transparent: true
    }));
    this.scene.add(inputLine);
    
    // Connect from output embeddings to first decoder layer
    const outputPoints = [];
    outputPoints.push(new THREE.Vector3(6, -1, 0));
    outputPoints.push(new THREE.Vector3(6, 0, 0));
    
    const outputGeometry = new THREE.BufferGeometry().setFromPoints(outputPoints);
    const outputLine = new THREE.Line(outputGeometry, new THREE.LineBasicMaterial({ 
      color: 0x8bb4fa,
      opacity: 0.4,
      transparent: true
    }));
    this.scene.add(outputLine);
    
    // Connect from last decoder layer to linear layer
    const finalPoints = [];
    finalPoints.push(new THREE.Vector3(6, layerCount * layerSpacing, 0));
    finalPoints.push(new THREE.Vector3(6, layerCount * layerSpacing + 1, 0));
    
    const finalGeometry = new THREE.BufferGeometry().setFromPoints(finalPoints);
    const finalLine = new THREE.Line(finalGeometry, new THREE.LineBasicMaterial({ 
      color: 0x8bb4fa,
      opacity: 0.4,
      transparent: true
    }));
    this.scene.add(finalLine);
    
    // Add labels for encoder and decoder
    this.addTextLabel("Encoder", -9, layerCount * layerSpacing / 2, 0);
    this.addTextLabel("Decoder", 9, layerCount * layerSpacing / 2, 0);
    
    // Add Nx labels indicating multiple layers
    this.addTextLabel("Nx", -7.5, layerCount * layerSpacing / 2, 0);
    this.addTextLabel("Nx", 7.5, layerCount * layerSpacing / 2, 0);
  }
  
  private addTextLabel(text: string, x: number, y: number, z: number): void {
    // In a real implementation, you would use a library like THREE.TextGeometry
    // or HTML overlay elements positioned with CSS to show these labels.
    // For simplicity, we're just creating a small indicator sphere for now.
    if (!this.scene) return;
    
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.05, 8, 8),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    sphere.position.set(x, y, z);
    this.scene.add(sphere);
    
    // In a full implementation, you would add the text here
  }
  
  private addVerticalConnections(x: number, startY: number, layerCount: number, layerSpacing: number, color: number): void {
    if (!this.scene) return;
    
    const lineMaterial = new THREE.LineBasicMaterial({ 
      color: color,
      opacity: 0.4,
      transparent: true
    });
    
    for (let i = 0; i < layerCount - 1; i++) {
      const points = [];
      points.push(new THREE.Vector3(x, startY + i * layerSpacing + 1, 0));
      points.push(new THREE.Vector3(x, startY + (i + 1) * layerSpacing, 0));
      
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geometry, lineMaterial);
      this.scene.add(line);
    }
  }
  
  private createBlock(width: number, height: number, depth: number, color: number, opacity: number): THREE.Mesh {
    const geometry = new THREE.BoxGeometry(width, height, depth);
    const material = new THREE.MeshPhongMaterial({
      color: color,
      opacity: opacity,
      transparent: true,
      side: THREE.DoubleSide,
      specular: 0x111111,
      shininess: 30
    });
    
    const mesh = new THREE.Mesh(geometry, material);
    // Add a subtle edge outline
    const edges = new THREE.EdgesGeometry(geometry);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x000000, opacity: 0.2, transparent: true })
    );
    mesh.add(line);
    
    return mesh;
  }
  
  private createNormIndicator(): THREE.Mesh {
    const geometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
    const material = new THREE.MeshBasicMaterial({ color: 0x24b47e }); // Green for normalization
    return new THREE.Mesh(geometry, material);
  }
  
  // D3 Visualizations
  private initAttentionVisualization(): void {
    if (!this.attentionCanvas) return;
    
    // Create SVG
    this.attentionSvg = d3.select(this.attentionCanvas.nativeElement)
      .append("svg")
      .attr("width", "100%")
      .attr("height", "100%");
      
    this.updateAttentionVisualization();
  }
  
  private updateAttentionVisualization(): void {
    if (!this.attentionSvg || !this.attentionCanvas) return;
    
    const width = Math.max(this.attentionCanvas.nativeElement.clientWidth, 400);
    const height = Math.max(this.attentionCanvas.nativeElement.clientHeight, 400);
    const padding = 50;
    
    // Clear previous visualization
    this.attentionSvg.selectAll("*").remove();
    
    const tokenCount = this.tokens.length;
    const cellSize = Math.max(
      Math.min(
        (width - padding * 2) / tokenCount,
        (height - padding * 2) / tokenCount
      ),
      20 // Minimum cell size
    );
    
    // Update SVG size
    this.attentionSvg
      .attr("width", width)
      .attr("height", height);
    
    // Add title
    this.attentionSvg.append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .attr("fill", "#ffffff") // White from design system
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text("Self-Attention Weights");
    
    // Create scales
    const xScale = d3.scaleBand()
      .domain(this.tokens.map((_, i) => i.toString()))
      .range([padding, padding + cellSize * tokenCount])
      .padding(0.05);
    
    const yScale = d3.scaleBand()
      .domain(this.tokens.map((_, i) => i.toString()))
      .range([padding, padding + cellSize * tokenCount])
      .padding(0.05);
    
    // Color scale for attention weights
    const colorScale = d3.scaleSequential(d3.interpolateBlues)
      .domain([0, 1]);
    
    // Create heatmap cells
    const cellGroup = this.attentionSvg.append("g");
    
    for (let i = 0; i < tokenCount; i++) {
      for (let j = 0; j < tokenCount; j++) {
        const cellX = xScale(i.toString());
        const cellY = yScale(j.toString());
        const cellWidth = xScale.bandwidth();
        const cellHeight = yScale.bandwidth();
        
        if (cellX !== undefined && cellY !== undefined && 
            cellWidth > 0 && cellHeight > 0) {
          cellGroup.append("rect")
            .attr("x", cellX)
            .attr("y", cellY)
            .attr("width", cellWidth)
            .attr("height", cellHeight)
            .attr("fill", colorScale(this.attentionWeights[i][j]))
            .attr("stroke", "#162a4a")
            .attr("stroke-width", 1)
            .attr("rx", 2)
            .attr("ry", 2)
            .on("mouseover", (event: MouseEvent) => {
              // Add tooltip with weight value
              this.attentionSvg.append("text")
                .attr("class", "tooltip")
                .attr("x", event.offsetX + 10)
                .attr("y", event.offsetY - 10)
                .attr("fill", "#e1e7f5") // Light gray from design system
                .text(`${this.tokens[i]} → ${this.tokens[j]}: ${this.attentionWeights[i][j].toFixed(2)}`);
            })
            .on("mouseout", () => {
              // Remove tooltip
              this.attentionSvg.selectAll(".tooltip").remove();
            });
            
          // Add text inside cell for high attention values
          if (this.attentionWeights[i][j] > 0.3) {
            cellGroup.append("text")
              .attr("x", cellX + cellWidth / 2)
              .attr("y", cellY + cellHeight / 2)
              .attr("text-anchor", "middle")
              .attr("dominant-baseline", "middle")
              .attr("fill", this.attentionWeights[i][j] > 0.5 ? "#ffffff" : "#000000")
              .style("font-size", "10px")
              .text(this.attentionWeights[i][j].toFixed(2));
          }
        }
      }
    }
    
    // Add token labels for x-axis
    this.attentionSvg.selectAll(".x-label")
      .data(this.tokens)
      .enter()
      .append("text")
      .attr("class", "x-label")
      .attr("x", (_: string, i: number) => {
        const pos = xScale(i.toString());
        return pos !== undefined ? pos + xScale.bandwidth() / 2 : 0;
      })
      .attr("y", padding - 10)
      .attr("text-anchor", "middle")
      .attr("fill", "#e1e7f5") // Light gray from design system
      .style("font-size", "12px")
      .text((d: string) => d);
    
    // Add token labels for y-axis
    this.attentionSvg.selectAll(".y-label")
      .data(this.tokens)
      .enter()
      .append("text")
      .attr("class", "y-label")
      .attr("x", padding - 10)
      .attr("y", (_: string, i: number) => {
        const pos = yScale(i.toString());
        return pos !== undefined ? pos + yScale.bandwidth() / 2 : 0;
      })
      .attr("text-anchor", "end")
      .attr("dominant-baseline", "middle")
      .attr("fill", "#e1e7f5") // Light gray from design system
      .style("font-size", "12px")
      .text((d: string) => d);
    
    // Add legend
    const legendWidth = Math.min(250, width - 40);
    const legendHeight = 20;
    const legendX = Math.max(width - legendWidth - 20, padding);
    const legendY = height - 40;
    
    const defs = this.attentionSvg.append("defs");
    
    const gradient = defs.append("linearGradient")
      .attr("id", "attention-gradient")
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "0%");
    
    gradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", colorScale(0));
    
    gradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", colorScale(1));
    
    this.attentionSvg.append("rect")
      .attr("x", legendX)
      .attr("y", legendY)
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", "url(#attention-gradient)");
    
    this.attentionSvg.append("text")
      .attr("x", legendX)
      .attr("y", legendY - 5)
      .attr("text-anchor", "start")
      .attr("fill", "#e1e7f5")
      .style("font-size", "10px")
      .text("0.0");
    
    this.attentionSvg.append("text")
      .attr("x", legendX + legendWidth)
      .attr("y", legendY - 5)
      .attr("text-anchor", "end")
      .attr("fill", "#e1e7f5")
      .style("font-size", "10px")
      .text("1.0");
    
    this.attentionSvg.append("text")
      .attr("x", legendX + legendWidth / 2)
      .attr("y", legendY - 5)
      .attr("text-anchor", "middle")
      .attr("fill", "#e1e7f5")
      .style("font-size", "10px")
      .text("Attention Weight");
  }
  
  private initPositionalEncodingVisualization(): void {
    if (!this.posEncodingCanvas) return;
    
    // Create SVG
    this.posEncodingSvg = d3.select(this.posEncodingCanvas.nativeElement)
      .append("svg")
      .attr("width", "100%")
      .attr("height", "100%");
      
    this.updatePositionalEncodingVisualization();
  }
  
  private updatePositionalEncodingVisualization(): void {
    if (!this.posEncodingSvg || !this.posEncodingCanvas) return;
    
    const width = Math.max(this.posEncodingCanvas.nativeElement.clientWidth, 400);
    const height = Math.max(this.posEncodingCanvas.nativeElement.clientHeight, 400);
    const padding = 50;
    
    // Clear previous visualization
    this.posEncodingSvg.selectAll("*").remove();
    
    // Update SVG size
    this.posEncodingSvg
      .attr("width", width)
      .attr("height", height);
    
    // Add title
    this.posEncodingSvg.append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .attr("fill", "#ffffff") // White from design system
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text("Positional Encoding Vectors");
    
    // Generate positional encoding data (using the formula from the paper)
    const maxTokens = 10;
    const dimensions = 20; // Just show a subset of dimensions for visualization
    const posEncodings: number[][] = [];
    
    for (let pos = 0; pos < maxTokens; pos++) {
      const encoding: number[] = [];
      
      for (let i = 0; i < dimensions; i++) {
        const value = i % 2 === 0 
          ? Math.sin(pos / Math.pow(10000, i / dimensions))
          : Math.cos(pos / Math.pow(10000, (i - 1) / dimensions));
        encoding.push(value);
      }
      
      posEncodings.push(encoding);
    }
    
    // Create heatmap for positional encodings
    const cellWidth = Math.max((width - padding * 2) / dimensions, 10);
    const cellHeight = Math.max((height - padding * 2) / maxTokens, 10);
    
    // Color scale for positional encoding values
    const colorScale = d3.scaleLinear<string>()
      .domain([-1, 0, 1])
      .range(["#ff6b6b", "#162a4a", "#24b47e"]);
    
    // Create cellGroup for better organization
    const cellGroup = this.posEncodingSvg.append("g");
    
    // Create heatmap cells
    for (let pos = 0; pos < maxTokens; pos++) {
      for (let dim = 0; dim < dimensions; dim++) {
        const cellX = padding + dim * cellWidth;
        const cellY = padding + pos * cellHeight;
        
        cellGroup.append("rect")
          .attr("x", cellX)
          .attr("y", cellY)
          .attr("width", cellWidth)
          .attr("height", cellHeight)
          .attr("fill", colorScale(posEncodings[pos][dim]))
          .attr("stroke", "#162a4a")
          .attr("stroke-width", 1)
          .attr("rx", 2)
          .attr("ry", 2)
          .on("mouseover", (event: MouseEvent) => {
            // Add tooltip with value
            this.posEncodingSvg.append("text")
              .attr("class", "tooltip")
              .attr("x", event.offsetX + 10)
              .attr("y", event.offsetY - 10)
              .attr("fill", "#e1e7f5") // Light gray from design system
              .text(`Position ${pos}, Dimension ${dim}: ${posEncodings[pos][dim].toFixed(3)}`);
          })
          .on("mouseout", () => {
            // Remove tooltip
            this.posEncodingSvg.selectAll(".tooltip").remove();
          });
      }
    }
    
    // Add position labels for y-axis
    this.posEncodingSvg.selectAll(".pos-label")
      .data(Array.from({ length: maxTokens }, (_: unknown, i: number) => i))
      .enter()
      .append("text")
      .attr("class", "pos-label")
      .attr("x", padding - 10)
      .attr("y", (d: number) => padding + d * cellHeight + cellHeight / 2)
      .attr("text-anchor", "end")
      .attr("dominant-baseline", "middle")
      .attr("fill", "#e1e7f5") // Light gray from design system
      .style("font-size", "12px")
      .text((d: number) => `Pos ${d}`);
    
    // Add dimension labels for x-axis
    this.posEncodingSvg.selectAll(".dim-label")
      .data(Array.from({ length: dimensions }, (_: unknown, i: number) => i))
      .enter()
      .append("text")
      .attr("class", "dim-label")
      .attr("x", (d: number) => padding + d * cellWidth + cellWidth / 2)
      .attr("y", padding - 10)
      .attr("text-anchor", "middle")
      .attr("fill", "#e1e7f5") // Light gray from design system
      .style("font-size", "10px")
      .text((d: number) => `${d}`);
    
    // Add legend
    const legendWidth = Math.min(250, width - 40);
    const legendHeight = 20;
    const legendX = Math.max(width - legendWidth - 20, padding);
    const legendY = height - 40;
    
    const defs = this.posEncodingSvg.append("defs");
    
    const gradient = defs.append("linearGradient")
      .attr("id", "pos-encoding-gradient")
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "0%");
    
    gradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", colorScale(-1));
    
    gradient.append("stop")
      .attr("offset", "50%")
      .attr("stop-color", colorScale(0));
      
    gradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", colorScale(1));
    
    this.posEncodingSvg.append("rect")
      .attr("x", legendX)
      .attr("y", legendY)
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", "url(#pos-encoding-gradient)");
    
    this.posEncodingSvg.append("text")
      .attr("x", legendX)
      .attr("y", legendY - 5)
      .attr("text-anchor", "start")
      .attr("fill", "#e1e7f5")
      .style("font-size", "10px")
      .text("-1.0");
    
    this.posEncodingSvg.append("text")
      .attr("x", legendX + legendWidth / 2)
      .attr("y", legendY - 5)
      .attr("text-anchor", "middle")
      .attr("fill", "#e1e7f5")
      .style("font-size", "10px")
      .text("0.0");
    
    this.posEncodingSvg.append("text")
      .attr("x", legendX + legendWidth)
      .attr("y", legendY - 5)
      .attr("text-anchor", "end")
      .attr("fill", "#e1e7f5")
      .style("font-size", "10px")
      .text("1.0");
    
    this.posEncodingSvg.append("text")
      .attr("x", legendX + legendWidth / 2)
      .attr("y", legendY + legendHeight + 15)
      .attr("text-anchor", "middle")
      .attr("fill", "#e1e7f5")
      .style("font-size", "10px")
      .text("Positional Encoding Value");
  }
  
  // Simulation methods
  updateSimulation(): void {
    // Reset all component colors
    this.resetComponentColors();
    
    // Update visualizations based on current step
    switch (this.currentStep) {
      case 0: // Input embedding
        this.highlightComponent(-6, -2, 0, 0x64b5f6, 1.5);
        break;
        
      case 1: // Positional encoding
        this.highlightComponent(-6, -1, 0, 0xff9d45, 1.5);
        break;
        
      case 2: // Self-attention in first encoder layer
        this.highlightEncoderComponent(0, 1, 0x4285f4, 1.5);
        break;
        
      case 3: // FFN in first encoder layer
        this.highlightEncoderComponent(0, 0, 0x7c4dff, 1.5);
        break;
        
      case 4: // Masked self-attention in first decoder layer
        this.highlightDecoderComponent(0, 2, 0x00c9ff, 1.5);
        break;
        
      case 5: // Encoder-decoder attention in first decoder layer
        this.highlightDecoderComponent(0, 1, 0xff9d45, 1.5);
        this.highlightConnectionLine(0);
        break;
        
      case 6: // Linear layer
        this.highlightComponent(6, 6 * 1.5 + 1, 0, 0x24b47e, 1.5);
        break;
        
      case 7: // Softmax layer
        this.highlightComponent(6, 6 * 1.5 + 2, 0, 0x24b47e, 1.5);
        break;
        
      default:
        // No highlighting for unknown steps
        break;
    }
  }
  
  private resetComponentColors(): void {
    if (!this.scene) return;
    
    // Reset all blocks to their original colors
    this.scene.traverse((object: THREE.Object3D) => {
      if (object instanceof THREE.Mesh && object.material instanceof THREE.MeshPhongMaterial) {
        object.material.emissive.set(0x000000);
        object.material.emissiveIntensity = 0;
      }
    });
    
    // Reset connection lines
    if (this.connectionLines && this.connectionLines.length > 0) {
      this.connectionLines.forEach((line: THREE.Line) => {
        if (line.material instanceof THREE.LineBasicMaterial) {
          line.material.color.set(0x8bb4fa);
          line.material.opacity = 0.4;
        }
      });
    }
  }
  
  private highlightComponent(x: number, y: number, z: number, color: number, intensity: number): void {
    if (!this.scene) return;
    
    // Find the component at the given position and highlight it
    this.scene.traverse((object: THREE.Object3D) => {
      if (object instanceof THREE.Mesh && 
          Math.abs(object.position.x - x) < 0.1 && 
          Math.abs(object.position.y - y) < 0.1 && 
          Math.abs(object.position.z - z) < 0.1) {
        if (object.material instanceof THREE.MeshPhongMaterial) {
          object.material.emissive.set(color);
          object.material.emissiveIntensity = intensity;
        }
      }
    });
  }
  
  private highlightEncoderComponent(layerIndex: number, componentIndex: number, color: number, intensity: number): void {
    if (!this.encoder || !this.encoder[layerIndex]) return;
    
    const encoderLayer = this.encoder[layerIndex];
    
    // Find the component in the encoder layer
    encoderLayer.traverse((object: THREE.Object3D) => {
      if (object instanceof THREE.Mesh && 
          Math.abs(object.position.y - componentIndex) < 0.1) {
        if (object.material instanceof THREE.MeshPhongMaterial) {
          object.material.emissive.set(color);
          object.material.emissiveIntensity = intensity;
        }
      }
    });
  }
  
  private highlightDecoderComponent(layerIndex: number, componentIndex: number, color: number, intensity: number): void {
    if (!this.decoder || !this.decoder[layerIndex]) return;
    
    const decoderLayer = this.decoder[layerIndex];
    
    // Find the component in the decoder layer
    decoderLayer.traverse((object: THREE.Object3D) => {
      if (object instanceof THREE.Mesh && 
          Math.abs(object.position.y - componentIndex) < 0.1) {
        if (object.material instanceof THREE.MeshPhongMaterial) {
          object.material.emissive.set(color);
          object.material.emissiveIntensity = intensity;
        }
      }
    });
  }
  
  private highlightConnectionLine(layerIndex: number): void {
    if (!this.connectionLines || 
        layerIndex < 0 || 
        layerIndex >= this.connectionLines.length) {
      return;
    }
    
    const connectionLine = this.connectionLines[layerIndex];
    if (connectionLine && connectionLine.material instanceof THREE.LineBasicMaterial) {
      connectionLine.material.color.set(0xff9d45); // Orange
      connectionLine.material.opacity = 1.0;
    }
  }
  
  // Simulation control methods
  startAutoPlay(): void {
    if (this.isPlaying) return;
    
    this.isPlaying = true;
    this.playInterval = setInterval(() => {
      this.nextStep();
    }, this.autoPlaySpeed);
  }
  
  stopAutoPlay(): void {
    if (!this.isPlaying) return;
    
    this.isPlaying = false;
    if (this.playInterval) {
      clearInterval(this.playInterval);
      this.playInterval = null;
    }
  }
  
  togglePlay(): void {
    if (this.isPlaying) {
      this.stopAutoPlay();
    } else {
      this.startAutoPlay();
    }
  }
  
  resetSimulation(): void {
    this.stopAutoPlay();
    this.currentStep = 0;
    this.updateSimulation();
  }
  
  nextStep(): void {
    if (this.currentStep < this.totalSteps - 1) {
      this.currentStep++;
    } else {
      this.currentStep = 0; // Loop back to beginning
    }
    this.updateSimulation();
  }
  
  prevStep(): void {
    if (this.currentStep > 0) {
      this.currentStep--;
    } else {
      this.currentStep = this.totalSteps - 1; // Loop to end
    }
    this.updateSimulation();
  }
  
  goToStep(step: number): void {
    if (step >= 0 && step < this.totalSteps) {
      this.currentStep = step;
      this.updateSimulation();
    }
  }
  
  // Navigation methods
  showSection(sectionName: string): void {
    this.activeSection = sectionName;
  }
}