import { Component, ElementRef, OnInit, ViewChild, AfterViewInit, OnDestroy, HostListener, PLATFORM_ID, Inject } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';

@Component({
  selector: 'app-lstm-simulator',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './lstm-simulator.component.html',
  styleUrls: ['./lstm-simulator.component.scss']
})
export class LstmSimulatorComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('simulationContainer') simulationContainer!: ElementRef;

  // Simulation dimensions
  private width = 1000;
  private height = 800;
  private margin = { top: 50, right: 50, bottom: 50, left: 50 };
  
  // Animation settings
  private animationSpeed = 1000; // ms
  private animationInterval: any;
  
  // LSTM data
  private defaultSequence = ['The', 'clouds', 'are', 'in', 'the', 'sky'];
  public sequence: string[] = [...this.defaultSequence];
  public customSequence = '';
  public timeSteps: LstmTimeStep[] = [];
  
  // D3 elements
  private svg: any;
  private mainGroup: any;
  
  // UI states
  isPlaying = false;
  currentStepIndex = 0;
  playbackSpeed = 1;
  showAdvancedView = false;
  activeTab = 'concepts';
  showCustomInput = false;
  
  // Explanation text
  explanationText = "Long Short-Term Memory (LSTM) networks are specialized RNNs designed to address the vanishing gradient problem, allowing them to effectively learn and remember information over long sequences. LSTMs use gate mechanisms to control information flow, making them powerful for tasks involving long-range dependencies.";
  
  // Mathematical formulas (using LaTeX notation for the view)
  mathFormulas = {
    forgetGate: "f_t = σ(W_f · [h_{t-1}, x_t] + b_f)",
    inputGate: "i_t = σ(W_i · [h_{t-1}, x_t] + b_i)",
    candidateState: "\\tilde{C}_t = tanh(W_C · [h_{t-1}, x_t] + b_C)",
    cellState: "C_t = f_t ⊙ C_{t-1} + i_t ⊙ \\tilde{C}_t",
    outputGate: "o_t = σ(W_o · [h_{t-1}, x_t] + b_o)",
    hiddenState: "h_t = o_t ⊙ tanh(C_t)"
  };
  
  // Research papers
  researchPapers = [
    {
      title: "Long Short-Term Memory",
      authors: "Sepp Hochreiter, Jürgen Schmidhuber",
      year: 1997,
      journal: "Neural Computation 9(8): 1735-1780",
      url: "https://www.bioinf.jku.at/publications/older/2604.pdf",
      description: "The original paper introducing LSTM networks to address the vanishing gradient problem in RNNs."
    },
    {
      title: "Learning to Forget: Continual Prediction with LSTM",
      authors: "Felix A. Gers, Jürgen Schmidhuber, Fred Cummins",
      year: 2000,
      journal: "Neural Computation 12(10): 2451-2471",
      url: "https://www.researchgate.net/publication/12292425_Learning_to_Forget_Continual_Prediction_with_LSTM",
      description: "Introduced the forget gate to the LSTM architecture, allowing the network to reset its state."
    },
    {
      title: "LSTM: A Search Space Odyssey",
      authors: "Klaus Greff, Rupesh K. Srivastava, Jan Koutník, Bas R. Steunebrink, Jürgen Schmidhuber",
      year: 2017,
      journal: "IEEE Transactions on Neural Networks and Learning Systems 28(10): 2222-2232",
      url: "https://arxiv.org/abs/1503.04069",
      description: "A comprehensive analysis of LSTM variants and their performance."
    },
    {
      title: "Sequence to Sequence Learning with Neural Networks",
      authors: "Ilya Sutskever, Oriol Vinyals, Quoc V. Le",
      year: 2014,
      journal: "Advances in Neural Information Processing Systems 27",
      url: "https://arxiv.org/abs/1409.3215",
      description: "Applied LSTMs to sequence-to-sequence problems, particularly machine translation."
    }
  ];
  
  // Applications data
  applications = [
    {
      name: "Machine Translation",
      icon: "translate",
      color: "#4285f4",
      description: "LSTMs power translation systems by capturing relationships between words across languages, even when they appear in different positions."
    },
    {
      name: "Speech Recognition",
      icon: "keyboard_voice",
      color: "#7c4dff",
      description: "LSTMs process audio features over time, recognizing phonemes, words, and phrases with their temporal context preserved."
    },
    {
      name: "Text Generation",
      icon: "text_fields",
      color: "#00c9ff",
      description: "LSTMs can generate coherent text by modeling the probability of each word given all previous words in the sequence."
    },
    {
      name: "Time Series Prediction",
      icon: "query_stats",
      color: "#ff9d45",
      description: "LSTMs predict future values in financial markets, sensor readings, and other time series by identifying patterns across time periods."
    },
    {
      name: "Sentiment Analysis",
      icon: "mood",
      color: "#24b47e",
      description: "LSTMs can understand the sentiment of text by capturing contextual information and long-range dependencies in reviews, social media, and more."
    },
    {
      name: "Music Generation",
      icon: "music_note",
      color: "#ae94ff",
      description: "LSTMs can learn musical patterns and generate new compositions by capturing temporal structures in musical sequences."
    },
    {
      name: "Healthcare",
      icon: "medical_services",
      color: "#ff6b6b",
      description: "LSTMs analyze medical time series data like ECG signals, predict disease progression, and assist in clinical decision making."
    },
    {
      name: "Video Analysis",
      icon: "videocam",
      color: "#64b5f6",
      description: "LSTMs process frame sequences to understand activities, detect anomalies, and track objects across time in video content."
    }
  ];
  
  // Advantages data
  advantages = [
    {
      name: "Long-Range Dependencies",
      icon: "memory",
      color: "#24b47e",
      description: "Unlike standard RNNs, LSTMs can learn dependencies spanning hundreds of time steps, thanks to their cell state's constant error carousel."
    },
    {
      name: "Vanishing Gradient Solution",
      icon: "trending_up",
      color: "#4285f4",
      description: "The cell state's linear path allows gradients to flow backward through time without vanishing, enabling effective training on long sequences."
    },
    {
      name: "Selective Memory",
      icon: "lock_open",
      color: "#7c4dff",
      description: "Gate mechanisms allow LSTMs to selectively remember or forget information, focusing on relevant features while discarding noise."
    },
    {
      name: "Architectural Flexibility",
      icon: "extension",
      color: "#00c9ff",
      description: "LSTMs can be extended with bidirectional layers, attention mechanisms, or stacked to create deeper networks for complex tasks."
    },
    {
      name: "Robust to Sequence Length",
      icon: "straighten",
      color: "#ff9d45",
      description: "LSTMs maintain consistent performance regardless of sequence length, making them suitable for variable-length inputs."
    }
  ];
  
  // Code examples for common frameworks
  codeExamples = {
    tensorflow: `
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Create a simple LSTM for sequence classification
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(units=128, dropout=0.2, recurrent_dropout=0.2),
    Dense(units=num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
`,
    pytorch: `
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Initialize and train model
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
`,
    javascript: `
// TensorFlow.js implementation of an LSTM
import * as tf from '@tensorflow/tfjs';

// Create an LSTM model for text generation
function createModel(vocabSize, embeddingDim, lstmUnits) {
  const model = tf.sequential();
  
  // Add embedding layer
  model.add(tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: embeddingDim,
    inputLength: 1
  }));
  
  // Add LSTM layer
  model.add(tf.layers.lstm({
    units: lstmUnits,
    returnSequences: false
  }));
  
  // Add output layer
  model.add(tf.layers.dense({
    units: vocabSize,
    activation: 'softmax'
  }));
  
  // Compile model
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

// Usage
const model = createModel(vocabSize, 128, 256);
`
  };
  
  // Additional learning resources
  learningResources = [
    {
      title: "Understanding LSTM Networks",
      author: "Christopher Olah",
      url: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/",
      type: "Blog Post",
      description: "An excellent visual explanation of LSTM networks and how they work."
    },
    {
      title: "Deep Learning Book - Chapter 10: Sequence Modeling",
      author: "Ian Goodfellow, Yoshua Bengio, Aaron Courville",
      url: "https://www.deeplearningbook.org/",
      type: "Book Chapter",
      description: "Comprehensive academic coverage of RNNs, LSTMs, and other sequence models."
    },
    {
      title: "CS231n: Convolutional Neural Networks - Lecture 10",
      author: "Stanford University",
      url: "http://cs231n.stanford.edu/",
      type: "Course Material",
      description: "Stanford's course materials on RNN, LSTM, and applications to visual recognition."
    },
    {
      title: "Illustrated Guide to LSTM's and GRU's",
      author: "Michael Phi",
      url: "https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21",
      type: "Tutorial",
      description: "A step-by-step visual guide to understanding LSTM and GRU architectures."
    },
    {
      title: "A Gentle Introduction to LSTM Networks",
      author: "Jason Brownlee",
      url: "https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/",
      type: "Tutorial",
      description: "Beginner-friendly introduction to LSTM concepts and implementation."
    }
  ];
  
  // Real-world examples
  realWorldExamples = [
    {
      name: "Google Translate",
      description: "Uses LSTM-based sequence-to-sequence models to translate between languages, handling complex grammar and context."
    },
    {
      name: "Apple Siri",
      description: "Employs LSTMs for speech recognition and natural language understanding to process user commands."
    },
    {
      name: "Netflix Recommendations",
      description: "Uses LSTMs to analyze viewing patterns over time to make personalized content recommendations."
    },
    {
      name: "Financial Forecasting",
      description: "Investment firms use LSTMs to predict stock prices by analyzing historical market data and trends."
    }
  ];
  
  constructor(@Inject(PLATFORM_ID) private platformId: Object) { }

  ngOnInit(): void {
    if (isPlatformBrowser(this.platformId)) {
      this.initializeTimeSteps();
    }
  }

  ngAfterViewInit(): void {
    if (isPlatformBrowser(this.platformId)) {
      setTimeout(() => {
        this.initializeVisualization();
      });
    }
  }

  ngOnDestroy(): void {
    // Clean up
    if (this.animationInterval) {
      clearInterval(this.animationInterval);
    }
  }

  @HostListener('window:resize')
  onResize(): void {
    this.handleResize();
  }

  private handleResize(): void {
    if (this.svg) {
      // Redraw visualization on window resize
      this.updateVisualizationSize();
      this.drawLstmCell(this.timeSteps[this.currentStepIndex]);
    }
  }

  private updateVisualizationSize(): void {
    if (!this.simulationContainer) return;
    
    const container = this.simulationContainer.nativeElement;
    if (!container) return;
    
    const containerWidth = container.clientWidth;
    
    // Update SVG size
    this.svg
      .attr('width', containerWidth)
      .attr('height', this.height)
      .attr('viewBox', `0 0 ${containerWidth} ${this.height}`);
    
    // Update width
    this.width = containerWidth;
  }

  // Apply custom input sequence
  applyCustomSequence(): void {
    if (!this.customSequence.trim()) {
      return;
    }
    
    // Parse the input string into words
    const words = this.customSequence
      .split(/\s+/)
      .filter(word => word.trim().length > 0)
      .slice(0, 10); // Limit to 10 words for visualization clarity
    
    if (words.length < 2) {
      alert('Please enter at least 2 words for the sequence.');
      return;
    }
    
    // Reset and reinitialize with new sequence
    this.pauseAnimation();
    this.sequence = words;
    this.timeSteps = [];
    this.currentStepIndex = 0;
    this.initializeTimeSteps();
    
    if (this.svg) {
      this.drawLstmCell(this.timeSteps[0]);
    }
  }

  // Reset to default sequence
  resetToDefaultSequence(): void {
    this.pauseAnimation();
    this.sequence = [...this.defaultSequence];
    this.customSequence = '';
    this.timeSteps = [];
    this.currentStepIndex = 0;
    this.initializeTimeSteps();
    
    if (this.svg) {
      this.drawLstmCell(this.timeSteps[0]);
    }
  }

  // Initialize LSTM time steps with sample data
  private initializeTimeSteps(): void {
    // Initial state
    let prevCellState = new Array(4).fill(0);
    let prevHiddenState = new Array(4).fill(0);
    
    // Create time steps for each word in the sequence
    this.sequence.forEach((word, index) => {
      const input = this.wordToVector(word);
      
      // Calculate gate values (simplified for visualization purposes)
      const forgetGate = this.calculateGateValues(input, prevHiddenState, 'forget', word);
      const inputGate = this.calculateGateValues(input, prevHiddenState, 'input', word);
      const outputGate = this.calculateGateValues(input, prevHiddenState, 'output', word);
      const candidateValues = this.calculateCandidateValues(input, prevHiddenState);
      
      // Calculate new cell state
      const newCellState = prevCellState.map((val, i) => 
        forgetGate[i] * val + inputGate[i] * candidateValues[i]
      );
      
      // Calculate new hidden state
      const newHiddenState = outputGate.map((val, i) => 
        val * Math.tanh(newCellState[i])
      );
      
      // Store the time step
      this.timeSteps.push({
        word,
        index,
        input,
        forgetGate,
        inputGate,
        outputGate,
        candidateValues,
        prevCellState: [...prevCellState],
        cellState: [...newCellState],
        prevHiddenState: [...prevHiddenState],
        hiddenState: [...newHiddenState],
        forgetGateImpact: prevCellState.map((val, i) => forgetGate[i] * val),
        inputGateImpact: candidateValues.map((val, i) => inputGate[i] * val)
      });
      
      // Update for next time step
      prevCellState = [...newCellState];
      prevHiddenState = [...newHiddenState];
    });
  }

  // Convert word to vector (simplified for visualization)
  private wordToVector(word: string): number[] {
    // Basic features based on the word properties
    const length = Math.min(word.length / 10, 1);  // Normalized length
    const firstChar = word.length > 0 ? word.toLowerCase().charCodeAt(0) % 100 / 100 : 0;
    
    // Simple semantic features - can be customized based on the sequence
    const isKeyword = ['clouds', 'sky', 'rain', 'sun', 'mountain', 'river'].includes(word.toLowerCase()) ? 0.9 : 0.2;
    
    // Character-based feature
    const hasCommonLetter = word.toLowerCase().includes('e') ? 0.8 : 0.3;
    
    return [length, firstChar, isKeyword, hasCommonLetter];
  }

  // Calculate gate values (simplified for visualization)
  private calculateGateValues(input: number[], prevHidden: number[], gateType: string, word: string): number[] {
    // Ensure consistent gate values across visualizations
    const results = input.map((val, i) => {
      let baseValue = (val + prevHidden[i]) / 2;
      
      // Add semantic understanding for visualization purposes
      if (gateType === 'forget') {
        // Forget gate should be high when we need to remember context
        if (word.toLowerCase() === 'clouds' && i === 2) return 0.95;  // Remember "clouds" in semantic dimension
        return this.sigmoid(baseValue * 1.5 + 0.3);
      } else if (gateType === 'input') {
        // Input gate should be high for important information
        if (['clouds', 'sky', 'rain', 'sun'].includes(word.toLowerCase()) && i === 2) return 0.9;
        return this.sigmoid(baseValue * 1.2);
      } else {
        // Output gate controls what to expose
        if (word.toLowerCase() === 'sky' && i === 2) return 0.95;
        
        // Fix the specific values to match the expected visualization
        if (word === 'The' || word === 'the') {
          if (i === 0) return 0.57;
          if (i === 1) return 0.64;
          if (i === 2) return 0.58;
          if (i === 3) return 0.61;
        }
        
        return this.sigmoid(baseValue * 1.1 + 0.1);
      }
    });
    
    return results;
  }

  // Calculate candidate values
  private calculateCandidateValues(input: number[], prevHidden: number[]): number[] {
    return input.map((val, i) => 
      Math.tanh((val + prevHidden[i]) / 2)
    );
  }

  // Sigmoid activation function
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  // Initialize the D3 visualization
  private initializeVisualization(): void {
    if (!this.simulationContainer || !this.simulationContainer.nativeElement) {
      console.error('Simulation container not found');
      return;
    }
    
    const element = this.simulationContainer.nativeElement;
    
    // Clear any existing SVG
    d3.select(element).selectAll('svg').remove();
    
    // Create SVG
    this.svg = d3.select(element)
      .append('svg')
      .attr('width', this.width)
      .attr('height', this.height)
      .attr('viewBox', `0 0 ${this.width} ${this.height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    
    // Define arrow marker for connections
    this.svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#e1e7f5');
    
    // Add main group
    this.mainGroup = this.svg.append('g')
      .attr('transform', `translate(${this.margin.left}, ${this.margin.top})`);
    
    // Draw initial state
    if (this.timeSteps.length > 0) {
      this.drawLstmCell(this.timeSteps[0]);
    } else {
      console.error('No time steps available to visualize');
    }
  }

  // Draw the LSTM cell for a specific time step
  private drawLstmCell(timeStep: LstmTimeStep): void {
    if (!this.mainGroup) return;
    
    // Clear previous drawing
    this.mainGroup.selectAll('*').remove();
    
    // Cell dimensions
    const cellWidth = this.width - this.margin.left - this.margin.right;
    const cellHeight = this.height - this.margin.top - this.margin.bottom;
    
    // Define areas - adjust for better fit
    const inputAreaWidth = cellWidth * 0.2;
    const gateAreaWidth = cellWidth * 0.15;
    const stateAreaWidth = cellWidth * 0.4;
    const outputAreaWidth = cellWidth * 0.15;
    
    // Colors from design system
    const primaryBlue = '#4285f4';
    const lightBlue = '#8bb4fa';
    const darkBlue = '#2c5cbd';
    const purple = '#7c4dff';
    const cyan = '#00c9ff';
    const orange = '#ff9d45';
    const darkBlueBg = '#162a4a';
    const mediumBlue = '#1e3a66';
    
    // Draw title
    this.mainGroup.append('text')
      .attr('x', cellWidth / 2)
      .attr('y', -25)
      .attr('text-anchor', 'middle')
      .attr('fill', '#ffffff')
      .attr('font-size', '28px')
      .attr('font-weight', 'bold')
      .text(`LSTM Cell - Processing "${timeStep.word}" (Step ${timeStep.index + 1}/${this.sequence.length})`);
    
    // Draw background
    this.mainGroup.append('rect')
      .attr('width', cellWidth)
      .attr('height', cellHeight)
      .attr('fill', darkBlueBg)
      .attr('rx', 12)
      .attr('ry', 12)
      .attr('stroke', mediumBlue)
      .attr('stroke-width', 2);
    
    // Draw sequence context
    this.drawSequenceContext(cellWidth, 50, timeStep.index);
    
    // Draw input vector - adjust position to prevent cutoff
    this.drawVector(
      inputAreaWidth * 1.1,
      cellHeight / 2, 
      timeStep.input, 
      'Input Vector', 
      primaryBlue,
      ['Length', '1st Char', 'Semantic', 'Letter "e"']
    );
    
    // Draw gates
    const gateY = cellHeight * 0.25;
    const gateYSpacing = cellHeight * 0.25;
    
    // Forget gate
    this.drawGate(
      inputAreaWidth + gateAreaWidth / 2,
      gateY,
      timeStep.forgetGate,
      'Forget Gate',
      orange,
      timeStep.forgetGateImpact
    );
    
    // Input gate
    this.drawGate(
      inputAreaWidth + gateAreaWidth / 2,
      gateY + gateYSpacing,
      timeStep.inputGate,
      'Input Gate',
      cyan,
      timeStep.inputGateImpact
    );
    
    // Candidate values
    this.drawVector(
      inputAreaWidth + gateAreaWidth / 2,
      gateY + gateYSpacing * 2,
      timeStep.candidateValues,
      'Candidate Values',
      lightBlue
    );
    
    // Output gate
    this.drawGate(
      inputAreaWidth + gateAreaWidth + stateAreaWidth + gateAreaWidth / 2,
      gateY + gateYSpacing,
      timeStep.outputGate,
      'Output Gate',
      purple,
      undefined,
      true
    );
    
    // Draw cell state - the "memory lane"
    this.drawCellState(
      inputAreaWidth + gateAreaWidth + stateAreaWidth / 2,
      gateY,
      timeStep.prevCellState,
      timeStep.cellState,
      timeStep.forgetGateImpact,
      timeStep.inputGateImpact,
      'Cell State (Memory)',
      darkBlue
    );
    
    // Draw hidden state
    this.drawState(
      inputAreaWidth + gateAreaWidth + stateAreaWidth / 2,
      gateY + gateYSpacing * 2,
      timeStep.prevHiddenState,
      timeStep.hiddenState,
      'Hidden State',
      primaryBlue
    );
    
    // Draw output
    this.drawVector(
      cellWidth - outputAreaWidth / 2,
      cellHeight / 2,
      timeStep.hiddenState,
      'Output',
      primaryBlue
    );
    
    // Draw connections
    this.drawConnections(cellWidth, cellHeight, inputAreaWidth, gateAreaWidth, stateAreaWidth, timeStep);
    
    // Draw explanation for current step
    this.drawStepExplanation(timeStep, cellWidth, cellHeight);
    
    // Add formula legend if in advanced view
    if (this.showAdvancedView) {
      this.drawFormulaLegend(cellWidth, cellHeight);
    }
  }

  // Draw formula legend
  private drawFormulaLegend(cellWidth: number, cellHeight: number): void {
    if (!this.mainGroup) return;
    
    const legendGroup = this.mainGroup.append('g')
      .attr('transform', `translate(${cellWidth - 330}, ${50})`);
    
    legendGroup.append('rect')
      .attr('width', 320)
      .attr('height', 145)
      .attr('fill', '#1e3a66')
      .attr('rx', 8)
      .attr('ry', 8)
      .attr('opacity', 0.9);
    
    legendGroup.append('text')
      .attr('x', 10)
      .attr('y', 25)
      .attr('fill', '#ffffff')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text('Key LSTM Equations:');
    
    const formulas = [
      {label: 'Forget Gate:', formula: this.mathFormulas.forgetGate},
      {label: 'Input Gate:', formula: this.mathFormulas.inputGate},
      {label: 'Cell State:', formula: this.mathFormulas.cellState}
    ];
    
    formulas.forEach((item, i) => {
      legendGroup.append('text')
        .attr('x', 15)
        .attr('y', 50 + i * 30)
        .attr('fill', '#e1e7f5')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .text(item.label);
        
      legendGroup.append('text')
        .attr('x', 100)
        .attr('y', 50 + i * 30)
        .attr('fill', '#e1e7f5')
        .attr('font-size', '14px')
        .text(item.formula);
    });
  }

  // Draw the sequence context
  private drawSequenceContext(width: number, y: number, currentIndex: number): void {
    if (!this.mainGroup) return;
    
    const group = this.mainGroup.append('g')
      .attr('transform', `translate(0, ${y})`);
    
    const boxWidth = width / this.sequence.length;
    const boxHeight = 40;
    
    this.sequence.forEach((word, index) => {
      const x = index * boxWidth;
      
      // Background
      group.append('rect')
        .attr('x', x)
        .attr('y', 0)
        .attr('width', boxWidth)
        .attr('height', boxHeight)
        .attr('fill', index === currentIndex ? '#4285f4' : '#1e3a66')
        .attr('rx', 8)
        .attr('ry', 8)
        .attr('stroke', index === currentIndex ? '#8bb4fa' : 'none')
        .attr('stroke-width', 2);
      
      // Word text
      group.append('text')
        .attr('x', x + boxWidth / 2)
        .attr('y', boxHeight / 2 + 5)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#ffffff')
        .attr('font-size', '16px')
        .attr('font-weight', index === currentIndex ? 'bold' : 'normal')
        .text(word);
    });
    
    // Add tooltip to explain this part
    group.append('rect')
      .attr('x', width - 20)
      .attr('y', -5)
      .attr('width', 20)
      .attr('height', 20)
      .attr('fill', '#7c4dff')
      .attr('rx', 10)
      .attr('ry', 10)
      .attr('cursor', 'pointer')
      .on('mouseover', (event: MouseEvent) => {
        const tooltip = this.mainGroup.append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${width - 200}, ${y - 50})`);
          
        tooltip.append('rect')
          .attr('width', 180)
          .attr('height', 40)
          .attr('fill', '#1e3a66')
          .attr('rx', 5)
          .attr('ry', 5)
          .attr('opacity', 0.9);
          
        tooltip.append('text')
          .attr('x', 10)
          .attr('y', 25)
          .attr('fill', '#ffffff')
          .attr('font-size', '12px')
          .text('Input sequence processed one word at a time');
      })
      .on('mouseout', (event: MouseEvent) => {
        this.mainGroup.selectAll('.tooltip').remove();
      });
      
    group.append('text')
      .attr('x', width - 10)
      .attr('y', 7)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', '#ffffff')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('?');
  }

  // Draw a vector as a series of cells
  private drawVector(
    x: number, 
    y: number, 
    values: number[], 
    label: string, 
    color: string,
    featureLabels?: string[]
  ): void {
    if (!this.mainGroup) return;
    
    const cellSize = 50;
    const spacing = 15;
    
    // Group for the vector
    const group = this.mainGroup.append('g')
      .attr('transform', `translate(${x - (values.length * (cellSize + spacing)) / 2}, ${y - cellSize / 2})`);
    
    // Draw label
    group.append('text')
      .attr('x', (values.length * (cellSize + spacing)) / 2)
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e1e7f5')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text(label);
    
    // Draw cells
    values.forEach((value, i) => {
      // Cell background
      group.append('rect')
        .attr('x', i * (cellSize + spacing))
        .attr('y', 0)
        .attr('width', cellSize)
        .attr('height', cellSize)
        .attr('fill', d3.interpolateBlues(Math.abs(value)))
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('rx', 8)
        .attr('ry', 8);
      
      // Cell value
      group.append('text')
        .attr('x', i * (cellSize + spacing) + cellSize / 2)
        .attr('y', cellSize / 2 + 5)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', value > 0.5 ? '#ffffff' : '#e1e7f5')
        .attr('font-size', '16px')
        .text(value.toFixed(2));
      
      // Index indicator (0-3) or feature label if provided
      const labelText = featureLabels && featureLabels[i] ? featureLabels[i] : `[${i}]`;
      
      group.append('text')
        .attr('x', i * (cellSize + spacing) + cellSize / 2)
        .attr('y', cellSize + 15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#8a9ab0')
        .attr('font-size', '12px')
        .text(labelText);
    });
  }

  // Draw a gate with sigmoid activation visualization
  private drawGate(
    x: number, 
    y: number, 
    values: number[], 
    label: string, 
    color: string, 
    impactValues?: number[], 
    fixAlignment: boolean = false
  ): void {
    if (!this.mainGroup) return;
    
    const cellSize = 50;
    const spacing = 15;
    
    // Group for the gate
    const group = this.mainGroup.append('g')
      .attr('transform', `translate(${x - (values.length * (cellSize + spacing)) / 2}, ${y - cellSize / 2})`);
    
    // Draw label
    group.append('text')
      .attr('x', (values.length * (cellSize + spacing)) / 2)
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e1e7f5')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text(label);
    
    // Draw gate symbol
    group.append('circle')
      .attr('cx', (values.length * (cellSize + spacing)) / 2 - ((values.length * (cellSize + spacing)) / 2 + 25))
      .attr('cy', cellSize / 2)
      .attr('r', 15)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2);
    
    group.append('text')
      .attr('x', (values.length * (cellSize + spacing)) / 2 - ((values.length * (cellSize + spacing)) / 2 + 25))
      .attr('y', cellSize / 2 + 5)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', color)
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('σ');
    
    // Draw cells
    values.forEach((value, i) => {
      // Cell background
      group.append('rect')
        .attr('x', i * (cellSize + spacing))
        .attr('y', 0)
        .attr('width', cellSize)
        .attr('height', cellSize)
        .attr('fill', d3.interpolate('#1e3a66', color)(value))
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('rx', 8)
        .attr('ry', 8);
      
      // Gate value
      group.append('text')
        .attr('x', i * (cellSize + spacing) + cellSize / 2)
        .attr('y', cellSize / 2 + 5)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#ffffff')
        .attr('font-size', '14px')
        .text(value.toFixed(2));
      
      // Index indicator
      group.append('text')
        .attr('x', i * (cellSize + spacing) + cellSize / 2)
        .attr('y', cellSize + 15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#8a9ab0')
        .attr('font-size', '12px')
        .text(`[${i}]`)
        .attr('class', fixAlignment ? 'fixed-alignment' : '');
      
      // Draw gate visualization (open/closed)
      if (this.showAdvancedView) {
        const gateHeight = 15;
        const gateOpen = cellSize * value;
        
        // Gate background
        group.append('rect')
          .attr('x', i * (cellSize + spacing))
          .attr('y', cellSize + 25)
          .attr('width', cellSize)
          .attr('height', gateHeight)
          .attr('fill', '#2a4980')
          .attr('rx', 2)
          .attr('ry', 2);
        
        // Gate open amount
        group.append('rect')
          .attr('x', i * (cellSize + spacing))
          .attr('y', cellSize + 25)
          .attr('width', gateOpen)
          .attr('height', gateHeight)
          .attr('fill', color)
          .attr('rx', 2)
          .attr('ry', 2);
        
        // Gate impact value (if provided)
        if (impactValues) {
          group.append('text')
            .attr('x', i * (cellSize + spacing) + cellSize / 2)
            .attr('y', cellSize + 25 + gateHeight + 15)
            .attr('text-anchor', 'middle')
            .attr('fill', '#e1e7f5')
            .attr('font-size', '12px')
            .text(`Impact: ${impactValues[i].toFixed(2)}`);
        }
      }
    });
  }

  // Draw cell state with special visualization of the "memory lane"
  private drawCellState(
    x: number, 
    y: number, 
    prevValues: number[], 
    newValues: number[], 
    forgetImpact: number[],
    inputImpact: number[],
    label: string, 
    color: string
  ): void {
    if (!this.mainGroup) return;
    
    const cellSize = 50;
    const spacing = 15;
    const laneHeight = 110;
    
    // Group for the cell state
    const group = this.mainGroup.append('g')
      .attr('transform', `translate(${x - (newValues.length * (cellSize + spacing)) / 2}, ${y - cellSize / 2})`);
    
    // Draw label
    group.append('text')
      .attr('x', (newValues.length * (cellSize + spacing)) / 2)
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e1e7f5')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text(label);
    
    // Draw memory lane
    group.append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', newValues.length * (cellSize + spacing) - spacing)
      .attr('height', laneHeight)
      .attr('fill', '#1e3a66')
      .attr('stroke', color)
      .attr('stroke-width', 2)
      .attr('rx', 8)
      .attr('ry', 8)
      .attr('opacity', 0.7);
    
    // Draw cell state values
    newValues.forEach((value, i) => {
      const prevValue = prevValues[i];
      const changed = Math.abs(value - prevValue) > 0.1;
      
      // Cell state value
      group.append('rect')
        .attr('x', i * (cellSize + spacing))
        .attr('y', laneHeight / 2 - cellSize / 2)
        .attr('width', cellSize)
        .attr('height', cellSize)
        .attr('fill', d3.interpolate('#1e3a66', color)(Math.abs(value)))
        .attr('stroke', changed ? '#ff9d45' : color)
        .attr('stroke-width', changed ? 3 : 2)
        .attr('rx', 8)
        .attr('ry', 8);
      
      // Cell value
      group.append('text')
        .attr('x', i * (cellSize + spacing) + cellSize / 2)
        .attr('y', laneHeight / 2 + 5)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#ffffff')
        .attr('font-size', '14px')
        .text(value.toFixed(2));
      
      // Index indicator
      group.append('text')
        .attr('x', i * (cellSize + spacing) + cellSize / 2)
        .attr('y', laneHeight + 15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#8a9ab0')
        .attr('font-size', '12px')
        .text(`[${i}]`);
      
      // Flow visualization
      if (this.showAdvancedView) {
        // Previous value indicator
        group.append('text')
          .attr('x', i * (cellSize + spacing) + cellSize / 2)
          .attr('y', 15)
          .attr('text-anchor', 'middle')
          .attr('fill', '#8a9ab0')
          .attr('font-size', '12px')
          .text(`Prev: ${prevValue.toFixed(2)}`);
        
        // Calculation breakdown
        group.append('text')
          .attr('x', i * (cellSize + spacing) + cellSize / 2)
          .attr('y', laneHeight - 10)
          .attr('text-anchor', 'middle')
          .attr('fill', '#8a9ab0')
          .attr('font-size', '10px')
          .text(`${forgetImpact[i].toFixed(2)} + ${inputImpact[i].toFixed(2)}`);
      }
      
      // Change visualization (arrow)
      if (changed) {
        const arrowY = laneHeight / 2;
        const arrowDirection = value > prevValue ? 1 : -1;
        
        group.append('path')
          .attr('d', `M${i * (cellSize + spacing) + cellSize - 5},${arrowY} L${i * (cellSize + spacing) + cellSize + 5},${arrowY} L${i * (cellSize + spacing) + cellSize},${arrowY + 5 * arrowDirection}Z`)
          .attr('fill', value > prevValue ? '#24b47e' : '#ff6b6b');
      }
    });
  }

  // Draw hidden state
  private drawState(x: number, y: number, prevValues: number[], newValues: number[], label: string, color: string): void {
    if (!this.mainGroup) return;
    
    const cellSize = 50;
    const spacing = 15;
    
    // Group for the state
    const group = this.mainGroup.append('g')
      .attr('transform', `translate(${x - (newValues.length * (cellSize + spacing)) / 2}, ${y - cellSize / 2})`);
    
    // Draw label
    group.append('text')
      .attr('x', (newValues.length * (cellSize + spacing)) / 2)
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e1e7f5')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text(label);
    
    // Draw new state cells
    newValues.forEach((value, i) => {
      // Cell background
      group.append('rect')
        .attr('x', i * (cellSize + spacing))
        .attr('y', 0)
        .attr('width', cellSize)
        .attr('height', cellSize)
        .attr('fill', d3.interpolate('#1e3a66', color)(Math.abs(value)))
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('rx', 8)
        .attr('ry', 8);
      
      // Cell value
      group.append('text')
        .attr('x', i * (cellSize + spacing) + cellSize / 2)
        .attr('y', cellSize / 2 + 5)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#ffffff')
        .attr('font-size', '14px')
        .text(value.toFixed(2));
      
      // Index indicator
      group.append('text')
        .attr('x', i * (cellSize + spacing) + cellSize / 2)
        .attr('y', cellSize + 15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#8a9ab0')
        .attr('font-size', '12px')
        .text(`[${i}]`);
      
      // Additional hidden state details
      if (this.showAdvancedView) {
        const prevValue = prevValues[i];
        group.append('text')
          .attr('x', i * (cellSize + spacing) + cellSize / 2)
          .attr('y', -30)
          .attr('text-anchor', 'middle')
          .attr('fill', '#8a9ab0')
          .attr('font-size', '10px')
          .text(`Prev: ${prevValue.toFixed(2)}`);
      }
    });
  }

  // Draw connections between components
  private drawConnections(
    cellWidth: number, 
    cellHeight: number, 
    inputAreaWidth: number, 
    gateAreaWidth: number, 
    stateAreaWidth: number,
    timeStep: LstmTimeStep
  ): void {
    if (!this.mainGroup) return;
    
    // Clear any previous lines to prevent visual artifacts
    this.mainGroup.selectAll('.connection-line').remove();
    
    // Define base positions
    const inputX = inputAreaWidth / 2;
    const gateX = inputAreaWidth + gateAreaWidth / 2;
    const stateX = inputAreaWidth + gateAreaWidth + stateAreaWidth / 2;
    const outputGateX = inputAreaWidth + gateAreaWidth + stateAreaWidth + gateAreaWidth / 2;
    const outputX = cellWidth - inputAreaWidth / 2;
    
    const inputY = cellHeight / 2;
    const forgetGateY = cellHeight * 0.3;
    const inputGateY = cellHeight * 0.3 + cellHeight * 0.2;
    const candidateY = cellHeight * 0.3 + cellHeight * 0.4;
    const cellStateY = cellHeight * 0.3;
    const hiddenStateY = cellHeight * 0.3 + cellHeight * 0.4;
    const outputGateY = cellHeight * 0.3 + cellHeight * 0.2;
    
    // Define connections with data flow strength
    const forgetActivity = d3.mean(timeStep.forgetGate) || 0;
    const inputActivity = d3.mean(timeStep.inputGate) || 0;
    const outputActivity = d3.mean(timeStep.outputGate) || 0;
    
    const connections = [
      // Input to gates
      { 
        x1: inputX, y1: inputY,
        x2: gateX, y2: forgetGateY,
        color: '#8a9ab0',
        width: 2,
        animate: false
      },
      { 
        x1: inputX, y1: inputY,
        x2: gateX, y2: inputGateY,
        color: '#8a9ab0',
        width: 2,
        animate: false 
      },
      { 
        x1: inputX, y1: inputY,
        x2: gateX, y2: candidateY,
        color: '#8a9ab0',
        width: 2,
        animate: false
      },
      
      // Gates to cell state
      {
        x1: gateX, y1: forgetGateY,
        x2: stateX - 120, y2: cellStateY,
        color: '#ff9d45',
        width: 1 + 3 * forgetActivity,
        animate: true
      },
      {
        x1: gateX, y1: inputGateY,
        x2: stateX - 120, y2: cellStateY + 30,
        color: '#00c9ff',
        width: 1 + 3 * inputActivity,
        animate: true
      },
      {
        x1: gateX, y1: candidateY,
        x2: stateX - 140, y2: cellStateY + 50,
        color: '#8bb4fa',
        width: 1 + 3 * inputActivity,
        animate: true
      },
      
      // Cell state to hidden state
      {
        x1: stateX, y1: cellStateY,
        x2: stateX, y2: hiddenStateY,
        color: '#2c5cbd',
        width: 3,
        animate: true
      },
      
      // Output gate to hidden state
      {
        x1: outputGateX, y1: outputGateY,
        x2: stateX + 80, y2: hiddenStateY,
        color: '#7c4dff',
        width: 1 + 3 * outputActivity,
        animate: true
      },
      
      // Hidden state to output
      {
        x1: stateX, y1: hiddenStateY,
        x2: outputX, y2: inputY,
        color: '#4285f4',
        width: 3,
        animate: true
      }
    ];
    
    // Draw connections
    connections.forEach((conn, i) => {
      // Create line
      const line = this.mainGroup.append('line')
        .attr('class', 'connection-line')
        .attr('x1', conn.x1)
        .attr('y1', conn.y1)
        .attr('x2', conn.x2)
        .attr('y2', conn.y2)
        .attr('stroke', conn.color)
        .attr('stroke-width', conn.width)
        .attr('stroke-opacity', 0.7)
        .attr('marker-end', 'url(#arrowhead)');
      
      // Add animation for data flow
      if (conn.animate) {
        this.animateConnection(line, conn.color);
      }
    });
  }

  // Animate connection to show data flow
  private animateConnection(line: any, color: string): void {
    if (!line) return;
    
    line.attr('stroke-dasharray', '5,5')
      .attr('stroke-dashoffset', 0)
      .transition()
      .duration(2000)
      .ease(d3.easeLinear)
      .attr('stroke-dashoffset', 50)
      .on('end', () => this.animateConnection(line, color));
  }

  // Draw explanation text for current step
  private drawStepExplanation(timeStep: LstmTimeStep, cellWidth: number, cellHeight: number): void {
    if (!this.mainGroup) return;
    
    // Calculate average gate activations for explanation
    const forgetGateAvg = d3.mean(timeStep.forgetGate) || 0;
    const inputGateAvg = d3.mean(timeStep.inputGate) || 0;
    const outputGateAvg = d3.mean(timeStep.outputGate) || 0;
    
    // Generate explanation based on time step
    let explanation = '';
    
    if (timeStep.word === 'The' || timeStep.word === 'the') {
      explanation = `Starting to process the sentence. The forget gate is moderately active (${(forgetGateAvg * 100).toFixed(0)}%), since there's no previous context to forget yet. The input gate is adding new information about "The" to the cell state.`;
    } else if (timeStep.word.toLowerCase() === 'clouds') {
      explanation = `Processing "clouds" - an important semantic word. The forget gate remains high (${(forgetGateAvg * 100).toFixed(0)}%), preserving context. The input gate is highly active (${(inputGateAvg * 100).toFixed(0)}%) as this word contains key information that will be needed later to predict related words like "sky".`;
    } else if (timeStep.word.toLowerCase() === 'are') {
      explanation = `For "are", the gates show moderate activity. The cell state maintains information about "clouds" from the previous step, demonstrating how LSTMs preserve important context over multiple time steps.`;
    } else if (timeStep.word.toLowerCase() === 'in') {
      explanation = `For the preposition "in", the gates show typical behavior for a connecting word. Notice how the information about "clouds" is still preserved in the cell state vectors.`;
    } else if (timeStep.word.toLowerCase() === 'sky') {
      explanation = `For "sky", the output gate is highly active (${(outputGateAvg * 100).toFixed(0)}%), as this is a key word. Notice the semantic relationship between "clouds" and "sky" - the LSTM's cell state maintained the semantic information about "clouds" through several time steps, enabling it to properly handle this long-range dependency.`;
    } else {
      // Generic explanation for custom inputs
      explanation = `Processing "${timeStep.word}". The forget gate activity is ${(forgetGateAvg * 100).toFixed(0)}%, the input gate is ${(inputGateAvg * 100).toFixed(0)}% active, and the output gate is ${(outputGateAvg * 100).toFixed(0)}% active. Watch how information flows through the cell state and affects the hidden state output.`;
    }
    
    // Add explanation text panel
    const explanationGroup = this.mainGroup.append('g')
      .attr('transform', `translate(${cellWidth - 330}, ${cellHeight - 155})`);
    
    explanationGroup.append('rect')
      .attr('width', 320)
      .attr('height', 145)
      .attr('fill', '#1e3a66')
      .attr('rx', 8)
      .attr('ry', 8)
      .attr('opacity', 0.9);
    
    explanationGroup.append('text')
      .attr('x', 10)
      .attr('y', 25)
      .attr('fill', '#ffffff')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text('Step Analysis:');
    
    // Add multiline explanation wrapping at 45 characters per line
    this.wrapText(explanationGroup, explanation, 10, 50, 300, 18);
  }

  // Helper to wrap text
  private wrapText(selection: any, text: string, x: number, y: number, width: number, lineHeight: number): void {
    if (!selection) return;
    
    const words = text.split(/\s+/).reverse();
    let word;
    let line: string[] = [];
    let lineNumber = 0;
    let tspan = selection.append('text')
      .attr('x', x)
      .attr('y', y)
      .attr('fill', '#e1e7f5')
      .attr('font-size', '14px')
      .append('tspan')
      .attr('x', x)
      .attr('y', y);
    
    while ((word = words.pop())) {
      line.push(word);
      tspan.text(line.join(' '));
      if (tspan.node().getComputedTextLength() > width) {
        line.pop();
        tspan.text(line.join(' '));
        line = [word];
        lineNumber++;
        tspan = selection.append('text')
          .attr('x', x)
          .attr('y', y)
          .attr('fill', '#e1e7f5')
          .attr('font-size', '14px')
          .attr('dy', lineNumber * lineHeight)
          .append('tspan')
          .attr('x', x)
          .text(word);
      }
    }
  }

  // Play/pause animation
  togglePlayback(): void {
    this.isPlaying = !this.isPlaying;
    
    if (this.isPlaying) {
      this.playAnimation();
    } else {
      this.pauseAnimation();
    }
  }

  // Start animation
  private playAnimation(): void {
    if (this.animationInterval) {
      clearInterval(this.animationInterval);
    }
    
    this.animationInterval = setInterval(() => {
      this.goToNextStep();
      
      // Loop back to start if at the end
      if (this.currentStepIndex >= this.timeSteps.length - 1) {
        this.pauseAnimation();
        this.isPlaying = false;
        
        // Add a small delay before resetting to the beginning
        setTimeout(() => {
          this.resetSimulation();
        }, 1000);
      }
    }, this.animationSpeed / this.playbackSpeed);
  }

  // Pause animation
  private pauseAnimation(): void {
    if (this.animationInterval) {
      clearInterval(this.animationInterval);
      this.animationInterval = null;
    }
  }

  // Go to next step
  goToNextStep(): void {
    if (this.currentStepIndex < this.timeSteps.length - 1) {
      this.currentStepIndex++;
      this.drawLstmCell(this.timeSteps[this.currentStepIndex]);
    }
  }

  // Go to previous step
  goToPrevStep(): void {
    if (this.currentStepIndex > 0) {
      this.currentStepIndex--;
      this.drawLstmCell(this.timeSteps[this.currentStepIndex]);
    }
  }

  // Adjust playback speed
  setPlaybackSpeed(speed: number): void {
    this.playbackSpeed = speed;
    
    // Update animation if playing
    if (this.isPlaying) {
      this.pauseAnimation();
      this.playAnimation();
    }
  }
  
  // Toggle advanced view with more details
  toggleAdvancedView(): void {
    this.showAdvancedView = !this.showAdvancedView;
    this.drawLstmCell(this.timeSteps[this.currentStepIndex]);
  }
  
  // Toggle custom input section
  toggleCustomInput(): void {
    this.showCustomInput = !this.showCustomInput;
  }
  
  // Set active tab
  setActiveTab(tab: string): void {
    this.activeTab = tab;
  }
  
  // Active framework for code examples
  activeFramework: string = 'tensorflow';
  
  // Show selected framework code
  showFramework(framework: string): void {
    this.activeFramework = framework;
  }

  // Reset simulation
  resetSimulation(): void {
    this.pauseAnimation();
    this.currentStepIndex = 0;
    this.drawLstmCell(this.timeSteps[this.currentStepIndex]);
  }
  
  // Download current view as SVG
  downloadSVG(): void {
    if (!this.svg || !isPlatformBrowser(this.platformId)) return;
    
    try {
      // Get SVG content
      const svgElement = this.svg.node();
      const serializer = new XMLSerializer();
      const svgString = serializer.serializeToString(svgElement);
      
      // Create download link
      const link = document.createElement('a');
      link.setAttribute('href', 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgString));
      link.setAttribute('download', `lstm-step-${this.currentStepIndex + 1}.svg`);
      link.style.display = 'none';
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Error downloading SVG:', error);
    }
  }
}

// Interface for LSTM time step data
interface LstmTimeStep {
  word: string;
  index: number;
  input: number[];
  forgetGate: number[];
  inputGate: number[];
  outputGate: number[];
  candidateValues: number[];
  prevCellState: number[];
  cellState: number[];
  prevHiddenState: number[];
  hiddenState: number[];
  forgetGateImpact: number[];
  inputGateImpact: number[];
}