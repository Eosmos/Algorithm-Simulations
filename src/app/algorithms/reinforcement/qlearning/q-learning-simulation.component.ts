import { Component, OnInit, ViewChild, ElementRef, AfterViewInit, NgZone, OnDestroy } from '@angular/core';
import * as d3 from 'd3';

interface GridCell {
  x: number;
  y: number;
  type: 'empty' | 'wall' | 'goal' | 'pitfall';
  reward: number;
}

interface QValue {
  up: number;
  down: number;
  left: number;
  right: number;
}

interface AgentState {
  x: number;
  y: number;
  totalReward: number;
  episode: number;
  step: number;
  isExploring: boolean;
  lastAction?: 'up' | 'down' | 'left' | 'right';
  tdError?: number;
}

@Component({
  selector: 'app-q-learning-simulation',
  templateUrl: './q-learning-simulation.component.html',
  styleUrls: ['./q-learning-simulation.component.scss']
})
export class QLearningSimulationComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('gridContainer') gridContainer!: ElementRef;
  @ViewChild('qValueChart') qValueChart!: ElementRef;
  @ViewChild('rewardChart') rewardChart!: ElementRef;
  @ViewChild('tdErrorChart') tdErrorChart!: ElementRef;

  // Grid configuration
  gridSize = 8;
  cellSize = 60;
  grid: GridCell[][] = [];
  
  // Q-learning parameters
  qTable: Record<string, QValue> = {};
  learningRate = 0.1;
  discountFactor = 0.9;
  epsilon = 0.3;
  epsilonDecayRate = 0.99;
  minEpsilon = 0.01;
  
  // Simulation state
  agent: AgentState = {
    x: 0,
    y: 0,
    totalReward: 0,
    episode: 0,
    step: 0,
    isExploring: false,
  };
  isRunning = false;
  isAutoPlaying = false;
  speed = 300; // milliseconds between steps
  timerId: any = null;
  episodeRewards: number[] = [];
  tdErrors: number[] = [];
  playbackSpeed = 1;
  maxEpisodes = 100;
  isSimulationComplete = false;
  simulationResults = '';
  
  // D3 visualization elements
  svg: any;
  qValueSvg: any;
  rewardChartSvg: any;
  tdErrorChartSvg: any;
  agentElement: any;
  qValueArrows: any = {};
  tooltipDiv: any;
  
  // Simulation explanation
  currentExplanation = 'Welcome to the Q-Learning Simulation. Press Play to start learning automatically, or use the Step button to move through the process manually.';
  
  constructor(private ngZone: NgZone) {}

  ngOnInit(): void {
    this.initializeGrid();
    this.initializeQTable();
  }

  ngAfterViewInit(): void {
    this.initializeVisualization();
    this.initializeQValueVisualization();
    this.initializeRewardChart();
    this.initializeTDErrorChart();
    this.createTooltip();
  }

  ngOnDestroy(): void {
    this.stopSimulation();
  }
  
  generateSimulationResults(): void {
    const avgReward = this.episodeRewards.reduce((sum, value) => sum + value, 0) / this.episodeRewards.length;
    const maxReward = Math.max(...this.episodeRewards);
    const minReward = Math.min(...this.episodeRewards);
    const last10AvgReward = this.episodeRewards.slice(-10).reduce((sum, value) => sum + value, 0) / 10;
    
    // Count successful episodes (ones that reached the goal)
    const successfulEpisodes = this.episodeRewards.filter(reward => reward > 0).length;
    const successRate = (successfulEpisodes / this.maxEpisodes) * 100;
    
    // Calculate average TD error and how it changed
    const avgTDErrorFirst20 = this.tdErrors.slice(0, 20).reduce((sum, value) => sum + Math.abs(value), 0) / 20;
    const avgTDErrorLast20 = this.tdErrors.slice(-20).reduce((sum, value) => sum + Math.abs(value), 0) / 20;
    const tdErrorImprovement = ((avgTDErrorFirst20 - avgTDErrorLast20) / avgTDErrorFirst20) * 100;
    
    // Find best path length
    let shortestSuccessfulPath = Infinity;
    for (let i = 0; i < this.episodeRewards.length; i++) {
      if (this.episodeRewards[i] > 0) {
        // Rough estimate of path length - higher reward means shorter path
        const pathLength = (100 + this.episodeRewards[i]) / 99; // Convert to approximate steps
        shortestSuccessfulPath = Math.min(shortestSuccessfulPath, pathLength);
      }
    }
    
    // Build the results text
    let results = `Q-LEARNING SIMULATION RESULTS\n`;
    results += `==========================\n\n`;
    results += `Total Episodes: ${this.maxEpisodes}\n`;
    results += `Success Rate: ${successRate.toFixed(1)}%\n`;
    results += `Average Reward: ${avgReward.toFixed(2)}\n`;
    results += `Best Episode Reward: ${maxReward.toFixed(2)}\n`;
    results += `Worst Episode Reward: ${minReward.toFixed(2)}\n`;
    results += `Average Reward (last 10 episodes): ${last10AvgReward.toFixed(2)}\n\n`;
    
    results += `LEARNING METRICS\n`;
    results += `---------------\n`;
    results += `TD Error Reduction: ${tdErrorImprovement.toFixed(1)}%\n`;
    results += `Shortest Successful Path: ~${Math.ceil(shortestSuccessfulPath)} steps\n`;
    results += `Final Exploration Rate (ε): ${this.epsilon.toFixed(4)}\n\n`;
    
    results += `Q-VALUES ANALYSIS\n`;
    results += `---------------\n`;
    
    // Find states with highest Q-values
    const stateQValues: { stateKey: string; maxQValue: number; bestAction: string }[] = [];
    for (const stateKey in this.qTable) {
      const qValues = this.qTable[stateKey];
      const actions = ['up', 'down', 'left', 'right'] as const;
      let bestAction: "up" | "down" | "left" | "right" = actions[0];
      let maxQValue = qValues[bestAction];
      
      actions.forEach(action => {
        if (qValues[action] > maxQValue) {
          maxQValue = qValues[action];
          bestAction = action;
        }
      });
      
      if (maxQValue > 0) {
        stateQValues.push({ stateKey, maxQValue, bestAction: bestAction });
      }
    }
    
    // Sort by highest Q-value and get top 5
    stateQValues.sort((a, b) => b.maxQValue - a.maxQValue);
    const top5States = stateQValues.slice(0, 5);
    
    results += `Top 5 Highest Q-Values:\n`;
    top5States.forEach((state, index) => {
      results += `${index + 1}. Position (${state.stateKey}) - Action: ${state.bestAction}, Value: ${state.maxQValue.toFixed(2)}\n`;
    });
    
    results += `\nCONCLUSION\n`;
    results += `----------\n`;
    if (successRate > 80) {
      results += `The agent learned an effective policy, finding the optimal path to the goal in most episodes.\n`;
    } else if (successRate > 50) {
      results += `The agent developed a moderate policy, sometimes finding the goal but not always optimally.\n`;
    } else {
      results += `The agent struggled to learn an effective policy for this environment.\n`;
    }
    
    results += `\nThe learning process demonstrates key Q-learning principles:\n`;
    results += `- Exploration vs. exploitation balance\n`;
    results += `- Value propagation from rewards\n`;
    results += `- Temporal difference learning\n`;
    results += `- Policy convergence over time\n`;
    
    this.simulationResults = results;
  }
  
  copyResultsToClipboard(): void {
    const textArea = document.createElement('textarea');
    textArea.value = this.simulationResults;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    
    // Show temporary toast or notification that could be implemented
    alert('Results copied to clipboard!');
  }

  private initializeGrid(): void {
    // Create a blank grid
    this.grid = Array(this.gridSize)
      .fill(null)
      .map((_, x) =>
        Array(this.gridSize)
          .fill(null)
          .map((_, y) => ({
            x,
            y,
            type: 'empty',
            reward: 0
          }))
      );

    // Set walls
    [
      { x: 1, y: 2 },
      { x: 2, y: 2 },
      { x: 3, y: 2 },
      { x: 4, y: 4 },
      { x: 5, y: 4 },
      { x: 6, y: 4 },
      { x: 3, y: 5 },
      { x: 3, y: 6 },
    ].forEach(pos => {
      if (pos.x < this.gridSize && pos.y < this.gridSize) {
        this.grid[pos.x][pos.y].type = 'wall';
      }
    });

    // Set goal
    this.grid[7][7].type = 'goal';
    this.grid[7][7].reward = 100;

    // Set pitfalls
    [
      { x: 2, y: 5 },
      { x: 5, y: 2 },
      { x: 6, y: 6 },
    ].forEach(pos => {
      if (pos.x < this.gridSize && pos.y < this.gridSize && this.grid[pos.x][pos.y].type === 'empty') {
        this.grid[pos.x][pos.y].type = 'pitfall';
        this.grid[pos.x][pos.y].reward = -50;
      }
    });

    // Set small negative reward for empty cells to encourage finding shortest path
    for (let x = 0; x < this.gridSize; x++) {
      for (let y = 0; y < this.gridSize; y++) {
        if (this.grid[x][y].type === 'empty') {
          this.grid[x][y].reward = -1;
        }
      }
    }

    // Position agent at start
    this.resetAgent();
  }

  private initializeQTable(): void {
    // Initialize Q-values to zeros for all state-action pairs
    for (let x = 0; x < this.gridSize; x++) {
      for (let y = 0; y < this.gridSize; y++) {
        if (this.grid[x][y].type !== 'wall') {
          const stateKey = `${x},${y}`;
          this.qTable[stateKey] = {
            up: 0,
            down: 0,
            left: 0,
            right: 0
          };
        }
      }
    }
  }

  private resetAgent(): void {
    // Start position (top-left corner)
    this.agent.x = 0;
    this.agent.y = 0;
    this.agent.totalReward = 0;
    this.agent.step = 0;
    this.agent.isExploring = false;
  }

  private getStateKey(x: number, y: number): string {
    return `${x},${y}`;
  }

  private getValidActions(x: number, y: number): ('up' | 'down' | 'left' | 'right')[] {
    const actions: ('up' | 'down' | 'left' | 'right')[] = [];
    
    // Check each direction
    if (y > 0 && this.grid[x][y - 1].type !== 'wall') actions.push('up');
    if (y < this.gridSize - 1 && this.grid[x][y + 1].type !== 'wall') actions.push('down');
    if (x > 0 && this.grid[x - 1][y].type !== 'wall') actions.push('left');
    if (x < this.gridSize - 1 && this.grid[x + 1][y].type !== 'wall') actions.push('right');
    
    return actions;
  }

  private chooseAction(x: number, y: number): { action: 'up' | 'down' | 'left' | 'right', isExploring: boolean } {
    const stateKey = this.getStateKey(x, y);
    const validActions = this.getValidActions(x, y);
    
    // With probability epsilon, explore (random action)
    if (Math.random() < this.epsilon) {
      const randomIndex = Math.floor(Math.random() * validActions.length);
      const randomAction = validActions[randomIndex];
      return { action: randomAction, isExploring: true };
    }
    
    // Otherwise, exploit (greedy action - choose best Q-value)
    const initialAction = validActions[0];
    let bestAction: 'up' | 'down' | 'left' | 'right' = initialAction;
    let bestQValue = this.qTable[stateKey][initialAction];
    
    for (const action of validActions) {
      if (this.qTable[stateKey][action] > bestQValue) {
        bestQValue = this.qTable[stateKey][action];
        bestAction = action;
      }
    }
    
    return { action: bestAction, isExploring: false };
  }

  private getNextState(x: number, y: number, action: 'up' | 'down' | 'left' | 'right'): { nextX: number, nextY: number } {
    let nextX = x;
    let nextY = y;
    
    switch (action) {
      case 'up':
        if (y > 0 && this.grid[x][y - 1].type !== 'wall') nextY = y - 1;
        break;
      case 'down':
        if (y < this.gridSize - 1 && this.grid[x][y + 1].type !== 'wall') nextY = y + 1;
        break;
      case 'left':
        if (x > 0 && this.grid[x - 1][y].type !== 'wall') nextX = x - 1;
        break;
      case 'right':
        if (x < this.gridSize - 1 && this.grid[x + 1][y].type !== 'wall') nextX = x + 1;
        break;
    }
    
    return { nextX, nextY };
  }

  private updateQValue(x: number, y: number, action: 'up' | 'down' | 'left' | 'right', nextX: number, nextY: number, reward: number): number {
    const stateKey = this.getStateKey(x, y);
    const nextStateKey = this.getStateKey(nextX, nextY);
    
    // Find max Q-value for next state
    let maxNextQ = -Infinity;
    if (this.qTable[nextStateKey]) {
      const nextQValues = Object.values(this.qTable[nextStateKey]);
      maxNextQ = Math.max(...nextQValues);
    } else {
      maxNextQ = 0; // Terminal state
    }
    
    // Calculate TD target
    const tdTarget = reward + this.discountFactor * maxNextQ;
    
    // Calculate TD error
    const tdError = tdTarget - this.qTable[stateKey][action];
    
    // Update Q-value using the Q-learning update rule
    this.qTable[stateKey][action] += this.learningRate * tdError;
    
    return tdError;
  }

  private isTerminalState(x: number, y: number): boolean {
    return this.grid[x][y].type === 'goal' || this.grid[x][y].type === 'pitfall';
  }

  private decayEpsilon(): void {
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecayRate);
  }

  // Step through one action in the simulation
  step(): void {
    const { x, y } = this.agent;
    
    // Choose an action
    const { action, isExploring } = this.chooseAction(x, y);
    
    // Get next state
    const { nextX, nextY } = this.getNextState(x, y, action);
    
    // Get reward
    const reward = this.grid[nextX][nextY].reward;
    
    // Update Q-value
    const tdError = this.updateQValue(x, y, action, nextX, nextY, reward);
    
    // Update agent state
    this.agent.x = nextX;
    this.agent.y = nextY;
    this.agent.totalReward += reward;
    this.agent.step++;
    this.agent.isExploring = isExploring;
    this.agent.lastAction = action;
    this.agent.tdError = tdError;
    
    // Record TD error for plotting
    this.tdErrors.push(tdError);
    
    // Update explanation
    this.updateExplanation(action, isExploring, nextX, nextY, reward, tdError);
    
    // Check if terminal state
    if (this.isTerminalState(nextX, nextY) || this.agent.step > 100) {
      this.episodeRewards.push(this.agent.totalReward);
      this.agent.episode++;
      
      // Decay epsilon
      this.decayEpsilon();
      
      // Check if we've reached max episodes
      if (this.agent.episode >= this.maxEpisodes) {
        this.stopSimulation();
        this.currentExplanation = `Simulation complete! Completed ${this.maxEpisodes} episodes.`;
        this.isSimulationComplete = true;
        this.generateSimulationResults();
        return;
      }
      
      // Reset agent for next episode
      this.resetAgent();
    }
    
    // Update visualization
    this.updateVisualization();
  }

  private updateExplanation(
    action: 'up' | 'down' | 'left' | 'right',
    isExploring: boolean,
    nextX: number,
    nextY: number,
    reward: number,
    tdError: number
  ): void {
    const stateKey = this.getStateKey(this.agent.x, this.agent.y);
    const qValue = this.qTable[stateKey][action].toFixed(2);
    
    let explanation = `Episode ${this.agent.episode + 1}, Step ${this.agent.step}: `;
    
    if (isExploring) {
      explanation += `EXPLORING: Randomly chose action "${action}". `;
    } else {
      explanation += `EXPLOITING: Chose best action "${action}" with Q-value ${qValue}. `;
    }
    
    explanation += `Moved to position (${nextX}, ${nextY}) and received reward ${reward}. `;
    explanation += `TD Error: ${tdError.toFixed(4)}. `;
    
    if (this.grid[nextX][nextY].type === 'goal') {
      explanation += 'Reached the goal! Starting new episode.';
    } else if (this.grid[nextX][nextY].type === 'pitfall') {
      explanation += 'Fell into a pitfall! Starting new episode.';
    }
    
    this.currentExplanation = explanation;
  }

  // Control functions
  toggleAutoplay(): void {
    if (this.isAutoPlaying) {
      this.stopAutoplay();
    } else {
      this.startAutoplay();
    }
  }

  startAutoplay(): void {
    this.isRunning = true;
    this.isAutoPlaying = true;
    
    this.ngZone.runOutsideAngular(() => {
      this.timerId = setInterval(() => {
        this.ngZone.run(() => {
          this.step();
          this.updateRewardChart();
          this.updateTDErrorChart();
        });
      }, this.speed / this.playbackSpeed);
    });
  }

  stopAutoplay(): void {
    this.isAutoPlaying = false;
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  startSimulation(): void {
    if (!this.isRunning) {
      this.isRunning = true;
      this.currentExplanation = 'Simulation started. The agent is learning...';
    }
  }

  stopSimulation(): void {
    this.isRunning = false;
    this.stopAutoplay();
    this.currentExplanation = 'Simulation paused. Press Play to continue.';
  }

  resetSimulation(): void {
    this.stopSimulation();
    this.initializeQTable();
    this.resetAgent();
    this.agent.episode = 0;
    this.epsilon = 0.3;
    this.episodeRewards = [];
    this.tdErrors = [];
    this.isSimulationComplete = false;
    this.simulationResults = '';
    this.updateVisualization();
    this.updateRewardChart();
    this.updateTDErrorChart();
    this.currentExplanation = 'Simulation reset. Press Play to start learning.';
  }

  changeSpeed(factor: number): void {
    this.playbackSpeed = factor;
    if (this.isAutoPlaying) {
      this.stopAutoplay();
      this.startAutoplay();
    }
  }

  manualStep(): void {
    if (!this.isRunning) {
      this.startSimulation();
    }
    this.step();
    this.updateRewardChart();
    this.updateTDErrorChart();
  }

  // Visualization methods
  private initializeVisualization(): void {
    const containerWidth = this.gridSize * this.cellSize;
    const containerHeight = this.gridSize * this.cellSize;
    
    this.svg = d3.select(this.gridContainer.nativeElement)
      .append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)
      .style('background', '#162a4a');
    
    // Draw grid cells
    for (let x = 0; x < this.gridSize; x++) {
      for (let y = 0; y < this.gridSize; y++) {
        const cell = this.grid[x][y];
        let color = '#1e3a66'; // Default empty cell
        
        switch (cell.type) {
          case 'wall':
            color = '#0c1428';
            break;
          case 'goal':
            color = '#24b47e';
            break;
          case 'pitfall':
            color = '#ff6b6b';
            break;
        }
        
        this.svg.append('rect')
          .attr('x', x * this.cellSize)
          .attr('y', y * this.cellSize)
          .attr('width', this.cellSize)
          .attr('height', this.cellSize)
          .attr('fill', color)
          .attr('stroke', '#2a4980')
          .attr('stroke-width', 1)
          .attr('rx', 8)
          .attr('ry', 8)
          .on('mouseover', (event: MouseEvent) => {
            if (cell.type !== 'wall') {
              const stateKey = this.getStateKey(x, y);
              const qValues = this.qTable[stateKey];
              
              let tooltipContent = `<div><strong>Position: (${x}, ${y})</strong></div>`;
              tooltipContent += `<div>Type: ${cell.type}</div>`;
              tooltipContent += `<div>Reward: ${cell.reward}</div>`;
              tooltipContent += `<div class="q-values">Q-Values:</div>`;
              tooltipContent += `<div>↑ Up: ${qValues?.up?.toFixed(2) || 'N/A'}</div>`;
              tooltipContent += `<div>↓ Down: ${qValues?.down?.toFixed(2) || 'N/A'}</div>`;
              tooltipContent += `<div>← Left: ${qValues?.left?.toFixed(2) || 'N/A'}</div>`;
              tooltipContent += `<div>→ Right: ${qValues?.right?.toFixed(2) || 'N/A'}</div>`;
              
              this.tooltipDiv.transition()
                .duration(200)
                .style('opacity', 0.9);
              
              this.tooltipDiv.html(tooltipContent)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
            }
          })
          .on('mouseout', () => {
            this.tooltipDiv.transition()
              .duration(500)
              .style('opacity', 0);
          });
          
        // Add coordinate text
        this.svg.append('text')
          .attr('x', x * this.cellSize + 5)
          .attr('y', y * this.cellSize + 15)
          .attr('font-size', '10px')
          .attr('fill', '#8a9ab0')
          .text(`(${x},${y})`);
          
        // Add reward text for non-empty cells
        if (cell.type !== 'empty' && cell.type !== 'wall') {
          this.svg.append('text')
            .attr('x', x * this.cellSize + this.cellSize / 2)
            .attr('y', y * this.cellSize + this.cellSize / 2 + 5)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .attr('fill', '#e1e7f5')
            .text(`R: ${cell.reward}`);
        }
          
        // Add visual cues for cell types
        if (cell.type === 'goal') {
          this.svg.append('text')
            .attr('x', x * this.cellSize + this.cellSize / 2)
            .attr('y', y * this.cellSize + this.cellSize / 2 - 5)
            .attr('text-anchor', 'middle')
            .attr('font-size', '16px')
            .attr('fill', '#e1e7f5')
            .text('🎯');
        } else if (cell.type === 'pitfall') {
          this.svg.append('text')
            .attr('x', x * this.cellSize + this.cellSize / 2)
            .attr('y', y * this.cellSize + this.cellSize / 2 - 5)
            .attr('text-anchor', 'middle')
            .attr('font-size', '16px')
            .attr('fill', '#e1e7f5')
            .text('⚠️');
        }
      }
    }
    
    // Create agent
    this.agentElement = this.svg.append('circle')
      .attr('cx', this.agent.x * this.cellSize + this.cellSize / 2)
      .attr('cy', this.agent.y * this.cellSize + this.cellSize / 2)
      .attr('r', this.cellSize / 3)
      .attr('fill', '#4285f4')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2);
      
    // Create Q-value arrows container
    this.updateQValueArrows();
  }

  private updateQValueArrows(): void {
    // Remove existing arrows
    this.svg.selectAll('.q-arrow').remove();
    
    // Draw arrows for each cell based on Q-values
    for (let x = 0; x < this.gridSize; x++) {
      for (let y = 0; y < this.gridSize; y++) {
        if (this.grid[x][y].type !== 'wall') {
          const stateKey = this.getStateKey(x, y);
          const qValues = this.qTable[stateKey];
          
          if (!qValues) continue;
          
          // Find max Q-value for this state
          const actions = ['up', 'down', 'left', 'right'] as const;
          let bestAction: 'up' | 'down' | 'left' | 'right' = actions[0];
          let maxQ = qValues[bestAction];
          
          actions.forEach(action => {
            if (qValues[action] > maxQ) {
              maxQ = qValues[action];
              bestAction = action;
            }
          });
          
          // Only draw arrow if Q-value is positive
          if (maxQ > 0) {
            const centerX = x * this.cellSize + this.cellSize / 2;
            const centerY = y * this.cellSize + this.cellSize / 2;
            const arrowLength = (Math.min(Math.abs(maxQ) / 10, 1)) * (this.cellSize / 3);
            
            // Draw arrow based on best action
            let dx = 0, dy = 0;
            switch (bestAction as 'up' | 'down' | 'left' | 'right') {
              case 'up': dy = -arrowLength; break;
              case 'down': dy = arrowLength; break;
              case 'left': dx = -arrowLength; break;
              case 'right': dx = arrowLength; break;
            }
            
            // Calculate arrow points
            const arrowX2 = centerX + dx;
            const arrowY2 = centerY + dy;
            
            // Draw the line
            this.svg.append('line')
              .attr('class', 'q-arrow')
              .attr('x1', centerX)
              .attr('y1', centerY)
              .attr('x2', arrowX2)
              .attr('y2', arrowY2)
              .attr('stroke', '#00c9ff')
              .attr('stroke-width', 2)
              .attr('marker-end', 'url(#arrowhead)');
              
            // Add arrowhead definition if it doesn't exist
            if (!this.svg.select('defs').node()) {
              this.svg.append('defs')
                .append('marker')
                .attr('id', 'arrowhead')
                .attr('markerWidth', 10)
                .attr('markerHeight', 7)
                .attr('refX', 10)
                .attr('refY', 3.5)
                .attr('orient', 'auto')
                .append('polygon')
                .attr('points', '0 0, 10 3.5, 0 7')
                .attr('fill', '#00c9ff');
            }
          }
        }
      }
    }
  }

  private initializeQValueVisualization(): void {
    const containerWidth = 300;
    const containerHeight = 200;
    
    this.qValueSvg = d3.select(this.qValueChart.nativeElement)
      .append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)
      .style('background', '#162a4a')
      .style('border-radius', '12px');
      
    // Title
    this.qValueSvg.append('text')
      .attr('x', containerWidth / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e1e7f5')
      .text('Q-Values for Current State');
      
    // Will be updated during simulation
  }

  private initializeRewardChart(): void {
    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const containerWidth = 400;
    const containerHeight = 200;
    const width = containerWidth - margin.left - margin.right;
    const height = containerHeight - margin.top - margin.bottom;
    
    this.rewardChartSvg = d3.select(this.rewardChart.nativeElement)
      .append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)
      .style('background', '#162a4a')
      .style('border-radius', '12px')
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
      
    // Title
    this.rewardChartSvg.append('text')
      .attr('x', width / 2)
      .attr('y', -5)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e1e7f5')
      .text('Total Reward per Episode');
      
    // X Axis
    this.rewardChartSvg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(d3.scaleLinear().range([0, width])))
      .style('color', '#8a9ab0');
      
    // Y Axis
    this.rewardChartSvg.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(d3.scaleLinear().range([height, 0])))
      .style('color', '#8a9ab0');
      
    // X Axis Label
    this.rewardChartSvg.append('text')
      .attr('x', width / 2)
      .attr('y', height + 35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#8a9ab0')
      .text('Episode');
      
    // Y Axis Label
    this.rewardChartSvg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -30)
      .attr('text-anchor', 'middle')
      .attr('fill', '#8a9ab0')
      .text('Total Reward');
  }

  private initializeTDErrorChart(): void {
    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const containerWidth = 400;
    const containerHeight = 200;
    const width = containerWidth - margin.left - margin.right;
    const height = containerHeight - margin.top - margin.bottom;
    
    this.tdErrorChartSvg = d3.select(this.tdErrorChart.nativeElement)
      .append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)
      .style('background', '#162a4a')
      .style('border-radius', '12px')
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
      
    // Title
    this.tdErrorChartSvg.append('text')
      .attr('x', width / 2)
      .attr('y', -5)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e1e7f5')
      .text('TD Error Over Time');
      
    // X Axis
    this.tdErrorChartSvg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(d3.scaleLinear().range([0, width])))
      .style('color', '#8a9ab0');
      
    // Y Axis
    this.tdErrorChartSvg.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(d3.scaleLinear().range([height, 0])))
      .style('color', '#8a9ab0');
      
    // X Axis Label
    this.tdErrorChartSvg.append('text')
      .attr('x', width / 2)
      .attr('y', height + 35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#8a9ab0')
      .text('Step');
      
    // Y Axis Label
    this.tdErrorChartSvg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -30)
      .attr('text-anchor', 'middle')
      .attr('fill', '#8a9ab0')
      .text('TD Error');
  }

  private createTooltip(): void {
    this.tooltipDiv = d3.select('body')
      .append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background-color', '#162a4a')
      .style('color', '#e1e7f5')
      .style('border', '1px solid #2a4980')
      .style('border-radius', '8px')
      .style('padding', '10px')
      .style('pointer-events', 'none')
      .style('z-index', '10');
  }

  private updateVisualization(): void {
    // Update agent position with animation
    this.agentElement
      .transition()
      .duration(200)
      .attr('cx', this.agent.x * this.cellSize + this.cellSize / 2)
      .attr('cy', this.agent.y * this.cellSize + this.cellSize / 2)
      .attr('fill', this.agent.isExploring ? '#ff9d45' : '#4285f4'); // Orange when exploring, blue when exploiting
      
    // Create "exploration" flash effect
    if (this.agent.isExploring) {
      this.svg.append('circle')
        .attr('cx', this.agent.x * this.cellSize + this.cellSize / 2)
        .attr('cy', this.agent.y * this.cellSize + this.cellSize / 2)
        .attr('r', this.cellSize / 3)
        .attr('fill', 'none')
        .attr('stroke', '#ff9d45')
        .attr('stroke-width', 3)
        .attr('opacity', 1)
        .transition()
        .duration(500)
        .attr('r', this.cellSize / 2)
        .attr('opacity', 0)
        .remove();
    }
    
    // Update Q-value arrows
    this.updateQValueArrows();
    
    // Update the Q-value chart for current state
    this.updateQValueChart();
  }

  private updateQValueChart(): void {
    const containerWidth = 300;
    const containerHeight = 200;
    const margin = { top: 30, right: 20, bottom: 30, left: 20 };
    const width = containerWidth - margin.left - margin.right;
    const height = containerHeight - margin.top - margin.bottom;
    
    // Clear previous content
    this.qValueSvg.selectAll('.q-chart-content').remove();
    
    const stateKey = this.getStateKey(this.agent.x, this.agent.y);
    const qValues = this.qTable[stateKey];
    
    if (!qValues) return;
    
    const chartGroup = this.qValueSvg.append('g')
      .attr('class', 'q-chart-content')
      .attr('transform', `translate(${margin.left},${margin.top})`);
      
    const actions = ['up', 'down', 'left', 'right'];
    const data = actions.map(action => ({
      action,
      value: qValues[action as keyof QValue]
    }));
    
    // Find max absolute Q-value for scaling
    const maxAbsValue = Math.max(
      Math.abs(d3.min(data, d => d.value) || 0),
      Math.abs(d3.max(data, d => d.value) || 0),
      0.1 // Minimum scale
    );
    
    // X scale for actions
    const x = d3.scaleBand()
      .domain(actions)
      .range([0, width])
      .padding(0.3);
      
    // Y scale for Q-values
    const y = d3.scaleLinear()
      .domain([-maxAbsValue, maxAbsValue])
      .range([height, 0]);
      
    // Add X axis
    chartGroup.append('g')
      .attr('transform', `translate(0,${y(0)})`) // Center at 0
      .call(d3.axisBottom(x))
      .style('color', '#8a9ab0');
      
    // Add Y axis
    chartGroup.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .style('color', '#8a9ab0');
      
    // Define interface for data items
    interface QValueDataItem {
      action: string;
      value: number;
    }
    
    // Draw the bars
    chartGroup.selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', (d: QValueDataItem) => x(d.action) || 0)
      .attr('width', x.bandwidth())
      .attr('y', (d: QValueDataItem) => d.value >= 0 ? y(d.value) : y(0))
      .attr('height', (d: QValueDataItem) => Math.abs(y(d.value) - y(0)))
      .attr('fill', (d: QValueDataItem) => {
        // Use a gradient based on value
        if (d.value > 0) {
          return '#00c9ff'; // Positive - Cyan
        } else {
          return '#ff6b6b'; // Negative - Red
        }
      });
      
    // Add value labels
    chartGroup.selectAll('.value-label')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'value-label')
      .attr('x', (d: QValueDataItem) => (x(d.action) || 0) + x.bandwidth() / 2)
      .attr('y', (d: QValueDataItem) => d.value >= 0 ? y(d.value) - 5 : y(d.value) + 15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e1e7f5')
      .attr('font-size', '10px')
      .text((d: QValueDataItem) => d.value.toFixed(2));
      
    // Highlight the last action taken
    if (this.agent.lastAction) {
      chartGroup.append('rect')
        .attr('x', x(this.agent.lastAction) || 0)
        .attr('y', 0)
        .attr('width', x.bandwidth())
        .attr('height', height)
        .attr('fill', 'none')
        .attr('stroke', '#7c4dff')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '3,3');
    }
  }

  private updateRewardChart(): void {
    if (this.episodeRewards.length === 0) return;
    
    const containerWidth = 400;
    const containerHeight = 200;
    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = containerWidth - margin.left - margin.right;
    const height = containerHeight - margin.top - margin.bottom;
    
    // X scale
    const x = d3.scaleLinear()
      .domain([0, this.episodeRewards.length])
      .range([0, width]);
      
    // Y scale with padding
    const minReward = Math.min(...this.episodeRewards);
    const maxReward = Math.max(...this.episodeRewards);
    const padding = Math.abs(maxReward - minReward) * 0.1;
    
    const y = d3.scaleLinear()
      .domain([minReward - padding, maxReward + padding])
      .range([height, 0]);
      
    // Update axes
    this.rewardChartSvg.select('.x-axis')
      .call(d3.axisBottom(x))
      .style('color', '#8a9ab0');
      
    this.rewardChartSvg.select('.y-axis')
      .call(d3.axisLeft(y))
      .style('color', '#8a9ab0');
      
    // Remove existing line and area
    this.rewardChartSvg.selectAll('.reward-line').remove();
    this.rewardChartSvg.selectAll('.reward-area').remove();
    this.rewardChartSvg.selectAll('.reward-point').remove();
    
    // Create line function
    const line = d3.line<number>()
      .x((d, i) => x(i))
      .y(d => y(d))
      .curve(d3.curveMonotoneX);
      
    // Create area function for the gradient under the line
    const area = d3.area<number>()
      .x((d, i) => x(i))
      .y0(height)
      .y1(d => y(d))
      .curve(d3.curveMonotoneX);
      
    // Add the area fill
    this.rewardChartSvg.append('path')
      .datum(this.episodeRewards)
      .attr('class', 'reward-area')
      .attr('fill', 'url(#reward-gradient)')
      .attr('d', area);
      
    // Add reward gradient if it doesn't exist
    if (!this.rewardChartSvg.select('#reward-gradient').node()) {
      const gradient = this.rewardChartSvg.append('defs')
        .append('linearGradient')
        .attr('id', 'reward-gradient')
        .attr('gradientUnits', 'userSpaceOnUse')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', 0)
        .attr('y2', height);
        
      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', '#7c4dff')
        .attr('stop-opacity', 0.8);
        
      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', '#7c4dff')
        .attr('stop-opacity', 0.1);
    }
    
    // Add the line
    this.rewardChartSvg.append('path')
      .datum(this.episodeRewards)
      .attr('class', 'reward-line')
      .attr('fill', 'none')
      .attr('stroke', '#7c4dff')
      .attr('stroke-width', 2)
      .attr('d', line);
      
    // Add points
    this.rewardChartSvg.selectAll('.reward-point')
      .data(this.episodeRewards)
      .enter()
      .append('circle')
      .attr('class', 'reward-point')
      .attr('cx', (d: number, i: number) => x(i))
      .attr('cy', (d: number) => y(d))
      .attr('r', 3)
      .attr('fill', '#00c9ff');
  }

  private updateTDErrorChart(): void {
    if (this.tdErrors.length === 0) return;
    
    // Only plot the last 100 TD errors
    const tdErrorsToPlot = this.tdErrors.slice(-100);
    
    const containerWidth = 400;
    const containerHeight = 200;
    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = containerWidth - margin.left - margin.right;
    const height = containerHeight - margin.top - margin.bottom;
    
    // X scale
    const x = d3.scaleLinear()
      .domain([Math.max(0, this.tdErrors.length - 100), this.tdErrors.length])
      .range([0, width]);
      
    // Y scale with padding
    const minError = Math.min(...tdErrorsToPlot);
    const maxError = Math.max(...tdErrorsToPlot);
    const padding = Math.abs(maxError - minError) * 0.1;
    
    const y = d3.scaleLinear()
      .domain([minError - padding, maxError + padding])
      .range([height, 0]);
      
    // Update axes
    this.tdErrorChartSvg.select('.x-axis')
      .call(d3.axisBottom(x))
      .style('color', '#8a9ab0');
      
    this.tdErrorChartSvg.select('.y-axis')
      .call(d3.axisLeft(y))
      .style('color', '#8a9ab0');
      
    // Remove existing line
    this.tdErrorChartSvg.selectAll('.td-error-line').remove();
    
    // Create line function
    const line = d3.line<number>()
      .x((d, i) => x(this.tdErrors.length - tdErrorsToPlot.length + i))
      .y(d => y(d))
      .curve(d3.curveMonotoneX);
      
    // Add the line
    this.tdErrorChartSvg.append('path')
      .datum(tdErrorsToPlot)
      .attr('class', 'td-error-line')
      .attr('fill', 'none')
      .attr('stroke', '#ff9d45')
      .attr('stroke-width', 2)
      .attr('d', line);
  }
}