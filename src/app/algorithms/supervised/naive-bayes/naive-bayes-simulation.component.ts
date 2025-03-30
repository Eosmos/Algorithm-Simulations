import { Component, ElementRef, OnInit, ViewChild, AfterViewInit, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as d3 from 'd3';

@Component({
  selector: 'app-naive-bayes-simulation',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './naive-bayes-simulation.component.html',
  styleUrls: ['./naive-bayes-simulation.component.scss']
})
export class NaiveBayesSimulationComponent implements OnInit, AfterViewInit {
  @ViewChild('distributionChart') distributionChart!: ElementRef;
  @ViewChild('decisionBoundary') decisionBoundary!: ElementRef;
  @ViewChild('textClassification') textClassification!: ElementRef;

  // Simulation state
  currentStep = 0;
  isPlaying = false;
  playInterval: any = null;
  animationSpeed = 1500; // ms between steps
  
  // Active simulation tab
  activeTab = 'gaussian';
  
  // Gaussian simulation data
  featureData = {
    male: { height: { mean: 175, stdDev: 7 }, weight: { mean: 78, stdDev: 10 } },
    female: { height: { mean: 162, stdDev: 6 }, weight: { mean: 62, stdDev: 9 } }
  };
  testPoint = { height: 170, weight: 70 };
  priorProbabilities = { male: 0.5, female: 0.5 };
  classProbabilities = { male: 0, female: 0 };
  
  // Text classification data
  emailData = [
    { label: 'spam', words: ['free', 'win', 'prize', 'million', 'winner'] },
    { label: 'spam', words: ['free', 'cash', 'prize', 'claim', 'today'] },
    { label: 'spam', words: ['win', 'lottery', 'million', 'claim', 'prize'] },
    { label: 'not_spam', words: ['meeting', 'tomorrow', 'project', 'report', 'team'] },
    { label: 'not_spam', words: ['hello', 'lunch', 'meeting', 'office', 'regards'] },
    { label: 'not_spam', words: ['report', 'data', 'analysis', 'results', 'project'] }
  ];
  testEmail = ['free', 'win', 'hello'];
  wordProbabilities: { 
    spam: { [key: string]: number },
    not_spam: { [key: string]: number } 
  } = { spam: {}, not_spam: {} };
  emailProbabilities = { spam: 0, not_spam: 0 };
  
  // Decision boundary data
  points: Array<{x: number, y: number, class: string}> = [];
  gridPoints: Array<{x: number, y: number, probA: number, predicted: string}> = [];
  
  constructor(private ngZone: NgZone) {}

  ngOnInit(): void {
    this.calculateWordProbabilities();
    this.generatePoints();
    this.generateGridPoints();
  }

  ngAfterViewInit(): void {
    setTimeout(() => {
      this.initDistributionChart();
      this.initDecisionBoundary();
      this.initTextClassification();
    }, 0);
  }

  // ----- Simulation Control Methods -----
  
  setActiveTab(tab: string): void {
    this.activeTab = tab;
    this.resetSimulation();
  }
  
  playSimulation(): void {
    if (this.isPlaying) return;
    
    this.isPlaying = true;
    this.ngZone.runOutsideAngular(() => {
      this.playInterval = setInterval(() => {
        this.ngZone.run(() => {
          if (this.advanceStep()) {
            clearInterval(this.playInterval);
            this.isPlaying = false;
          }
        });
      }, this.animationSpeed);
    });
  }
  
  pauseSimulation(): void {
    if (this.playInterval) {
      clearInterval(this.playInterval);
      this.isPlaying = false;
    }
  }
  
  resetSimulation(): void {
    this.pauseSimulation();
    this.currentStep = 0;
    this.updateSimulation();
  }
  
  previousStep(): void {
    if (this.currentStep > 0) {
      this.currentStep--;
      this.updateSimulation();
    }
  }
  
  nextStep(): void {
    this.advanceStep();
  }
  
  advanceStep(): boolean {
    const maxSteps = this.activeTab === 'gaussian' ? 5 : (this.activeTab === 'text' ? 4 : 3);
    
    if (this.currentStep < maxSteps) {
      this.currentStep++;
      this.updateSimulation();
      return false;
    }
    return true; // Reached the end
  }
  
  updateSimulation(): void {
    switch (this.activeTab) {
      case 'gaussian':
        this.updateGaussianSimulation();
        break;
      case 'text':
        this.updateTextSimulation();
        break;
      case 'decision':
        this.updateDecisionBoundarySimulation();
        break;
    }
  }
  
  // ----- Gaussian Distribution Simulation Methods -----
  
  initDistributionChart(): void {
    const svg = d3.select(this.distributionChart.nativeElement);
    svg.selectAll('*').remove();
    
    // Setup container and margins
    const margin = { top: 20, right: 30, bottom: 50, left: 50 };
    const width = +svg.attr('width') - margin.left - margin.right;
    const height = +svg.attr('height') - margin.top - margin.bottom;
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // X axis
    const x = d3.scaleLinear()
      .domain([150, 190]) // Height range for the plot
      .range([0, width]);
    
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x));
    
    g.append('text')
      .attr('transform', `translate(${width/2},${height + 40})`)
      .style('text-anchor', 'middle')
      .text('Height (cm)');
    
    // Y axis
    const y = d3.scaleLinear()
      .domain([0, 0.07]) // Probability density
      .range([height, 0]);
    
    g.append('g')
      .call(d3.axisLeft(y));
    
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -40)
      .attr('x', -height/2)
      .style('text-anchor', 'middle')
      .text('Probability Density');
    
    // Male distribution line
    const maleLine = d3.line<number>()
      .x(d => x(d))
      .y(d => y(this.normalPDF(d, this.featureData.male.height.mean, this.featureData.male.height.stdDev)));
    
    const maleData = d3.range(150, 190, 0.5);
    
    g.append('path')
      .datum(maleData)
      .attr('fill', 'none')
      .attr('stroke', '#4285f4')
      .attr('stroke-width', 2)
      .attr('d', maleLine);
    
    // Female distribution line
    const femaleLine = d3.line<number>()
      .x(d => x(d))
      .y(d => y(this.normalPDF(d, this.featureData.female.height.mean, this.featureData.female.height.stdDev)));
    
    g.append('path')
      .datum(maleData) // Reuse the same range
      .attr('fill', 'none')
      .attr('stroke', '#7c4dff')
      .attr('stroke-width', 2)
      .attr('d', femaleLine);
    
    // Legend
    g.append('circle').attr('cx', width - 120).attr('cy', 20).attr('r', 6).style('fill', '#4285f4');
    g.append('text').attr('x', width - 100).attr('y', 20).text('Male').style('font-size', '15px').attr('alignment-baseline', 'middle').style('fill', '#e1e7f5');
    g.append('circle').attr('cx', width - 120).attr('cy', 50).attr('r', 6).style('fill', '#7c4dff');
    g.append('text').attr('x', width - 100).attr('y', 50).text('Female').style('font-size', '15px').attr('alignment-baseline', 'middle').style('fill', '#e1e7f5');
    
    // Add the test point
    g.append('circle')
      .attr('id', 'test-point')
      .attr('cx', x(this.testPoint.height))
      .attr('cy', height)
      .attr('r', 8)
      .style('fill', '#00c9ff')
      .style('opacity', 0);
    
    // Add vertical lines from test point to curves
    g.append('line')
      .attr('id', 'male-prob-line')
      .attr('x1', x(this.testPoint.height))
      .attr('y1', height)
      .attr('x2', x(this.testPoint.height))
      .attr('y2', y(this.normalPDF(this.testPoint.height, this.featureData.male.height.mean, this.featureData.male.height.stdDev)))
      .style('stroke', '#4285f4')
      .style('stroke-width', 2)
      .style('stroke-dasharray', '5,5')
      .style('opacity', 0);
    
    g.append('line')
      .attr('id', 'female-prob-line')
      .attr('x1', x(this.testPoint.height))
      .attr('y1', height)
      .attr('x2', x(this.testPoint.height))
      .attr('y2', y(this.normalPDF(this.testPoint.height, this.featureData.female.height.mean, this.featureData.female.height.stdDev)))
      .style('stroke', '#7c4dff')
      .style('stroke-width', 2)
      .style('stroke-dasharray', '5,5')
      .style('opacity', 0);
    
    // Probability circles
    g.append('circle')
      .attr('id', 'male-prob-point')
      .attr('cx', x(this.testPoint.height))
      .attr('cy', y(this.normalPDF(this.testPoint.height, this.featureData.male.height.mean, this.featureData.male.height.stdDev)))
      .attr('r', 6)
      .style('fill', '#4285f4')
      .style('opacity', 0);
    
    g.append('circle')
      .attr('id', 'female-prob-point')
      .attr('cx', x(this.testPoint.height))
      .attr('cy', y(this.normalPDF(this.testPoint.height, this.featureData.female.height.mean, this.featureData.female.height.stdDev)))
      .attr('r', 6)
      .style('fill', '#7c4dff')
      .style('opacity', 0);
  }
  
  updateGaussianSimulation(): void {
    const svg = d3.select(this.distributionChart.nativeElement);
    
    // Reset probability calculations
    this.classProbabilities = { male: 0, female: 0 };
    
    switch (this.currentStep) {
      case 0: // Reset
        svg.select('#test-point').style('opacity', 0);
        svg.select('#male-prob-line').style('opacity', 0);
        svg.select('#female-prob-line').style('opacity', 0);
        svg.select('#male-prob-point').style('opacity', 0);
        svg.select('#female-prob-point').style('opacity', 0);
        break;
      case 1: // Show test point
        svg.select('#test-point').style('opacity', 1);
        break;
      case 2: // Show male probability
        svg.select('#male-prob-line').style('opacity', 1);
        svg.select('#male-prob-point').style('opacity', 1);
        
        // Calculate male likelihood for height
        const maleLikelihood = this.normalPDF(
          this.testPoint.height, 
          this.featureData.male.height.mean, 
          this.featureData.male.height.stdDev
        );
        this.classProbabilities.male = maleLikelihood;
        break;
      case 3: // Show female probability
        svg.select('#female-prob-line').style('opacity', 1);
        svg.select('#female-prob-point').style('opacity', 1);
        
        // Calculate female likelihood for height
        const femaleLikelihood = this.normalPDF(
          this.testPoint.height, 
          this.featureData.female.height.mean, 
          this.featureData.female.height.stdDev
        );
        this.classProbabilities.female = femaleLikelihood;
        break;
      case 4: // Multiply by prior probabilities
        this.classProbabilities.male *= this.priorProbabilities.male;
        this.classProbabilities.female *= this.priorProbabilities.female;
        break;
      case 5: // Normalize to get posterior probabilities
        const total = this.classProbabilities.male + this.classProbabilities.female;
        this.classProbabilities.male /= total;
        this.classProbabilities.female /= total;
        break;
    }
  }
  
  // ----- Text Classification Simulation Methods -----
  
  calculateWordProbabilities(): void {
    // Count words by class
    const wordCounts: { 
      spam: { [key: string]: number }, 
      not_spam: { [key: string]: number } 
    } = { 
      spam: {}, 
      not_spam: {} 
    };
    const classCounts: { spam: number, not_spam: number } = { spam: 0, not_spam: 0 };
    
    // Initialize with Laplace smoothing
    const allWords = new Set<string>();
    this.emailData.forEach(email => {
      email.words.forEach(word => allWords.add(word));
      classCounts[email.label as keyof typeof classCounts]++;
    });
    
    allWords.forEach(word => {
      wordCounts.spam[word] = 1; // Add 1 for Laplace smoothing
      wordCounts.not_spam[word] = 1;
    });
    
    // Count word occurrences
    this.emailData.forEach(email => {
      email.words.forEach(word => {
        if (email.label === 'spam') {
          wordCounts.spam[word]++;
        } else if (email.label === 'not_spam') {
          wordCounts.not_spam[word]++;
        }
      });
    });
    
    // Calculate probabilities
    const wordTotals = {
      spam: Object.values(wordCounts.spam).reduce((a, b) => a + b, 0),
      not_spam: Object.values(wordCounts.not_spam).reduce((a, b) => a + b, 0)
    };
    
    this.wordProbabilities = { spam: {}, not_spam: {} };
    
    allWords.forEach(word => {
      this.wordProbabilities.spam[word] = wordCounts.spam[word] / wordTotals.spam;
      this.wordProbabilities.not_spam[word] = wordCounts.not_spam[word] / wordTotals.not_spam;
    });
  }
  
  initTextClassification(): void {
    // Implementation requires DOM manipulation that would be handled in updateTextSimulation
    this.updateTextSimulation();
  }
  
  updateTextSimulation(): void {
    const container = d3.select(this.textClassification.nativeElement);
    container.selectAll('*').remove();
    
    // Reset probabilities
    this.emailProbabilities = { spam: Math.log(0.5), not_spam: Math.log(0.5) };
    
    const table = container.append('table')
      .attr('class', 'word-probability-table');
    
    // Header row
    const thead = table.append('thead');
    thead.append('tr')
      .selectAll('th')
      .data(['Word', 'P(word|spam)', 'P(word|not spam)'])
      .enter()
      .append('th')
      .text(d => d);
    
    // Data rows for test email words
    const tbody = table.append('tbody');
    const rows = tbody.selectAll('tr')
      .data(this.testEmail)
      .enter()
      .append('tr');
    
    // Word column
    rows.append('td')
      .text(d => d)
      .attr('class', 'word-cell');
    
    // P(word|spam) column
    rows.append('td')
      .text(d => this.wordProbabilities.spam[d]?.toFixed(4) || '0.0000')
      .attr('class', 'probability-cell')
      .style('opacity', this.currentStep >= 1 ? 1 : 0.3);
    
    // P(word|not spam) column
    rows.append('td')
      .text(d => this.wordProbabilities.not_spam[d]?.toFixed(4) || '0.0000')
      .attr('class', 'probability-cell')
      .style('opacity', this.currentStep >= 1 ? 1 : 0.3);
    
    // Create probability calculation display
    const calcDiv = container.append('div')
      .attr('class', 'probability-calculation');
    
    if (this.currentStep >= 2) {
      calcDiv.append('h4').text('Probability Calculation (using log probabilities to avoid underflow):');
      
      const spamCalc = calcDiv.append('div').attr('class', 'calc-row');
      spamCalc.append('span').text('log P(spam) = log(0.5) = ').attr('class', 'calc-label');
      spamCalc.append('span').text(this.emailProbabilities.spam.toFixed(4)).attr('class', 'calc-value');
      
      const notSpamCalc = calcDiv.append('div').attr('class', 'calc-row');
      notSpamCalc.append('span').text('log P(not spam) = log(0.5) = ').attr('class', 'calc-label');
      notSpamCalc.append('span').text(this.emailProbabilities.not_spam.toFixed(4)).attr('class', 'calc-value');
    }
    
    // Calculate probabilities for each word
    if (this.currentStep >= 3) {
      this.testEmail.forEach(word => {
        // Add log probabilities to avoid numerical underflow
        this.emailProbabilities.spam += Math.log(this.wordProbabilities.spam[word] || 0.01);
        this.emailProbabilities.not_spam += Math.log(this.wordProbabilities.not_spam[word] || 0.01);
        
        // Display the calculation
        const spamWordCalc = calcDiv.append('div').attr('class', 'calc-row');
        spamWordCalc.append('span').text(`log P(spam) += log P("${word}"|spam) = ${Math.log(this.wordProbabilities.spam[word] || 0.01).toFixed(4)} → `).attr('class', 'calc-label');
        spamWordCalc.append('span').text(this.emailProbabilities.spam.toFixed(4)).attr('class', 'calc-value');
        
        const notSpamWordCalc = calcDiv.append('div').attr('class', 'calc-row');
        notSpamWordCalc.append('span').text(`log P(not spam) += log P("${word}"|not spam) = ${Math.log(this.wordProbabilities.not_spam[word] || 0.01).toFixed(4)} → `).attr('class', 'calc-label');
        notSpamWordCalc.append('span').text(this.emailProbabilities.not_spam.toFixed(4)).attr('class', 'calc-value');
      });
    }
    
    // Show final classification
    if (this.currentStep >= 4) {
      const resultDiv = calcDiv.append('div').attr('class', 'classification-result');
      
      // Convert from log probabilities back to probabilities
      const spamProb = Math.exp(this.emailProbabilities.spam);
      const notSpamProb = Math.exp(this.emailProbabilities.not_spam);
      const total = spamProb + notSpamProb;
      
      const normalizedSpam = spamProb / total;
      const normalizedNotSpam = notSpamProb / total;
      
      resultDiv.append('h4').text('Final Classification:');
      
      const resultTable = resultDiv.append('table').attr('class', 'result-table');
      const resultHeader = resultTable.append('thead').append('tr');
      resultHeader.append('th').text('Class');
      resultHeader.append('th').text('Probability');
      
      const resultBody = resultTable.append('tbody');
      
      const spamRow = resultBody.append('tr');
      spamRow.append('td').text('Spam');
      spamRow.append('td').text(normalizedSpam.toFixed(4));
      
      const notSpamRow = resultBody.append('tr');
      notSpamRow.append('td').text('Not Spam');
      notSpamRow.append('td').text(normalizedNotSpam.toFixed(4));
      
      // Highlight the winner
      const prediction = normalizedSpam > normalizedNotSpam ? 'Spam' : 'Not Spam';
      const predictionClass = normalizedSpam > normalizedNotSpam ? 'spam' : 'not-spam';
      
      resultDiv.append('div')
        .attr('class', `prediction ${predictionClass}`)
        .text(`Prediction: ${prediction}`);
    }
  }
  
  // ----- Decision Boundary Simulation Methods -----
  
  generatePoints(): void {
    // Generate random points for two classes
    this.points = [];
    
    // Class A (centered at [25, 25])
    for (let i = 0; i < 50; i++) {
      this.points.push({
        x: this.normalRandom(25, 5),
        y: this.normalRandom(25, 5),
        class: 'A'
      });
    }
    
    // Class B (centered at [75, 75])
    for (let i = 0; i < 50; i++) {
      this.points.push({
        x: this.normalRandom(75, 5),
        y: this.normalRandom(75, 5),
        class: 'B'
      });
    }
  }
  
  generateGridPoints(): void {
    // Generate a grid of points and calculate probabilities for each
    this.gridPoints = [];
    
    const classAPoints = this.points.filter(p => p.class === 'A');
    const classBPoints = this.points.filter(p => p.class === 'B');
    
    // Calculate mean and variance for each feature in each class
    const stats = {
      A: {
        x: { mean: d3.mean(classAPoints, d => d.x) || 0, variance: d3.variance(classAPoints, d => d.x) || 1 },
        y: { mean: d3.mean(classAPoints, d => d.y) || 0, variance: d3.variance(classAPoints, d => d.y) || 1 }
      },
      B: {
        x: { mean: d3.mean(classBPoints, d => d.x) || 0, variance: d3.variance(classBPoints, d => d.x) || 1 },
        y: { mean: d3.mean(classBPoints, d => d.y) || 0, variance: d3.variance(classBPoints, d => d.y) || 1 }
      }
    };
    
    // Add small amount to variances to avoid division by zero
    stats.A.x.variance = Math.max(stats.A.x.variance, 0.1);
    stats.A.y.variance = Math.max(stats.A.y.variance, 0.1);
    stats.B.x.variance = Math.max(stats.B.x.variance, 0.1);
    stats.B.y.variance = Math.max(stats.B.y.variance, 0.1);
    
    // Prior probabilities
    const priorA = classAPoints.length / this.points.length;
    const priorB = classBPoints.length / this.points.length;
    
    // Generate grid points
    for (let x = 0; x <= 100; x += 5) {
      for (let y = 0; y <= 100; y += 5) {
        // Calculate likelihood for each class
        const likelihoodA = 
          this.normalPDF(x, stats.A.x.mean, Math.sqrt(stats.A.x.variance)) * 
          this.normalPDF(y, stats.A.y.mean, Math.sqrt(stats.A.y.variance));
        
        const likelihoodB = 
          this.normalPDF(x, stats.B.x.mean, Math.sqrt(stats.B.x.variance)) * 
          this.normalPDF(y, stats.B.y.mean, Math.sqrt(stats.B.y.variance));
        
        // Calculate posterior probabilities
        const posteriorA = likelihoodA * priorA;
        const posteriorB = likelihoodB * priorB;
        
        // Normalize
        const probA = posteriorA / (posteriorA + posteriorB);
        
        this.gridPoints.push({
          x,
          y,
          probA,
          predicted: probA > 0.5 ? 'A' : 'B'
        });
      }
    }
  }
  
  initDecisionBoundary(): void {
    const svg = d3.select(this.decisionBoundary.nativeElement);
    svg.selectAll('*').remove();
    
    // Setup container and margins
    const margin = { top: 20, right: 30, bottom: 50, left: 50 };
    const width = +svg.attr('width') - margin.left - margin.right;
    const height = +svg.attr('height') - margin.top - margin.bottom;
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Scales
    const x = d3.scaleLinear()
      .domain([0, 100])
      .range([0, width]);
    
    const y = d3.scaleLinear()
      .domain([0, 100])
      .range([height, 0]);
    
    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x));
    
    g.append('text')
      .attr('transform', `translate(${width/2},${height + 40})`)
      .style('text-anchor', 'middle')
      .text('Feature 1');
    
    g.append('g')
      .call(d3.axisLeft(y));
    
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -40)
      .attr('x', -height/2)
      .style('text-anchor', 'middle')
      .text('Feature 2');
    
    // Create color gradient for decision boundary
    const color = d3.scaleSequential(d3.interpolateBlues)
      .domain([0, 1]);
    
    // Add grid points (only for showing the decision boundary)
    if (this.currentStep >= 1) {
      g.selectAll('.grid-point')
        .data(this.gridPoints)
        .enter()
        .append('rect')
        .attr('class', 'grid-point')
        .attr('x', d => x(d.x) - 2.5)
        .attr('y', d => y(d.y) - 2.5)
        .attr('width', 5)
        .attr('height', 5)
        .style('fill', d => d.predicted === 'A' ? color(d.probA) : color(1 - d.probA))
        .style('opacity', 0.7);
    }
    
    // Add decision boundary contour (probability = 0.5)
    if (this.currentStep >= 2) {
      const boundaryPoints = this.gridPoints.filter(p => 
        Math.abs(p.probA - 0.5) < 0.05
      );
      
      g.selectAll('.boundary-point')
        .data(boundaryPoints)
        .enter()
        .append('circle')
        .attr('class', 'boundary-point')
        .attr('cx', d => x(d.x))
        .attr('cy', d => y(d.y))
        .attr('r', 3)
        .style('fill', '#00c9ff')
        .style('stroke', 'white')
        .style('stroke-width', 1);
    }
    
    // Add data points
    if (this.currentStep >= 3) {
      g.selectAll('.data-point')
        .data(this.points)
        .enter()
        .append('circle')
        .attr('class', 'data-point')
        .attr('cx', d => x(d.x))
        .attr('cy', d => y(d.y))
        .attr('r', 5)
        .style('fill', d => d.class === 'A' ? '#4285f4' : '#7c4dff')
        .style('stroke', 'white')
        .style('stroke-width', 1);
      
      // Legend
      g.append('circle').attr('cx', width - 120).attr('cy', 20).attr('r', 6).style('fill', '#4285f4');
      g.append('text').attr('x', width - 100).attr('y', 20).text('Class A').style('font-size', '15px').attr('alignment-baseline', 'middle').style('fill', '#e1e7f5');
      g.append('circle').attr('cx', width - 120).attr('cy', 50).attr('r', 6).style('fill', '#7c4dff');
      g.append('text').attr('x', width - 100).attr('y', 50).text('Class B').style('font-size', '15px').attr('alignment-baseline', 'middle').style('fill', '#e1e7f5');
    }
  }
  
  updateDecisionBoundarySimulation(): void {
    // Redraw the chart at each step
    this.initDecisionBoundary();
  }
  
  // ----- Utility Methods -----
  
  normalPDF(x: number, mean: number, stdDev: number): number {
    const variance = stdDev * stdDev;
    return (1 / (Math.sqrt(2 * Math.PI * variance))) * 
           Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
  }
  
  normalRandom(mean: number, stdDev: number): number {
    // Box-Muller transform
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0 * stdDev + mean;
  }
}