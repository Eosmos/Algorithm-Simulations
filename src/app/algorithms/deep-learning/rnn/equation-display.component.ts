import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-equation-display',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="equation-container">
      <!-- Hidden State Update Equation -->
      @if (type === 'hiddenState') {
        <div class="equation-content">
          <div class="equation-row">
            <span class="variable">h<sub>t</sub></span>
            <span class="operator">=</span>
            <span class="function">tanh</span>
            <span class="parenthesis">(</span>
            <span class="variable">W<sub>xh</sub></span>
            <span class="variable">x<sub>t</sub></span>
            <span class="operator">+</span>
            <span class="variable">W<sub>hh</sub></span>
            <span class="variable">h<sub>t-1</sub></span>
            <span class="operator">+</span>
            <span class="variable">b<sub>h</sub></span>
            <span class="parenthesis">)</span>
          </div>
        </div>
      }
      
      <!-- Output Calculation Equation -->
      @if (type === 'output') {
        <div class="equation-content">
          <div class="equation-row">
            <span class="variable">y<sub>t</sub></span>
            <span class="operator">=</span>
            <span class="function">softmax</span>
            <span class="parenthesis">(</span>
            <span class="variable">W<sub>hy</sub></span>
            <span class="variable">h<sub>t</sub></span>
            <span class="operator">+</span>
            <span class="variable">b<sub>y</sub></span>
            <span class="parenthesis">)</span>
          </div>
        </div>
      }
    </div>
  `,
  styles: [`
    .equation-container {
      background-color: #1e3a66;
      border-radius: 8px;
      padding: 16px;
      margin: 16px 0;
      display: flex;
      justify-content: center;
    }
    
    .equation-content {
      color: #e1e7f5;
      font-size: 1.2em;
      font-family: 'Times New Roman', serif;
    }
    
    .equation-row {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      justify-content: center;
      gap: 4px;
    }
    
    .variable {
      font-style: italic;
    }
    
    .function {
      font-style: normal;
      margin-right: 2px;
    }
    
    .operator {
      margin: 0 4px;
    }
    
    .parenthesis {
      font-size: 1.2em;
    }
    
    sub {
      font-size: 0.7em;
      position: relative;
      bottom: -0.2em;
    }
  `]
})
export class EquationDisplayComponent {
  @Input() type: 'hiddenState' | 'output' = 'hiddenState';
}