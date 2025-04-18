import { Component, Input, ElementRef, AfterViewInit, OnChanges, SimpleChanges, ViewChild, NgZone } from '@angular/core';
import { MathRenderingService } from './math-rendering.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-math-equation',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="math-wrapper">
      <div #mathContainer class="math-content"></div>
    </div>
  `,
  styles: [`
    .math-wrapper {
      width: 100%;
      margin: 16px 0;
    }
    
    .math-content {
      padding: 16px;
      background-color: #1e3a66;
      border-radius: 8px;
      overflow-x: auto;
      min-height: 50px;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    
    :host ::ng-deep .MathJax {
      color: #e1e7f5 !important;
    }
  `]
})
export class MathEquationComponent implements AfterViewInit, OnChanges {
  @Input() equation: string = '';
  @Input() display: boolean = true; // True for display math, false for inline
  
  @ViewChild('mathContainer') mathContainer!: ElementRef<HTMLDivElement>;
  
  private hasRendered = false;
  private renderRetryCount = 0;
  private maxRetries = 5;
  
  constructor(
    private mathService: MathRenderingService,
    private ngZone: NgZone
  ) {}

  ngAfterViewInit(): void {
    this.renderEquation();
  }
  
  ngOnChanges(changes: SimpleChanges): void {
    if (changes['equation'] || changes['display']) {
      // Only re-render if view is already initialized
      if (this.mathContainer?.nativeElement) {
        this.renderRetryCount = 0;
        setTimeout(() => this.renderEquation(), 0);
      }
    }
  }
  
  private renderEquation(): void {
    if (!this.equation || !this.mathContainer?.nativeElement) {
      return;
    }
    
    // Format the equation with proper delimiters
    const formattedEquation = this.formatEquation(this.equation);
    
    // Clear previous content first
    const container = this.mathContainer.nativeElement;
    container.innerHTML = formattedEquation;
    
    // Add a small delay to ensure the DOM is updated
    setTimeout(() => {
      this.ngZone.runOutsideAngular(() => {
        try {
          this.mathService.render(container)
            .then(() => {
              this.hasRendered = true;
              this.renderRetryCount = 0;
            })
            .catch((error) => {
              console.error('Error rendering equation:', error);
              this.handleRenderError();
            });
        } catch (error) {
          console.error('Error initiating equation rendering:', error);
          this.handleRenderError();
        }
      });
    }, 100);
  }
  
  private handleRenderError(): void {
    if (this.renderRetryCount < this.maxRetries) {
      this.renderRetryCount++;
      const delay = 300 * this.renderRetryCount;
      console.log(`Retrying equation render (attempt ${this.renderRetryCount}) in ${delay}ms`);
      setTimeout(() => this.renderEquation(), delay);
    } else {
      console.error(`Failed to render equation after ${this.maxRetries} attempts:`, this.equation);
    }
  }
  
  private formatEquation(equation: string): string {
    if (!equation) return '';
    
    // Clean up any problematic characters
    let cleanEquation = equation.trim()
      .replace(/\\{/g, '\\lbrace')
      .replace(/\\}/g, '\\rbrace');
    
    // Format based on display or inline mode
    if (this.display) {
      // For displayed equations
      if (!cleanEquation.startsWith('\\[') && !cleanEquation.startsWith('$$')) {
        cleanEquation = '$$' + cleanEquation + '$$';
      }
    } else {
      // For inline equations
      if (!cleanEquation.startsWith('\\(') && !cleanEquation.startsWith('$')) {
        cleanEquation = '$' + cleanEquation + '$';
      }
    }
    
    return cleanEquation;
  }
}