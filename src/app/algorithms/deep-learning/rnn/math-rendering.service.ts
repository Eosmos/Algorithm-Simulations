import { Injectable, NgZone } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class MathRenderingService {
  private isScriptLoaded = false;
  private isReady = false;
  private renderQueue: Array<() => void> = [];
  
  constructor(private ngZone: NgZone) {
    this.loadMathJax();
  }
  
  /**
   * Load MathJax library from CDN
   */
  private loadMathJax(): void {
    // Don't load if already loaded
    if (document.getElementById('mathjax-script')) {
      this.isScriptLoaded = true;
      return;
    }
    
    // Configure MathJax
    window.MathJax = {
      options: {
        enableMenu: false,       // Disable the menu
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
      },
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true,
        packages: {'[+]': ['noerrors']}
      },
      svg: {
        fontCache: 'global',
        scale: 1.0
      },
      startup: {
        typeset: false,
        ready: () => {
          this.ngZone.run(() => {
            console.log('MathJax is ready');
            this.isReady = true;
            this.processRenderQueue();
          });
        }
      }
    };
    
    // Create and append the script
    const script = document.createElement('script');
    script.id = 'mathjax-script';
    script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
    script.async = true;
    script.onload = () => {
      this.isScriptLoaded = true;
    };
    document.head.appendChild(script);
  }
  
  /**
   * Render LaTeX in the provided element
   * @param element The DOM element containing LaTeX
   */
  public render(element: HTMLElement, retryCount = 0): void {
    if (!element) return;
    
    const renderAction = () => {
      try {
        if (window.MathJax?.typesetPromise) {
          console.log('Attempting to render:', element);
          
          // Use try-catch to avoid errors breaking the app
          this.ngZone.runOutsideAngular(() => {
            window.MathJax.typesetPromise([element])
              .then(() => {
                console.log('MathJax typesetting complete');
              })
              .catch((err: Error) => {
                console.error('MathJax typesetting error:', err);
                
                // Retry a few times with increasing delay
                if (retryCount < 3) {
                  setTimeout(() => {
                    this.render(element, retryCount + 1);
                  }, 500 * (retryCount + 1));
                }
              });
          });
        } else {
          console.warn('MathJax typesetPromise not available yet');
          if (retryCount < 5) {
            setTimeout(() => {
              this.render(element, retryCount + 1);
            }, 300 * (retryCount + 1));
          }
        }
      } catch (error) {
        console.error('MathJax rendering error:', error);
      }
    };
    
    if (this.isReady) {
      renderAction();
    } else {
      this.renderQueue.push(renderAction);
    }
  }
  
  /**
   * Process the queued rendering actions
   */
  private processRenderQueue(): void {
    console.log(`Processing ${this.renderQueue.length} queued MathJax renders`);
    if (this.renderQueue.length === 0) return;
    
    // Process with a small delay between items to avoid overwhelming MathJax
    let index = 0;
    const processNext = () => {
      if (index < this.renderQueue.length) {
        const action = this.renderQueue[index++];
        action();
        setTimeout(processNext, 100);
      } else {
        // Clear the queue when done
        this.renderQueue = [];
      }
    };
    
    processNext();
  }
}

// Add MathJax typing
declare global {
  interface Window {
    MathJax: any;
  }
}