import { Injectable, NgZone } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class MathRenderingService {
  private isScriptLoaded = false;
  private isReady = false;
  private renderQueue: Array<() => void> = [];
  private scriptLoadPromise: Promise<void> | null = null;
  
  constructor(private ngZone: NgZone) {
    this.loadMathJax();
  }
  
  /**
   * Load MathJax library from CDN
   * @returns Promise that resolves when MathJax is ready
   */
  private loadMathJax(): Promise<void> {
    // Return existing promise if already loading
    if (this.scriptLoadPromise) {
      return this.scriptLoadPromise;
    }
    
    // Create promise to load MathJax
    this.scriptLoadPromise = new Promise<void>((resolve, reject) => {
      // Don't load if already loaded
      if (document.getElementById('mathjax-script')) {
        this.isScriptLoaded = true;
        if (window.MathJax && window.MathJax.typesetPromise) {
          this.isReady = true;
          resolve();
          return;
        }
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
              resolve();
            });
          }
        }
      };
      
      try {
        // Create and append the script
        const script = document.createElement('script');
        script.id = 'mathjax-script';
        script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
        script.async = true;
        script.onload = () => {
          this.isScriptLoaded = true;
        };
        script.onerror = (error) => {
          console.error('Failed to load MathJax:', error);
          reject(new Error('Failed to load MathJax script'));
        };
        document.head.appendChild(script);
      } catch (error) {
        console.error('Error appending MathJax script:', error);
        reject(error);
        this.scriptLoadPromise = null;
      }
    });
    
    return this.scriptLoadPromise;
  }
  
  /**
   * Render LaTeX in the provided element
   * @param element The DOM element containing LaTeX
   * @param retryCount Number of retry attempts made
   * @returns Promise that resolves when rendering is complete
   */
  public render(element: HTMLElement, retryCount = 0): Promise<void> {
    if (!element) {
      return Promise.reject(new Error('No element provided for rendering'));
    }
    
    // Return a Promise for better async handling
    return new Promise<void>((resolve, reject) => {
      const renderAction = () => {
        try {
          if (!window.MathJax?.typesetPromise) {
            const error = new Error('MathJax typesetPromise not available');
            console.warn(error.message);
            
            // Retry with increasing delay
            if (retryCount < 5) {
              setTimeout(() => {
                this.render(element, retryCount + 1)
                  .then(resolve)
                  .catch(reject);
              }, 300 * (retryCount + 1));
            } else {
              reject(error);
            }
            return;
          }
          
          console.log('Attempting to render:', element);
          
          // Use try-catch to avoid errors breaking the app
          this.ngZone.runOutsideAngular(() => {
            window.MathJax.typesetPromise([element])
              .then(() => {
                console.log('MathJax typesetting complete');
                resolve();
              })
              .catch((err: Error) => {
                console.error('MathJax typesetting error:', err);
                
                // Retry a few times with increasing delay
                if (retryCount < 3) {
                  setTimeout(() => {
                    this.render(element, retryCount + 1)
                      .then(resolve)
                      .catch(reject);
                  }, 500 * (retryCount + 1));
                } else {
                  reject(err);
                }
              });
          });
        } catch (error) {
          console.error('MathJax rendering error:', error);
          reject(error);
        }
      };
      
      if (this.isReady) {
        renderAction();
      } else {
        // Queue the render action and ensure MathJax is loaded
        this.renderQueue.push(() => {
          renderAction();
        });
        
        // Make sure MathJax is loading
        this.loadMathJax()
          .catch(error => {
            console.error('Failed to load MathJax:', error);
            reject(error);
          });
      }
    });
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