import { Component, Input, OnInit } from '@angular/core';
import { NgIf, NgClass } from '@angular/common';

@Component({
  selector: 'app-rnn-visualization-guide',
  templateUrl: './rnn-simulation-note.component.html',
  styleUrls: ['./rnn-simulation-note.component.scss'],
  imports: [NgIf, NgClass],
  standalone: true
})
export class RnnVisualizationGuideComponent implements OnInit {
  @Input() currentView: string = 'unrolled';
  
  isExpanded = true; // Start expanded by default
  
  constructor() { }

  ngOnInit(): void {
    // Check if the user has previously closed the guide
    try {
      const guideState = localStorage.getItem('rnnGuideExpanded');
      if (guideState !== null) {
        this.isExpanded = guideState === 'true';
      }
    } catch (error) {
      console.error('Error accessing localStorage:', error);
      // Continue with default value if localStorage fails
    }
  }
  
  toggleGuide(): void {
    this.isExpanded = !this.isExpanded;
    // Save the user preference
    try {
      localStorage.setItem('rnnGuideExpanded', this.isExpanded.toString());
    } catch (error) {
      console.error('Error saving to localStorage:', error);
      // Continue without saving if localStorage fails
    }
  }
}