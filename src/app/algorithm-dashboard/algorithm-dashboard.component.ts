import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

interface Algorithm {
  id: string;
  name: string;
  description: string;
  icon: string;
  routePath: string;
}

interface SubCategory {
  id: string;
  name: string;
  algorithms: Algorithm[];
}

interface Category {
  id: string;
  name: string;
  subCategories: SubCategory[];
}

@Component({
  selector: 'app-algorithm-dashboard',
  templateUrl: './algorithm-dashboard.component.html',
  styleUrls: ['./algorithm-dashboard.component.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class AlgorithmDashboardComponent implements OnInit {
  allCategories: Category[] = [];  // Store original data
  categories: Category[] = [];     // Filtered data to display
  expandedCategories: { [key: string]: boolean } = {};
  expandedSubCategories: { [key: string]: boolean } = {};
  hoveredCard: string | null = null;
  searchTerm: string = '';

  constructor(private router: Router) {}

  ngOnInit(): void {
    this.initializeData();
    this.allCategories = JSON.parse(JSON.stringify(this.categories)); // Deep copy the original data
    
    // Initially expand the first category and its subcategories
    if (this.categories.length > 0) {
      this.expandedCategories[this.categories[0].id] = true;
      this.categories[0].subCategories.forEach(subCategory => {
        this.expandedSubCategories[subCategory.id] = true;
      });
    }
  }

  navigateTo(routePath: string): void {
    this.router.navigate([routePath]);
  }

  toggleCategory(categoryId: string, event: Event): void {
    event.stopPropagation();
    // Toggle the category expansion state
    this.expandedCategories[categoryId] = !this.expandedCategories[categoryId];
    
    // Find the category
    const category = this.categories.find(cat => cat.id === categoryId);
    
    // If expanding the category, also expand all its subcategories
    if (this.expandedCategories[categoryId] && category) {
      category.subCategories.forEach(subCategory => {
        this.expandedSubCategories[subCategory.id] = true;
      });
    }
    // If collapsing the category, also collapse all its subcategories
    else if (category) {
      category.subCategories.forEach(subCategory => {
        this.expandedSubCategories[subCategory.id] = false;
      });
    }
  }

  toggleSubCategory(subCategoryId: string, event: Event): void {
    event.stopPropagation();
    this.expandedSubCategories[subCategoryId] = !this.expandedSubCategories[subCategoryId];
  }

  isCategoryExpanded(categoryId: string): boolean {
    return this.expandedCategories[categoryId] || false;
  }

  isSubCategoryExpanded(subCategoryId: string): boolean {
    return this.expandedSubCategories[subCategoryId] || false;
  }

  setHoveredCard(algorithmId: string | null): void {
    this.hoveredCard = algorithmId;
  }
  
  getComplexityLevel(algorithmId: string): string {
    const complexityMap: { [key: string]: string } = {
      'linear-regression': 'Beginner',
      'logistic-regression': 'Beginner',
      'decision-trees': 'Intermediate',
      'random-forests': 'Advanced',
      'svm': 'Advanced',
      'naive-bayes': 'Beginner',
      'kmeans': 'Intermediate',
      'pca': 'Intermediate',
      'autoencoders': 'Advanced',
      'qlearning': 'Advanced',
      'policy-gradient': 'Expert',
      'cnn': 'Advanced',
      'rnn': 'Advanced',
      'lstm': 'Expert',
      'gan': 'Expert',
      'transformers': 'Expert'
    };
    
    return complexityMap[algorithmId] || 'Intermediate';
  }
  
  isPopular(algorithmId: string): boolean {
    const popularAlgorithms = [
      'linear-regression',
      'random-forests',
      'kmeans',
      'cnn',
      'transformers'
    ];
    
    return popularAlgorithms.includes(algorithmId);
  }
  
  // Statistics methods
  getTotalAlgorithmsCount(): number {
    let count = 0;
    this.allCategories.forEach(category => {
      category.subCategories.forEach(subCategory => {
        count += subCategory.algorithms.length;
      });
    });
    return count;
  }
  
  getTotalSubCategoriesCount(): number {
    let count = 0;
    this.allCategories.forEach(category => {
      count += category.subCategories.length;
    });
    return count;
  }
  
  // Search functionality
  onSearch(event: Event): void {
    const searchValue = (event.target as HTMLInputElement).value.toLowerCase().trim();
    this.searchTerm = searchValue;
    
    if (!searchValue) {
      // If search is empty, restore original data
      this.categories = JSON.parse(JSON.stringify(this.allCategories));
      return;
    }
    
    // Filter categories and their content based on search term
    this.categories = this.allCategories.map(category => {
      // Deep clone the category to avoid modifying the original
      const filteredCategory: Category = { 
        ...category, 
        subCategories: [] 
      };
      
      // Filter subcategories and their algorithms
      filteredCategory.subCategories = category.subCategories
        .map(subCategory => {
          // Deep clone the subcategory
          const filteredSubCategory: SubCategory = { 
            ...subCategory, 
            algorithms: [] 
          };
          
          // Filter algorithms based on search term
          filteredSubCategory.algorithms = subCategory.algorithms.filter(algorithm => 
            algorithm.name.toLowerCase().includes(searchValue) || 
            algorithm.description.toLowerCase().includes(searchValue)
          );
          
          return filteredSubCategory;
        })
        .filter(subCategory => subCategory.algorithms.length > 0); // Keep only subcategories with matching algorithms
      
      return filteredCategory;
    }).filter(category => category.subCategories.length > 0); // Keep only categories with matching subcategories
    
    // Auto-expand all categories and subcategories when searching
    this.categories.forEach(category => {
      this.expandedCategories[category.id] = true;
      category.subCategories.forEach(subCategory => {
        this.expandedSubCategories[subCategory.id] = true;
      });
    });
  }

  private initializeData(): void {
    this.categories = [
      {
        id: 'machine-learning',
        name: 'Machine Learning Algorithms',
        subCategories: [
          {
            id: 'supervised-learning',
            name: 'Supervised Learning',
            algorithms: [
              {
                id: 'linear-regression',
                name: 'Linear Regression',
                description: 'Predicts continuous outputs (e.g., house prices) using a linear equation.',
                icon: 'chart-line',
                routePath: '/linear-regression'
              },
              {
                id: 'logistic-regression',
                name: 'Logistic Regression',
                description: 'Binary classification (e.g., spam detection) using the sigmoid function.',
                icon: 'code-branch',
                routePath: '/logistic-regression'
              },
              {
                id: 'decision-trees',
                name: 'Decision Trees',
                description: 'Classification or regression via hierarchical splits in data.',
                icon: 'tree',
                routePath: '/decision-trees'
              },
              {
                id: 'random-forests',
                name: 'Random Forests',
                description: 'Enhances decision trees with ensemble methods for better prediction.',
                icon: 'sitemap',
                routePath: '/random-forests'
              },
              {
                id: 'svm',
                name: 'Support Vector Machines',
                description: 'Classification by maximizing margin between classes.',
                icon: 'project-diagram',
                routePath: '/svm'
              },
              {
                id: 'naive-bayes',
                name: 'Naive Bayes',
                description: 'Classification using probabilistic independence assumptions.',
                icon: 'pie-chart',
                routePath: '/naive-bayes'
              }
            ]
          },
          {
            id: 'unsupervised-learning',
            name: 'Unsupervised Learning',
            algorithms: [
              {
                id: 'kmeans',
                name: 'K-means Clustering',
                description: 'Groups data into k clusters based on similarity.',
                icon: 'object-group',
                routePath: '/kmeans'
              },
              {
                id: 'pca',
                name: 'Principal Component Analysis',
                description: 'Reduces dimensionality while retaining variance in data.',
                icon: 'compress-arrows-alt',
                routePath: '/pca'
              },
              {
                id: 'autoencoders',
                name: 'Autoencoders',
                description: 'Neural networks that learn compressed data representations.',
                icon: 'compress',
                routePath: '/autoencoders'
              }
            ]
          },
          {
            id: 'reinforcement-learning',
            name: 'Reinforcement Learning',
            algorithms: [
              {
                id: 'qlearning',
                name: 'Q-learning',
                description: 'Learns optimal policies in discrete environments through rewards.',
                icon: 'brain',
                routePath: '/qlearning'
              },
              {
                id: 'policy-gradient',
                name: 'Policy Gradient Methods',
                description: 'Optimizes policies in continuous spaces using gradient ascent.',
                icon: 'chart-line',
                routePath: '/policy-gradient'
              }
            ]
          }
        ]
      },
      {
        id: 'deep-learning',
        name: 'Deep Learning Algorithms',
        subCategories: [
          {
            id: 'neural-networks',
            name: 'Neural Network Models',
            algorithms: [
              {
                id: 'cnn',
                name: 'Convolutional Neural Networks',
                description: 'Specialized for image processing with filters and pooling layers.',
                icon: 'image',
                routePath: '/cnn'
              },
              {
                id: 'rnn',
                name: 'Recurrent Neural Networks',
                description: 'Process sequences with memory of previous inputs.',
                icon: 'sync',
                routePath: '/rnn'
              },
              {
                id: 'lstm',
                name: 'Long Short-Term Memory Networks',
                description: 'RNNs with special gates for long-term sequence modeling.',
                icon: 'memory',
                routePath: '/lstm'
              },
              {
                id: 'gan',
                name: 'Generative Adversarial Networks',
                description: 'Generate realistic data through competition of two networks.',
                icon: 'yin-yang',
                routePath: '/gan'
              },
              {
                id: 'transformers',
                name: 'Transformers',
                description: 'NLP architecture using self-attention mechanisms.',
                icon: 'file-alt',
                routePath: '/transformers'
              }
            ]
          }
        ]
      }
    ];
  }
}