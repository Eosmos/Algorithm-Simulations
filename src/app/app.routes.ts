import { Routes } from '@angular/router';

export const routes: Routes = [
    { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
    { 
      path: 'dashboard', 
      loadComponent: () => import('./algorithm-dashboard/algorithm-dashboard.component')
        .then(c => c.AlgorithmDashboardComponent) 
    },
    
    // Supervised Learning algorithms
    { 
      path: 'linear-regression',
      loadComponent: () => import('./algorithms/supervised/linear-regression/linear-regression-simulator.component')
        .then(c => c.LinearRegressionSimulatorComponent)
    },
    { 
      path: 'logistic-regression',
      loadComponent: () => import('./algorithms/supervised/logistic-regression-simulation/logistic-regression-simulation.component')
        .then(c => c.LogisticRegressionSimulationComponent)
    },
    { 
      path: 'decision-trees',
      loadComponent: () => import('./algorithms/supervised/decision-tree/decision-tree.component')
        .then(c => c.DecisionTreeComponent)
    },
    { 
      path: 'random-forests',
      loadComponent: () => import('./algorithms/supervised/random-forest-simulation/random-forest-visualization.component')
        .then(c => c.RandomForestVisualizationComponent)
    },
    { 
      path: 'svm',
      loadComponent: () => import('./algorithms/supervised/svm-visualization/svm-visualization.component')
        .then(c => c.SvmVisualizationComponent)
    },
    { 
      path: 'naive-bayes',
      loadComponent: () => import('./algorithms/supervised/naive-bayes/naive-bayes-simulation.component')
        .then(c => c.NaiveBayesSimulationComponent)
    },
    
    // Unsupervised Learning algorithms
    { 
      path: 'kmeans',
      loadComponent: () => import('./algorithms/unsupervised/kmeans/k-means-simulation.component')
        .then(c => c.KMeansSimulationComponent)
    },
    { 
      path: 'pca',
      loadComponent: () => import('./algorithms/unsupervised/pca/pca-simulation.component')
        .then(c => c.PcaSimulationComponent)
    },
    { 
      path: 'autoencoders',
      loadComponent: () => import('./algorithms/unsupervised/autoencoders/autoencoder-simulation.component')
        .then(c => c.AutoencoderSimulationComponent)
    },
    
    // Reinforcement Learning algorithms
    { 
      path: 'qlearning',
      loadComponent: () => import('./algorithms/reinforcement/qlearning/q-learning-simulation.component')
        .then(c => c.QLearningSimulationComponent)
    },
    { 
      path: 'policy-gradient',
      loadComponent: () => import('./algorithms/reinforcement/policy-gradient/policy-gradient-simulation.component')
        .then(c => c.PolicyGradientSimulationComponent)
    },
    
    // Deep Learning algorithms
    { 
      path: 'cnn',
      loadComponent: () => import('./algorithms/deep-learning/cnn/cnn-visualization.component')
        .then(c => c.CnnVisualizationComponent)
    },
    { 
      path: 'rnn',
      loadComponent: () => import('./algorithms/deep-learning/rnn/rnn-simulation.component')
        .then(c => c.RnnSimulationComponent)
    },
    { 
      path: 'lstm',
      loadComponent: () => import('./algorithms/deep-learning/lstm/lstm-simulator.component')
        .then(c => c.LstmSimulatorComponent)
    },
    { 
      path: 'gan',
      loadComponent: () => import('./algorithms/deep-learning/gan/gan-simulation.component')
        .then(c => c.GanSimulationComponent)
    },
    { 
      path: 'transformers',
      loadComponent: () => import('./algorithms/deep-learning/transformers/transformer-network.component')
        .then(c => c.TransformerNetworkComponent)
    },
    
    // Wildcard route - redirect to dashboard if path doesn't match
    { path: '**', redirectTo: '/dashboard' }
  ];