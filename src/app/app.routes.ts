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
      loadComponent: () => import('./algorithms/supervised/linear-regression/linear-regression.component')
        .then(c => c.LinearRegressionComponent)
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
      loadComponent: () => import('./algorithms/supervised/random-forest-simulation/random-forest-simulation.component')
        .then(c => c.RandomForestSimulationComponent)
    },
    { 
      path: 'svm',
      loadComponent: () => import('./algorithms/supervised/svm-visualization/svm-visualization.component')
        .then(c => c.SvmVisualizationComponent)
    },
    // { 
    //   path: 'naive-bayes',
    //   loadComponent: () => import('./algorithms/supervised/naive-bayes/naive-bayes.component')
    //     .then(c => c.NaiveBayesComponent)
    // },
    
    // // Unsupervised Learning algorithms
    // { 
    //   path: 'kmeans',
    //   loadComponent: () => import('./algorithms/unsupervised/kmeans/kmeans.component')
    //     .then(c => c.KmeansComponent)
    // },
    // { 
    //   path: 'pca',
    //   loadComponent: () => import('./algorithms/unsupervised/pca/pca.component')
    //     .then(c => c.PcaComponent)
    // },
    // { 
    //   path: 'autoencoders',
    //   loadComponent: () => import('./algorithms/unsupervised/autoencoders/autoencoders.component')
    //     .then(c => c.AutoencodersComponent)
    // },
    
    // // Reinforcement Learning algorithms
    // { 
    //   path: 'qlearning',
    //   loadComponent: () => import('./algorithms/reinforcement/qlearning/qlearning.component')
    //     .then(c => c.QlearningComponent)
    // },
    // { 
    //   path: 'policy-gradient',
    //   loadComponent: () => import('./algorithms/reinforcement/policy-gradient/policy-gradient.component')
    //     .then(c => c.PolicyGradientComponent)
    // },
    
    // // Deep Learning algorithms
    // { 
    //   path: 'cnn',
    //   loadComponent: () => import('./algorithms/deep-learning/cnn/cnn.component')
    //     .then(c => c.CnnComponent)
    // },
    // { 
    //   path: 'rnn',
    //   loadComponent: () => import('./algorithms/deep-learning/rnn/rnn.component')
    //     .then(c => c.RnnComponent)
    // },
    // { 
    //   path: 'lstm',
    //   loadComponent: () => import('./algorithms/deep-learning/lstm/lstm.component')
    //     .then(c => c.LstmComponent)
    // },
    // { 
    //   path: 'gan',
    //   loadComponent: () => import('./algorithms/deep-learning/gan/gan.component')
    //     .then(c => c.GanComponent)
    // },
    // { 
    //   path: 'transformers',
    //   loadComponent: () => import('./algorithms/deep-learning/transformers/transformers.component')
    //     .then(c => c.TransformersComponent)
    // },
    
    // Wildcard route - redirect to dashboard if path doesn't match
    { path: '**', redirectTo: '/dashboard' }
  ];