import { Component } from '@angular/core';
import { RouterLink, RouterOutlet } from '@angular/router';
import { LinearRegressionComponent } from "./algorithms/supervised/linear-regression/linear-regression.component";
import { DecisionTreeComponent } from "./algorithms/supervised/decision-tree/decision-tree.component";
import { LogisticRegressionSimulationComponent } from "./algorithms/supervised/logistic-regression-simulation/logistic-regression-simulation.component";
import { RandomForestSimulationComponent } from "./algorithms/supervised/random-forest-simulation/random-forest-simulation.component";
import { SvmVisualizationComponent } from "./algorithms/supervised/svm-visualization/svm-visualization.component";
import { AlgorithmDashboardComponent } from "./algorithm-dashboard/algorithm-dashboard.component";
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet,RouterLink,CommonModule, LinearRegressionComponent, DecisionTreeComponent, LogisticRegressionSimulationComponent, RandomForestSimulationComponent, SvmVisualizationComponent, AlgorithmDashboardComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'AI Algorithm Explorer';
}
