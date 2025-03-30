import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { LinearRegressionComponent } from "./linear-regression/linear-regression.component";
import { DecisionTreeComponent } from "./decision-tree/decision-tree.component";
import { LogisticRegressionSimulationComponent } from "./logistic-regression-simulation/logistic-regression-simulation.component";
import { RandomForestSimulationComponent } from "./random-forest-simulation/random-forest-simulation.component";
import { SvmVisualizationComponent } from "./svm-visualization/svm-visualization.component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, LinearRegressionComponent, DecisionTreeComponent, LogisticRegressionSimulationComponent, RandomForestSimulationComponent, SvmVisualizationComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'Algorithm-Simulations';
}
