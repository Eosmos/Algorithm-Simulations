import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { LinearRegressionComponent } from "./linear-regression/linear-regression.component";
import { DecisionTreeComponent } from "./decision-tree/decision-tree.component";
import { LogisticRegressionSimulationComponent } from "./logistic-regression-simulation/logistic-regression-simulation.component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, LinearRegressionComponent, DecisionTreeComponent, LogisticRegressionSimulationComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'Algorithm-Simulations';
}
