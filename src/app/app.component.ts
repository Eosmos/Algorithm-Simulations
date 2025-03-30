import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { LinearRegressionComponent } from "./linear-regression/linear-regression.component";
import { DecisionTreeComponent } from "./decision-tree/decision-tree.component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, LinearRegressionComponent, DecisionTreeComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'Algorithm-Simulations';
}
