import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { LinearRegressionComponent } from "./linear-regression/linear-regression.component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, LinearRegressionComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'Algorithm-Simulations';
}
