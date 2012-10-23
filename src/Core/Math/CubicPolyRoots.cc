/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "CubicPolyRoots.h"
#include <cmath>
#include <iostream>
#include <cassert>

#define PI 3.14159265358979323846
bool test(double b, double c, double d, double x);
double newtons_method(double b, double c, double d, double x);

int cubic_poly_roots(double b, double c, double d,
			double& x0, double& x1, double& x2)
{
  /* convert x^3 + b*x^2 + c*x + d = 0 to y^3 + m*x = n */
  
  // x = y - b/3
  // x^3 + b*x^2 + c*x + d = (y-b/3)^3 + b*(y-b/3)^2 + c(y-b/3) + d =
  // y^3 + [3*(-b/3) + b] * y^2 + [3*(b/3)^2 - 2*b*b/3 + c] * y
  //     + [-(b/3)^3 + b*(b/3)^2 - b*c/3 + d] = 0
  // y^3 + [-(b^2)/3 + c] * y = [-2*(b^3)/27 + b*c/3 - d]
  // m = c - (b^2)/3, n = -2*(b^3)/27 + b*c/3 - d

  double b2 = b*b;
  double b3 = b*b2;

  double m = c - b2/3;
  double n = -2*b3/27 + b*c/3 - d;

  //cout << "y^3 + " << m << "*y = " << n << endl; 

  // (t - u)^3 = t^3 - 3*t^2*u + 3*t*u^2 - u^3
  // (t-u)^3 + 3*t*u*(t-u) = t^3 - u^3
  // y = t-u, m = 3*t*u, n = t^3 - u^3
  
  // n*(t^3) = (t^3 - u^3) * t^3 = t^6 - (t*u)^3 = t^6 - (m^3)/27
  // (t^3)^2 - n*(t^3) - (m^3)/27
  // by quadratic formula:
  // t^3 = n/2 + sqrt((n^2)/4 + (m^3)/27)

  double n_half = n/2;
  double t3deter = n_half*n_half + m*m*m/27;

  if (t3deter <= 0) {
    // t^3 is complex
    double t3_real = n_half;
    double t3_im = sqrt(-t3deter);

    // n = t^3 - u^3  -->  u^3 = t^3 - n
    // t^3 = n/2 + i*t3_im = t3_real + i*t3_im
    // u^3 = -n/2 + i*t3_im = -t3_real + i*t3_im
    // let t = r * exp(i*theta_t) = r * cos(theta_t) + i * r * sin(theta_t)
    //     u = r * exp(i*theta_u) = r * cos(theta_u) + i * r * sin(theta_u)
    // /* |t| = |u| = r is KEY to rest of this!!! */
    // if y = t - u is real, Im(t) = Im(u) ->
    //   sin(theta_t) = sin(theta_u) ->
    //   theta_t = theta_u + 2*n*pi or theta_t = pi - theta_u + 2*n*pi ->
    //   cos(theta_t) = cos(theta_u) or cos(theta_t) = -cos(theta_u) ->
    //   t - u = 0 (trival case already handled) or t - u = 2*r*cos(theta_t)
    // let theta = theta_t (for simplicity of names)
    // let t^3 = r^3 * exp(i * theta_3)
    // t = r * exp(i * (theta_3 / 3 + 2*n*pi / 3)), n = 0, 1, 2
    // theta = theta_3 / 3 + 2*n*pi / 3, n = 0, 1, 2
    // Note: if t3deter <= 0, all 3 roots are real (I don't know
    // the proof but it says that here:
    // http://mathworld.wolfram.com/CubicEquation.html)
    
    double r = pow(sqrt(t3_real*t3_real + t3_im*t3_im), 1/3.0);
    double theta = atan2(t3_im, t3_real)/3;   

    double pi_2thirds = 2*PI/3;
    double b_thirds = b/3;
    x0 = 2*r*cos(theta) - b_thirds;
    x1 = 2*r*cos(theta + pi_2thirds) - b_thirds;
    x2 = 2*r*cos(theta + 2*pi_2thirds) - b_thirds;
    //assert(test(b, c, d, x0));
    //assert(test(b, c, d, x1));
    //assert(test(b, c, d, x2));
    return 3;
  }
  else
  {
    // Note: if t3deter > 0, there is one real root
    // (according to http://mathworld.wolfram.com/CubicEquation.html)

    double t3 = n_half + sqrt(t3deter);

    // n = t^3 - u^3  -->  u^3 = t^3 - n
    double u3 = t3 - n;

    double t = (t3 < 0) ? -pow(-t3, 1/3.0) : pow(t3, 1/3.0);
    double u = (u3 < 0) ? -pow(-u3, 1/3.0) : pow(u3, 1/3.0);

    // y = t - u
    // x = y - b/3 = t - u - b/3
    x0 = t - u - b/3;
    //assert(test(b, c, d, x0));

    return 1;
  }
}

inline double evaluate_cubic(double a, double b, double c, double d, double x)
{
  double x2 = x*x;
  double x3 = x*x2;
  return a*x3 + b*x2 + c*x + d;
}

inline double evaluate_cubic(double b, double c, double d, double x)
{
  double x2 = x*x;
  double x3 = x*x2;
  return x3 + b*x2 + c*x + d;
}

inline double evaluate_quadratic(double a, double b, double c, double x)
{
  return a*x*x + b*x + c;
}

bool test(double b, double c, double d, double x)
{
  // df/dx = 3*x^2 + 2*b*x + c
  // (uncertainty in f = f) = (uncertainty in x) * df/dx
  // check uncertainty in x relative to max(x, d, 1)
  // (to try to make it relative to the scale of the numbers)
  double f = evaluate_cubic(b, c, d, x);
  double df_dx = evaluate_quadratic(3, 2*b, c, x);
  double Sx = fabs(df_dx) < 1 ? fabs(f) : fabs(f/df_dx); // uncertainty in x

  // not the best way -- oh, well
  double Sx_relative;
  if (fabs(x) < 1 && fabs(d) < 1)
    Sx_relative = Sx;
  else if (fabs(x) > fabs(d))
    Sx_relative = Sx/fabs(x);
  else
    Sx_relative = Sx/fabs(d);
    
  //cout << "relative Sx = " << Sx_relative << endl;
  if (Sx_relative < 1e-3)
    return true;
  else {
    //newtons_method(b, c, d, x);
    return false;
  }
}

/* for testing
double newtons_method(double b, double c, double d, double x)
{
  // f(x) = x^3 + b*x^2 + c*x + d
  // f'(x) = 3*x^2 + 2*b*x + c
  cout << "Newton's Method\n";
  cout << "x = " << x << endl;;
  double f = evaluate_cubic(b, c, d, x);
  cout << "Error = " << f << endl;
  while (fabs(f) > 1e-5) {
    double lastx = x;
    x = x - f / evaluate_quadratic(3, 2*b, c, x);
    cout << "x = " << x << endl;

    f = evaluate_cubic(b, c, d, x);

    cout << "Error = " << f << endl;
    if (lastx - x < 1e-25) {
      break;
    }
  }
  return x;
}
*/

