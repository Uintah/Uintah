/* The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

#include <CCA/Components/MD/ShiftedCardinalBSpline.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Util/FancyAssert.h>

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <string>

using namespace Uintah;
using std::vector;

ShiftedCardinalBSpline::ShiftedCardinalBSpline()
{

}

ShiftedCardinalBSpline::~ShiftedCardinalBSpline()
{

}

ShiftedCardinalBSpline::ShiftedCardinalBSpline(const int order)
{
  if (order < 2) {
    std::string message = "MD::SPME: Interpolation spline order cannot be less than 2!";
    throw InvalidValue(message, __FILE__, __LINE__);
  }
  if (order % 2 != 0) {
    std::string message = "MD::SPME: Interpolation spline order should be a multiple of 2!";
    throw InvalidValue(message, __FILE__, __LINE__);
  }

  d_splineOrder = order;
}

vector<double> ShiftedCardinalBSpline::evaluateGridAligned(const double X) const
{
  vector<double> splineArray = evaluateInternal(X, d_splineOrder);
  return splineArray;
}

vector<double> ShiftedCardinalBSpline::derivativeGridAligned(double X) const
{

  /*
   * dS (x)
   *   n
   * ------  =  S  (x) - S  (x-1)
   *   dx        n-1      n-1
   */

  // Calculate S_(n-1) array
  vector<double> derivativeArray = evaluateInternal(X, d_splineOrder - 1);

  // dS_n[d_splineOrder] = -S_(n-1)[d_splineOrder-1]
  derivativeArray.push_back(-derivativeArray[d_splineOrder - 1]);

  for (size_t Index = d_splineOrder - 1; Index > 0; --Index) {
    derivativeArray[Index] -= derivativeArray[Index - 1];
    // dA[k] is already S_(n-1)(x+k), so just subtract S_(n-1)(x+k-1) = dA[k-1]
  }
  // dA[0] is S_(n-1)(x); S_(n-1)(x-1) is 0 by definition, so nothing to do for the 0th element
  return derivativeArray;
}

// Private Methods
vector<double> ShiftedCardinalBSpline::evaluateInternal(const double X,
                                                        const int splineOrder) const
{
  vector<double> splineArray(splineOrder + 1, 0.0);

  // Initialize the second order spline from which to recurse
  splineArray[0] = S2(X);
  splineArray[1] = S2(X + 1);
  splineArray[2] = S2(X + 2);

  /* Generically:
   *              x              n-x
   *   S (x)  =  ---  S  (x) +   ---  S  (x)
   *    n        n-1   n-1       n-1   n-1
   *
   *              1
   *   S (x+k)  =--- ( (x+k) S  (x+k) + (n-(x+k)) S  (x+k-1) )
   *    n        n-1          n-1                  n-1
   *
   *   The array of grid aligned points are S(x), S(x+1), S(x+2), ... S(x+d_splineOrder-1)
   */
  // After a complete pass, splineArray[q] contains the value of S_n(X+q)
  double denominator = 1.0;  // Initialize the 1/(n-1) term and accumulate it throughout
  for (int n = 3; n <= splineOrder; ++n) {
    int n_minus_1 = n - 1;
    denominator /= static_cast<double>(n_minus_1);
    splineArray[n] = -X * splineArray[n_minus_1];  // For k=n -> S_(n-1)(x+n) is zero by definition and n-(x+n) == -x
    double n_double = static_cast<double>(n);
    for (int k = n - 1; k > 0; --k) {
      // Must traverse backwards to avoid overwriting spline of previous order before we're done with it
      double X_plus_k = X + static_cast<double>(k);
      splineArray[k] = X_plus_k * splineArray[k] + (n_double - X_plus_k) * splineArray[k - 1];
    }
    splineArray[0] = X * splineArray[0];  // For k=0 -> S_(n-1)(x-1) is zero by definition
  }

  // Spline is now correct except for the single multiplicative factor of 1/(n-1)!
  int SplineSize = splineArray.size();
  for (int Index = 0; Index < SplineSize; ++Index) {
    splineArray[Index] *= denominator;
  }

  return splineArray;
}
