/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#include <CCA/Components/MD/Electrostatics/Ewald/InverseSpace/SPME/ShiftedCardinalBSpline.h>
#include <CCA/Components/MD/MDUtil.h>
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

void ShiftedCardinalBSpline::evaluateThroughSecondDerivative(
    const SCIRun::Vector& inputValues,
    std::vector<SCIRun::Vector>& base,
    std::vector<SCIRun::Vector>& first,
    std::vector<SCIRun::Vector>& second) const
{
	// Generate the raw splines of two orders lower than our base order
	vector<SCIRun::Vector> rawSpline(d_splineOrder + 1, SCIRun::Vector(0.0));
	evaluateVectorizedInternalInPlace(inputValues, d_splineOrder - 2, rawSpline);

	// Set the index backstop for the raw Spline's current order
	size_t currentMaxIndex = d_splineOrder - 2; 
	
	SCIRun::Vector p(d_splineOrder);
	double firstDenom = 1.0/static_cast<double> (d_splineOrder - 2);
	double baseDenom = firstDenom / static_cast<double> (d_splineOrder -1);

	// rawSpline is currently order d_splineOrder - 2 to allow calculation 
	//   of 2nd derivative

    /*	d2S
      	---  =  S   (u) - 2S   (u-1) + S   (u-2)
    	du2      p-2        p-2         p-2  */

	/*  dS       1 /                                                \
	    ---  =  ---| u * S   (u) + (p-2u)*S   (u-1) + (p-u)S   (u-2) |
	    du      p-2\      p-2              p-2              p-2     /  */

	/*           1    1  /  2                                      2        \
	    S(u) =  ---  --- | u S   (u) + (2u(p-u)-p)S   (u-1) + (p-u) S   (u-2)|
	           (p-2)(p-1)\    p-2                  p-2               p-2    / */

	SCIRun::Vector u = inputValues;
	second[0]=                                  rawSpline[0];
    first[0] =                                u*rawSpline[0];
    base[0]  =                              u*u*rawSpline[0];

    u = u + MDConstants::V_ONE;
    SCIRun::Vector pMinusu = p - u;
    second[1]=         rawSpline[1]  -                2*rawSpline[0];
	first[1] =       u*rawSpline[1]  +          (p-2*u)*rawSpline[0];
	base[1]  =     u*u*rawSpline[1]  +  (2*u*pMinusu-p)*rawSpline[0];
	for (size_t Index=2; Index <= currentMaxIndex; ++Index) {
	    u = u + MDConstants::V_ONE;
	    pMinusu = p - u;
		second[Index] =   rawSpline[Index]
		              - 2*rawSpline[Index-1]
		              +   rawSpline[Index-2];

		first[Index]  =          u*rawSpline[Index]
		               +   (p-2*u)*rawSpline[Index-1]
		               -   pMinusu*rawSpline[Index-2];

		base[Index]   =              u*u*rawSpline[Index]
		               + (2*u*pMinusu-p)*rawSpline[Index-1]
		               + pMinusu*pMinusu*rawSpline[Index-2];
	}

    u = u + MDConstants::V_ONE;
    pMinusu = p - u;
	second[currentMaxIndex+1] =               -2*rawSpline[currentMaxIndex]
	                            +                rawSpline[currentMaxIndex-1];
    first[currentMaxIndex+1]  =          (p-2*u)*rawSpline[currentMaxIndex]
                                -      (pMinusu)*rawSpline[currentMaxIndex-1];
    base[currentMaxIndex+1]   =  (2*u*pMinusu-p)*rawSpline[currentMaxIndex]
                                +pMinusu*pMinusu*rawSpline[currentMaxIndex-1];

    u = u + MDConstants::V_ONE;
    pMinusu = p - u;
	second[currentMaxIndex+2] =                 rawSpline[currentMaxIndex];
    first[currentMaxIndex+2]  =        -pMinusu*rawSpline[currentMaxIndex];
    base[currentMaxIndex+2]   = pMinusu*pMinusu*rawSpline[currentMaxIndex];

    for (size_t Index=0; Index < base.size(); ++Index)
    {
      base[Index] *= baseDenom;
      first[Index]*= firstDenom;
    }
//	// Bring raw Spline up to order d_splineOrder - 1
////           u            (p-1)-u
////	S (u) = --- S   (u) + ------- S   (u-1)
////	 p      p-2  p-2        p-2    p-2
//    ++currentMaxIndex;
//	rawSpline[currentMaxIndex] = -inputValues * rawSpline[currentMaxIndex - 1];
//	SCIRun::Vector rightmost = SCIRun::Vector(static_cast<double>(d_splineOrder - 1));
//	for ( int k = currentMaxIndex - 1; k > 0; --k) {
//		SCIRun::Vector input_plus_k = inputValues + SCIRun::Vector(static_cast<double>(k));
//		rawSpline[k] = input_plus_k * rawSpline[k] + (rightmost - input_plus_k) * rawSpline[k-1];
//	}
//	rawSpline[0] = inputValues * rawSpline[0];
//	double denom = 1.0/static_cast<double> (d_splineOrder - 1);
//	for (size_t Index = 0; Index <= currentMaxIndex; ++Index) {
//		rawSpline[Index] *= denom;
//	}
//
//	//raw Spline is now order p-1
////	dS
////	-- = S   (u) - S   (u-1)
////	du    p-1       p-1
//	first[0] = rawSpline[0];
//	for (size_t Index=1; Index <= currentMaxIndex; ++Index) {
//		first[Index] = rawSpline[Index] - rawSpline[Index-1];
//	}
//	first[currentMaxIndex+1] = - rawSpline[currentMaxIndex];
//
//	// Bring raw Spline up to order d_splineOrder
////	          u            p - u
////	S  (u) = --- S   (u) + ----- S   (u-1)
////	  p      p-1  p-1      p - 1  p-1
//	++currentMaxIndex;
//	rawSpline[currentMaxIndex] = -inputValues * rawSpline[currentMaxIndex - 1];
//	rightmost = SCIRun::Vector(static_cast<double> (d_splineOrder));
//	for ( int k = currentMaxIndex - 1; k > 0; --k) {
//		SCIRun::Vector input_plus_k = inputValues + SCIRun::Vector(static_cast<double>(k));
//		rawSpline[k] = input_plus_k * rawSpline[k] + (rightmost - input_plus_k) * rawSpline[k-1];
//	}
//	rawSpline[0] = inputValues * rawSpline[0];
//	denom = 1.0/static_cast<double> (d_splineOrder);
//	for (size_t Index = 0; Index <= currentMaxIndex; ++Index) {
//		base[Index] = rawSpline[Index]*denom;
//	}
    return;
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
void ShiftedCardinalBSpline::evaluateVectorizedInternalInPlace
     ( const SCIRun::Vector& offset,
       const int Order,
       vector<SCIRun::Vector>& array) const
{
  // Initialize the second order spline into the array
  int arraySize = array.size();
  assert(arraySize >= (Order + 1));

  array.assign(arraySize,SCIRun::Vector(0.0,0.0,0.0));
  array[0] = S2(offset);
  array[1] = S2(offset + SCIRun::Vector(1));
  array[2] = S2(offset + SCIRun::Vector(2));

  double denominator = 1.0;
  for (int n = 3; n <= Order; ++n) {
    int n_minus_1 = n - 1;
    denominator /= static_cast<double> (n_minus_1);
    array[n] = -offset * array[n_minus_1];
    SCIRun::Vector n_double = SCIRun::Vector(n);
    for (int k = n - 1; k > 0; --k) {
      SCIRun::Vector offset_plus_k = offset + SCIRun::Vector(k);
      array[k] = offset_plus_k * array[k] +
                   (n_double - offset_plus_k) * array[k-1];
    }
    array[0] = offset * array[0];
  }
  size_t splineSize = array.size();
  for (size_t Index = 0; Index < splineSize; ++Index) {
    array[Index] *= denominator;
  }
}

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
   *   S (x)  =  ---  S  (x) +   ---  S  (x-1)
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
