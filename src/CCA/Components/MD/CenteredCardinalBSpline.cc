/*
 * The MIT License
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

#include <CCA/Components/MD/CenteredCardinalBSpline.h>

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <Core/Util/FancyAssert.h>

#ifdef DEBUG
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#endif

using namespace Uintah;

CenteredCardinalBSpline::CenteredCardinalBSpline()
{

}

CenteredCardinalBSpline::~CenteredCardinalBSpline()
{

}

CenteredCardinalBSpline::CenteredCardinalBSpline(const int order) :
    d_splineOrder(order)
{
  // For ease of use, we actually generate quantities to calculate the spline
  //   of order SplineOrder - 1
  d_basisShifts = generateBasisShifts(order - 1);
  d_prefactorValues = generatePrefactorValues(order - 1);
  d_prefactorMap = generatePrefactorMap(order - 1, d_basisShifts);

#ifdef DEBUG
  std::cout << " Basis Shifts: " << std::endl;
  for (size_t idx = 0; idx < d_basisShifts.size(); ++idx) {
    std::cout << std::setw(5) << idx;
  }
  std::cout << std::endl;
  for (size_t idx = 0; idx < d_basisShifts.size(); ++idx) {
    std::cout << std::setw(5) << d_basisShifts[idx];
  }
  std::cout << std::endl;
  std::cout << " Prefactor Values: " << std::endl;
  for (size_t idx = 0; idx < d_prefactorValues.size(); ++idx) {
    std::cout << std::setw(10) << idx;
  }
  std::cout << std::endl;
  for (size_t idx = 0; idx < d_prefactorValues.size(); ++idx) {
    std::cout << std::setw(10) << std::setprecision(5) << d_prefactorValues[idx];
  }
  std::cout << std::endl;
  std::cout << " Prefactor Map: " << std::endl;
  int origin = 0;
  int terms = static_cast<int>(pow(2.0, order - 1));
  for (int pass = 0; pass < order - 1; ++pass) {
    std::cout << " Order " << std::setw(3) << pass << ": ";
    for (size_t termidx = 0; termidx < terms; ++termidx) {
      std::cout << std::setw(5) << d_prefactorMap[origin + termidx];
    }
    std::cout << std::endl;
    origin += terms;
    terms = terms >> 1;
  }
#endif
}

std::vector<double> CenteredCardinalBSpline::evaluateGridAligned(const double x) const
{
  // Expect standardized input between -1.0 and 0.0 (inclusive)
  ASSERTRANGE(x, -1.0, 1e-15);

  // For uniformity we will embed the spline in a vector that extends to the maximum support for the entire domain.
  int HalfMax = this->getHalfMaxSupport();

  // To center the vector, we need to include support up to maximum HalfMax on each side
  std::vector<double> paddedSpline(2 * HalfMax + 1, 0.0);

  // First let's calculate the base spline values across the entire possible spline support by stripping off any shifts of the spline
  int xShift = static_cast<int>(x);
  double xBase = x - xShift;

  // xBase should be between -1.0 and 1.0 now, however the spline is defined on the interval (-0.5,0.5).  Let's wrap xBase into that
  if (xBase < -0.5) {
    xBase += 1.0;
    xShift -= 1;
  };
  if (xBase > 0.5) {
    xBase -= 1.0;
    xShift += 1;
  };

  /*
   * Consider the following cases: Spline Order = 6 (Spline is defined from -3.5 to 3.5)
   *   x = -1.75; xShift = int (-1.75) = -1; xBase = -1.75 - (-1) = -0.75
   *   xBase < -0.5 ? xBase += 1.0 : xBase = 0.25; xShift -= 1 : xShift = -2
   * Original support points          :  -2.75 <- -1.75 -> -0.75 -> 0.25 -> 1.25 -> 2.25 -> 3.25
   * Original support array positions :    -1        0       1       2       3       4       5
   *
   * Unshifted support points         :  -2.75 <- -1.75 <- -0.75 <- 0.25 -> 1.25 -> 2.25 -> 3.25
   * Unshifted support array positiosn:   -3       -2       -1       0       1       2       3
   *
   * Array is the same, but indices are shifted by -xShift
   */
  // subSpline is the spline of order d_splineOrder - 1
  std::vector<double> subSpline;
  if (d_splineOrder % 2 == 1) {
    subSpline = evaluateInternal(xBase + 0.5, d_splineOrder - 1, d_basisShifts, d_prefactorMap, d_prefactorValues);
  } else {
    subSpline = evaluateInternal(xBase - 0.5, d_splineOrder - 1, d_basisShifts, d_prefactorMap, d_prefactorValues);
  }
  /*
   *  S_p(x) = (1.0/p) [ {(p+1)/2 + x} * S_p-1(x+1/2) + {(p-1)/2 - x}*S_p-1(x-1/2) ]
   *
   *  With shifts ( x -> x + 0.5*q), q = ..., -6, -4, -2, 0, 2, 4, 6, ...
   *    (We shift only unit intervals to coincide with grid points,
   *       and use 0.5 * 2 * the shift to align with the p+/-1 term in the prefactor)
   *
   *  S_p( x + q/2) = (1.0/p) [ {(p+1)/2 + (x + q/2)} * S_p-1(x + q/2 + 1/2) + {(p+1)/2 - (x + q/2)} * S_p-1(x + q/2 - 1/2) ]
   *                = (1.0/p) [ {(p+1+q)/2 + x } * S_p-1(x + q/2 + 1/2) + {(p+1-q)/2 + x } * S_p-1(x + q/2 + 1/2) ]
   *   let l_p = (p+q+1)/2 + x
   *       r_p = (p-q+1)/2 - x
   */

  /* Left support is how many values we have from zero to the left edge of the spline.
   *   2 * number of integral shifts = q
   */
  int q = 2 * leftSupport(xBase, d_splineOrder);
  double l_p = static_cast<double>(d_splineOrder + q + 1) / 2.0 + xBase;
  double r_p = static_cast<double>(d_splineOrder - q + 1) / 2.0 + xBase;
  double scale = 1.0 / (static_cast<double>(d_splineOrder));

  // Calculate spline of p^th order
  std::vector<double> splineCurrentOrder;
  // subSpline[0] represents the left most part of the subSpline; if it is x+1/2 then x-1/2 must be zero, so the first point
  //   in the total spline has only the S_p-1(x+1/2) term (i.e. the left term, paradoxically)
  splineCurrentOrder.push_back(scale * (l_p * subSpline[0]));
  // from here until we get to the last term, all terms of the p^th order spline have two sub-terms from the (p-1)^th order spline
  size_t subTerms = subSpline.size();
  for (size_t subIndex = 1; subIndex < subTerms; ++subIndex) {
    r_p -= 1.0;
    l_p += 1.0;
    // At any Index, subSpline[Index] = S_(p-1)(x+1/2), subSpline[Index-1] = S_(p-1)(x-1/2)
    splineCurrentOrder.push_back(scale * (l_p * subSpline[subIndex] + r_p * subSpline[subIndex - 1]));
  }
  // At the right end of the sub-spline, S_(p-1)(x+1/2) is zero.
  r_p -= 1.0;
  splineCurrentOrder.push_back(scale * (r_p * subSpline[subTerms - 1]));

  // We now have the non-shifted spline, let's embed it in an appropriate zero-padded vector
  int dataSize = splineCurrentOrder.size();
  for (size_t baseIndex = 0; baseIndex < dataSize; ++baseIndex) {
    ASSERTRANGE(baseIndex-xShift, 0, paddedSpline.size());
    paddedSpline[baseIndex - xShift] = splineCurrentOrder[baseIndex];  //FIXME Not right
  }
  return paddedSpline;
}

std::vector<double> CenteredCardinalBSpline::evaluate(const double x) const
{
  // Expect standardized input between -1.0 and 0.0 (inclusive)
  assert(x >= -1.0 && x <= 0.0);

  std::vector<double> subSpline;
  if (d_splineOrder % 2 == 1) {
    subSpline = evaluateInternal(x + 0.5, d_splineOrder - 1, d_basisShifts, d_prefactorMap, d_prefactorValues);
  } else {
    subSpline = evaluateInternal(x - 0.5, d_splineOrder - 1, d_basisShifts, d_prefactorMap, d_prefactorValues);
  }

  int leftQ = 2 * leftSupport(x, d_splineOrder);
  double leftPrefactor = static_cast<double>(d_splineOrder + leftQ + 1) / 2.0 + x;
  double rightPrefactor = static_cast<double>(d_splineOrder - leftQ + 1) / 2.0 - x;

  std::vector<double> fullSpline;
  double scale = 1.0 / (static_cast<double>(d_splineOrder));

  // X is at the edge of the support, so X-1/2 contributes 0
  fullSpline.push_back(scale * (leftPrefactor * subSpline[0]));

  size_t subTerms = subSpline.size();
  for (size_t idx = 1; idx < subTerms; ++idx) {
    rightPrefactor -= 1.0;
    leftPrefactor += 1.0;
    // At any Index, SubSpline[Index] = S_(p-1)(x+1/2), SubSpline[Index-1] = S_(p-1)(x-1/2)
    fullSpline.push_back(scale * (leftPrefactor * subSpline[idx] + rightPrefactor * subSpline[idx - 1]));
  }
  rightPrefactor -= 1.0;
  leftPrefactor += 1.0;
  fullSpline.push_back(scale * (rightPrefactor * subSpline[subTerms - 1]));

  return fullSpline;
}

std::vector<double> CenteredCardinalBSpline::derivativeGridAligned(const double x) const
{
  // Expect standardized input between -1.0 and 0.0 (inclusive)
  ASSERTRANGE(x, -1.0, 1e-15);

  int HalfMax = this->getHalfMaxSupport();
  std::vector<double> paddedDeriv(2 * HalfMax + 1, 0.0);

  int xShift = static_cast<int>(x);
  double xBase = x - xShift;

  if (xBase < -0.5) {
    xBase += 1.0;
    xShift -= 1;
  };
  if (xBase > 0.5) {
    xBase -= 1.0;
    xShift += 1;
  };

  std::vector<double> subSpline = evaluateInternal(x + 0.5, d_splineOrder - 1, d_basisShifts, d_prefactorMap, d_prefactorValues);
  std::vector<double> splineDeriv;

  splineDeriv.push_back(2.0 * subSpline[0]);

  size_t subTerms = subSpline.size();
  for (size_t subIndex = 1; subIndex < subTerms; ++subIndex) {
    splineDeriv.push_back(2.0 * (subSpline[subIndex] - subSpline[subIndex - 1]));
  }

  splineDeriv.push_back(-2.0 * subSpline[subTerms - 1]);

  int dataSize = splineDeriv.size();
  for (size_t baseIndex = 0; baseIndex < dataSize; ++baseIndex) {
    ASSERTRANGE(baseIndex-xShift, 0, paddedDeriv.size());
    paddedDeriv[baseIndex - xShift] = splineDeriv[baseIndex];
  }
  return paddedDeriv;

}
std::vector<double> CenteredCardinalBSpline::derivative(const double x) const
{
  // Expect standardized input between -1.0 and 0.0 (inclusive)
  assert(x >= -1.0 && x <= 0.0);

  std::vector<double> subSpline = evaluateInternal(x + 0.5, d_splineOrder - 1, d_basisShifts, d_prefactorMap, d_prefactorValues);

  std::vector<double> splineDeriv;

  // Derivative of the centered cardinal B spline, S_p(x) is 2.0*(S_(p-1)(x+1/2) - S_(p-1)(x-1/2))

  // First term, S_(p-1)(x-1/2) is 0
  splineDeriv.push_back(2.0 * subSpline[0]);

  size_t subTerms = subSpline.size();
  for (size_t idx = 1; idx < subTerms; ++idx) {
    splineDeriv.push_back(2.0 * (subSpline[idx] - subSpline[idx - 1]));
  }

  // Last term, S_(p-1)(x+1/2) is 0
  splineDeriv.push_back(-2.0 * subSpline[subTerms - 1]);

  return splineDeriv;
}

std::vector<int> CenteredCardinalBSpline::generateBasisShifts(const int order)
{
  int NumberOfTerms = static_cast<int>(pow(2.0, order));
  std::vector<int> shifts(NumberOfTerms, 0);

  int numPartitions = 1;
  int termsPerPartition = NumberOfTerms >> 1;
  for (int currOrder = 0; currOrder < order; ++currOrder) {
    for (int Partition = 0; Partition < numPartitions; ++Partition) {
      int origin = 2 * termsPerPartition * Partition;
      for (int Term = origin; Term < origin + termsPerPartition; ++Term) {
        shifts[Term] += 1;
        shifts[Term + termsPerPartition] -= 1;
      }
    }
    numPartitions = numPartitions << 1;
    termsPerPartition = termsPerPartition >> 1;
  }
  return shifts;
}

std::vector<int> CenteredCardinalBSpline::generatePrefactorMap(const int order,
                                                               const std::vector<int>& shifts)
{
  // Generates the offset map to map the appropriate prefactor into the constant shift value array
  size_t mapSize = static_cast<int>(pow(2.0, order));
  std::vector<int> map(2 * mapSize, 0);

  int origin = 0;
  size_t stepSize = 1;
  size_t numberOfTerms = mapSize;
  int currentOrder = 0;

  for (int pass = 0; pass < order; ++pass) {
    for (size_t term = 0; term < numberOfTerms; term += 2 * stepSize) {
      int leftSum = 0;
      int rightSum = 0;
      for (size_t intraTerm = 0; intraTerm < stepSize; ++intraTerm) {
        leftSum += shifts[term + intraTerm];
        rightSum += shifts[term + intraTerm + stepSize];
      }
      map[origin] = currentOrder + leftSum / stepSize;
      origin++;

      map[origin] = currentOrder - rightSum / stepSize;
      origin++;
    }
    stepSize = stepSize << 1;
    ++currentOrder;
  }
  return map;
}

std::vector<double> CenteredCardinalBSpline::generatePrefactorValues(const int order)
{
  // Max in either direction is Order+1 integral gradations, for 2*(Order+1) terms with
  //   a term every 0.5
  int maxTerm = 2 * (order + 1);

  std::vector<double> values;
  for (int idx = -maxTerm; idx <= maxTerm; ++idx) {
    values.push_back(static_cast<double>(idx) / 2.0);
  }

  return values;
}

std::vector<double> CenteredCardinalBSpline::evaluateInternal(const double x,
                                                              const int order,
                                                              const std::vector<int>& shifts,
                                                              const std::vector<int>& map,
                                                              const std::vector<double>& values) const
{
  int leftOffset = leftSupport(x, order);
  int rightOffset = rightSupport(x, order);
  int totalOffset = (rightOffset - leftOffset) + 1;
  size_t shiftSize = values.size();

  std::vector<double> shiftPlusX = values;
  std::vector<double> shiftMinusX(shiftSize, 0.0);

  for (size_t shiftIdx = 0; shiftIdx < shiftSize; ++shiftIdx) {
    shiftPlusX[shiftIdx] += x;
    // Invert the order of the ShiftMinusX array
    shiftMinusX[shiftIdx] = values[(shiftSize - 1) - shiftIdx] - x;
  }

  // Construct the mask for the extension of the spline value to nearby
  //   grid points.
  std::vector<int> mask;
  for (int maskIdx = leftOffset; maskIdx <= rightOffset; ++maskIdx) {
    mask.push_back(2 * maskIdx);
  }

  int zeroIndex = (shiftSize + 1) >> 1;

  double* left = &shiftPlusX[zeroIndex];
  double* right = &shiftMinusX[(shiftSize - 1) - zeroIndex];

  size_t numTerms = shifts.size();

  // Row - One complete spline array for a given term
  // Column - All terms comprising the subsplines used to make the current order
  std::vector < std::vector<double> > valTemp2D;
  for (size_t vtidx = 0; vtidx < numTerms; ++vtidx) {
    std::vector<double> vtRow(totalOffset, 0.0);
    valTemp2D.push_back(vtRow);
  }

  // Populate the array with the initial shifted basis function values
  for (size_t termIdx = 0; termIdx < numTerms; ++termIdx) {
    for (int supportIdx = leftOffset; supportIdx <= rightOffset; ++supportIdx) {
      valTemp2D[termIdx][supportIdx - leftOffset] = S0(
          x + 0.5 * (static_cast<double>(shifts[termIdx] + mask[supportIdx - leftOffset])));
    }
  }

  // Propagate the recursive calculation of splines from the above initial basis values
  int origin = 0;
  double scale = 1.0;
  for (int pass = 0; pass < order; ++pass) {
    scale *= (static_cast<double>(pass + 1));
    for (size_t TermIndex = 0; TermIndex < numTerms; TermIndex += 2) {
      int leftBase = map[origin + TermIndex];
      int rightBase = -map[origin + TermIndex + 1];
      for (int supportIdx = leftOffset; supportIdx <= rightOffset; ++supportIdx) {
        int shiftedSupport = supportIdx - leftOffset;
        valTemp2D[TermIndex / 2][shiftedSupport] = (left[leftBase + mask[shiftedSupport]] * valTemp2D[TermIndex][shiftedSupport]
                                                    + right[rightBase + mask[shiftedSupport]]
                                                      * valTemp2D[TermIndex + 1][shiftedSupport]);
      }
    }
    origin += numTerms;
    numTerms = numTerms >> 1;
  }
  std::vector<double> results = valTemp2D[0];
  for (size_t Idx = 0; Idx < results.size(); ++Idx) {
    results[Idx] /= scale;
  }
  return results;
}
