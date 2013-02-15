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

#include <CCA/Components/MD/CenteredCardinalBSpline.h>

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace Uintah;

double S0(const double x)
{
  if (x < -0.5) {
    return 0;
  }
  if (x >= 0.5) {
    return 0;
  }
  return 1;
}

CenteredCardinalBSpline::CenteredCardinalBSpline()
{

}

CenteredCardinalBSpline::~CenteredCardinalBSpline()
{

}

CenteredCardinalBSpline::CenteredCardinalBSpline(int order) :
    splineOrder(order)
{
  int oddShift = splineOrder % 2;
  splineSupport = splineOrder + oddShift + 1;
  splineHalfSupport = splineSupport / 2;

  //Initialize calculation arrays; these need only be calculated once per spline order
  basisOffsets = generateBasisOffsets(splineOrder);
  prefactorMap = generatePrefactorMap(splineOrder, basisOffsets);
  prefactorValues = generatePrefactorValues(splineOrder);

  //Initialize derivative arrays
  derivativeOffsets = generateBasisOffsets(splineOrder - 1);
  derivativeMap = generatePrefactorMap(splineOrder - 1, basisOffsets);
  derivativeValues = generatePrefactorValues(splineOrder - 1);
}

std::vector<double> CenteredCardinalBSpline::evaluate(const double x) const
{
  std::vector<double> results = evaluateInternal(x, splineOrder, prefactorMap, basisOffsets, prefactorValues);
  return results;
}

std::vector<double> CenteredCardinalBSpline::derivative(const double x) const
{
  std::cout << "Beginning Derivative. " << std::endl;
  std::vector<double> deriv = evaluateInternal(x + 0.5, splineOrder - 1, derivativeMap, derivativeOffsets, derivativeValues);
  std::cout << "Evaluated Spline. " << std::endl;
  std::cout.flush();

  size_t size = deriv.size();
  for (size_t idx = 0; idx < size; ++idx) {
    std::cout << std::setw(12) << std::setprecision(5) << deriv[idx];
  }
  std::cout << std::endl;

  // If the original spline order is odd, then the integral support of the spline is larger than that of the
  //   derivative, which is of order SplineOrder - 1.  In that case we need to zero-pad the derivative array
  //   before finding the derivative.
  if (splineOrder % 2 == 1) {
    std::cout << "Start padding.. " << std::endl;
    std::vector<double> padded((splineOrder + 2), 0);
    std::cout << "Padded Size: " << padded.size() << " -- Deriv Size: " << size << std::endl;
    for (size_t derivIndex = 0; derivIndex < size; ++derivIndex) {
      padded[1 + derivIndex] = deriv[derivIndex];
    }
    deriv = padded;
    std::cout << "2 .. " << std::endl;
  }

  size = deriv.size();

  for (size_t idx = size - 1; idx > 0; --idx) {
    std::cout << "DerivIndex:   " << idx << ":  " << deriv[idx] << "  --  " << "DerivIndex-1: " << idx - 1 << ":  "
              << deriv[idx - 1] << std::endl;
    deriv[idx] -= deriv[idx - 1];
    deriv[idx] *= 2.0;
  }
  deriv[0] *= 2.0;
  std::cout << "3 .. " << std::endl;

  return deriv;
}

std::vector<int> CenteredCardinalBSpline::generateBasisOffsets(const int order)
{
  // Generates the shift offset values which get fed into the basis functions
  //   The numerical values are the same for either right term or left term prefactors, the difference
  //     occurs when the particular functional value is added/subtracted.  That is calculated in the
  //     Evaluate routine.
  int terms = static_cast<int>(pow(2.0, order));
  std::vector<int> offsets(terms, 0);

  int numPartitions = 1;
  int termExtent = terms >> 1;
  for (int order_Current = 0; order_Current < order; ++order_Current) {
    for (int partition = 0; partition < numPartitions; ++partition) {
      int origin = 2 * termExtent * partition;
      for (int shiftIndex = origin; shiftIndex < origin + termExtent; ++shiftIndex) {
        offsets[shiftIndex] += 1.0;
        offsets[shiftIndex + termExtent] -= 1.0;
      }
    }
    numPartitions = numPartitions << 1;
    termExtent = termExtent >> 1;
  }
  return offsets;
}

std::vector<int> CenteredCardinalBSpline::generatePrefactorMap(const int Order,
                                                               const std::vector<int>& Offset)
{
  // Generates the offsets necessary to calculate the prefactors which multiply the basis functions
  int MapSize = static_cast<int>(pow(2.0, Order));
  std::vector<int> Map(2 * MapSize, 0);

  int Origin = 0;
  int StepSize = 1;
  int NumTerms = MapSize;
  int CurrentOrder = 0;

  for (int Pass = 0; Pass < Order; ++Pass) {
    for (int ShiftIndex = 0; ShiftIndex < NumTerms; ShiftIndex += 2 * StepSize) {
      int LeftSum = 0;
      int RightSum = 0;
      for (int SubIndex = 0; SubIndex < StepSize; ++SubIndex) {
        LeftSum += Offset[ShiftIndex + SubIndex];
        RightSum += Offset[ShiftIndex + SubIndex + StepSize];
      }
      Map[Origin] = CurrentOrder + LeftSum / StepSize;
      Origin++;
      Map[Origin] = CurrentOrder - RightSum / StepSize;
      Origin++;
    }
    StepSize = StepSize << 1;
    ++CurrentOrder;
  }
  return Map;
}

std::vector<double> CenteredCardinalBSpline::generatePrefactorValues(const int order)
{
  // There are innately p-n terms for the spline centered at X=0.
  // To return a vector which will allow for calculation of all the supported points
  //   the spline covers, we need to expand this to account for the support of the spline.
  // The support of a p^th order spline is int<(p+1)/2> in both directions.
  // Therefore there are 2p+p%2-n total terms.

  // For now, we assume n=0 to decompose into selection functions.

  std::vector<double> Values(2 * order + order % 2);

  double Shift = static_cast<double>(-(order - 2) + 1) / 2.0 - static_cast<double>((order + 1) / 2);
  size_t size = Values.size();
  for (size_t idx = 0; idx < size; ++idx) {
    Values[idx] = Shift;
    Shift += 1.0;
  }
  return Values;
}

std::vector<double> CenteredCardinalBSpline::evaluateInternal(const double x,
                                                              const int order,
                                                              const std::vector<int>& map,
                                                              const std::vector<int>& offsets,
                                                              const std::vector<double>& values) const
{
  // Internal evaluation routines so that Evaluate() and Derivative()
  // can be wrappers to a single unified routine.

  size_t shiftSize = values.size();
  int oddShift = order % 2;
  int zeroIndex = (order - 1 - oddShift) / 2 + (order + 1) / 2;

  //  Number of points in total support range and half support range
  size_t support = order + 1 + oddShift;
  int halfSupport = support / 2;

  std::vector<int> mask(support, 0);
  for (int Index = -halfSupport; Index <= halfSupport; ++Index) {
    mask[Index + halfSupport] = 2 * Index;
  }

  // Set up the shift vectors which store the proper multiplicative factors for the spline recursion
  std::vector<double> shiftPlus = values;
  std::vector<double> shiftMinus(shiftSize, 0);

  std::cout << " X2 ... " << std::endl;
  for (size_t Index = 0; Index <= shiftSize; ++Index) {
    shiftMinus[Index] = values[shiftSize - 1 - Index] - x;
    shiftPlus[Index] += x;
  }

  for (size_t Idx = 0; Idx < shiftMinus.size(); ++Idx) {
    std::cout << std::setw(10) << Idx;
  }
  std::cout << std::endl;
  for (size_t Idx = 0; Idx < shiftMinus.size(); ++Idx) {
    std::cout << std::setw(10) << std::setprecision(5) << shiftMinus[Idx];
  }
  std::cout << std::endl;
  std::cout << " X3 ... " << std::endl;
  double* left = &shiftPlus[zeroIndex];
  double* right = &shiftMinus[(shiftSize - 1) - zeroIndex];

  // Rows are complete spline arrays for a particular term
  //   Columns are the sub-terms used to composite the final spline point array
  size_t terms = offsets.size();
  std::vector<std::vector<double> > valTemp2D(terms, std::vector<double>(support, 0.0));

  // Populate the array with the initial shifted basis function values
  for (size_t TermIndex = 0; TermIndex < terms; ++TermIndex) {
    for (size_t SupportIndex = 0; SupportIndex < support; ++SupportIndex) {
      valTemp2D[TermIndex][SupportIndex] = S0(x + 0.5 * (static_cast<double>(offsets[TermIndex] + mask[SupportIndex])));
    }
  }
  std::cout << " X4 ... " << std::endl;

  int origin = 0;
  // Propagate up the recursive structure, summing the current two order o contributions
  //   into one term of order (o+1).
  for (size_t Pass = 0; Pass < order; ++Pass) {
    double Scale = (1.0 / static_cast<double>(Pass + 1));
    for (size_t TermIndex = 0; TermIndex < terms; TermIndex += 2) {
      int LeftBase = (map[origin + TermIndex] + oddShift) / 2;
      int RightBase = -(map[origin + TermIndex + 1] + oddShift) / 2;

      // Calculate all support points at the current level
      for (int SupportIndex = -halfSupport; SupportIndex <= halfSupport; ++SupportIndex) {
        int ShiftedIndex = SupportIndex + halfSupport;  // Shift to be 0 anchored
        valTemp2D[TermIndex / 2][ShiftedIndex] = Scale
                                                 * (left[LeftBase + SupportIndex] * valTemp2D[TermIndex][ShiftedIndex]
                                                    + right[RightBase + SupportIndex] * valTemp2D[TermIndex + 1][ShiftedIndex]);
      }
    }
    origin += terms;
    terms = terms >> 1;
  }
  std::cout << " X5 ... " << std::endl;

  return valTemp2D[0];
}
