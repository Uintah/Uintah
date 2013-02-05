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
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <fstream>

using namespace Uintah;
using namespace SCIRun;

CenteredCardinalBSpline::CenteredCardinalBSpline()
{

}

CenteredCardinalBSpline::~CenteredCardinalBSpline()
{

}

double S0(const double& X)
{
  if (X < -0.5)
    return 0;
  if (X >= 0.5)
    return 0;
  return 1;
}

// Spline related functions
double S1(const double& X)
{
  if (X <= -1.0)
    return 0;
  if (X > 1.0)
    return 0;

  return (1.0 - abs(X));
}

CenteredCardinalBSpline::CenteredCardinalBSpline(const int& Order)
{

  splineOrder = Order;

  // Initialize calculation arrays; these need only be calculated once per spline order.
  basisOffsets = generateBasisOffsets(splineOrder);
  prefactorMap = generatePrefactorMap(splineOrder);
  prefactorValues = generatePrefactorValues(splineOrder);

  // Initialize derivative arrays
  derivativeOffsets = generateBasisOffsets(splineOrder - 1);
  derivativeMap = generatePrefactorMap(splineOrder - 1);
  derivativeValues = generatePrefactorValues(splineOrder - 1);

}

vector<double> CenteredCardinalBSpline::generatePrefactorValues(const int& SplineOrder)
{
  // There are innately p-n terms for the spline centered at X=0.
  // To return a vector which will allow for calculation of all the supported points
  //   the spline covers, we need to expand this to account for the support of the spline.
  // The support of a p^th order spline is int<(p+1)/2> in both directions.
  // Therefore there are 2p+p%2-n total terms.

  // For now, we assume n=0 to decompose into selection functions.

  vector<double> PrefactorValues(2 * SplineOrder + SplineOrder % 2);

  double Shift = static_cast<double>(-(SplineOrder - 2) + 1) / 2.0 - static_cast<double>((SplineOrder + 1) / 2);
  for (int Index = 0; Index < PrefactorValues.size(); ++Index) {
    PrefactorValues[Index] = Shift;
    Shift += 1.0;
  }
  return PrefactorValues;
}

vector<int> CenteredCardinalBSpline::generateBasisOffsets(const int& SplineOrder)
{
  // Generates the shift offset values which get fed into the basis functions
  //   The numerical values are the same for either right term or left term prefactors, the difference
  //     occurs when the particular functional value is added/subtracted.  That is calculated in the
  //     Evaluate routine.
  int Terms = static_cast<int>(pow(2.0, SplineOrder));
  vector<int> Offsets(Terms, 0);

  int NumberPartitions = 1;
  int TermExtent = Terms >> 1;
  for (int Order_Current = 0; Order_Current < SplineOrder; ++Order_Current) {
    for (int Partition = 0; Partition < NumberPartitions; ++Partition) {
      int Origin = 2 * TermExtent * Partition;
      for (int ShiftIndex = Origin; ShiftIndex < Origin + TermExtent; ++ShiftIndex) {
        Offsets[ShiftIndex] += 1.0;
        Offsets[ShiftIndex + TermExtent] -= 1.0;
      }
    }
    NumberPartitions = NumberPartitions << 1;
    TermExtent = TermExtent >> 1;
  }
  return Offsets;
}

vector<int> CenteredCardinalBSpline::generatePrefactorMap(const int& SplineOrder)
{
  // Generates the offsets necessary to calculate the prefactors which multiply the basis functions
  int MapSize = static_cast<int>(pow(2.0, SplineOrder));
  vector<int> PrefactorMap(2 * MapSize, 0);

  int Origin = 0;
  int StepSize = 1;
  int NumTerms = MapSize;
  int CurrentOrder = 0;

  for (int Pass = 0; Pass < SplineOrder; ++Pass) {
    for (int ShiftIndex = 0; ShiftIndex < NumTerms; ShiftIndex += 2 * StepSize) {
      int LeftSum = 0;
      int RightSum = 0;
      for (int SubIndex = 0; SubIndex < StepSize; ++SubIndex) {
        LeftSum += basisOffsets[ShiftIndex + SubIndex];
        RightSum += basisOffsets[ShiftIndex + SubIndex + StepSize];
      }
      PrefactorMap[Origin] = CurrentOrder + LeftSum / StepSize;
      Origin++;
      PrefactorMap[Origin] = CurrentOrder - RightSum / StepSize;
      Origin++;
    }
    StepSize = StepSize << 1;
    ++CurrentOrder;
  }
  return PrefactorMap;
}

vector<double> CenteredCardinalBSpline::derivative(const double& X)
{
  // Evaluates the centered cardinal B spline of order SplineOrder-1 at X+0.5 for each support point.
  // The derivative is the difference of the value at X+0.5 and X-0.5. (S_x-1(X+0.5)-S_x-1(X-0.5))
  // JBH-NOTE:  Double check this; the derivative might be S_x-1(X)-S_x-1(X-1)  FIXME

  int DerivativeOrder = splineOrder - 1;
  int ShiftSize = derivativeValues.size();
  int OddShift = DerivativeOrder % 2;
  int Support = (DerivativeOrder + 1 + OddShift);
  int HalfSupport = Support / 2;
  int ZeroIndex = (DerivativeOrder - 1 - OddShift) / 2 + (DerivativeOrder + 1) / 2;

  vector<double> Mask(Support, 0);
  for (int Index = -HalfSupport; Index <= HalfSupport; ++Index) {
    Mask[Index + HalfSupport] = 2 * Index;
  }

  vector<double> ShiftPlus = derivativeValues;
  for (size_t Index = 0; Index < ShiftSize; ++Index) {
    ShiftPlus[Index] += (X + 0.5);
  }
  double* Left = &ShiftPlus[ZeroIndex];

  // Store right coefficients backwards so that increasing mask value has the correct effect
  vector<double> ShiftMinus = derivativeValues;
  for (size_t Index = 0; Index < ShiftSize; ++Index) {
    ShiftMinus[Index] = derivativeValues[ShiftSize - 1 - Index] - (X + 0.5);
  }
  double* Right = &ShiftMinus[(ShiftSize - 1) - ZeroIndex];

  int Terms = derivativeOffsets.size();
  vector<vector<double> > ValTemp2D(Terms, vector<double>(Support, 0.0));

  // Populate the initial array with the appropriate basis response
  for (size_t TermIndex = 0; TermIndex < Terms; ++TermIndex) {
    for (size_t SupportIndex = 0; SupportIndex < Support; ++SupportIndex) {
      ValTemp2D[TermIndex][SupportIndex] = S0(
          (X + 0.5) + 0.5 * (static_cast<double>(derivativeOffsets[TermIndex] + Mask[SupportIndex])));
    }
  }

  int Origin = 0;
  for (size_t Pass = 0; Pass < DerivativeOrder; ++Pass) {
    double Scale = (1.0 / static_cast<double>(Pass + 1));
    for (size_t TermIndex = 0; TermIndex < Terms; TermIndex += 2) {
      int LeftBase = (derivativeMap[Origin + TermIndex] + OddShift) / 2;
      int RightBase = -(derivativeMap[Origin + TermIndex + 1] + OddShift) / 2;

      for (int SupportIndex = -HalfSupport; SupportIndex <= HalfSupport; ++SupportIndex) {
        ValTemp2D[TermIndex / 2][SupportIndex + HalfSupport] = Scale
                                                               * (Left[LeftBase + SupportIndex]
                                                                  * ValTemp2D[TermIndex][SupportIndex + HalfSupport]
                                                                  + Right[RightBase + SupportIndex]
                                                                    * ValTemp2D[TermIndex + 1][SupportIndex + HalfSupport]);
      }
    }
    Origin += Terms;
    Terms = Terms >> 1;
  }

  vector<double> Deriv = ValTemp2D[0];
  for (int SupportIndex = Support; SupportIndex > 0; --SupportIndex) {
    Deriv[SupportIndex] -= Deriv[SupportIndex - 1];
  }  // d(S_x) = S_(x-1)(X+0.5) - S_(x-1)(X-0.5)
  return Deriv;
}

vector<double> CenteredCardinalBSpline::evaluate(const double& X)
{
  // Evaluates the centered cardinal B spline for the value of X at each of the support points
  //   on the grid from -(p+1)/2 ... (p+1)/2

  int ShiftSize = prefactorValues.size();
  int OddShift = splineOrder % 2;
  int Support = (splineOrder + 1 + OddShift);
  int HalfSupport = Support / 2;
  int ZeroIndex = (splineOrder - 1 - OddShift) / 2 + (splineOrder + 1) / 2;

  vector<double> Mask(Support, 0);
  for (int Index = -HalfSupport; Index <= HalfSupport; ++Index) {
    Mask[Index + HalfSupport] = 2 * Index;
  }

  vector<double> ShiftPlus = prefactorValues;
  for (size_t Index = 0; Index < ShiftSize; ++Index) {
    ShiftPlus[Index] += X;
  }
  double* Left = &ShiftPlus[ZeroIndex];

  vector<double> ShiftMinus = prefactorValues;
  // Store right coefficients backwards so we have forward contiguous memory to work from in the loop
  for (size_t Index = 0; Index < ShiftSize; ++Index) {
    ShiftMinus[Index] = prefactorValues[ShiftSize - 1 - Index] - X;
  }
  double* Right = &ShiftMinus[(ShiftSize - 1) - ZeroIndex];

  int Terms = basisOffsets.size();
  // Rows are complete spline arrays for a particular point
  // Columns are the individual sub-terms
  vector<vector<double> > ValTemp2D(Terms, vector<double>(Support, 0.0));

  // Populate the initial array with the appropriate basis response
  for (size_t TermIndex = 0; TermIndex < Terms; ++TermIndex) {
    for (size_t SupportIndex = 0; SupportIndex < Support; ++SupportIndex) {
      ValTemp2D[TermIndex][SupportIndex] = S0(X + 0.5 * (static_cast<double>(basisOffsets[TermIndex] + Mask[SupportIndex])));
    }
  }

  int Origin = 0;
  for (size_t Pass = 0; Pass < splineOrder; ++Pass) {
    double Scale = (1.0 / static_cast<double>(Pass + 1));
    for (size_t TermIndex = 0; TermIndex < Terms; TermIndex += 2) {
      int LeftBase = (prefactorMap[Origin + TermIndex] + OddShift) / 2;
      int RightBase = -(prefactorMap[Origin + TermIndex + 1] + OddShift) / 2;

      for (int SupportIndex = -HalfSupport; SupportIndex <= HalfSupport; ++SupportIndex) {
        double value = Scale
                       * (Left[LeftBase + SupportIndex] * ValTemp2D[TermIndex][SupportIndex + HalfSupport]
                          + Right[RightBase + SupportIndex] * ValTemp2D[TermIndex + 1][SupportIndex + HalfSupport]);
        ValTemp2D[TermIndex / 2][SupportIndex + HalfSupport] = value;
      }
    }
    Origin += Terms;
    Terms = Terms >> 1;
  }
  return ValTemp2D[0];
}

//int main(int argc, char* argv[])
//{
//  vector<string> args(argv,argv+argc);
//
//  int SplineOrder, BasisOrder=0;
//  double X=0.425;
//  switch (argc) {
//    case 1: {
//      cout << "Enter the spline order (p): ";
//      cin  >> SplineOrder;
//      break;
//    }
//    case 2: {
//      istringstream OrderArg(args[1]);
//      OrderArg >> SplineOrder;
//      break;
//    }
//    case 3: {
//      istringstream OrderArg(args[1]);
//      OrderArg >> SplineOrder;
//      istringstream XArg(args[2]);
//      XArg >> X;
//      break;
//    }
//    default: {
//      cout << "Usage:  " << args[0] << " [Spline Order (p) ]  [Anchor Point (X)]" << endl;
//      return 0;
//      break;
//    }
//  }
//
//  CenteredCardinalBSpline Spline(SplineOrder);
//
//
//  vector<double> Results;
//  Results=Spline.Evaluate(X);
//  int Width=20;
//  int Precision=Width-8;
//  cout << "Size of returned results vector: " << Results.size() << endl;
//  for (size_t Index=0; Index < Results.size(); ++Index) { cout << setw(Width) << Index; } cout << endl;
//  for (int Support=-(Results.size()/2); Support <= (Results.size()/2); ++Support) { cout << setw(Width) << Support; } cout << endl;
//  for (size_t Terms=0; Terms < Results.size(); ++Terms) { cout << setw(Width) << setprecision(Precision) << Results[Terms]; } cout << endl;
//
//  return 0;
//}
