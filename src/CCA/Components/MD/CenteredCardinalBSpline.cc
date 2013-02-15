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

using namespace Uintah;
using namespace SCIRun;

double S0(const double& X)
{
  if (X < -0.5)
    return 0;
  if (X >= 0.5)
    return 0;
  return 1;
}

CenteredCardinalBSpline::CenteredCardinalBSpline()
{

}

CenteredCardinalBSpline::~CenteredCardinalBSpline()
{

}

CenteredCardinalBSpline::CenteredCardinalBSpline(int splineOrder) :
    SplineOrder(splineOrder)
{
  int OddShift = SplineOrder % 2;
  SplineSupport = SplineOrder + OddShift + 1;
  SplineHalfSupport = SplineSupport / 2;

  //Initialize calculation arrays; these need only be calculated once per spline order
  BasisOffsets = CenteredCardinalBSpline::GenerateBasisOffsets(SplineOrder);
  PrefactorMap = CenteredCardinalBSpline::GeneratePrefactorMap(SplineOrder);
  PrefactorValues = CenteredCardinalBSpline::GeneratePrefactorValues(SplineOrder);

  //Initialize derivative arrays
  DerivativeOffsets = GenerateBasisOffsets(SplineOrder - 1);
  DerivativeMap = GeneratePrefactorMap(SplineOrder - 1);
  DerivativeValues = GeneratePrefactorValues(SplineOrder - 1);

}

vector<double> CenteredCardinalBSpline::GeneratePrefactorValues(const int order)
{
  // There are innately p-n terms for the spline centered at X=0 where p is the
  //   spline order, and n is the order of the basis splines.  For this code, n=0.
  // To return a vector which accounts for all points in the spline support, we
  //   must expand this number to account for the support itself in both directions.
}
