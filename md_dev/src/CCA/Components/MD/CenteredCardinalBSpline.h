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

#ifndef UINTAH_MD_CENTEREDCARDINALBSPLINE_H
#define UINTAH_MD_CENTEREDCARDINALBSPLINE_H

#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>

#include <vector>

namespace Uintah {

typedef int particleIndex;
typedef int particleId;

using SCIRun::Vector;
using SCIRun::IntVector;

class Point;
class Vector;
class IntVector;

/**
 *  @class CenteredCardinalBSpline
 *  @ingroup MD
 *  @author Alan Humphrey and Justin Hooper
 *  @date   January, 2013
 *
 *  @brief
 *
 *  @param
 */
class CenteredCardinalBSpline {

  public:

    /**
     * @brief
     * @param
     */
    CenteredCardinalBSpline();

    /**
     * @brief
     * @param
     */
    CenteredCardinalBSpline::~CenteredCardinalBSpline()

    /**
     * @brief
     * @param
     */
    CenteredCardinalBSpline(const int& _SplineOrder);

    /**
     * @brief
     * @param
     */
    vector<double> evaluate(const double& X);

    /**
     * @brief
     * @param
     */
    vector<double> derivative(const double& X);

  private:

    /**
     * @brief
     * @param
     */
    vector<int> generateBasisOffsets(const int& SplineOrder);

    /**
     * @brief
     * @param
     */
    vector<int> generatePrefactorMap(const int& SplineOrder);

    /**
     * @brief
     * @param
     */
    vector<double> generatePrefactorValues(const int& SplineOrder);

    int splineOrder;

    // For calculating values of the spline
    vector<double> prefactorValues;
    vector<int> prefactorMap, basisOffsets;

    // For calculating derivatives of the spline (=difference of spline of order SplineOrder - 1)
    vector<double> derivativeValues;
    vector<int> derivativeMap;
    vector<int> derivativeOffsets;

};

}  // End namespace Uintah

#endif
