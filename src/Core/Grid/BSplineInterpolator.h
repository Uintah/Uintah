/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef BSPLINE_INTERPOLATOR_H
#define BSPLINE_INTERPOLATOR_H

#include <Core/Grid/ParticleInterpolator.h>

namespace Uintah {

  class Patch;

  class BSplineInterpolator : public ParticleInterpolator {
    
  public:
    
    BSplineInterpolator();
    BSplineInterpolator(const Patch* patch);
    virtual ~BSplineInterpolator();
    
    virtual BSplineInterpolator* clone(const Patch*);
    
    virtual int findCellAndWeights(const Point& p,std::vector<IntVector>& ni,
                                    std::vector<double>& S, const Matrix3& size,
                                    const Matrix3& defgrad);

    virtual int findCellAndShapeDerivatives(const Point& pos,
                                             std::vector<IntVector>& ni,
                                             std::vector<Vector>& d_S,
                                             const Matrix3& size,
                                             const Matrix3& defgrad);
    virtual int findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                       std::vector<IntVector>& ni,
                                                       std::vector<double>& S,
                                                       std::vector<Vector>& d_S,
                                                       const Matrix3& size,
                                                       const Matrix3& defgrad);
    virtual int size();

    void findNodeComponents(const int& idx, int* xn, int& count,
                            const int& low, const int& hi);

    void getBSplineWeights(double* Sd, const int* xn,
                           const int& low, const int& hi,
                           const int& count, const double& cellpos);

    void getBSplineGrads(double* dSd, const int* xn,
                         const int& low, const int& hi, const int& count,
                         const double& cellpos);

    double evalType1BSpline(const double& cp);
    double evalType2BSpline(const double& cp);
    double evalType3BSpline(const double& cp);

    double evalType1BSplineGrad(const double& cp);
    double evalType2BSplineGrad(const double& cp);
    double evalType3BSplineGrad(const double& cp);

  private:
    const Patch* d_patch;
    int d_size;
    
  };
}

#endif
