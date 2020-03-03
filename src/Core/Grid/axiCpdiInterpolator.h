/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#ifndef AXICPDI_INTERPOLATOR_H
#define AXICPDI_INTERPOLATOR_H

#include <Core/Grid/ParticleInterpolator.h>
#include <Core/Grid/cpdiInterpolator.h>

namespace Uintah {

  class Patch;

  class axiCpdiInterpolator : public cpdiInterpolator {
    
  public:
    
    axiCpdiInterpolator();
    axiCpdiInterpolator(const Patch* patch);
    virtual ~axiCpdiInterpolator();
    
    virtual axiCpdiInterpolator* clone(const Patch*);
    
    virtual int findCellAndWeights(const Point& p,std::vector<IntVector>& ni,
                                   std::vector<double>& S, const Matrix3& size);

    virtual int findCellAndShapeDerivatives(const Point& pos,
                                            std::vector<IntVector>& ni,
                                            std::vector<Vector>& d_S,
                                            const Matrix3& size);

    virtual int findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                     std::vector<IntVector>& ni,
                                                     std::vector<double>& S,
                                                     std::vector<Vector>& d_S,
                                                     const Matrix3& size);

    virtual int size();
    
  private:
    const Patch* d_patch;
    int d_size;
    
  };
}

#endif
