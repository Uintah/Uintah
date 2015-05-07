/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef AXI_LINEAR_INTERPOLATOR_H
#define AXI_LINEAR_INTERPOLATOR_H

#include <Core/Math/MiscMath.h>
#include <Core/Grid/ParticleInterpolator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <vector>

namespace Uintah {

  using namespace SCIRun;

  class AxiLinearInterpolator : public ParticleInterpolator {
    
  public:
    
    AxiLinearInterpolator();
    AxiLinearInterpolator(const Patch* patch);
    virtual ~AxiLinearInterpolator();
    
    virtual AxiLinearInterpolator* clone(const Patch*);

    virtual void findCellAndWeights(const Point& p,
                                    std::vector<IntVector>& ni,
                                    std::vector<double>& S,
                                    const Matrix3& size,
                                    const Matrix3& defgrad);
                                
    virtual void findCellAndShapeDerivatives(const Point& pos,
                                             std::vector<IntVector>& ni,
                                             std::vector<Vector>& d_S,
                                             const Matrix3& size,
                                             const Matrix3& defgrad);
                                        
    virtual void findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                       std::vector<IntVector>& ni,
                                                       std::vector<double>& S,
                                                       std::vector<Vector>& d_S,
                                                       const Matrix3& size,
                                                       const Matrix3& defgrad);
    virtual int size();

  private:
    const Patch* d_patch;
    int d_size;
  };
}

#endif

