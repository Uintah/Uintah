/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef PARTICLE_INTERPOLATOR_H
#define PARTICLE_INTERPOLATOR_H

#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <vector>

#include <Core/Grid/Variables/NCVariable.h>
namespace Uintah {

  class Patch;
  class Stencil7;
  using SCIRun::Vector;
  using SCIRun::IntVector;
  using SCIRun::Point;
  using std::vector;

  class ParticleInterpolator {
    
  public:
    
    ParticleInterpolator() {};
    virtual ~ParticleInterpolator() {};
    
    virtual ParticleInterpolator* clone(const Patch*) = 0;
    
    virtual void findCellAndWeights(const Point& p,
                                    vector<IntVector>& ni, 
                                    vector<double>& S,
                                    const Matrix3& size, 
                                    const Matrix3& defgrad) = 0;
                                    

                                    
    virtual void findCellAndShapeDerivatives(const Point& pos,
                                             vector<IntVector>& ni,
                                             vector<Vector>& d_S,
                                             const Matrix3& size, 
                                             const Matrix3& defgrad=Matrix3(0)) = 0;
                                             
    virtual void findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                       vector<IntVector>& ni,
                                                       vector<double>& S,
                                                       vector<Vector>& d_S,
                                                       const Matrix3& size,
                                                       const Matrix3& defgrad) = 0;

   //__________________________________
   //  Needed for AMRMPM
    virtual void findCellAndWeights(const Point& p,vector<IntVector>& ni,
                                    vector<double>& S,
                                    constNCVariable<Stencil7>& zoi,
                                    constNCVariable<Stencil7>& zoi_fine,
                                    const bool& getFiner,
                                    int& num_cur,int& num_fine,int& num_coarse,                                     
                                    const Vector& size, bool coarse_part,
                                    const Patch* patch) {};
                                    
    virtual void findCellAndWeights_CFI(const Point& pos,
                                        vector<IntVector>& ni,
                                        vector<double>& S,
                                        constNCVariable<Stencil7>& zoi) {};
                                    
    virtual void findCellAndWeightsAndShapeDerivatives_CFI(
                                            const Point& pos,
                                            vector<IntVector>& CFI_ni,
                                            vector<double>& S,
                                            vector<Vector>& d_S,
                                            constNCVariable<Stencil7>& zoi) {};
    virtual int size() = 0;
  };
}

#endif

