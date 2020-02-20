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

/* Convected Particle Tetrahedral Domain (CPTI) Integrator

Based on the CPDI integrator in the paper by Sadeghirad, Burghardt and Brannon
"A convected particle domain interpolation technique to extend applicability of 
the material point method for problems involving massive deformations"
International Journal for Numerical Methods in Engineering
Volume 86, Issue 12, pages 1435–1456, 24 June 2011

CPTI is an extension to allow for conforming particle domain polygons,
in this case a triangular/tetrahedral particle domain description.

An additional feature of this implementation, not described above, is the
ability to restrict the particle domains from exceeding a user specified
length, defined here as "lcrit".  An algorithm developed by Michael Homel and adapted by
Rebecca Brannon and Brian Leavy for tetrahedron, is used to scale the deformed particle
such that no corners of that particle will fall outside of a sphere with radius lcrit
with the particle.  This feature was added to avoid particles from getting
so large that they have influence with nodes that lie beyond the ghost nodes
of neighboring patches, or outside of the computational domain, as they approach
node boundaries.  Note that lcrit is a dimension relative to the cell size.
Thus, lcrit=1 implies that a particle can have no length as measured from the
center to any corner that exceeds the side length of a computational cell.

*/

#ifndef CPTI_INTERPOLATOR_H
#define CPTI_INTERPOLATOR_H

#include <Core/Grid/ParticleInterpolator.h>

namespace Uintah {

  class Patch;

  class cptiInterpolator : public ParticleInterpolator {
    
  public:
    
    cptiInterpolator();
    cptiInterpolator(const Patch* patch);
    cptiInterpolator(const Patch* patch, const double lcrit);
    virtual ~cptiInterpolator();
    
    virtual cptiInterpolator* clone(const Patch*);
    
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

    virtual void setLcrit(double lcrit){
      d_lcrit = lcrit;
    }
    
  private:
    const Patch* d_patch;
    int d_size;
    double d_lcrit;
  };
}

#endif

