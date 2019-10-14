/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#include "ParticleWallBC.h"

void
ParticleWallBC::
evaluate()
{
  using namespace SpatialOps;
  using namespace std;
  ParticleField& f = this->value();
  if ((this->boundaryParticles_)->size() > 0) {
    const IntVec& unitNormal = this->bndNormal_;
    const double sign = (unitNormal[0] >=0 && unitNormal[1] >= 0 && unitNormal[2] >= 0) ? 1.0 : -1.0;
    vector<int>::const_iterator pIt = (this->boundaryParticles_)->begin();
    vector<int>::const_iterator pItEnd = (this->boundaryParticles_)->end();
    
    if (transverseVelocity_) {
      for (; pIt != pItEnd; ++pIt) {
        double& vel = f(*pIt, 0, 0);
        vel *= restCoef_;
      }
    } else {
      for (; pIt != pItEnd; ++pIt) {
        double& vel = f(*pIt, 0, 0);
        if ( sign*vel > 0.0 ) { // if the particle is attempting to leave this boundary, then reflect it :)
          vel *= -restCoef_;
        }
      }
    }
  }
}
