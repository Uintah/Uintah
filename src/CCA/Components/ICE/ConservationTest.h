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

#ifndef _CONSERVATIONTEST_H
#define _CONSERVATIONTEST_H

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>

namespace Uintah {
/*______________________________________________________________________
Function  conservation test
Purpose:  computes sum( mass *q_CC interior + mass*q_CC fluxing)
______________________________________________________________________*/
template<class T>
void  conservationTest(const Patch* patch,
                       const double& delT,
                       CCVariable<T>& mass_q_CC,
                       constSFCXVariable<double>& uvel_FC,
                       constSFCYVariable<double>& vvel_FC,
                       constSFCZVariable<double>& wvel_FC,
                       T& sum)
{
  //__________________________________
  // sum all the interior
  T zero(0.0);
  T sum_interior(zero);

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
    IntVector c = *iter;
    sum_interior += mass_q_CC[c];
  }

  //__________________________________
  // sum the fluxes crossing the boundaries of the
  // computational domain
  T sum_fluxes(zero);

  vector<Patch::FaceType>::const_iterator iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  for (iter  = bf.begin(); iter != bf.end(); ++iter){
    Patch::FaceType face = *iter;

    IntVector axes = patch->getFaceAxes(face);
    int P_dir = axes[0];  // principal direction
    double plus_minus_one = (double) patch->faceDirection(face)[P_dir];
    
    Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;    

    if (face == Patch::xminus || face == Patch::xplus) {    // X faces
      for(CellIterator iter=patch->getFaceIterator(face, MEC); 
        !iter.done();iter++) {
        IntVector c = *iter;
        sum_fluxes -= plus_minus_one*uvel_FC[c]*mass_q_CC[c];
      }
    }
    if (face == Patch::yminus || face == Patch::yplus) {    // Y faces
      for(CellIterator iter=patch->getFaceIterator(face, MEC);
        !iter.done();iter++) {
        IntVector c = *iter;
        sum_fluxes -= plus_minus_one*vvel_FC[c]*mass_q_CC[c];
      }
    }
    if (face == Patch::zminus || face == Patch::zplus) {    // Z faces
      for(CellIterator iter=patch->getFaceIterator(face, MEC);
        !iter.done();iter++) {
        IntVector c = *iter;
        sum_fluxes -= plus_minus_one*wvel_FC[c]*mass_q_CC[c];
      }
    }
  }
  sum = sum_interior + delT * sum_fluxes;

  //cout<< " sum: interior " << sum_interior
  //    << " faces " << delT * sum_fluxes
  //    << " sum " << sum << endl;
}
}// End namespace Uintah
#endif
 
