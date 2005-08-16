#ifndef _CONSERVATIONTEST_H
#define _CONSERVATIONTEST_H

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {
/*______________________________________________________________________
Function  conservation test
Purpose:  computes sum( mass *q_CC interior + mass*q_CC fluxing)
______________________________________________________________________*/
template<class T>
void  conservationTest(const Patch* patch,
                       const double& delT,
                       const constCCVariable<T>& mass_q_CC,
                       const constSFCXVariable<double>& uvel_FC,
                       const constSFCYVariable<double>& vvel_FC,
                       const constSFCZVariable<double>& wvel_FC,
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

  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;

    IntVector axes = patch->faceAxes(face);
    int P_dir = axes[0];  // principal direction
    double plus_minus_one = (double) patch->faceDirection(face)[P_dir];

    if (face == Patch::xminus || face == Patch::xplus) {    // X faces
      for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
        !iter.done();iter++) {
        IntVector c = *iter;
        sum_fluxes -= plus_minus_one*uvel_FC[c]*mass_q_CC[c];
      }
    }
    if (face == Patch::yminus || face == Patch::yplus) {    // Y faces
      for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
        !iter.done();iter++) {
        IntVector c = *iter;
        sum_fluxes -= plus_minus_one*vvel_FC[c]*mass_q_CC[c];
      }
    }
    if (face == Patch::zminus || face == Patch::zplus) {    // Z faces
      for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
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
 
