#include <Packages/Uintah/CCA/Components/ICE/ConservationTest.h>
#include <Packages/Uintah/Core/Variables/CellIterator.h>

using namespace Uintah;
namespace Uintah {
/*______________________________________________________________________
Function  conservation test
Purpose:  computes sum( mass *q_CC interior + mass*q_CC fluxing)
______________________________________________________________________*/
void  conservationTest(const Patch* patch,
                       const double& delT,
                       CCVariable<double>& mass_q_CC,
                       constSFCXVariable<double>& uvel_FC,
                       constSFCYVariable<double>& vvel_FC,
                       constSFCZVariable<double>& wvel_FC,
                       double& sum)
{
  //__________________________________
  // sum all the interior
  double sum_interior = 0;

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
    IntVector c = *iter;
    sum_interior += mass_q_CC[c];
  }

  //__________________________________
  // sum the fluxes crossing the boundaries of the
  // computational domain
  double sum_fluxes = 0.0;

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
}  // using namespace Uintah
