#include <Packages/Uintah/CCA/Components/ICE/SmagorinskyModel.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

using namespace Uintah;

Smagorinsky_Model::Smagorinsky_Model(ProblemSpecP& ps)
{
  //__________________________________
  //typically filter_width=grid spacing(uniform) for implicit filter.
  ps->require("model_constant",d_model_constant);
  ps->require("filter_width",d_filter_width);
//  ps->require("turb_Pr",d_turbPr);  

}

Smagorinsky_Model::Smagorinsky_Model()
{
}

Smagorinsky_Model::~Smagorinsky_Model()
{
}

/* ---------------------------------------------------------------------
  Function~  computeTurbViscosity
  Purpose~ Calculate the turbulent viscosity
  -----------------------------------------------------------------------  */
void Smagorinsky_Model::computeTurbViscosity(DataWarehouse* new_dw,
                                            const Patch* patch,
                                            const CCVariable<Vector>& /*vel_CC*/,
                                            const SFCXVariable<double>& uvel_FC,
                                            const SFCYVariable<double>& vvel_FC,
                                            const SFCZVariable<double>& wvel_FC,
                                            const CCVariable<double>& rho_CC,
                                            const int indx,
                                            SimulationStateP&  d_sharedState,
                                            CCVariable<double>& turb_viscosity)
{
  //__________________________________
  //implicit filter, filter_width=(dx*dy*dz)**(1.0/3.0), 
  //don't use d_filter_width given in input file now, 
  //keep the parameter here for future use
  
  Vector dx = patch->dCell();
  filter_width = pow((dx.x()*dx.y()*dx.z()), 1.0/3.0);
  double term = (d_model_constant * filter_width)
               *(d_model_constant * filter_width);
  
  StaticArray<CCVariable<double> > SIJ(6);
  for (int comp = 0; comp <= 5; comp++) {
    new_dw->allocateTemporary(SIJ[comp], patch,Ghost::AroundCells,2);
    SIJ[comp].initialize(0.0);
  }
   
  computeStrainRate(patch, uvel_FC, vvel_FC, wvel_FC, indx, d_sharedState, SIJ);

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
  
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    IntVector c = *iter;    
       
    turb_viscosity[c]=rho_CC[c] * term * 
    sqrt(2.0 * (SIJ[0][c]*SIJ[0][c] + SIJ[1][c]*SIJ[1][c] + SIJ[2][c]*SIJ[2][c] + 
         2.0 * (SIJ[3][c]*SIJ[3][c] + SIJ[4][c]*SIJ[4][c] + SIJ[5][c]*SIJ[5][c])));

   }
}


/* ---------------------------------------------------------------------
  Function~  computeStrainRate
  Purpose~ Calculate the grid strain rate
  -----------------------------------------------------------------------  */  
void Smagorinsky_Model::computeStrainRate(const Patch* patch,
                                    const SFCXVariable<double>& uvel_FC,
                                    const SFCYVariable<double>& vvel_FC,
                                    const SFCZVariable<double>& wvel_FC,
                                    const int indx,
                                    SimulationStateP&  d_sharedState,
                                    StaticArray<CCVariable<double> >& SIJ)
{
  Vector dx = patch->dCell();
  double delX = dx.x();
  double delY = dx.y();
  double delZ = dx.z();

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by twe cells in ghostCells
  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlus2Ghost = patch->addGhostCell_Iter(iter,2);
  
  for(CellIterator iter = iterPlus2Ghost; !iter.done(); iter++) {
    IntVector c = *iter;
    int i = c.x();
    int j = c.y();
    int k = c.z();

    IntVector right = IntVector(i+1,j,k);
    IntVector top   = IntVector(i,j+1,k);
    IntVector front = IntVector(i,j,k+1);             
    //__________________________________
    //calculate the grid strain rate tensor
    SIJ[0][c] = (uvel_FC[right] - uvel_FC[c])/delX;
    SIJ[1][c] = (vvel_FC[top]   - vvel_FC[c])/delY;
    SIJ[2][c] = (wvel_FC[front] - wvel_FC[c])/delZ;
    
    SIJ[3][c] = 0.5 * ((uvel_FC[right] - uvel_FC[c])/delY 
                       + (vvel_FC[top] - vvel_FC[c])/delX);
    SIJ[4][c] = 0.5 * ((uvel_FC[right] - uvel_FC[c])/delZ 
                     + (wvel_FC[front] - wvel_FC[c])/delX);
    SIJ[5][c] = 0.5 * ((vvel_FC[top] - vvel_FC[c])/delZ 
                   + (wvel_FC[front] - wvel_FC[c])/delY);
  }
  
  for (int comp = 0; comp < 6; comp ++ ) {
    setBC(SIJ[comp],"zeroNeumann",patch, d_sharedState, indx);
  } 
 
}
  

