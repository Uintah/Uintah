/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//----- RHSSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/RHSSolver.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/Discretization.h>
#include <CCA/Components/Arches/PressureSolverV2.h>
#include <CCA/Components/Arches/Source.h>
#include <CCA/Components/Arches/StencilMatrix.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
using namespace std;

#include <CCA/Components/Arches/fortran/explicit_scalar_fort.h>

//****************************************************************************
// Default constructor for RHSSolver
//****************************************************************************
RHSSolver::RHSSolver()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
RHSSolver::~RHSSolver()
{
}

//****************************************************************************
// compute hat velocity for explicit projection
//****************************************************************************
void 
RHSSolver::calculateHatVelocity(const Patch* patch,
                                double delta_t,
                                CellInformation* cellinfo,
                                ArchesVariables* vars,
                                ArchesConstVariables* constvars)

{
  // ignore faces that lie on the edge of the computational domain
  // in the principal direction
  IntVector noNeighborsLow = patch->noNeighborsLow();
  IntVector noNeighborsHigh = patch->noNeighborsHigh();
  
  //__________________________________
  //    X dir
  IntVector shift(-1,0,0);  //one cell inward.  Only offset at the edge of the computational domain.
  IntVector loPt = patch->getSFCXLowIndex() - noNeighborsLow * shift;
  IntVector hiPt = patch->getSFCXHighIndex()+ noNeighborsHigh * shift;
  CellIterator iter = CellIterator(loPt, hiPt);
  
  explicitUpdate_stencilMatrix<SFCXVariable<double> > (iter,  shift,                         
                               vars->uVelNonlinearSrc,                  
                               constvars->uVelocity,                    
                               vars->uVelRhoHat,               
                               constvars->new_density,         
                               vars->uVelocityCoeff,                    
                               cellinfo->sewu, cellinfo->sns, cellinfo->stb,          
                               delta_t);
  //__________________________________
  //    Y dir
  shift = IntVector(0,-1,0);  // one cell inward.  Only offset at the edge of the computational domain.
  loPt = patch->getSFCYLowIndex() - noNeighborsLow * shift;
  hiPt = patch->getSFCYHighIndex()+ noNeighborsHigh * shift;
  iter = CellIterator(loPt, hiPt);
  
  explicitUpdate_stencilMatrix<SFCYVariable<double> > (iter,  shift,                         
                               vars->vVelNonlinearSrc,                  
                               constvars->vVelocity,                    
                               vars->vVelRhoHat,               
                               constvars->new_density,         
                               vars->vVelocityCoeff,                    
                               cellinfo->sew, cellinfo->snsv, cellinfo->stb,          
                               delta_t);
  //__________________________________
  //    Z dir       
  shift = IntVector(0,0,-1); //one cell inward.  Only offset at the edge of the computational domain. 
  loPt = patch->getSFCZLowIndex() - noNeighborsLow * shift;
  hiPt = patch->getSFCZHighIndex()+ noNeighborsHigh * shift;
  iter = CellIterator(loPt, hiPt);
  
  explicitUpdate_stencilMatrix<SFCZVariable<double> > (iter,  shift,                         
                               vars->wVelNonlinearSrc,                  
                               constvars->wVelocity,                    
                               vars->wVelRhoHat,               
                               constvars->new_density,         
                               vars->wVelocityCoeff,                    
                               cellinfo->sew, cellinfo->sns, cellinfo->stbw,          
                               delta_t);

}

//****************************************************************************
// compute velocity from hat velocity and pressure gradient
//****************************************************************************
void 
RHSSolver::calculateVelocity(const Patch* patch,
                             double delta_t,
                             CellInformation* cellinfo,
                             ArchesVariables* vars,
                             constCCVariable<double>& rho_CC,
                             constCCVariable<double>& press_CC)
{
  // ignore faces that lie on the edge of the computational domain
  // in the principal direction
  IntVector noNeighborsLow = patch->noNeighborsLow();
  IntVector noNeighborsHigh = patch->noNeighborsHigh();
  
  //__________________________________
  //  X-Velocity
  IntVector shift(-1,0,0);  //one cell inward.  Only offset at the edge of the computational domain.
  IntVector loPt = patch->getSFCXLowIndex() - noNeighborsLow * shift;
  IntVector hiPt = patch->getSFCXHighIndex()+ noNeighborsHigh * shift;
  CellIterator iter = CellIterator(loPt, hiPt);
  
  for (; !iter.done(); iter++){
    IntVector c = *iter;
    int i = c.x();
    IntVector adj = c + shift;
    double rho_ave = 0.5 * (rho_CC[adj] + rho_CC[c]);
    vars->uVelRhoHat[c] = delta_t*(press_CC[adj] - press_CC[c])/cellinfo->dxpw[i]/rho_ave + vars->uVelRhoHat[c];
  }

  //__________________________________
  //  Y-Velocity
  shift = IntVector(0,-1,0);  // one cell inward.  Only offset at the edge of the computational domain.
  loPt = patch->getSFCYLowIndex() - noNeighborsLow * shift;
  hiPt = patch->getSFCYHighIndex()+ noNeighborsHigh * shift;
  iter = CellIterator(loPt, hiPt);
  
  for (; !iter.done(); iter++){
    IntVector c = *iter;
    IntVector adj = c + shift;
    int j = c.y();
    double rho_ave = 0.5 * (rho_CC[adj] + rho_CC[c]);
    vars->vVelRhoHat[c] = delta_t*(press_CC[adj] - press_CC[c])/cellinfo->dyps[j]/rho_ave + vars->vVelRhoHat[c];
  }
  
  //__________________________________
  //  Z-Velocity  
  shift = IntVector(0,0,-1); // Only offset iterator at the edge of the computational domain. 
  loPt = patch->getSFCZLowIndex() - noNeighborsLow * shift;
  hiPt = patch->getSFCZHighIndex()+ noNeighborsHigh * shift;
  iter = CellIterator(loPt, hiPt);
  
  for (; !iter.done(); iter++){
    IntVector c = *iter;
    IntVector adj = c + shift;
    int k = c.z();
    double rho_ave = 0.5 * (rho_CC[adj] + rho_CC[c]);
    vars->wVelRhoHat[c] = delta_t*(press_CC[adj] - press_CC[c])/cellinfo->dzpb[k]/rho_ave + vars->wVelRhoHat[c];  
  }
}

//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
RHSSolver::scalarLisolve(const ProcessorGroup*,
                          const Patch* patch,
                          double delta_t,
                          ArchesVariables* vars,
                          ArchesConstVariables* constvars,
                          CellInformation* cellinfo)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

    fort_explicit_scalar(idxLo, idxHi, vars->scalar, constvars->old_scalar,
                  constvars->scalarCoeff[Arches::AE], 
                  constvars->scalarCoeff[Arches::AW], 
                  constvars->scalarCoeff[Arches::AN], 
                  constvars->scalarCoeff[Arches::AS], 
                  constvars->scalarCoeff[Arches::AT], 
                  constvars->scalarCoeff[Arches::AB], 
                  constvars->scalarCoeff[Arches::AP], 
                  constvars->scalarNonlinearSrc, constvars->density_guess,
                  cellinfo->sew, cellinfo->sns, cellinfo->stb, delta_t);

}
//------------------------------------------------------
// Explicit update of any cell centered scalar
void 
RHSSolver::scalarExplicitUpdate(const ProcessorGroup*,
                              const Patch* patch,
                              double delta_t,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars,
                              CellInformation* cellinfo, 
                              bool doingMM, int intrusionVal)
{
  CellIterator iter = patch->getCellIterator();
  explicitUpdate<constCCVariable<Stencil7>,CCVariable<double>,constCCVariable<double> >(iter,
                                                            constvars->scalarTotCoef, 
                                                            constvars->scalarNonlinearSrc, 
                                                            constvars->density_guess, 
                                                            constvars->old_scalar, 
                                                            vars->scalar, 
                                                            constvars->cellType,
                                                            cellinfo, 
                                                            delta_t, doingMM, intrusionVal);
}
//--------------------------------------------------------
// Generic explicit solver
// Solves Ax = b for an explicit scheme. 
// 
template<class T_mtrx, class T_varmod, class T_varconst> void
RHSSolver::explicitUpdate(CellIterator iter, 
                          T_mtrx& A,
                          T_varconst source, 
                          constCCVariable<double> old_den, 
                          T_varconst old_phi,
                          T_varmod& new_phi,  
                          constCCVariable<int>  cellType,
                          CellInformation* cellinfo,
                          double delta_t, 
                          bool doingMM, int intrusionVal)
{
  if (!doingMM) {
    for (; !iter.done(); iter++) {
      IntVector curr = *iter;

      double vol = cellinfo->sew[curr.x()]*cellinfo->sns[curr.y()]*cellinfo->stb[curr.z()];
      double apo = old_den[curr]*vol/delta_t;
      double rhs = A[curr].e*old_phi[curr+IntVector(1,0,0)] + 
                   A[curr].w*old_phi[curr-IntVector(1,0,0)] + 
                   A[curr].n*old_phi[curr+IntVector(0,1,0)] + 
                   A[curr].s*old_phi[curr-IntVector(0,1,0)] + 
                   A[curr].t*old_phi[curr+IntVector(0,0,1)] + 
                   A[curr].b*old_phi[curr-IntVector(0,0,1)] +
                   source[curr] - A[curr].p*old_phi[curr];

      new_phi[curr] = rhs/apo;
    } 
  } else {
    for (; !iter.done(); iter++) {
      IntVector curr = *iter;

      double vol = cellinfo->sew[curr.x()]*cellinfo->sns[curr.y()]*cellinfo->stb[curr.z()];
      double apo = old_den[curr]*vol/delta_t;
      double rhs = A[curr].e*old_phi[curr+IntVector(1,0,0)] + 
                   A[curr].w*old_phi[curr-IntVector(1,0,0)] + 
                   A[curr].n*old_phi[curr+IntVector(0,1,0)] + 
                   A[curr].s*old_phi[curr-IntVector(0,1,0)] + 
                   A[curr].t*old_phi[curr+IntVector(0,0,1)] + 
                   A[curr].b*old_phi[curr-IntVector(0,0,1)] +
                   source[curr] - A[curr].p*old_phi[curr];

      new_phi[curr] = rhs/apo;

      if (cellType[curr] == intrusionVal) 
        new_phi[curr] = 0.0; 
    } 
  } 
}
//______________________________________________________________________
// Generic explicit solver
// Solves Ax = b for an explicit scheme. 
//  This will eventually be consolidated with the other explicitUpdate.
template<class T> void
RHSSolver::explicitUpdate_stencilMatrix(CellIterator iter, 
                                        IntVector shift,
                                        const T& source, 
                                        const T& old_phi,
                                        T& new_phi,
                                        constCCVariable<double> density,
                                        StencilMatrix<T>& A,
                                        const OffsetArray1<double>& sew,
                                        const OffsetArray1<double>& sns,
                                        const OffsetArray1<double>& stb,
                                        double delta_t)
{
  for (; !iter.done(); iter++) {
    IntVector c = *iter;                                                            
    IntVector adj = c + shift;
    int i = c.x();
    int j = c.y();
    int k = c.z();                                                 
    double vol = sew[i]*sns[j]*stb[k];    
    double apo = 0.5 * (density[c] + density[adj])*vol/delta_t;
                                                   
    double rhs = A[Arches::AE][c] * old_phi[c + IntVector(1,0,0)] +                             
                 A[Arches::AW][c] * old_phi[c - IntVector(1,0,0)] +                             
                 A[Arches::AN][c] * old_phi[c + IntVector(0,1,0)] +                             
                 A[Arches::AS][c] * old_phi[c - IntVector(0,1,0)] +                             
                 A[Arches::AT][c] * old_phi[c + IntVector(0,0,1)] +                             
                 A[Arches::AB][c] * old_phi[c - IntVector(0,0,1)] +                             
                 source[c] - A[Arches::AP][c] * old_phi[c];
                 
    new_phi[c] = rhs/apo;                                                           
  } 
}


//****************************************************************************
// Enthalpy Solve
//****************************************************************************

void 
RHSSolver::enthalpyLisolve(const ProcessorGroup*,
                          const Patch* patch,
                          double delta_t,
                          ArchesVariables* vars,
                          ArchesConstVariables* constvars,
                          CellInformation* cellinfo)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

    fort_explicit_scalar(idxLo, idxHi, vars->enthalpy, constvars->old_enthalpy,
                  constvars->scalarCoeff[Arches::AE], 
                  constvars->scalarCoeff[Arches::AW], 
                  constvars->scalarCoeff[Arches::AN], 
                  constvars->scalarCoeff[Arches::AS], 
                  constvars->scalarCoeff[Arches::AT], 
                  constvars->scalarCoeff[Arches::AB], 
                  constvars->scalarCoeff[Arches::AP], 
                  constvars->scalarNonlinearSrc, constvars->density_guess,
                  cellinfo->sew, cellinfo->sns, cellinfo->stb, delta_t);
     
}

