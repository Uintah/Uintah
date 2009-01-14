//----- RHSSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/RHSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

#include <Packages/Uintah/CCA/Components/Arches/fortran/explicit_scalar_fort.h>

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
  //__________________________________
  //    X dir
  IntVector shift(-1,0,0);  // ignore outer edge/plane of computational domain
  IntVector loPt =  patch->getExtraLowIndex( Patch::XFaceBased,shift);
  IntVector hiPt =  patch->getExtraHighIndex(Patch::XFaceBased,shift);
  CellIterator iter = CellIterator(loPt, hiPt);
  
  explicitUpdate_stencilMatrix<SFCXVariable<double> > (iter,  shift,                         
                               vars->uVelNonlinearSrc,                  
                               constvars->uVelocity,                    
                               vars->uVelRhoHat,               
                               constvars->new_density,         
                               vars->uVelocityCoeff,                    
                               cellinfo->sewu, cellinfo->sns, cellinfo->stb,          
                               delta_t);

                                                   
  //MMS conv and diff force term collection
  // Since the collection (along with velFmms allocation) 
  // is conditional on d_doMMS, 
  // I put the force term calculation in the fortran 
  // and hence the "verification" doesn't completely 
  // check the explicit update!
  if (d_doMMS) {
    for (; !iter.done(); iter++){
      IntVector c = *iter;
      int i = c.x();
      int j = c.y();
      int k = c.z();

      IntVector E(i+1, j, k);    IntVector W(i-1, j, k);
      IntVector N(i, j+1, k);    IntVector S(i, j-1, k);
      IntVector T(i, j, k+1);    IntVector B(i, j, k-1);

      vars->uFmms[c] = vars->uVelocityCoeff[Arches::AE][c]*constvars->uVelocity[E] +
        vars->uVelocityCoeff[Arches::AW][c]*constvars->uVelocity[W] +
        vars->uVelocityCoeff[Arches::AN][c]*constvars->uVelocity[N] +
        vars->uVelocityCoeff[Arches::AS][c]*constvars->uVelocity[S] +
        vars->uVelocityCoeff[Arches::AT][c]*constvars->uVelocity[T] +
        vars->uVelocityCoeff[Arches::AB][c]*constvars->uVelocity[B] - 
        vars->uVelocityCoeff[Arches::AP][c]*constvars->uVelocity[c] +
        vars->uVelNonlinearSrc[c] - 
        0.5*(constvars->new_density[c] + constvars->new_density[W])*constvars->uVelocity[c];

      vars->uFmms[c] = vars->uFmms[c]/cellinfo->sew[i]*cellinfo->sns[j]*cellinfo->stb[k];

    }
  }
  //__________________________________
  //    Y dir
  shift = IntVector(0,-1,0);  // ignore outer edge/plane of computational domain
  loPt =  patch->getExtraLowIndex( Patch::YFaceBased,shift);
  hiPt =  patch->getExtraHighIndex(Patch::YFaceBased,shift);
  iter = CellIterator(loPt, hiPt);
  
  explicitUpdate_stencilMatrix<SFCYVariable<double> > (iter,  shift,                         
                               vars->vVelNonlinearSrc,                  
                               constvars->vVelocity,                    
                               vars->vVelRhoHat,               
                               constvars->new_density,         
                               vars->vVelocityCoeff,                    
                               cellinfo->sew, cellinfo->snsv, cellinfo->stb,          
                               delta_t);


  //MMS conv and diff force term collection
  // Since the collection (along with velFmms allocation) 
  // is conditional on d_doMMS, 
  // I put the force term caluculation in the fortran 
  // and hence the "verification" doesn't completely 
  // check the explicit update!
  if (d_doMMS) {
    for (; !iter.done(); iter++){
      IntVector c = *iter;
      int i = c.x();
      int j = c.y();
      int k = c.z();

      IntVector E(i+1, j, k);    IntVector W(i-1, j, k);
      IntVector N(i, j+1, k);    IntVector S(i, j-1, k);
      IntVector T(i, j, k+1);    IntVector B(i, j, k-1);

      vars->vFmms[c] = vars->vVelocityCoeff[Arches::AE][c]*constvars->vVelocity[E] +
        vars->vVelocityCoeff[Arches::AW][c]*constvars->vVelocity[W] +
        vars->vVelocityCoeff[Arches::AN][c]*constvars->vVelocity[N] +
        vars->vVelocityCoeff[Arches::AS][c]*constvars->vVelocity[S] +
        vars->vVelocityCoeff[Arches::AT][c]*constvars->vVelocity[T] +
        vars->vVelocityCoeff[Arches::AB][c]*constvars->vVelocity[B] - 
        vars->vVelocityCoeff[Arches::AP][c]*constvars->vVelocity[c] +
        vars->vVelNonlinearSrc[c] - 
        0.5*(constvars->new_density[c] + constvars->new_density[S])*constvars->vVelocity[c]/delta_t;
    }
  }

  //__________________________________
  //    Z dir       
  shift = IntVector(0,0,-1);  // ignore outer edge/plane of computational domain
  loPt =  patch->getExtraLowIndex( Patch::ZFaceBased,shift);
  hiPt =  patch->getExtraHighIndex(Patch::ZFaceBased,shift);
  iter = CellIterator(loPt, hiPt);
  
  explicitUpdate_stencilMatrix<SFCZVariable<double> > (iter,  shift,                         
                               vars->wVelNonlinearSrc,                  
                               constvars->wVelocity,                    
                               vars->wVelRhoHat,               
                               constvars->new_density,         
                               vars->wVelocityCoeff,                    
                               cellinfo->sew, cellinfo->sns, cellinfo->stbw,          
                               delta_t);

  //MMS conv and diff force term collection
  // Since the collection (along with velFmms allocation) 
  // is conditional on d_doMMS, 
  // I put the force term caluculation in the fortran 
  // and hence the "verification" doesn't completely 
  // check the explicit update!
  if (d_doMMS) {
    for (; !iter.done(); iter++){
      IntVector c = *iter;
      int i = c.x();
      int j = c.y();
      int k = c.z();

      IntVector E(i+1, j, k);    IntVector W(i-1, j, k);
      IntVector N(i, j+1, k);    IntVector S(i, j-1, k);
      IntVector T(i, j, k+1);    IntVector B(i, j, k-1);

      vars->wFmms[c] = vars->wVelocityCoeff[Arches::AE][c]*constvars->wVelocity[E] +
        vars->wVelocityCoeff[Arches::AW][c]*constvars->wVelocity[W] +
        vars->wVelocityCoeff[Arches::AN][c]*constvars->wVelocity[N] +
        vars->wVelocityCoeff[Arches::AS][c]*constvars->wVelocity[S] +
        vars->wVelocityCoeff[Arches::AT][c]*constvars->wVelocity[T] +
        vars->wVelocityCoeff[Arches::AB][c]*constvars->wVelocity[B] - 
        vars->wVelocityCoeff[Arches::AP][c]*constvars->wVelocity[c] +
        vars->wVelNonlinearSrc[c] - 
        0.5*(constvars->new_density[c] + constvars->new_density[B])*constvars->wVelocity[c]/delta_t;
    }
  }
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
  //__________________________________
  //  X-Velocity
  IntVector shift(-1,0,0);  // ignore outer edge/plane of computational domain
  IntVector loPt =  patch->getExtraLowIndex( Patch::XFaceBased,shift);
  IntVector hiPt =  patch->getExtraHighIndex(Patch::XFaceBased,shift);
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
  shift = IntVector(0,-1,0); // ignore outer edge/plane of computational domain
  loPt =  patch->getExtraLowIndex( Patch::YFaceBased,shift);
  hiPt =  patch->getExtraHighIndex(Patch::YFaceBased,shift);
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
  shift = IntVector(0,0,-1); // ignore outer edge/plan of computational domain
  loPt =  patch->getExtraLowIndex( Patch::ZFaceBased,shift);
  hiPt =  patch->getExtraHighIndex(Patch::ZFaceBased,shift);
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
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

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
  CellIterator iter = patch->getCellIterator__New();
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
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

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

