//----- RHSSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/RHSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
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
#include <Packages/Uintah/CCA/Components/Arches/fortran/explicit_vel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/computeVel_fort.h>

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
RHSSolver::calculateHatVelocity(const ProcessorGroup* /*pc*/,
                            const Patch* patch,
                            double delta_t,
                            CellInformation* cellinfo,
                            ArchesVariables* vars,
                            ArchesConstVariables* constvars)

{
  // Get the patch bounds and the variable bounds
  IntVector idxLo;
  IntVector idxHi;
  // for explicit solver
  int ioff, joff, koff;
  //__________________________________
  //    X dir
  idxLo = patch->getSFCXFORTLowIndex();
  idxHi = patch->getSFCXFORTHighIndex();
  ioff = 1; joff = 0; koff = 0;

  fort_explicit_vel(idxLo, idxHi, 
                    vars->uVelRhoHat,
                    constvars->uVelocity,
                    vars->uVelocityCoeff[Arches::AE], 
                    vars->uVelocityCoeff[Arches::AW], 
                    vars->uVelocityCoeff[Arches::AN], 
                    vars->uVelocityCoeff[Arches::AS], 
                    vars->uVelocityCoeff[Arches::AT], 
                    vars->uVelocityCoeff[Arches::AB], 
                    vars->uVelocityCoeff[Arches::AP], 
                    vars->uVelNonlinearSrc,
                    constvars->new_density,
                    cellinfo->sewu, cellinfo->sns, cellinfo->stb,
                    delta_t, ioff, joff, koff);


  //MMS conv and diff force term collection
  // Since the collection (along with velFmms allocation) 
  // is conditional on d_doMMS, 
  // I put the force term calculation in the fortran 
  // and hence the "verification" doesn't completely 
  // check the explicit update!
  if (d_doMMS) {
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector eastCell(colX+1, colY, colZ);
          IntVector westCell(colX-1, colY, colZ);
          IntVector northCell(colX, colY+1, colZ);
          IntVector southCell(colX, colY-1, colZ);
          IntVector topCell(colX, colY, colZ+1);
          IntVector bottomCell(colX, colY, colZ-1);

          vars->uFmms[currCell] = vars->uVelocityCoeff[Arches::AE][currCell]*constvars->uVelocity[eastCell] +
            vars->uVelocityCoeff[Arches::AW][currCell]*constvars->uVelocity[westCell] +
            vars->uVelocityCoeff[Arches::AN][currCell]*constvars->uVelocity[northCell] +
            vars->uVelocityCoeff[Arches::AS][currCell]*constvars->uVelocity[southCell] +
            vars->uVelocityCoeff[Arches::AT][currCell]*constvars->uVelocity[topCell] +
            vars->uVelocityCoeff[Arches::AB][currCell]*constvars->uVelocity[bottomCell] - 
            vars->uVelocityCoeff[Arches::AP][currCell]*constvars->uVelocity[currCell] +
            vars->uVelNonlinearSrc[currCell] - 
            0.5*(constvars->new_density[currCell] + constvars->new_density[westCell])*constvars->uVelocity[currCell];

          vars->uFmms[currCell] = vars->uFmms[currCell]/cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];

        }
      }
    }
  }
  //__________________________________
  //    Y dir
  idxLo = patch->getSFCYFORTLowIndex();
  idxHi = patch->getSFCYFORTHighIndex();
  ioff = 0; joff = 1; koff = 0;

  fort_explicit_vel(idxLo, idxHi, 
                    vars->vVelRhoHat,
                    constvars->vVelocity,
                    vars->vVelocityCoeff[Arches::AE], 
                    vars->vVelocityCoeff[Arches::AW], 
                    vars->vVelocityCoeff[Arches::AN], 
                    vars->vVelocityCoeff[Arches::AS], 
                    vars->vVelocityCoeff[Arches::AT], 
                    vars->vVelocityCoeff[Arches::AB], 
                    vars->vVelocityCoeff[Arches::AP], 
                    vars->vVelNonlinearSrc,
                    constvars->new_density,
                    cellinfo->sew, cellinfo->snsv, cellinfo->stb,
                    delta_t, ioff, joff, koff);

  //MMS conv and diff force term collection
  // Since the collection (along with velFmms allocation) 
  // is conditional on d_doMMS, 
  // I put the force term caluculation in the fortran 
  // and hence the "verification" doesn't completely 
  // check the explicit update!
  if (d_doMMS) {
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector eastCell(colX+1, colY, colZ);
          IntVector westCell(colX-1, colY, colZ);
          IntVector northCell(colX, colY+1, colZ);
          IntVector southCell(colX, colY-1, colZ);
          IntVector topCell(colX, colY, colZ+1);
          IntVector bottomCell(colX, colY, colZ-1);

          vars->vFmms[currCell] = vars->vVelocityCoeff[Arches::AE][currCell]*constvars->vVelocity[eastCell] +
            vars->vVelocityCoeff[Arches::AW][currCell]*constvars->vVelocity[westCell] +
            vars->vVelocityCoeff[Arches::AN][currCell]*constvars->vVelocity[northCell] +
            vars->vVelocityCoeff[Arches::AS][currCell]*constvars->vVelocity[southCell] +
            vars->vVelocityCoeff[Arches::AT][currCell]*constvars->vVelocity[topCell] +
            vars->vVelocityCoeff[Arches::AB][currCell]*constvars->vVelocity[bottomCell] - 
            vars->vVelocityCoeff[Arches::AP][currCell]*constvars->vVelocity[currCell] +
            vars->vVelNonlinearSrc[currCell] - 
            0.5*(constvars->new_density[currCell] + constvars->new_density[southCell])*constvars->vVelocity[currCell]/delta_t;


        }
      }
    }
  }

  //__________________________________
  //    Z dir
  idxLo = patch->getSFCZFORTLowIndex();
  idxHi = patch->getSFCZFORTHighIndex();
  ioff = 0; joff = 0; koff = 1;

  fort_explicit_vel(idxLo, idxHi, 
                    vars->wVelRhoHat,
                    constvars->wVelocity,
                    vars->wVelocityCoeff[Arches::AE], 
                    vars->wVelocityCoeff[Arches::AW], 
                    vars->wVelocityCoeff[Arches::AN], 
                    vars->wVelocityCoeff[Arches::AS], 
                    vars->wVelocityCoeff[Arches::AT], 
                    vars->wVelocityCoeff[Arches::AB], 
                    vars->wVelocityCoeff[Arches::AP], 
                    vars->wVelNonlinearSrc,
                    constvars->new_density,
                    cellinfo->sew, cellinfo->sns, cellinfo->stbw,
                    delta_t, ioff, joff, koff);

  //MMS conv and diff force term collection
  // Since the collection (along with velFmms allocation) 
  // is conditional on d_doMMS, 
  // I put the force term caluculation in the fortran 
  // and hence the "verification" doesn't completely 
  // check the explicit update!
  if (d_doMMS) {
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector eastCell(colX+1, colY, colZ);
          IntVector westCell(colX-1, colY, colZ);
          IntVector northCell(colX, colY+1, colZ);
          IntVector southCell(colX, colY-1, colZ);
          IntVector topCell(colX, colY, colZ+1);
          IntVector bottomCell(colX, colY, colZ-1);

          vars->wFmms[currCell] = vars->wVelocityCoeff[Arches::AE][currCell]*constvars->wVelocity[eastCell] +
            vars->wVelocityCoeff[Arches::AW][currCell]*constvars->wVelocity[westCell] +
            vars->wVelocityCoeff[Arches::AN][currCell]*constvars->wVelocity[northCell] +
            vars->wVelocityCoeff[Arches::AS][currCell]*constvars->wVelocity[southCell] +
            vars->wVelocityCoeff[Arches::AT][currCell]*constvars->wVelocity[topCell] +
            vars->wVelocityCoeff[Arches::AB][currCell]*constvars->wVelocity[bottomCell] - 
            vars->wVelocityCoeff[Arches::AP][currCell]*constvars->wVelocity[currCell] +
            vars->wVelNonlinearSrc[currCell] - 
            0.5*(constvars->new_density[currCell] + constvars->new_density[bottomCell])*constvars->wVelocity[currCell]/delta_t;


        }
      }
    }
  }
}

//****************************************************************************
// compute velocity from hat velocity and pressure gradient
//****************************************************************************

void 
RHSSolver::calculateVelocity(const ProcessorGroup* ,
                             const Patch* patch,
                             double delta_t,
                             CellInformation* cellinfo,
                             ArchesVariables* vars,
                             ArchesConstVariables* constvars)
{
  
  int ioff, joff, koff;
  IntVector idxLoU;
  IntVector idxHiU;
  
  //__________________________________
  idxLoU = patch->getSFCXFORTLowIndex();
  idxHiU = patch->getSFCXFORTHighIndex();
  ioff = 1; joff = 0; koff = 0;

  fort_computevel(idxLoU, idxHiU, vars->uVelRhoHat, constvars->pressure,
                  constvars->density, delta_t,
                  ioff, joff, koff, cellinfo->dxpw);
  //__________________________________
  idxLoU = patch->getSFCYFORTLowIndex();
  idxHiU = patch->getSFCYFORTHighIndex();
  ioff = 0; joff = 1; koff = 0;

  fort_computevel(idxLoU, idxHiU, vars->vVelRhoHat, constvars->pressure,
                  constvars->density, delta_t,
                  ioff, joff, koff, cellinfo->dyps);
  //__________________________________
  idxLoU = patch->getSFCZFORTLowIndex();
  idxHiU = patch->getSFCZFORTHighIndex();
  ioff = 0; joff = 0; koff = 1;

  fort_computevel(idxLoU, idxHiU, vars->wVelRhoHat, constvars->pressure,
                  constvars->density, delta_t,
                  ioff, joff, koff, cellinfo->dzpb);
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

