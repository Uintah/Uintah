/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
                                ArchesConstVariables* constvars,
                                constCCVariable<double>& volFrac)

{
  // ignore faces that lie on the edge of the computational domain
  // in the principal direction
  IntVector noNeighborsLow = patch->noNeighborsLow();
  IntVector noNeighborsHigh = patch->noNeighborsHigh();

  Vector DX = patch->dCell();

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
                               DX,
                               volFrac,
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
                               DX,
                               volFrac,
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
                               DX,
                               volFrac,
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
                             constCCVariable<double>& press_CC,
                             constCCVariable<double>& volFrac_CC)
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
    //WARNING: use of the volume fraction is a bit of a hack here.
    //Ideally the coefficients would be modified for the presence of walls
    //but this was a quick and dirty solution without rewriting the entire
    //coefficient calculation.
    if ( volFrac_CC[c] > 1e-16 && volFrac_CC[c+shift] > 1e-16 )
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
    if ( volFrac_CC[c] > 1e-16 && volFrac_CC[c+shift] > 1.e-16 )
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
    if ( volFrac_CC[c] > 1e-16 && volFrac_CC[c+shift] > 1e-16 )
      vars->wVelRhoHat[c] = delta_t*(press_CC[adj] - press_CC[c])/cellinfo->dzpb[k]/rho_ave + vars->wVelRhoHat[c];
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
                                        const Vector Dx,
                                        constCCVariable<double>& volFraction,
                                        double delta_t)
{

  double vol = Dx.x()*Dx.y()*Dx.z();

  for (; !iter.done(); iter++) {

    IntVector c = *iter;
    IntVector adj = c + shift;

    double apo = 0.5 * (density[c] + density[adj])*vol/delta_t;

    double rhs = A[Arches::AE][c] * old_phi[c + IntVector(1,0,0)] +
                 A[Arches::AW][c] * old_phi[c - IntVector(1,0,0)] +
                 A[Arches::AN][c] * old_phi[c + IntVector(0,1,0)] +
                 A[Arches::AS][c] * old_phi[c - IntVector(0,1,0)] +
                 A[Arches::AT][c] * old_phi[c + IntVector(0,0,1)] +
                 A[Arches::AB][c] * old_phi[c - IntVector(0,0,1)] +
                 source[c] - A[Arches::AP][c] * old_phi[c];

    rhs *= volFraction[c] * volFraction[c+shift];
    
    //Note: volume fraction use here is a hack until the momentum solver is rewritten.
    if ( std::abs(apo) > 1e-16 )
      new_phi[c] = rhs/apo;

  }
}
