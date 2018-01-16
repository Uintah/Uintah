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

//----- Discretization.cc ----------------------------------------------

#include <CCA/Components/Arches/Discretization.h>
#include <CCA/Components/Arches/Arches.h>
#include <Core/Geometry/Vector.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/StencilMatrix.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Task.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>
#include <cmath>
#include <iostream>

using namespace std;
using namespace Uintah;

#ifdef divergenceconstraint
#include <CCA/Components/Arches/fortran/prescoef_var_fort.h>
#endif
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/fortran/uvelcoef_fort.h>
#include <CCA/Components/Arches/fortran/uvelcoef_central_fort.h>
#include <CCA/Components/Arches/fortran/uvelcoef_upwind_fort.h>
#include <CCA/Components/Arches/fortran/uvelcoef_mixed_fort.h>
#include <CCA/Components/Arches/fortran/uvelcoef_hybrid_fort.h>
#include <CCA/Components/Arches/fortran/vvelcoef_fort.h>
#include <CCA/Components/Arches/fortran/vvelcoef_central_fort.h>
#include <CCA/Components/Arches/fortran/vvelcoef_upwind_fort.h>
#include <CCA/Components/Arches/fortran/vvelcoef_mixed_fort.h>
#include <CCA/Components/Arches/fortran/vvelcoef_hybrid_fort.h>
#include <CCA/Components/Arches/fortran/wvelcoef_fort.h>
#include <CCA/Components/Arches/fortran/wvelcoef_central_fort.h>
#include <CCA/Components/Arches/fortran/wvelcoef_upwind_fort.h>
#include <CCA/Components/Arches/fortran/wvelcoef_mixed_fort.h>
#include <CCA/Components/Arches/fortran/wvelcoef_hybrid_fort.h>


//****************************************************************************
// Default constructor for Discretization
//****************************************************************************
Discretization::Discretization(PhysicalConstants* physConst)
{
  d_filter = 0;
  d_physicalConsts = physConst;
}

//****************************************************************************
// Destructor
//****************************************************************************
Discretization::~Discretization()
{
}

//****************************************************************************
// Velocity stencil weights
//****************************************************************************
void
Discretization::calculateVelocityCoeff(const Patch* patch,
                                       double delta_t,
                                       bool lcentral,
                                       CellInformation* cellinfo,
                                       ArchesVariables* coeff_vars,
                                       ArchesConstVariables* coeff_constvars,
                                       constCCVariable<double>* volFraction,
                                       SFCXVariable<double>* conv_scheme_x,
                                       SFCYVariable<double>* conv_scheme_y,
                                       SFCZVariable<double>* conv_scheme_z,
                                       MOMCONV conv_scheme,
                                       double Re_limit )
{
   // ignore faces that lie on the edge of the computational domain
  // in the principal direction
  IntVector noNeighborsLow = patch->noNeighborsLow();
  IntVector noNeighborsHigh = patch->noNeighborsHigh();
  //__________________________________
  //  X DIR
  IntVector oci(-1,0,0);  //one cell inward.  Only offset at the edge of the computational domain.
  IntVector idxLoU = patch->getSFCXLowIndex() - noNeighborsLow * oci;
  IntVector idxHiU = patch->getSFCXHighIndex()+ noNeighborsHigh * oci - IntVector(1,1,1);

  Vector DX = patch->dCell();
  double dx = DX.x();
  double dy = DX.y();
  double dz = DX.z();

  int wall1 = BoundaryCondition::WALL;
  int wall2 = BoundaryCondition::INTRUSION;

  Vector gravity = d_physicalConsts->getGravity();
  double grav = gravity.x();

  // Calculate the coeffs
  switch (conv_scheme){
    case Discretization::CENTRAL:
      fort_uvelcoef_central( coeff_constvars->uVelocity,
                             coeff_vars->uVelocityConvectCoeff[Arches::AE],
                             coeff_vars->uVelocityConvectCoeff[Arches::AW],
                             coeff_vars->uVelocityConvectCoeff[Arches::AN],
                             coeff_vars->uVelocityConvectCoeff[Arches::AS],
                             coeff_vars->uVelocityConvectCoeff[Arches::AT],
                             coeff_vars->uVelocityConvectCoeff[Arches::AB],
                             coeff_vars->uVelocityCoeff[Arches::AP],
                             coeff_vars->uVelocityCoeff[Arches::AE],
                             coeff_vars->uVelocityCoeff[Arches::AW],
                             coeff_vars->uVelocityCoeff[Arches::AN],
                             coeff_vars->uVelocityCoeff[Arches::AS],
                             coeff_vars->uVelocityCoeff[Arches::AT],
                             coeff_vars->uVelocityCoeff[Arches::AB],
                             coeff_constvars->vVelocity, coeff_constvars->wVelocity,
                             coeff_constvars->density, coeff_constvars->viscosity,
                             coeff_constvars->denRefArray, coeff_vars->uVelNonlinearSrc,
                             coeff_constvars->old_density, coeff_constvars->old_uVelocity,
                             *volFraction,
                             delta_t, grav, dx, dy, dz,
                             idxLoU, idxHiU );
      break;
    case Discretization::UPWIND:
      fort_uvelcoef_upwind( coeff_constvars->uVelocity,
                            coeff_vars->uVelocityConvectCoeff[Arches::AE],
                            coeff_vars->uVelocityConvectCoeff[Arches::AW],
                            coeff_vars->uVelocityConvectCoeff[Arches::AN],
                            coeff_vars->uVelocityConvectCoeff[Arches::AS],
                            coeff_vars->uVelocityConvectCoeff[Arches::AT],
                            coeff_vars->uVelocityConvectCoeff[Arches::AB],
                            coeff_vars->uVelocityCoeff[Arches::AP],
                            coeff_vars->uVelocityCoeff[Arches::AE],
                            coeff_vars->uVelocityCoeff[Arches::AW],
                            coeff_vars->uVelocityCoeff[Arches::AN],
                            coeff_vars->uVelocityCoeff[Arches::AS],
                            coeff_vars->uVelocityCoeff[Arches::AT],
                            coeff_vars->uVelocityCoeff[Arches::AB],
                            coeff_constvars->vVelocity, coeff_constvars->wVelocity,
                            coeff_constvars->density, coeff_constvars->viscosity,
                            coeff_constvars->denRefArray, coeff_vars->uVelNonlinearSrc,
                            *volFraction,
                            delta_t, grav, dx, dy, dz,
                            idxLoU, idxHiU);
      break;
    case Discretization::WALLUPWIND:
      fort_uvelcoef_mixed( coeff_constvars->uVelocity, coeff_constvars->cellType,
                           coeff_vars->uVelocityConvectCoeff[Arches::AE],
                           coeff_vars->uVelocityConvectCoeff[Arches::AW],
                           coeff_vars->uVelocityConvectCoeff[Arches::AN],
                           coeff_vars->uVelocityConvectCoeff[Arches::AS],
                           coeff_vars->uVelocityConvectCoeff[Arches::AT],
                           coeff_vars->uVelocityConvectCoeff[Arches::AB],
                           coeff_vars->uVelocityCoeff[Arches::AP],
                           coeff_vars->uVelocityCoeff[Arches::AE],
                           coeff_vars->uVelocityCoeff[Arches::AW],
                           coeff_vars->uVelocityCoeff[Arches::AN],
                           coeff_vars->uVelocityCoeff[Arches::AS],
                           coeff_vars->uVelocityCoeff[Arches::AT],
                           coeff_vars->uVelocityCoeff[Arches::AB],
                           coeff_constvars->vVelocity, coeff_constvars->wVelocity,
                           coeff_constvars->density, coeff_constvars->viscosity,
                           coeff_constvars->denRefArray, coeff_vars->uVelNonlinearSrc,
                           *volFraction,
                           delta_t, grav, dx, dy, dz,
                           wall1, wall2, Re_limit,
                           idxLoU, idxHiU);
      break;
    case Discretization::HYBRID:
      fort_uvelcoef_hybrid( coeff_constvars->uVelocity, coeff_constvars->cellType,
                           coeff_vars->uVelocityConvectCoeff[Arches::AE],
                           coeff_vars->uVelocityConvectCoeff[Arches::AW],
                           coeff_vars->uVelocityConvectCoeff[Arches::AN],
                           coeff_vars->uVelocityConvectCoeff[Arches::AS],
                           coeff_vars->uVelocityConvectCoeff[Arches::AT],
                           coeff_vars->uVelocityConvectCoeff[Arches::AB],
                           coeff_vars->uVelocityCoeff[Arches::AP],
                           coeff_vars->uVelocityCoeff[Arches::AE],
                           coeff_vars->uVelocityCoeff[Arches::AW],
                           coeff_vars->uVelocityCoeff[Arches::AN],
                           coeff_vars->uVelocityCoeff[Arches::AS],
                           coeff_vars->uVelocityCoeff[Arches::AT],
                           coeff_vars->uVelocityCoeff[Arches::AB],
                           coeff_constvars->vVelocity, coeff_constvars->wVelocity,
                           coeff_constvars->density, coeff_constvars->viscosity,
                           coeff_constvars->denRefArray, coeff_vars->uVelNonlinearSrc,
                           *volFraction,
                           *conv_scheme_x,
                           delta_t, grav, dx, dy, dz,
                           wall1, wall2, Re_limit,
                           idxLoU, idxHiU);
      break;
    case Discretization::OLD:
      fort_uvelcoef(coeff_constvars->uVelocity,
                    coeff_vars->uVelocityConvectCoeff[Arches::AE],
                    coeff_vars->uVelocityConvectCoeff[Arches::AW],
                    coeff_vars->uVelocityConvectCoeff[Arches::AN],
                    coeff_vars->uVelocityConvectCoeff[Arches::AS],
                    coeff_vars->uVelocityConvectCoeff[Arches::AT],
                    coeff_vars->uVelocityConvectCoeff[Arches::AB],
                    coeff_vars->uVelocityCoeff[Arches::AP],
                    coeff_vars->uVelocityCoeff[Arches::AE],
                    coeff_vars->uVelocityCoeff[Arches::AW],
                    coeff_vars->uVelocityCoeff[Arches::AN],
                    coeff_vars->uVelocityCoeff[Arches::AS],
                    coeff_vars->uVelocityCoeff[Arches::AT],
                    coeff_vars->uVelocityCoeff[Arches::AB],
                    coeff_constvars->vVelocity, coeff_constvars->wVelocity,
                    coeff_constvars->density, coeff_constvars->viscosity,
                    coeff_constvars->denRefArray, coeff_vars->uVelNonlinearSrc,
                    coeff_constvars->old_density, coeff_constvars->old_uVelocity,
                    *volFraction,
                    delta_t, grav, lcentral,
                    cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
                    cellinfo->cnn, cellinfo->csn, cellinfo->css,
                    cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
                    cellinfo->sewu, cellinfo->sew, cellinfo->sns,
                    cellinfo->stb, cellinfo->dxepu, cellinfo->dxpwu,
                    cellinfo->dxpw, cellinfo->dynp, cellinfo->dyps,
                    cellinfo->dztp, cellinfo->dzpb, cellinfo->fac1u,
                    cellinfo->fac2u, cellinfo->fac3u, cellinfo->fac4u,
                    cellinfo->iesdu, cellinfo->iwsdu, cellinfo->nfac,
                    cellinfo->sfac, cellinfo->tfac, cellinfo->bfac,
                    cellinfo->fac1ns, cellinfo->fac2ns, cellinfo->fac3ns,
                    cellinfo->fac4ns, cellinfo->n_shift, cellinfo->s_shift,
                    cellinfo->fac1tb, cellinfo->fac2tb, cellinfo->fac3tb,
                    cellinfo->fac4tb, cellinfo->t_shift, cellinfo->b_shift,
                    idxLoU, idxHiU);

    default:
      break;
  }

  //__________________________________
  //      Y DIR
  oci = IntVector(0,-1,0);  // one cell inward.  Only offset at the edge of the computational domain.
  IntVector idxLoV = patch->getSFCYLowIndex() - noNeighborsLow * oci;
  IntVector idxHiV = patch->getSFCYHighIndex()+ noNeighborsHigh * oci - IntVector(1,1,1);

  grav = gravity.y();

  switch (conv_scheme){
    case Discretization::CENTRAL:
      fort_vvelcoef_central( coeff_constvars->vVelocity,
                             coeff_vars->vVelocityConvectCoeff[Arches::AE],
                             coeff_vars->vVelocityConvectCoeff[Arches::AW],
                             coeff_vars->vVelocityConvectCoeff[Arches::AN],
                             coeff_vars->vVelocityConvectCoeff[Arches::AS],
                             coeff_vars->vVelocityConvectCoeff[Arches::AT],
                             coeff_vars->vVelocityConvectCoeff[Arches::AB],
                             coeff_vars->vVelocityCoeff[Arches::AP],
                             coeff_vars->vVelocityCoeff[Arches::AE],
                             coeff_vars->vVelocityCoeff[Arches::AW],
                             coeff_vars->vVelocityCoeff[Arches::AN],
                             coeff_vars->vVelocityCoeff[Arches::AS],
                             coeff_vars->vVelocityCoeff[Arches::AT],
                             coeff_vars->vVelocityCoeff[Arches::AB],
                             coeff_constvars->uVelocity, coeff_constvars->wVelocity,
                             coeff_constvars->density, coeff_constvars->viscosity,
                             coeff_constvars->denRefArray, coeff_vars->vVelNonlinearSrc,
                             *volFraction,
                             delta_t, grav, dx, dy, dz,
                             idxLoV, idxHiV);
      break;
    case Discretization::UPWIND:
      fort_vvelcoef_upwind( coeff_constvars->vVelocity,
                            coeff_vars->vVelocityConvectCoeff[Arches::AE],
                            coeff_vars->vVelocityConvectCoeff[Arches::AW],
                            coeff_vars->vVelocityConvectCoeff[Arches::AN],
                            coeff_vars->vVelocityConvectCoeff[Arches::AS],
                            coeff_vars->vVelocityConvectCoeff[Arches::AT],
                            coeff_vars->vVelocityConvectCoeff[Arches::AB],
                            coeff_vars->vVelocityCoeff[Arches::AP],
                            coeff_vars->vVelocityCoeff[Arches::AE],
                            coeff_vars->vVelocityCoeff[Arches::AW],
                            coeff_vars->vVelocityCoeff[Arches::AN],
                            coeff_vars->vVelocityCoeff[Arches::AS],
                            coeff_vars->vVelocityCoeff[Arches::AT],
                            coeff_vars->vVelocityCoeff[Arches::AB],
                            coeff_constvars->uVelocity, coeff_constvars->wVelocity,
                            coeff_constvars->density, coeff_constvars->viscosity,
                            coeff_constvars->denRefArray, coeff_vars->vVelNonlinearSrc,
                            *volFraction,
                            delta_t, grav, dx, dy, dz,
                            idxLoV, idxHiV);
      break;
    case Discretization::WALLUPWIND:
      fort_vvelcoef_mixed( coeff_constvars->vVelocity, coeff_constvars->cellType,
                           coeff_vars->vVelocityConvectCoeff[Arches::AE],
                           coeff_vars->vVelocityConvectCoeff[Arches::AW],
                           coeff_vars->vVelocityConvectCoeff[Arches::AN],
                           coeff_vars->vVelocityConvectCoeff[Arches::AS],
                           coeff_vars->vVelocityConvectCoeff[Arches::AT],
                           coeff_vars->vVelocityConvectCoeff[Arches::AB],
                           coeff_vars->vVelocityCoeff[Arches::AP],
                           coeff_vars->vVelocityCoeff[Arches::AE],
                           coeff_vars->vVelocityCoeff[Arches::AW],
                           coeff_vars->vVelocityCoeff[Arches::AN],
                           coeff_vars->vVelocityCoeff[Arches::AS],
                           coeff_vars->vVelocityCoeff[Arches::AT],
                           coeff_vars->vVelocityCoeff[Arches::AB],
                           coeff_constvars->uVelocity, coeff_constvars->wVelocity,
                           coeff_constvars->density, coeff_constvars->viscosity,
                           coeff_constvars->denRefArray, coeff_vars->vVelNonlinearSrc,
                           *volFraction,
                           delta_t, grav, dx, dy, dz,
                           wall1, wall2, Re_limit,
                           idxLoV, idxHiV);
      break;
    case Discretization::HYBRID:
      fort_vvelcoef_hybrid( coeff_constvars->vVelocity, coeff_constvars->cellType,
                           coeff_vars->vVelocityConvectCoeff[Arches::AE],
                           coeff_vars->vVelocityConvectCoeff[Arches::AW],
                           coeff_vars->vVelocityConvectCoeff[Arches::AN],
                           coeff_vars->vVelocityConvectCoeff[Arches::AS],
                           coeff_vars->vVelocityConvectCoeff[Arches::AT],
                           coeff_vars->vVelocityConvectCoeff[Arches::AB],
                           coeff_vars->vVelocityCoeff[Arches::AP],
                           coeff_vars->vVelocityCoeff[Arches::AE],
                           coeff_vars->vVelocityCoeff[Arches::AW],
                           coeff_vars->vVelocityCoeff[Arches::AN],
                           coeff_vars->vVelocityCoeff[Arches::AS],
                           coeff_vars->vVelocityCoeff[Arches::AT],
                           coeff_vars->vVelocityCoeff[Arches::AB],
                           coeff_constvars->uVelocity, coeff_constvars->wVelocity,
                           coeff_constvars->density, coeff_constvars->viscosity,
                           coeff_constvars->denRefArray, coeff_vars->vVelNonlinearSrc,
                           *volFraction,
                           *conv_scheme_y,
                           delta_t, grav, dx, dy, dz,
                           wall1, wall2, Re_limit,
                           idxLoV, idxHiV);
      break;
    case Discretization::OLD:
      fort_vvelcoef(coeff_constvars->vVelocity,
                    coeff_vars->vVelocityConvectCoeff[Arches::AE],
                    coeff_vars->vVelocityConvectCoeff[Arches::AW],
                    coeff_vars->vVelocityConvectCoeff[Arches::AN],
                    coeff_vars->vVelocityConvectCoeff[Arches::AS],
                    coeff_vars->vVelocityConvectCoeff[Arches::AT],
                    coeff_vars->vVelocityConvectCoeff[Arches::AB],
                    coeff_vars->vVelocityCoeff[Arches::AP],
                    coeff_vars->vVelocityCoeff[Arches::AE],
                    coeff_vars->vVelocityCoeff[Arches::AW],
                    coeff_vars->vVelocityCoeff[Arches::AN],
                    coeff_vars->vVelocityCoeff[Arches::AS],
                    coeff_vars->vVelocityCoeff[Arches::AT],
                    coeff_vars->vVelocityCoeff[Arches::AB],
                    coeff_constvars->uVelocity, coeff_constvars->wVelocity,
                    coeff_constvars->density, coeff_constvars->viscosity,
                    coeff_constvars->denRefArray, coeff_vars->vVelNonlinearSrc,
                    *volFraction,
                    delta_t, grav, lcentral,
                    cellinfo->cee, cellinfo->cwe, cellinfo->cww,
                    cellinfo->cnnv, cellinfo->csnv, cellinfo->cssv,
                    cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
                    cellinfo->sew, cellinfo->snsv, cellinfo->sns,
                    cellinfo->stb, cellinfo->dxep, cellinfo->dxpw,
                    cellinfo->dynpv, cellinfo->dypsv, cellinfo->dyps,
                    cellinfo->dztp, cellinfo->dzpb, cellinfo->fac1v,
                    cellinfo->fac2v, cellinfo->fac3v, cellinfo->fac4v,
                    cellinfo->jnsdv, cellinfo->jssdv, cellinfo->efac,
                    cellinfo->wfac, cellinfo->tfac, cellinfo->bfac,
                    cellinfo->fac1ew, cellinfo->fac2ew, cellinfo->fac3ew,
                    cellinfo->fac4ew, cellinfo->e_shift, cellinfo->w_shift,
                    cellinfo->fac1tb, cellinfo->fac2tb, cellinfo->fac3tb,
                    cellinfo->fac4tb, cellinfo->t_shift, cellinfo->b_shift,
                    idxLoV, idxHiV);
    default:
      break;
  }

  //__________________________________
  //    Z DIR
  oci = IntVector(0,0,-1); //one cell inward.  Only offset at the edge of the computational domain.
  IntVector idxLoW = patch->getSFCZLowIndex() - noNeighborsLow * oci;
  IntVector idxHiW = patch->getSFCZHighIndex()+ noNeighborsHigh * oci - IntVector(1,1,1);

  grav = gravity.z();

  switch (conv_scheme){
    case Discretization::CENTRAL:
      fort_wvelcoef_central( coeff_constvars->wVelocity,
                             coeff_vars->wVelocityConvectCoeff[Arches::AE],
                             coeff_vars->wVelocityConvectCoeff[Arches::AW],
                             coeff_vars->wVelocityConvectCoeff[Arches::AN],
                             coeff_vars->wVelocityConvectCoeff[Arches::AS],
                             coeff_vars->wVelocityConvectCoeff[Arches::AT],
                             coeff_vars->wVelocityConvectCoeff[Arches::AB],
                             coeff_vars->wVelocityCoeff[Arches::AP],
                             coeff_vars->wVelocityCoeff[Arches::AE],
                             coeff_vars->wVelocityCoeff[Arches::AW],
                             coeff_vars->wVelocityCoeff[Arches::AN],
                             coeff_vars->wVelocityCoeff[Arches::AS],
                             coeff_vars->wVelocityCoeff[Arches::AT],
                             coeff_vars->wVelocityCoeff[Arches::AB],
                             coeff_constvars->uVelocity, coeff_constvars->vVelocity,
                             coeff_constvars->density, coeff_constvars->viscosity,
                             coeff_constvars->denRefArray, coeff_vars->wVelNonlinearSrc,
                             *volFraction,
                             delta_t, grav, dx, dy, dz,
                             idxLoW, idxHiW);
      break;
    case Discretization::UPWIND:
      fort_wvelcoef_upwind( coeff_constvars->wVelocity,
                            coeff_vars->wVelocityConvectCoeff[Arches::AE],
                            coeff_vars->wVelocityConvectCoeff[Arches::AW],
                            coeff_vars->wVelocityConvectCoeff[Arches::AN],
                            coeff_vars->wVelocityConvectCoeff[Arches::AS],
                            coeff_vars->wVelocityConvectCoeff[Arches::AT],
                            coeff_vars->wVelocityConvectCoeff[Arches::AB],
                            coeff_vars->wVelocityCoeff[Arches::AP],
                            coeff_vars->wVelocityCoeff[Arches::AE],
                            coeff_vars->wVelocityCoeff[Arches::AW],
                            coeff_vars->wVelocityCoeff[Arches::AN],
                            coeff_vars->wVelocityCoeff[Arches::AS],
                            coeff_vars->wVelocityCoeff[Arches::AT],
                            coeff_vars->wVelocityCoeff[Arches::AB],
                            coeff_constvars->uVelocity, coeff_constvars->vVelocity,
                            coeff_constvars->density, coeff_constvars->viscosity,
                            coeff_constvars->denRefArray, coeff_vars->wVelNonlinearSrc,
                            *volFraction,
                            delta_t, grav, dx, dy, dz,
                            idxLoW, idxHiW);
      break;
    case Discretization::WALLUPWIND:
      fort_wvelcoef_mixed( coeff_constvars->wVelocity, coeff_constvars->cellType,
                           coeff_vars->wVelocityConvectCoeff[Arches::AE],
                           coeff_vars->wVelocityConvectCoeff[Arches::AW],
                           coeff_vars->wVelocityConvectCoeff[Arches::AN],
                           coeff_vars->wVelocityConvectCoeff[Arches::AS],
                           coeff_vars->wVelocityConvectCoeff[Arches::AT],
                           coeff_vars->wVelocityConvectCoeff[Arches::AB],
                           coeff_vars->wVelocityCoeff[Arches::AP],
                           coeff_vars->wVelocityCoeff[Arches::AE],
                           coeff_vars->wVelocityCoeff[Arches::AW],
                           coeff_vars->wVelocityCoeff[Arches::AN],
                           coeff_vars->wVelocityCoeff[Arches::AS],
                           coeff_vars->wVelocityCoeff[Arches::AT],
                           coeff_vars->wVelocityCoeff[Arches::AB],
                           coeff_constvars->uVelocity, coeff_constvars->vVelocity,
                           coeff_constvars->density, coeff_constvars->viscosity,
                           coeff_constvars->denRefArray, coeff_vars->wVelNonlinearSrc,
                           *volFraction,
                           delta_t, grav, dx, dy, dz,
                           wall1, wall2, Re_limit,
                           idxLoW, idxHiW);
      break;
    case Discretization::HYBRID:
      fort_wvelcoef_hybrid( coeff_constvars->wVelocity, coeff_constvars->cellType,
                           coeff_vars->wVelocityConvectCoeff[Arches::AE],
                           coeff_vars->wVelocityConvectCoeff[Arches::AW],
                           coeff_vars->wVelocityConvectCoeff[Arches::AN],
                           coeff_vars->wVelocityConvectCoeff[Arches::AS],
                           coeff_vars->wVelocityConvectCoeff[Arches::AT],
                           coeff_vars->wVelocityConvectCoeff[Arches::AB],
                           coeff_vars->wVelocityCoeff[Arches::AP],
                           coeff_vars->wVelocityCoeff[Arches::AE],
                           coeff_vars->wVelocityCoeff[Arches::AW],
                           coeff_vars->wVelocityCoeff[Arches::AN],
                           coeff_vars->wVelocityCoeff[Arches::AS],
                           coeff_vars->wVelocityCoeff[Arches::AT],
                           coeff_vars->wVelocityCoeff[Arches::AB],
                           coeff_constvars->uVelocity, coeff_constvars->vVelocity,
                           coeff_constvars->density, coeff_constvars->viscosity,
                           coeff_constvars->denRefArray, coeff_vars->wVelNonlinearSrc,
                           *volFraction,
                           *conv_scheme_z,
                           delta_t, grav, dx, dy, dz,
                           wall1, wall2, Re_limit,
                           idxLoW, idxHiW);
      break;
    case Discretization::OLD:
      fort_wvelcoef(coeff_constvars->wVelocity,
                    coeff_vars->wVelocityConvectCoeff[Arches::AE],
                    coeff_vars->wVelocityConvectCoeff[Arches::AW],
                    coeff_vars->wVelocityConvectCoeff[Arches::AN],
                    coeff_vars->wVelocityConvectCoeff[Arches::AS],
                    coeff_vars->wVelocityConvectCoeff[Arches::AT],
                    coeff_vars->wVelocityConvectCoeff[Arches::AB],
                    coeff_vars->wVelocityCoeff[Arches::AP],
                    coeff_vars->wVelocityCoeff[Arches::AE],
                    coeff_vars->wVelocityCoeff[Arches::AW],
                    coeff_vars->wVelocityCoeff[Arches::AN],
                    coeff_vars->wVelocityCoeff[Arches::AS],
                    coeff_vars->wVelocityCoeff[Arches::AT],
                    coeff_vars->wVelocityCoeff[Arches::AB],
                    coeff_constvars->uVelocity, coeff_constvars->vVelocity,
                    coeff_constvars->density, coeff_constvars->viscosity,
                    coeff_constvars->denRefArray, coeff_vars->wVelNonlinearSrc,
                    *volFraction,
                    delta_t, grav, lcentral,
                    cellinfo->cee, cellinfo->cwe, cellinfo->cww,
                    cellinfo->cnn, cellinfo->csn, cellinfo->css,
                    cellinfo->cttw, cellinfo->cbtw, cellinfo->cbbw,
                    cellinfo->sew, cellinfo->sns, cellinfo->stbw,
                    cellinfo->stb, cellinfo->dxep, cellinfo->dxpw,
                    cellinfo->dynp, cellinfo->dyps, cellinfo->dztpw,
                    cellinfo->dzpbw, cellinfo->dzpb, cellinfo->fac1w,
                    cellinfo->fac2w, cellinfo->fac3w, cellinfo->fac4w,
                    cellinfo->ktsdw, cellinfo->kbsdw, cellinfo->efac,
                    cellinfo->wfac, cellinfo->nfac, cellinfo->sfac,
                    cellinfo->fac1ew, cellinfo->fac2ew, cellinfo->fac3ew,
                    cellinfo->fac4ew, cellinfo->e_shift, cellinfo->w_shift,
                    cellinfo->fac1ns, cellinfo->fac2ns, cellinfo->fac3ns,
                    cellinfo->fac4ns, cellinfo->n_shift, cellinfo->s_shift,
                    idxLoW, idxHiW);


    default:
      break;
  }
}

//****************************************************************************
// Calculate the diagonal term A.p in the matrix
//****************************************************************************
template<class T> void
Discretization::compute_Ap(CellIterator iter,
                           CCVariable<Stencil7>& A,
                           T& source)
{
  for(; !iter.done();iter++) {
    IntVector c = *iter;
    Stencil7&  A_tmp=A[c];
    A_tmp.p = -(A_tmp.e + A_tmp.w + A_tmp.n + A_tmp.s + A_tmp.t + A_tmp.b) - source[c];
  }
}




//****************************************************************************
// Calculate the diagonal term A.p in the matrix
//****************************************************************************
template<class T> void
Discretization::compute_Ap_stencilMatrix(CellIterator iter,
                                         StencilMatrix<T>& A,
                                         T& source)
{
  for(; !iter.done();iter++) {
    IntVector c = *iter;
    A[Arches::AP][c] = A[Arches::AE][c] + A[Arches::AW][c]
                     + A[Arches::AN][c] + A[Arches::AS][c]
                     + A[Arches::AT][c] + A[Arches::AB][c] - source[c];
  }
}

template<class T>
struct computeADiagonal{

       computeADiagonal(T &_A_east,
                        T &_A_west,
                        T &_A_north,
                        T &_A_south,
                        T &_A_top,
                        T &_A_bot,
                        T &_A_diag, //or A_center
                        T &_source) :
                        A_east(_A_east),
                        A_west(_A_west),
                        A_north(_A_north),
                        A_south(_A_south),
                        A_top(_A_top),
                        A_bot(_A_bot),
                        A_diag(_A_diag),
                        source(_source)  {  }

       void operator()(int i , int j, int k ) const {
       A_diag(i,j,k) = A_east(i,j,k)  + A_west(i,j,k)
                     + A_north(i,j,k) + A_south(i,j,k)
                     + A_top(i,j,k)   + A_bot(i,j,k)- source(i,j,k);
       }

  private:
       T &A_east;
       T &A_west;
       T &A_north;
       T &A_south;
       T &A_top;
       T &A_bot;
       T &A_diag;
       T &source;
};

//****************************************************************************
// Calculate the diagonal terms (velocity)
//****************************************************************************
void
Discretization::calculateVelDiagonal(const Patch* patch,
                                     ArchesVariables* coeff_vars)
{

  Uintah::BlockRange rangex(patch->getSFCXLowIndex(),patch->getSFCXHighIndex());
  Uintah::BlockRange rangey(patch->getSFCYLowIndex(),patch->getSFCYHighIndex());
  Uintah::BlockRange rangez(patch->getSFCZLowIndex(),patch->getSFCZHighIndex());

  computeADiagonal<SFCXVariable<double> >  doADiagonalX(coeff_vars->uVelocityCoeff[Arches::AE],
                                                        coeff_vars->uVelocityCoeff[Arches::AW],
                                                        coeff_vars->uVelocityCoeff[Arches::AN],
                                                        coeff_vars->uVelocityCoeff[Arches::AS],
                                                        coeff_vars->uVelocityCoeff[Arches::AT],
                                                        coeff_vars->uVelocityCoeff[Arches::AB],
                                                        coeff_vars->uVelocityCoeff[Arches::AP],
                                                        coeff_vars->uVelLinearSrc);
  computeADiagonal<SFCYVariable<double> >  doADiagonalY(coeff_vars->vVelocityCoeff[Arches::AE],
                                                        coeff_vars->vVelocityCoeff[Arches::AW],
                                                        coeff_vars->vVelocityCoeff[Arches::AN],
                                                        coeff_vars->vVelocityCoeff[Arches::AS],
                                                        coeff_vars->vVelocityCoeff[Arches::AT],
                                                        coeff_vars->vVelocityCoeff[Arches::AB],
                                                        coeff_vars->vVelocityCoeff[Arches::AP],
                                                        coeff_vars->vVelLinearSrc);
  computeADiagonal<SFCZVariable<double> >  doADiagonalZ(coeff_vars->wVelocityCoeff[Arches::AE],
                                                        coeff_vars->wVelocityCoeff[Arches::AW],
                                                        coeff_vars->wVelocityCoeff[Arches::AN],
                                                        coeff_vars->wVelocityCoeff[Arches::AS],
                                                        coeff_vars->wVelocityCoeff[Arches::AT],
                                                        coeff_vars->wVelocityCoeff[Arches::AB],
                                                        coeff_vars->wVelocityCoeff[Arches::AP],
                                                        coeff_vars->wVelLinearSrc);
  Uintah::parallel_for( rangex, doADiagonalX);
  Uintah::parallel_for( rangey, doADiagonalY);
  Uintah::parallel_for( rangez, doADiagonalZ);
}

//****************************************************************************
// Pressure diagonal
//****************************************************************************
void
Discretization::calculatePressDiagonal(const Patch* patch,
                                       ArchesVariables* coeff_vars)
{
  CellIterator iter = patch->getCellIterator();
  compute_Ap<CCVariable<double> >(iter,coeff_vars->pressCoeff,
                                       coeff_vars->pressLinearSrc);
}
