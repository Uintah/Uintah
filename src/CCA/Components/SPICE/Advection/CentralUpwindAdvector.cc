/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#include <CCA/Components/SPICE/Advection/CentralUpwindAdvector.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

using namespace Uintah;
using namespace Uintah::centralUpwind;
using namespace std;

/* ---------------------------------------------------------------------
 Kurganov-Tadmor central-upwind advection operator.

 Method:  SPICE/Docs/PIMPLE_KT_method.pdf, Section 6 (eqs. 11-12).
 Math:    SPICE/Core/CentralUpwind.h (reconstruction, limiters, upwind
          pick, ktStencilFlags).
 The face mass fluxes phiOwn/phiNei are computed by
 SPICE::computeCentralFluxes and passed in through setCentralFluxes().

 References:
   Kurganov & Tadmor, J. Comput. Phys. 160 (2000).
   Kurganov, Noelle & Petrova, SIAM J. Sci. Comput. 23 (2001).
   Kraposhin et al., "A hybrid pressure-based/central-upwind solver for
   OpenFOAM" (reactingPimpleCentralFoam).
 ---------------------------------------------------------------------  */

namespace {

  //______________________________________________________________________
  //  Clamp a cell index to the level's cell range (non-periodic
  //  directions only).  Keeps reads in bounds when looping two cells out
  //  at a domain boundary that has only one layer of extra cells
  inline IntVector clampToLevel( const IntVector & c,
                                 const IntVector & lo,
                                 const IntVector & hi,
                                 const IntVector & periodic )
  {
    IntVector r = c;
    for( int d = 0; d < 3; d++ ){
      if( periodic[d] == 0 ){
        if( r[d] < lo[d] )     { r[d] = lo[d]; }
        if( r[d] > hi[d] - 1 ) { r[d] = hi[d] - 1; }
      }
    }
    return r;
  }

}  // anonymous namespace

//______________________________________________________________________
CentralUpwindAdvector::CentralUpwindAdvector()
{
}

CentralUpwindAdvector::CentralUpwindAdvector( Limiter limiter )
  : d_limiter( limiter )
{
}

CentralUpwindAdvector::CentralUpwindAdvector( Limiter limiter,
                                              DataWarehouse* /*new_dw*/,
                                              const Patch* patch,
                                              const bool /*isNewGrid*/ )
  : d_limiter( limiter ), d_patch( patch )
{
  if( patch ){
    patch->getLevel()->findCellIndexRange( d_levelLo, d_levelHi );
    d_periodic = patch->getLevel()->getPeriodicBoundaries();
    d_dx       = patch->dCell();
  }
}

CentralUpwindAdvector::~CentralUpwindAdvector()
{
}

CentralUpwindAdvector*
CentralUpwindAdvector::clone( DataWarehouse* new_dw,
                              const Patch*   patch,
                              const bool     isNewGrid )
{
  return scinew CentralUpwindAdvector( d_limiter, new_dw, patch, isNewGrid );
}

//______________________________________________________________________
//  Outflux volumes and the outflux > cell volume check, as in
//  FirstOrderAdvector.  VB->OFS is not used by the KT advect methods but
//  is filled for callers that expect it.  Also stores delT, dx and the
//  face velocities for the intensive advectQ (vol_frac), which needs u_f.
//  The pointers are valid only for the task that called this function.
void CentralUpwindAdvector::inFluxOutFluxVolume(
                        const SFCXVariable<double>& uvel_FC,
                        const SFCYVariable<double>& vvel_FC,
                        const SFCZVariable<double>& wvel_FC,
                        const double& delT,
                        const Patch* patch,
                        const int& indx,
                        const bool& bulletProof_test,
                        DataWarehouse* new_dw,
                        advectVarBasket* VB)
{
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom,delX_right, delX_left, delZ_front, delZ_back;
  double delX = dx.x(), delY = dx.y(), delZ = dx.z();

  d_delT    = delT;
  d_dx      = dx;
  d_uvel_FC = &uvel_FC;
  d_vvel_FC = &vvel_FC;
  d_wvel_FC = &wvel_FC;

  new_dw->allocateTemporary(VB->OFS,patch, Ghost::AroundCells,1);
  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  bool error = false;

  int NGC =1;  // number of ghostCells
  for(CellIterator iter = patch->getExtraCellIterator(NGC); !iter.done(); iter++) {
    const IntVector& c = *iter;

    delY_top    = std::max(0.0, (vvel_FC[c+IntVector(0,1,0)] * delT));
    delY_bottom = std::max(0.0,-(vvel_FC[c                 ] * delT));
    delX_right  = std::max(0.0, (uvel_FC[c+IntVector(1,0,0)] * delT));
    delX_left   = std::max(0.0,-(uvel_FC[c                 ] * delT));
    delZ_front  = std::max(0.0, (wvel_FC[c+IntVector(0,0,1)] * delT));
    delZ_back   = std::max(0.0,-(wvel_FC[c                 ] * delT));

    //__________________________________
    //   SLAB outfluxes
    double delX_Z = delX * delZ;
    double delX_Y = delX * delY;
    double delY_Z = delY * delZ;
    fflux& ofs = VB->OFS[c];

    ofs.fflux[TOP]   = delY_top   * delX_Z;
    ofs.fflux[BOTTOM]= delY_bottom* delX_Z;
    ofs.fflux[RIGHT] = delX_right * delY_Z;
    ofs.fflux[LEFT]  = delX_left  * delY_Z;
    ofs.fflux[FRONT] = delZ_front * delX_Y;
    ofs.fflux[BACK]  = delZ_back  * delX_Y;

    //__________________________________
    //  Bullet proofing
    double total_fluxout = 0.0;
    for(int face = TOP; face <= BACK; face++ )  {
      total_fluxout  += ofs.fflux[face];
    }
    if(total_fluxout > vol){
      error = true;
    }
  }  //cell iterator

  bool rts = new_dw->recomputeTimeStep();

  if (error && bulletProof_test && !rts) {
    vector<IntVector> badCells;
    vector<fflux>  badOutflux;

    for(CellIterator iter = patch->getExtraCellIterator(NGC); !iter.done(); iter++) {
      IntVector c = *iter;
      double total_fluxout = 0.0;
      fflux& ofs = VB->OFS[c];

      for(int face = TOP; face <= BACK; face++ )  {
        total_fluxout  += VB->OFS[c].fflux[face];
        VB->OFS[c].fflux[face] = 0.0;
      }
      // keep track of which cells are bad
      if (vol - total_fluxout < 0.0) {
        badCells.push_back(c);
        badOutflux.push_back(ofs);
      }
    }  // cell iter
    warning_recomputeTimeStep( badCells,badOutflux, vol, indx, patch);
    cout << "\nA time step recompute has been requested \n " << endl;
  }  // if total_fluxout > vol

  if (error && bulletProof_test) {
    new_dw->put( bool_or_vartype(true), VarLabel::find(abortTimeStep_name));
    new_dw->put( bool_or_vartype(true), VarLabel::find(recomputeTimeStep_name));
  }

  if (error && !bulletProof_test) {
    std::cout << " WARNING: ICE Advection operator Influx/Outflux volume error:"
              << " Patch " << patch->getID()
              << ", Level " << patch->getLevel()->getIndex()<< std::endl;
  }
}

//______________________________________________________________________
void CentralUpwindAdvector::setCentralFluxes( const SFCXVariable<double>* phiOwnX,
                                              const SFCYVariable<double>* phiOwnY,
                                              const SFCZVariable<double>* phiOwnZ,
                                              const SFCXVariable<double>* phiNeiX,
                                              const SFCYVariable<double>* phiNeiY,
                                              const SFCZVariable<double>* phiNeiZ )
{
  d_phiOwnX = phiOwnX;
  d_phiOwnY = phiOwnY;
  d_phiOwnZ = phiOwnZ;
  d_phiNeiX = phiNeiX;
  d_phiNeiY = phiNeiY;
  d_phiNeiZ = phiNeiZ;
  d_hasCentralFluxes = true;
}

//______________________________________________________________________
void CentralUpwindAdvector::checkNoRefluxing( const advectVarBasket* vb ) const
{
  if( vb->doRefluxing ){
    throw InternalError(
      "CentralUpwind advection does not support AMR refluxing",
      __FILE__, __LINE__);
  }
}

/*_____________________________________________________________________
     M A S S
 dm_f = delT * A_f * (phiOwn + phiNei) at each face (positive = flow in
 the +dir direction); q_adv[c] = sum over the 3 directions of
 (dm_lowFace - dm_highFace).  PDF eqs. 11-12 with q = 1.
_____________________________________________________________________*/
void CentralUpwindAdvector::advectMass(const CCVariable<double>& mass,
                                       CCVariable<double>& q_advected,
                                       advectVarBasket* vb)
{
  checkNoRefluxing(vb);

  if( !d_hasCentralFluxes ){
    throw InternalError(
      "CentralUpwindAdvector::advectMass: no KT fluxes -- "
      "SPICE::advectAndAdvanceInTime must call setCentralFluxes() before calling advectMass()",
      __FILE__, __LINE__);
  }

  const Patch* patch = vb->patch;
  Vector dx = patch->dCell();
  double Ax = dx.y()*dx.z();
  double Ay = dx.x()*dx.z();
  double Az = dx.x()*dx.y();

  bool error = false;
  vector<IntVector> badCells;

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    const IntVector& c = *iter;
    IntVector xHi = c + IntVector(1,0,0);
    IntVector yHi = c + IntVector(0,1,0);
    IntVector zHi = c + IntVector(0,0,1);

    double dm_xLo = d_delT*Ax*( (*d_phiOwnX)[c]   + (*d_phiNeiX)[c]   );
    double dm_xHi = d_delT*Ax*( (*d_phiOwnX)[xHi] + (*d_phiNeiX)[xHi] );
    double dm_yLo = d_delT*Ay*( (*d_phiOwnY)[c]   + (*d_phiNeiY)[c]   );
    double dm_yHi = d_delT*Ay*( (*d_phiOwnY)[yHi] + (*d_phiNeiY)[yHi] );
    double dm_zLo = d_delT*Az*( (*d_phiOwnZ)[c]   + (*d_phiNeiZ)[c]   );
    double dm_zHi = d_delT*Az*( (*d_phiOwnZ)[zHi] + (*d_phiNeiZ)[zHi] );

    q_advected[c] = (dm_xLo - dm_xHi) + (dm_yLo - dm_yHi) + (dm_zLo - dm_zHi);

    //__________________________________
    //  Bullet proofing: total mass leaving the cell can't exceed the
    //  mass it starts the step with (mirrors FirstOrderAdvector's
    //  outflux-vs-cellVolume test, in mass units since the KT fluxes
    //  are already physical mass fluxes).
    double outflow = std::max(0.0,-dm_xLo) + std::max(0.0, dm_xHi) +
                     std::max(0.0,-dm_yLo) + std::max(0.0, dm_yHi) +
                     std::max(0.0,-dm_zLo) + std::max(0.0, dm_zHi);
    if( outflow > mass[c] ){
      error = true;
      badCells.push_back(c);
    }
  }

  bool rts = vb->new_dw->recomputeTimeStep();
  if( error && !rts ){
    cout << " ERROR: CentralUpwindAdvector::advectMass: outflow > mass[c] in "
         << badCells.size() << " cells on patch " << patch->getID()
         << ", matl " << vb->indx << endl;
    for( unsigned int i = 0; i < badCells.size(); i++ ){
      cout << "   cell " << badCells[i] << endl;
    }
    vb->new_dw->put( bool_or_vartype(true), VarLabel::find(abortTimeStep_name));
    vb->new_dw->put( bool_or_vartype(true), VarLabel::find(recomputeTimeStep_name));
    cout << "\nA time step recompute has been requested \n " << endl;
  }
}

/*_____________________________________________________________________
   D O U B L E   /   V E C T O R   (mass-specific: mom, int_eng, KE,
   sp_vol, transported vars)
 qSpec = q_CC/mass reconstructed independently at every face; the KT
 upwind blend F = phiOwn*upwind(phiOwn,qOwn,qNei) +
                  phiNei*upwind(phiNei,qOwn,qNei)
 transfer_f = delT*A_f*F;  q_adv[c] = sum over the 3 directions of
 (transfer_lowFace - transfer_highFace).  PDF eqs. 11-12.
_____________________________________________________________________*/
template<class T>
void CentralUpwindAdvector::advectQ_massSpecific( const CCVariable<T>&      q_CC,
                                                  const CCVariable<double>& mass,
                                                  CCVariable<T>&            q_advected,
                                                  advectVarBasket*          vb )
{
  checkNoRefluxing(vb);

  if( !d_hasCentralFluxes ){
    throw InternalError(
      "CentralUpwindAdvector::advectQ: no KT fluxes -- "
      "SPICE::advectAndAdvanceInTime must call setCentralFluxes() before calling advectQ()",
      __FILE__, __LINE__);
  }

  const Patch* patch = vb->patch;
  Vector dx = patch->dCell();
  double Ax = dx.y()*dx.z();
  double Ay = dx.x()*dx.z();
  double Az = dx.x()*dx.y();
  const double tiny = 1.0e-100;

  //__________________________________
  //  qSpec = q_CC/mass, the specific quantity that is reconstructed
  //  (e.g. velocity for momentum)
  CCVariable<T> qSpec;
  vb->new_dw->allocateTemporary( qSpec, patch, Ghost::AroundCells, 2 );
  for(CellIterator iter = patch->getExtraCellIterator(2); !iter.done(); iter++){
    const IntVector& c = *iter;
    // clamp: at a non-periodic domain boundary with fewer real extra
    // cells than the requested ghost (2), q_CC/mass aren't populated
    // this far out -- reuse the nearest valid cell (harmless: the KT
    // stencil never actually uses qSpec there, see ktStencilFlags below).
    IntVector cSafe = clampToLevel(c, d_levelLo, d_levelHi, d_periodic);
    qSpec[c] = q_CC[cSafe] * (1.0/std::max( mass[cSafe], tiny ));
  }

  vector<IntVector> adj_offset(3);
  adj_offset[0] = IntVector(-1, 0, 0);
  adj_offset[1] = IntVector(0, -1, 0);
  adj_offset[2] = IntVector(0,  0, -1);

  //__________________________________
  //  Face flux F at face N (direction dir), PDF eq. 11.  Computed on
  //  demand rather than stored in an SFC array: a patch's SFC iterator
  //  does not include the high face shared with a higher-index neighbor
  //  patch, but the last interior cell still needs it.  qSpec and
  //  phiOwn/phiNei both have ghost data there.
  auto faceF = [&]( const IntVector & N, int dir ) -> T {
    IntVector P   = N + adj_offset[dir];
    IntVector Pm1 = P + adj_offset[dir];
    IntVector Np1 = N - adj_offset[dir];

    bool firstOrderOwn, firstOrderNei;
    ktStencilFlags(Pm1, Np1, dir, d_levelLo, d_levelHi, d_periodic,
                  firstOrderOwn, firstOrderNei);

    // see the Pm1Safe/Np1Safe comment in SPICE::computeCentralWeights
    const IntVector & Pm1Safe = firstOrderOwn ? P : Pm1;
    const IntVector & Np1Safe = firstOrderNei ? N : Np1;

    T qOwn, qNei;
    ownNei( qSpec[Pm1Safe], qSpec[P], qSpec[N], qSpec[Np1Safe],
           d_limiter, firstOrderOwn, firstOrderNei, qOwn, qNei );

    //  FSPICE no-penetration face: the donor's own specific value
    if( d_interfaceFaceX ){
      const double marker = (dir==0) ? (*d_interfaceFaceX)[N] :
                            (dir==1) ? (*d_interfaceFaceY)[N] :
                                       (*d_interfaceFaceZ)[N];
      if( marker >= 0.0 ){
        qOwn = qSpec[P];
        qNei = qSpec[N];
      }
    }

    double phiOwn = (dir==0) ? (*d_phiOwnX)[N] :
                    (dir==1) ? (*d_phiOwnY)[N] :
                               (*d_phiOwnZ)[N];
    double phiNei = (dir==0) ? (*d_phiNeiX)[N] :
                    (dir==1) ? (*d_phiNeiY)[N] :
                               (*d_phiNeiZ)[N];

    return phiOwn*upwindValue( phiOwn, qOwn, qNei ) +
           phiNei*upwindValue( phiNei, qOwn, qNei );
  };

  //__________________________________
  //  q_adv[c] = in - out   (extensive, no /cellVol)
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    const IntVector& c = *iter;
    IntVector xHi = c + IntVector(1,0,0);
    IntVector yHi = c + IntVector(0,1,0);
    IntVector zHi = c + IntVector(0,0,1);

    T transfer_xLo = d_delT*Ax*faceF(c,   0);
    T transfer_xHi = d_delT*Ax*faceF(xHi, 0);
    T transfer_yLo = d_delT*Ay*faceF(c,   1);
    T transfer_yHi = d_delT*Ay*faceF(yHi, 1);
    T transfer_zLo = d_delT*Az*faceF(c,   2);
    T transfer_zHi = d_delT*Az*faceF(zHi, 2);

    q_advected[c] = (transfer_xLo - transfer_xHi) +
                    (transfer_yLo - transfer_yHi) +
                    (transfer_zLo - transfer_zHi);
  }
}

void CentralUpwindAdvector::advectQ(const CCVariable<double>& q_CC,
                                    const CCVariable<double>& mass,
                                    CCVariable<double>& q_advected,
                                    advectVarBasket* vb)
{
  advectQ_massSpecific<double>(q_CC, mass, q_advected, vb);
}

void CentralUpwindAdvector::advectQ(const CCVariable<Vector>& q_CC,
                                    const CCVariable<double>& mass,
                                    CCVariable<Vector>& q_advected,
                                    advectVarBasket* vb)
{
  advectQ_massSpecific<Vector>(q_CC, mass, q_advected, vb);
}

/*_____________________________________________________________________
   S P E C I A L I Z E D   D O U B L E   (intensive, e.g. vol_frac)
 Volume transfer dV_f = delT*A_f*u_f with the velocities given to
 inFluxOutFluxVolume(); qFace = upwind(u_f, qOwn, qNei) with qOwn/qNei
 reconstructed straight from q_CC (own/nei, no mass division -- q_CC is
 already intensive).  q_adv[c] = (sum in - out of qFace*dV_f)/cellVol.
 PDF Section 6, intensive fields.
_____________________________________________________________________*/
void CentralUpwindAdvector::advectQ(const CCVariable<double>& q_CC,
                                    const Patch* patch,
                                    CCVariable<double>& q_advected,
                                    advectVarBasket* vb,
                                    SFCXVariable<double>& q_XFC,
                                    SFCYVariable<double>& q_YFC,
                                    SFCZVariable<double>& q_ZFC,
                                    DataWarehouse* /*new_dw*/)
{
  checkNoRefluxing(vb);

  Vector dx = patch->dCell();
  double Ax = dx.y()*dx.z();
  double Ay = dx.x()*dx.z();
  double Az = dx.x()*dx.y();
  double invVol = 1.0/(dx.x()*dx.y()*dx.z());

  vector<IntVector> adj_offset(3);
  adj_offset[0] = IntVector(-1, 0, 0);
  adj_offset[1] = IntVector(0, -1, 0);
  adj_offset[2] = IntVector(0,  0, -1);

  //__________________________________
  //  Upwind face value at face N (direction dir).  Computed on demand,
  //  not read back from q_XFC/YFC/ZFC, for the same reason as faceF in
  //  advectQ_massSpecific.
  auto faceQ = [&]( const IntVector & N, int dir ) -> double {
    IntVector P   = N + adj_offset[dir];
    IntVector Pm1 = P + adj_offset[dir];
    IntVector Np1 = N - adj_offset[dir];

    bool firstOrderOwn, firstOrderNei;
    ktStencilFlags(Pm1, Np1, dir, d_levelLo, d_levelHi, d_periodic,
                  firstOrderOwn, firstOrderNei);

    const IntVector & Pm1Safe = firstOrderOwn ? P : Pm1;
    const IntVector & Np1Safe = firstOrderNei ? N : Np1;

    double qOwn, qNei;
    ownNei( q_CC[Pm1Safe], q_CC[P], q_CC[N], q_CC[Np1Safe],
           d_limiter, firstOrderOwn, firstOrderNei, qOwn, qNei );

    double u_f = (dir==0) ? (*d_uvel_FC)[N] :
                 (dir==1) ? (*d_vvel_FC)[N] :
                            (*d_wvel_FC)[N];

    //  FSPICE no-penetration face partition (SPICE::computeDelPressAndUpdatePressCC)
    if( d_faceFracX ){
      const double ovr = (dir==0) ? (*d_faceFracX)[N] :
                         (dir==1) ? (*d_faceFracY)[N] :
                                    (*d_faceFracZ)[N];
      if( ovr >= 0.0 ){
        return ovr;
      }
    }

    return upwindValue( u_f, qOwn, qNei );
  };

  //__________________________________
  //  Save qFace on every face this patch owns (same coverage as
  //  FirstOrderAdvector).  Output only; consumers read only owned faces.
  for( int dir = 0; dir < 3; dir++ ){
    CellIterator it = (dir==0) ? patch->getSFCXIterator() :
                      (dir==1) ? patch->getSFCYIterator() :
                                 patch->getSFCZIterator();
    for(; !it.done(); it++){
      IntVector N = *it;
      double qFace = faceQ(N, dir);

      if( dir==0 )      { q_XFC[N] = qFace; }
      else if( dir==1 ) { q_YFC[N] = qFace; }
      else              { q_ZFC[N] = qFace; }
    }
  }

  //__________________________________
  //  q_adv[c] = (sum_faces qFace*dV_f)/cellVol
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    const IntVector& c = *iter;
    IntVector xHi = c + IntVector(1,0,0);
    IntVector yHi = c + IntVector(0,1,0);
    IntVector zHi = c + IntVector(0,0,1);

    double dV_xLo = d_delT*Ax*(*d_uvel_FC)[c];
    double dV_xHi = d_delT*Ax*(*d_uvel_FC)[xHi];
    double dV_yLo = d_delT*Ay*(*d_vvel_FC)[c];
    double dV_yHi = d_delT*Ay*(*d_vvel_FC)[yHi];
    double dV_zLo = d_delT*Az*(*d_wvel_FC)[c];
    double dV_zHi = d_delT*Az*(*d_wvel_FC)[zHi];

    double in_out = faceQ(c,   0)*dV_xLo - faceQ(xHi, 0)*dV_xHi +
                    faceQ(c,   1)*dV_yLo - faceQ(yHi, 1)*dV_yHi +
                    faceQ(c,   2)*dV_zLo - faceQ(zHi, 2)*dV_zHi;

    q_advected[c] = in_out * invVol;
  }
}
