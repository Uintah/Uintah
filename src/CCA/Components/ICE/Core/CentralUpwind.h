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

#ifndef UINTAH_ICE_CORE_CENTRALUPWIND_H
#define UINTAH_ICE_CORE_CENTRALUPWIND_H

//______________________________________________________________________
//  Kurganov-Tadmor central-upwind building blocks.
//
//  Pure math: no Patch, no DataWarehouse.  The formulas follow
//  reactingPimpleCentralFoam (Kraposhin's hybrid KT / PIMPLE solver):
//
//    own/nei reconstruction   fvc::interpolate(q, own|nei, "reconstruct(q)")
//                             with the vanAlbada limitedScheme
//    wave speeds              ap = max(u_own + c_own, u_nei + c_nei, 0)
//                             am = min(u_own - c_own, u_nei - c_nei, 0)
//    weights (Kurganov)       alpha_own = ap/(ap - am),  aSf = am*alpha_own
//    weights (Tadmor)         alpha_own = 1/2,           aSf = -amax/2
//    blending                 kappa = min( Ma_f / Co_ac, 1 )
//                             (standardMachToAcCourantRatio)
//
//  Face convention: the face lies between cell P (low side) and cell N
//  (high side) along the face normal.  "own" is the value reconstructed
//  from the P side, "nei" the value reconstructed from the N side.
//  qm1 = q[P-1], qP = q[P], qN = q[N], qp2 = q[N+1].
//______________________________________________________________________

#include <Core/Geometry/Vector.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace Uintah {
namespace centralUpwind {

  enum class Limiter { vanAlbada, vanLeer, minmod, MC, upwind };
  enum class FluxScheme { Kurganov, Tadmor };
  enum class KappaMode { MachToAcCourant, one };

  //______________________________________________________________________
  //  string -> enum.  Return false for an unknown name so the caller can
  //  throw a ProblemSetupException with its own context.
  inline bool limiterFromString( const std::string & name, Limiter & lim )
  {
    if( name == "vanAlbada" ){ lim = Limiter::vanAlbada; return true; }
    if( name == "vanLeer" )  { lim = Limiter::vanLeer;   return true; }
    if( name == "minmod" )   { lim = Limiter::minmod;    return true; }
    if( name == "MC" )       { lim = Limiter::MC;        return true; }
    if( name == "upwind" )   { lim = Limiter::upwind;    return true; }
    return false;
  }

  inline bool fluxSchemeFromString( const std::string & name, FluxScheme & fs )
  {
    if( name == "Kurganov" ){ fs = FluxScheme::Kurganov; return true; }
    if( name == "Tadmor" )  { fs = FluxScheme::Tadmor;   return true; }
    return false;
  }

  inline bool kappaModeFromString( const std::string & name, KappaMode & km )
  {
    if( name == "MachToAcCourant" ){ km = KappaMode::MachToAcCourant; return true; }
    if( name == "one" )            { km = KappaMode::one;             return true; }
    return false;
  }

  //______________________________________________________________________
  //  TVD limiter psi(r).  psi = 0 for r <= 0 (local extremum -> first order).
  //  vanAlbada is OpenFOAM's definition: r(r+1)/(r^2+1).
  inline double limiter( const double r, const Limiter lim )
  {
    if( r <= 0.0 ){
      return 0.0;
    }
    //  r can be huge (or +inf) when the jump across the face is tiny next
    //  to a finite upwind jump -- routine for trace species at a front.
    //  The textbook forms then evaluate inf/inf = NaN, so for r > 1 use the
    //  algebraically identical forms written in 1/r.
    switch( lim ){
      case Limiter::vanAlbada:
        if( r <= 1.0 ){
          return r*(r + 1.0)/(r*r + 1.0);
        }
        return (1.0 + 1.0/r)/(1.0 + 1.0/(r*r));        // -> 1 as r -> inf
      case Limiter::vanLeer:
        if( r <= 1.0 ){
          return 2.0*r/(1.0 + r);                       // (r+|r|)/(1+|r|), r > 0
        }
        return 2.0/(1.0 + 1.0/r);                       // -> 2 as r -> inf
      case Limiter::minmod:
        return std::min( r, 1.0 );
      case Limiter::MC:
        return std::min( std::min( 2.0*r, 0.5*(1.0 + r) ), 2.0 );
      case Limiter::upwind:
      default:
        return 0.0;
    }
  }

  //______________________________________________________________________
  //  Limited own/nei reconstruction of a scalar at the face between P and N.
  //    own = qP + 1/2 psi(rP) (qN - qP),   rP = (qP  - qm1)/(qN - qP)
  //    nei = qN - 1/2 psi(rN) (qN - qP),   rN = (qp2 - qN )/(qN - qP)
  //  On a uniform mesh this is identical to OpenFOAM's gradient-based
  //  r = 2 (d . grad(q)_P)/(qN - qP) - 1 with a central cell gradient.
  //
  //  firstOrderOwn / firstOrderNei force psi = 0 on that side; the caller
  //  sets them when qm1 (resp. qp2) lies outside the domain's extra-cell
  //  layer, i.e. the stencil is not available.
  inline void ownNei( const double qm1,
                      const double qP,
                      const double qN,
                      const double qp2,
                      const Limiter lim,
                      const bool firstOrderOwn,
                      const bool firstOrderNei,
                      double & own,
                      double & nei )
  {
    const double dq = qN - qP;
    const double scale = std::max( std::fabs(qP), std::fabs(qN) );

    // no jump across the face -> nothing to limit
    if( std::fabs(dq) <= 1.0e-14*scale || dq == 0.0 ){
      own = qP;
      nei = qN;
      return;
    }

    double psiP = 0.0;
    double psiN = 0.0;
    if( !firstOrderOwn ){
      psiP = limiter( (qP - qm1)/dq, lim );
    }
    if( !firstOrderNei ){
      psiN = limiter( (qp2 - qN)/dq, lim );
    }

    own = qP + 0.5*psiP*dq;
    nei = qN - 0.5*psiN*dq;
  }

  //______________________________________________________________________
  //  Vector overload: each component is limited independently.
  //  (OpenFOAM's vanAlbadaV uses a single limiter from the projection on
  //  the jump direction; componentwise is the standard structured-grid
  //  equivalent and reduces to it in 1D.)
  inline void ownNei( const Vector & qm1,
                      const Vector & qP,
                      const Vector & qN,
                      const Vector & qp2,
                      const Limiter lim,
                      const bool firstOrderOwn,
                      const bool firstOrderNei,
                      Vector & own,
                      Vector & nei )
  {
    for( int i = 0; i < 3; i++ ){
      double o = 0.0;
      double n = 0.0;
      ownNei( qm1[i], qP[i], qN[i], qp2[i], lim, firstOrderOwn, firstOrderNei, o, n );
      own[i] = o;
      nei[i] = n;
    }
  }

  //______________________________________________________________________
  //  Upwind selection of an own/nei pair by the sign of a face flux.
  //  flux >= 0 flows P -> N, so the donor is the own (P-side) value.
  template<class T>
  inline T upwindValue( const double flux, const T & own, const T & nei )
  {
    return ( flux >= 0.0 ) ? own : nei;
  }

  //______________________________________________________________________
  //  Central-upwind weights at one face.
  //    uOwn, uNei : face-normal velocity reconstructed from each side
  //    cOwn, cNei : sound speed reconstructed from each side
  //  Outputs
  //    alphaOwn   : weight of the own side (alphaNei = 1 - alphaOwn)
  //    aSf        : KT dissipation coefficient, <= 0.  The numerical flux is
  //                 F = alphaOwn*F_own + alphaNei*F_nei + aSf*(U_nei - U_own)
  //    amax       : max(ap, -am), the largest local wave speed
  inline void weights( const double uOwn,
                       const double uNei,
                       const double cOwn,
                       const double cNei,
                       const FluxScheme scheme,
                       double & alphaOwn,
                       double & aSf,
                       double & amax )
  {
    const double ap = std::max( std::max( uOwn + cOwn, uNei + cNei ), 0.0 );
    const double am = std::min( std::min( uOwn - cOwn, uNei - cNei ), 0.0 );

    amax = std::max( ap, -am );

    if( scheme == FluxScheme::Tadmor ){
      alphaOwn = 0.5;
      aSf      = -0.5*amax;
      return;
    }

    // Kurganov
    const double denom = ap - am;               // >= 0
    if( denom <= 1.0e-300 ){
      alphaOwn = 0.5;
      aSf      = 0.0;
      return;
    }
    alphaOwn = ap/denom;
    aSf      = am*alphaOwn;
  }

  //______________________________________________________________________
  //  Hybrid blending coefficient (standardMachToAcCourantRatio):
  //    kappa = min( Ma_f / Co_ac, 1 )
  //  kappa -> 1 : full compressible KT flux (shocks, detonation fronts)
  //  kappa -> 0 : pressure-based, upwind-limited flux (low Mach / large
  //               acoustic Courant number)
  inline double kappa( const double maFace,
                       const double coAc,
                       const KappaMode mode )
  {
    if( mode == KappaMode::one ){
      return 1.0;
    }
    const double k = maFace/std::max( coAc, 1.0e-300 );
    return std::max( 0.0, std::min( k, 1.0 ) );
  }

}  // namespace centralUpwind
}  // namespace Uintah

#endif
