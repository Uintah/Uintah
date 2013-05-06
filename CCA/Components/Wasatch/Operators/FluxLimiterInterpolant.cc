/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

#include "FluxLimiterInterpolant.h"
#include "OperatorTypes.h"
#include <CCA/Components/Wasatch/FieldAdaptor.h>

#include <cmath>

#include "spatialops/SpatialOpsDefs.h"
#include "spatialops/structured/FVTools.h"


//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
FluxLimiterInterpolant<PhiVolT,PhiFaceT>::
FluxLimiterInterpolant( const std::vector<int>& dim,
                        const std::vector<bool> hasPlusFace,
                        const std::vector<bool> hasMinusBoundary )
: hasPlusBoundary_ (false),
  hasMinusBoundary_(false)
{
  
  const size_t direction = PhiFaceT::Location::FaceDir::value;
  
  switch (direction) {
    case SpatialOps::XDIR::value:
      unitNormal_ = SpatialOps::structured::IntVec(1,0,0);
      hasMinusBoundary_ = hasMinusBoundary[0];
      hasPlusBoundary_ = hasPlusFace[0];
      break;
    case SpatialOps::YDIR::value:
      unitNormal_ = SpatialOps::structured::IntVec(0,1,0);
      hasMinusBoundary_ = hasMinusBoundary[1];
      hasPlusBoundary_ = hasPlusFace[1];
      break;
    case SpatialOps::ZDIR::value:
      unitNormal_ = SpatialOps::structured::IntVec(0,0,1);
      hasMinusBoundary_ = hasMinusBoundary[2];
      hasPlusBoundary_ = hasPlusFace[2];
      break;
    default:
      unitNormal_ = SpatialOps::structured::IntVec(0,0,0);
      break;
  }
  
}

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
void
FluxLimiterInterpolant<PhiVolT,PhiFaceT>::
set_advective_velocity( const PhiFaceT &theAdvectiveVelocity )
{
  // !!! NOT THREAD SAFE !!! USE LOCK
  advectiveVelocity_ = &theAdvectiveVelocity;
}

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
void
FluxLimiterInterpolant<PhiVolT,PhiFaceT>::
set_flux_limiter_type( Wasatch::ConvInterpMethods limiterType )
{
  limiterType_ = limiterType;
}

//--------------------------------------------------------------------

double calculate_flux_limiter_function( double r, Wasatch::ConvInterpMethods limiterType ) {
  // r is the ratio of successive gradients on the mesh
  // limiterType holds the name of the limiter function to be used
  const double infinity_ = 1.0e10;
  if ( r < -infinity_ ) return 0.0;
  
  double psi = 0.0;
  switch (limiterType) {
      
    case Wasatch::UPWIND:
      psi = 0.0;
      break;
      
    case Wasatch::SUPERBEE:
      if ( r < infinity_ ) {
        psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
        psi = std::max( 0.0, psi );
      } else {
        psi = 2.0;
      }
      break;
      
    case Wasatch::CHARM:
      if ( r < infinity_ ) {
        if (r > 0.0) psi = r*(3.0*r + 1)/((r+1)*(r+1));
        else psi = 0.0;
      } else {
        psi = 3.0;
      }
      break;
      
    case Wasatch::KOREN:
      if ( r < infinity_ ) {
        psi = std::min(2.0*r, (r + 2.0)/3.0);
        psi = std::min(psi, 2.0);
        psi = std::max(0.0, psi);
      } else {
        psi = 2.0;
      }
      break;
      
    case Wasatch::MC:
      if ( r < infinity_ ) {
        psi = std::min(2.0*r, 0.5*(r + 1.0));
        psi = std::min(psi, 2.0);
        psi = std::max(0.0, psi);
      } else {
        psi = 2.0;
      }
      break;
      
    case Wasatch::OSPRE:
      psi=( r < infinity_ ) ? 1.5*(r*r + r)/(r*r + r + 1.0) : 1.5;
      break;
      
    case Wasatch::SMART:
      if ( r < infinity_ ) {
        psi = std::min(2.0*r, 0.75*r + 0.25);
        psi = std::min(psi, 4.0);
        psi = std::max(0.0, psi);
      } else {
        psi = 4.0;
      }
      break;
      
    case Wasatch::VANLEER:
      psi=( r < infinity_ ) ? (r + std::abs(r))/(1 + std::abs(r)) : 2.0;
      break;
      
    case Wasatch::HCUS:
      psi=( r < infinity_ ) ? ( 1.5*(r + std::abs(r)) )/(r + 2.0) : 3.0;
      break;
      
    case Wasatch::MINMOD:
      psi=( r < infinity_ ) ? std::max( 0.0, std::min(1.0, r) ) : 1.0;
      break;
      
    case Wasatch::HQUICK:
      psi=( r < infinity_ ) ? ( 2*(r + std::abs(r)) )/(r + 3.0) : 4.0;
      break;
      
    default:
      psi = 0.0;
      break;
  }
  return psi;
}

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
FluxLimiterInterpolant<PhiVolT,PhiFaceT>::
~FluxLimiterInterpolant()
{}

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
void
FluxLimiterInterpolant<PhiVolT,PhiFaceT>::
apply_embedded_boundaries( const PhiVolT &src, PhiFaceT &dest ) const {
  
  using namespace SpatialOps;
  using namespace SpatialOps::structured;
  build_src_iterators(src);

  // now do interior
  const MemoryWindow& wdest = dest.window_with_ghost(); // used for velocity & interpolated phi
  IntVec destExtent = wdest.extent() - unitNormal_*3 - unitNormal_ * (hasPlusBoundary_ ? 1 : 0);
  IntVec destBaseOffset = wdest.offset() + unitNormal_*2;
  // this is the destination field value - always on the boundary
  const MemoryWindow wd( wdest.glob_dim(),
                        destBaseOffset,
                        destExtent,
                        wdest.has_bc(0), wdest.has_bc(1), wdest.has_bc(2) );
  
  PhiFaceT    d( wd, &dest[0], ExternalStorage  );
  PhiFaceT aVel( wd, &((*advectiveVelocity_)[0]), ExternalStorage );
  
  typename PhiFaceT::iterator      id   = d.begin();
  typename PhiFaceT::iterator      ide  = d.end();
  typename PhiFaceT::iterator      iav  = aVel.begin();
  typename PhiVolT::const_iterator vfracmm = srcIters_[0];
  typename PhiVolT::const_iterator vfracpp = srcIters_[3];
  std::cout << "-------------------------------------------\n";
  for (; id != ide; ++id, ++iav, ++vfracmm, ++vfracpp) {
    if ( *iav > 0.0 && *vfracmm == 0.0 ) *id = 0.0;
    else if( *iav < 0.0 && *vfracpp == 0.0 ) *id = 0.0;
  }
}

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
void
FluxLimiterInterpolant<PhiVolT,PhiFaceT>::
build_src_iterators( const PhiVolT& src ) const {
  using namespace SpatialOps;
  using namespace SpatialOps::structured;

  srcIters_.clear();
  const MemoryWindow& wsrc = src.window_with_ghost();
  for (int i=0; i<4; i++) {
    // i = 0: minus-minus
    // i = 1: minus
    // i = 2: plus
    // i = 3: plus-plus
    //  below is a depiction of the nomenclature, cells and faces...
    //  | minus-minus |  minus  || plus  | plus-plus
    const MemoryWindow srcwin( wsrc.glob_dim(),
                               wsrc.offset() + unitNormal_*i,
                               wsrc.extent() - unitNormal_*3,
                               wsrc.has_bc(0), wsrc.has_bc(1), wsrc.has_bc(2) );
    PhiVolT field(srcwin,src);
    srcIters_.push_back(field.begin());
  }
}


//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
void
FluxLimiterInterpolant<PhiVolT,PhiFaceT>::
apply_to_field( const PhiVolT &src, PhiFaceT &dest ) const
{
  // This will calculate the flux limiter function psi. The convective flux is
  // written as: phi_face = phi_lo - psi*(phi_lo - phi_hi) where
  // phi_lo is a low order interpolant (i.e. Upwind)
  // phi_hi is a high order interpolant (i.e. central).
  /* Algorithm: TSAAD - TODO - DESCRIBE ALGORITHM IN DETAIL
   * Loop over faces
   */
  using namespace SpatialOps;
  using namespace SpatialOps::structured;

  const MemoryWindow& wsrc  = src.window_with_ghost();
  const MemoryWindow& wdest = dest.window_with_ghost(); // used for velocity & interpolated phi
  
  int pm[2]={1,-1}; // plus or minus face
  int zo[2]={0,1};  // zero and one

  IntVec extent = wsrc.extent() - unitNormal_*wsrc.glob_dim() + unitNormal_;
  IntVec destExtent = wdest.extent() - unitNormal_*wdest.glob_dim() + unitNormal_;
  IntVec baseOffset;
  IntVec destBaseOffset;
  for (int direc=0; direc<2; direc++) {
    baseOffset = wsrc.offset() + (unitNormal_*wsrc.glob_dim() - unitNormal_ )* zo[direc];
    destBaseOffset = wdest.offset() + (unitNormal_*wdest.glob_dim() - unitNormal_ )* zo[direc] + unitNormal_*(1-zo[direc]);
    
    // this is the destination field value - always on the boundary
    const MemoryWindow wd( wdest.glob_dim(),
                          destBaseOffset,
                          destExtent,
                          wdest.has_bc(0), wdest.has_bc(1), wdest.has_bc(2) );
    
    // ghost cell: on a minus face, this is src-minus. on a plus face, this is src-plus
    const MemoryWindow ws1( wsrc.glob_dim(),
                           baseOffset,
                           extent,
                           wsrc.has_bc(0), wsrc.has_bc(1), wsrc.has_bc(2) );
    
    // first interior cell: on a minus face, this is src-plus. on a plus face this is src-minus
    const MemoryWindow ws2( wsrc.glob_dim(),
                           baseOffset + unitNormal_ * pm[direc],
                           extent,
                           wsrc.has_bc(0), wsrc.has_bc(1), wsrc.has_bc(2) );
        
    // second interior cell: on a minus face, this is src-plus-plus. on a plus face, this is src-minus-minus
    const MemoryWindow ws3( wsrc.glob_dim(),
                           baseOffset  + unitNormal_ * pm[direc] * 2,
                           extent,
                           wsrc.has_bc(0), wsrc.has_bc(1), wsrc.has_bc(2) );
    
    
    
    PhiFaceT    d( wd, &dest[0], ExternalStorage );
    PhiFaceT aVel( wd, &((*advectiveVelocity_)[0]), ExternalStorage );
    PhiVolT    s1( ws1, &src[0], ExternalStorage );
    PhiVolT    s2( ws2, &src[0], ExternalStorage );
    PhiVolT    s3( ws3, &src[0], ExternalStorage );
    
    
    typename PhiFaceT::iterator      id  = d .begin();
    typename PhiFaceT::iterator      ide = d .end();
    typename PhiFaceT::iterator      iav = aVel.begin();
    typename PhiVolT::const_iterator is1 = s1.begin();
    typename PhiVolT::const_iterator is2 = s2.begin();
    typename PhiVolT::const_iterator is3 = s3.begin();
    const bool isBoundaryFace = (hasMinusBoundary_ && direc==0) || (hasPlusBoundary_ && direc==1);
    for (; id != ide; ++id, ++iav, ++is1, ++is2, ++is3) {
      const double flowDir = -pm[direc] * *iav;
      if     ( flowDir > 0.0 ) { // flow is coming out of the patch. use limiter
        // calculate flux limiter function value
        const double r = (*is3 - *is2)/(*is2 - *is1);
        *id = calculate_flux_limiter_function(r, limiterType_);
      }
      else if( flowDir < 0.0 ) *id = ( isBoundaryFace ) ? 1.0 : 0.0; // flow is coming into the patch. use central differencing if we are at a physical boundary.
      else                     *id = 1.0;
    }
  }

  // now do interior
  build_src_iterators(src);
  destExtent = wdest.extent() - unitNormal_*3 - wdest.has_bc()*unitNormal_;
  destBaseOffset = wdest.offset() + unitNormal_*2;
  // this is the destination field value - always on the boundary
  const MemoryWindow wd( wdest.glob_dim(),
                        destBaseOffset,
                        destExtent,
                        wdest.has_bc(0), wdest.has_bc(1), wdest.has_bc(2) );

  PhiFaceT    d( wd, &dest[0], ExternalStorage  );
  PhiFaceT aVel( wd, &((*advectiveVelocity_)[0]), ExternalStorage );
  
  typename PhiFaceT::iterator      id   = d.begin();
  typename PhiFaceT::iterator      ide  = d.end();
  typename PhiFaceT::iterator      iav  = aVel.begin();
  typename PhiVolT::const_iterator ismm = srcIters_[0];
  typename PhiVolT::const_iterator ism  = srcIters_[1];
  typename PhiVolT::const_iterator isp  = srcIters_[2];
  typename PhiVolT::const_iterator ispp = srcIters_[3];
  
  for (; id != ide; ++id, ++iav, ++ismm, ++ism, ++isp, ++ispp) {    
    if     ( *iav > 0.0 ) {
      const double r = (*ism - *ismm)/(*isp - *ism);
      *id = calculate_flux_limiter_function(r, limiterType_);
    }
    else if( *iav < 0.0 ) {
      const double r = (*ispp - *isp)/(*isp - *ism);
      *id = calculate_flux_limiter_function(r, limiterType_);
    }
    else                     *id = 1.0;
  }
}

//==================================================================
// Explicit template instantiation
namespace SS = SpatialOps::structured;

template class FluxLimiterInterpolant< SS::SVolField, SS::SSurfXField >;
template class FluxLimiterInterpolant< SS::SVolField, SS::SSurfYField >;
template class FluxLimiterInterpolant< SS::SVolField, SS::SSurfZField >;

template class FluxLimiterInterpolant< SS::XVolField, SS::XSurfXField >;
template class FluxLimiterInterpolant< SS::XVolField, SS::XSurfYField >;
template class FluxLimiterInterpolant< SS::XVolField, SS::XSurfZField >;

template class FluxLimiterInterpolant< SS::YVolField, SS::YSurfXField >;
template class FluxLimiterInterpolant< SS::YVolField, SS::YSurfYField >;
template class FluxLimiterInterpolant< SS::YVolField, SS::YSurfZField >;

template class FluxLimiterInterpolant< SS::ZVolField, SS::ZSurfXField >;
template class FluxLimiterInterpolant< SS::ZVolField, SS::ZSurfYField >;
template class FluxLimiterInterpolant< SS::ZVolField, SS::ZSurfZField >;
//==================================================================
