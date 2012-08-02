/*
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
                       const std::vector<bool> hasMinusBoundary):
hasPlusBoundary_(false), hasMinusBoundary_(false)
{
  stride_ = calculate_stride(dim, hasPlusFace);
  
  faceCount_.resize(3);
  volIncr_.resize(3);
  faceIncr_.resize(3);
  
  bndFaceCount_.resize(3);
  bndVolIncr_.resize(3);
  bndFaceIncr_.resize(3);
  
  const int nGhost = 2*Wasatch::get_n_ghost<PhiVolT>();
  for (int i=0;i<=2;i++) {
    faceIncr_[i] = 0;
    volIncr_[i] = 0;
    faceCount_[i] = dim[i] + nGhost;
    //
    bndFaceCount_[i] = dim[i] + nGhost;
    bndVolIncr_[i] = dim[i] + nGhost;
    bndFaceIncr_[i] = dim[i] + nGhost;
  }
  
  const size_t direction = PhiFaceT::Location::FaceDir::value;
  switch (direction) {
      
    case SpatialOps::XDIR::value:
      faceIncr_[1] = 3;
      faceIncr_[2] = 0;
      volIncr_[1] = 3;
      volIncr_[2] = 0;
      if (hasPlusFace[0]) faceIncr_[1] += 1;    
      faceCount_[0] = dim[0] - 1;
      
      // boundary counters
      bndPlusStrideCoef_ = dim[0] + nGhost - 1;
      
      bndFaceCount_[0] = 1;
      bndFaceCount_[1] = dim[1] + nGhost;
      bndFaceCount_[2] = dim[2] + nGhost;
      
      bndVolIncr_[0] = 0;
      bndVolIncr_[1] = dim[0] + nGhost;
      bndVolIncr_[2] = 0;
      
      bndFaceIncr_[0] = 0;
      bndFaceIncr_[1] = dim[0] + nGhost;
      bndFaceIncr_[2] = 0;
      
      if (hasPlusFace[0]) bndFaceIncr_[1] += 1;
      if (hasPlusFace[0]) hasPlusBoundary_ = true;
      if (hasMinusBoundary[0]) hasMinusBoundary_ = true;
      break;
      
    case SpatialOps::YDIR::value:
      faceIncr_[1] = 0;
      faceIncr_[2] = 3*stride_;
      volIncr_[1] = 0;
      volIncr_[2] = 3*stride_;
      if (hasPlusFace[1]) faceIncr_[2] += stride_;
      faceCount_[1] = dim[1] -1;
      
      // boundary counters
      bndPlusStrideCoef_ = dim[1] + nGhost -1;
      
      bndFaceCount_[0] = dim[0] + nGhost;
      bndFaceCount_[1] = 1;
      bndFaceCount_[2] = dim[2] + nGhost;
      
      bndVolIncr_[0] = 1;
      bndVolIncr_[1] = 0;
      bndVolIncr_[2] = (dim[0]+nGhost)*(dim[1]+nGhost -1);
      
      bndFaceIncr_[0] = 1;
      bndFaceIncr_[1] = 0;
      bndFaceIncr_[2] = (dim[0]+nGhost)*(dim[1]+nGhost -1);
      
      if (hasPlusFace[1]) bndFaceIncr_[2] += stride_;
      if (hasPlusFace[1]) hasPlusBoundary_ = true;
      if (hasMinusBoundary[1]) hasMinusBoundary_ = true;
      break;
      
    case SpatialOps::ZDIR::value:
      // NOTE: for the z direction, xyzVolIncr & xyzFaceIncr are all zero.
      // no need to set them here as they are initialized to zero previously.
      faceIncr_[1] = 0;
      faceIncr_[2] = 0;
      volIncr_[1] = 0;
      volIncr_[2] = 0;
      //if (hasPlusFace[2]) faceIncr_[2] += stride_;
      //faceCount_[0] = dim[0] + nGhost;
      faceCount_[2] = dim[2] -1;
      
      // boundary counters
      bndPlusStrideCoef_ = dim[2] + nGhost - 1;
      
      bndFaceCount_[0] = dim[0] + nGhost;
      bndFaceCount_[1] = dim[1] + nGhost;
      bndFaceCount_[2] = 1;
      
      bndVolIncr_[0] = 1;
      bndVolIncr_[1] = 0;
      bndVolIncr_[2] = 0;
      
      bndFaceIncr_[0] = 1;
      bndFaceIncr_[1] = 0;
      bndFaceIncr_[2] = 0;
      
      if (hasPlusFace[2]) hasPlusBoundary_ = true;
      if (hasMinusBoundary[2]) hasMinusBoundary_ = true;
      //if (hasPlusFace[2]) bndFaceIncr_[2] += stride_;
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
  
  // Source field on the minus side of a face
  typename PhiVolT::const_iterator vFracMinus = src.begin() + stride_;
  typename PhiVolT::const_iterator vFracMinusMinus = src.begin();
  // Source field on the plus side of a face
  typename PhiVolT::const_iterator vFracPlus = src.begin() + 2 * stride_;
  // Source field on the plus, plus side of a face
  typename PhiVolT::const_iterator vFracPlusPlus = src.begin() + 3*stride_;
  // here the destination field is the flux limiting function
  typename PhiFaceT::iterator      destFld       = dest.begin() + 2*stride_;
  typename PhiFaceT::const_iterator advVel       = advectiveVelocity_->begin() + 2*stride_;
  
  for (size_t k=1; k<=faceCount_[2]; k++) { // count zCount times
    
    for (size_t j=1; j<=faceCount_[1]; j++) { // count yCount times
      
      for (size_t i=1; i<=faceCount_[0]; i++) { // count xCount times
        
        if ((*advVel) > 0.0) {
          if ( *vFracMinusMinus == 0 ) *destFld = 0.0;
        }
        
        else if ((*advVel) < 0.0) {
          if ( *vFracPlusPlus == 0 ) *destFld = 0.0;
        }
        
        ++vFracMinus;
        ++vFracMinusMinus;
        ++vFracPlus;
        ++vFracPlusPlus;
        ++destFld;
        ++advVel;
      }
      
      vFracMinus += volIncr_[1];
      vFracMinusMinus += volIncr_[1];
      vFracPlus  += volIncr_[1];
      vFracPlusPlus += volIncr_[1];
      destFld += faceIncr_[1];
      advVel  += faceIncr_[1];
    }
    
    vFracMinus += volIncr_[2];
    vFracMinusMinus += volIncr_[2];
    vFracPlus  += volIncr_[2];
    vFracPlusPlus += volIncr_[2];
    destFld += faceIncr_[2];
    advVel  += faceIncr_[2];
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
  //
  // Source field on the minus side of a face
  typename PhiVolT::const_iterator srcFieldMinus = src.begin();
  // Source field on the plus side of a face
  typename PhiVolT::const_iterator srcFieldPlus = src.begin() + stride_;
  // Source field on the plus, plus side of a face
  typename PhiVolT::const_iterator srcFieldPlusPlus = src.begin() + stride_ + stride_;
  // Destination field (face). Starts on the first face for that particular field
  typename PhiFaceT::iterator destFld = dest.begin() + stride_;
  // Whether it is x, y, or z face field. So its
  // face index will start at zero, a face for which we cannot compute the flux.
  // So we add stride to it. In x direction, it will be face 1.
  // In y direction, it will be nx. In z direction, it will be nx*ny
  typename PhiFaceT::const_iterator advVel = advectiveVelocity_->begin() + stride_;
  
  // first, treat boundary faces - start with minus side (i.e. x-, y-, z-).
  for (size_t k=1; k<=bndFaceCount_[2]; k++) { // count zCount times
    
    for (size_t j=1; j<=bndFaceCount_[1]; j++) { // count yCount times
      
      for (size_t i =1; i<=bndFaceCount_[0]; i++) { // count xCount times
        
        if ((*advVel) > 0.0) {
          // for a minus face, if there is a physical boundary present with an 
          // inlet (u_n > 0), then use central (psi = 1). Otherwise, if it is
          // a periodic boundary then use upwind (psi = 0.0).
          *destFld = hasMinusBoundary_ ? 1.0 : 0.0;
        }
        
        else if ((*advVel) < 0.0) {
          // calculate the ratio of gradients between successive cells
          const double r = (*srcFieldPlusPlus - *srcFieldPlus)/(*srcFieldPlus - *srcFieldMinus);
          *destFld = calculate_flux_limiter_function(r, limiterType_);
        }
        
        else *destFld = 1.0; // default to central when velocity = 0.0
        
        srcFieldMinus += bndVolIncr_[0];
        srcFieldPlus += bndVolIncr_[0];
        srcFieldPlusPlus += bndVolIncr_[0];
        destFld += bndFaceIncr_[0];
        advVel += bndFaceIncr_[0];
      }
      
      srcFieldMinus += bndVolIncr_[1];
      srcFieldPlus  += bndVolIncr_[1];
      srcFieldPlusPlus += bndVolIncr_[1];
      destFld += bndFaceIncr_[1];
      advVel  += bndFaceIncr_[1];
    }
    
    srcFieldMinus += bndVolIncr_[2];
    srcFieldPlus  += bndVolIncr_[2];
    srcFieldPlusPlus += bndVolIncr_[2];
    destFld += bndFaceIncr_[2];
    advVel  += bndFaceIncr_[2];
  }
  
  //
  // now for the plus side (i.e. x+, y+, z+).
  destFld       = dest.begin() + bndPlusStrideCoef_*stride_;
  srcFieldMinus = src.begin() + (bndPlusStrideCoef_-1)*stride_;
  advVel  = advectiveVelocity_->begin() + bndPlusStrideCoef_*stride_;
  // Source field on the minus, minus side of a face
  typename PhiVolT::const_iterator srcFieldMinusMinus = src.begin() + (bndPlusStrideCoef_ - 2)*stride_;
  srcFieldPlus = src.begin() + bndPlusStrideCoef_*stride_;
  
  for (size_t k=1; k<=bndFaceCount_[2]; k++) { // count zCount times
    
    for (size_t j=1; j<=bndFaceCount_[1]; j++) { // count yCount times
      
      for (size_t i =1; i<=bndFaceCount_[0]; i++) { // count xCount times
        
        if ((*advVel) < 0.0) {
          // for a plus face, if there is a physical boundary present along with
          // an inlet (u.n < 0), then use central (psi = 1.0). Otherwise,
          // use upwind (psi = 0).
          *destFld = hasPlusBoundary_ ? 1.0 : 0.0;
        }
        
        else if ((*advVel) > 0.0) {
          // calculate the gradient between successive cells
          const double r = (*srcFieldMinus - *srcFieldMinusMinus)/(*srcFieldPlus - *srcFieldMinus);
          *destFld = calculate_flux_limiter_function(r, limiterType_);          
        }
        
        else *destFld = 1.0; // default to central when velocity = 0.0
        
        srcFieldMinus += bndVolIncr_[0];
        srcFieldPlus += bndVolIncr_[0];
        srcFieldMinusMinus += bndVolIncr_[0];
        destFld += bndFaceIncr_[0];
        advVel += bndFaceIncr_[0];
      }
      
      srcFieldMinus += bndVolIncr_[1];
      srcFieldPlus  += bndVolIncr_[1];
      srcFieldMinusMinus += bndVolIncr_[1];
      destFld += bndFaceIncr_[1];
      advVel  += bndFaceIncr_[1];
    }
    
    srcFieldMinus += bndVolIncr_[2];
    srcFieldPlus  += bndVolIncr_[2];
    srcFieldMinusMinus += bndVolIncr_[2];
    destFld += bndFaceIncr_[2];
    advVel  += bndFaceIncr_[2];
  }
  
  //
  // now for the internal faces
  srcFieldMinus      = src.begin() + stride_;
  srcFieldMinusMinus = src.begin();
  srcFieldPlus       = src.begin() + stride_ + stride_;
  srcFieldPlusPlus   = src.begin() + stride_ + stride_ + stride_;
  destFld            = dest.begin() + stride_ + stride_;
  advVel = advectiveVelocity_->begin() + stride_ + stride_;
  for (size_t k=1; k<=faceCount_[2]; k++) { // count zCount times
    
    for (size_t j=1; j<=faceCount_[1]; j++) { // count yCount times
      
      for (size_t i=1; i<=faceCount_[0]; i++) { // count xCount times
        
        if ((*advVel) > 0.0) {
          // calculate the ratio of gradient between successive cells
          const double r = (*srcFieldMinus - *srcFieldMinusMinus)/(*srcFieldPlus - *srcFieldMinus);
          *destFld = calculate_flux_limiter_function(r, limiterType_);
        }
        
        else if ((*advVel) < 0.0) {
          // calculate the ratio of gradients between successive cells
          const double r = (*srcFieldPlusPlus - *srcFieldPlus)/(*srcFieldPlus - *srcFieldMinus);
          *destFld = calculate_flux_limiter_function(r, limiterType_);
        }
        
        else *destFld = 1.0; // default to central when velocity = 0.0
        
        ++srcFieldMinus;
        ++srcFieldMinusMinus;
        ++srcFieldPlus;
        ++srcFieldPlusPlus;
        ++destFld;
        ++advVel;
      }
      
      srcFieldMinus += volIncr_[1];
      srcFieldMinusMinus += volIncr_[1];
      srcFieldPlus  += volIncr_[1];
      srcFieldPlusPlus += volIncr_[1];
      destFld += faceIncr_[1];
      advVel  += faceIncr_[1];
    }
    
    srcFieldMinus += volIncr_[2];
    srcFieldMinusMinus += volIncr_[2];
    srcFieldPlus  += volIncr_[2];
    srcFieldPlusPlus += volIncr_[2];
    destFld += faceIncr_[2];
    advVel  += faceIncr_[2];
  }
  
}

//--------------------------------------------------------------------

template<typename PhiVolT, typename PhiFaceT>
int
FluxLimiterInterpolant<PhiVolT,PhiFaceT>::
calculate_stride(const std::vector<int>& dim,
                 const std::vector<bool> hasPlusFace) const
{
  const size_t direction = PhiFaceT::Location::FaceDir::value;
  int n = 0;
  switch (direction) {
    case SpatialOps::XDIR::value:
      n=1;
      break;
    case SpatialOps::YDIR::value:
      n = SpatialOps::structured::get_nx_with_ghost<PhiVolT>(dim[0],hasPlusFace[0]);
      break;
    case SpatialOps::ZDIR::value:
      n = SpatialOps::structured::get_nx_with_ghost<PhiVolT>(dim[0],hasPlusFace[0]) * SpatialOps::structured::get_ny_with_ghost<PhiVolT>(dim[1],hasPlusFace[1]);
      break;
  }
  return n;
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
