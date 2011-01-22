#include "UpwindInterpolant.h"
#include "OperatorTypes.h"

#include <cmath>
#include "spatialops/SpatialOpsDefs.h"
#include "spatialops/structured/FVTools.h"
#include "spatialops/structured/FVStaggeredIndexHelper.h"

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
UpwindInterpolant<PhiVolT,PhiFaceT>::
UpwindInterpolant( const std::vector<int>& dim,
                   const std::vector<bool> hasPlusFace )
{
  const SpatialOps::structured::IndexHelper<PhiVolT,PhiFaceT> indexHelper( dim, hasPlusFace[0], hasPlusFace[1], hasPlusFace[2] );
  stride_ = indexHelper.calculate_stride();
  faceCount_.resize(3);
  volIncr_.resize(3);
  faceIncr_.resize(3);  
  
  int nGhost = 2*SrcGhost::NGHOST;
  for (int i=0;i<=2;i++) {
    faceIncr_[i] = 0;
    volIncr_[i] = 0;
    faceCount_[i] = dim[i] + nGhost;
  }
  
  const size_t direction = PhiFaceT::Location::FaceDir::value;
  switch (direction) {
      
    case SpatialOps::XDIR::value:
      faceIncr_[1] = 1;
      volIncr_[1] = 1;
      if (hasPlusFace[0]) faceIncr_[1] += 1;
      faceCount_[0] -= 1;
      break;
      
    case SpatialOps::YDIR::value:
      faceIncr_[2] = stride_;
      volIncr_[2] = stride_;      
      if (hasPlusFace[1]) faceIncr_[2] += stride_;
      faceCount_[1] -= 1;
      break;
      
    case SpatialOps::ZDIR::value:
      // NOTE: for the z direction, xyzVolIncr & xyzFaceIncr are all zero.
      // no need to set them here as they are initialized to zero previously.
      if (hasPlusFace[2]) faceIncr_[2] += stride_;
      faceCount_[2] -= 1;
      break;
  }  
}

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
void
UpwindInterpolant<PhiVolT,PhiFaceT>::
set_advective_velocity( const PhiFaceT &theAdvectiveVelocity )
{
  // !!! NOT THREAD SAFE !!! USE LOCK
  advectiveVelocity_ = &theAdvectiveVelocity;  
}

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
UpwindInterpolant<PhiVolT,PhiFaceT>::
~UpwindInterpolant()
{}  

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
void 
UpwindInterpolant<PhiVolT,PhiFaceT>::
apply_to_field( const PhiVolT &src, PhiFaceT &dest ) const
{   
  /* Algorithm: TSAAD - TODO - DESCRIBE ALGORITHM IN DETAIL
   * Loop over faces
   */
  //
  // Source field on the minus side of a face
  typename PhiVolT::const_iterator srcFieldMinus = src.begin();
  // Source field on the plus side of a face
  typename PhiVolT::const_iterator srcFieldPlus = src.begin() + stride_; 
  // Destination field (face). Starts on the first face for that particular field
  typename PhiFaceT::iterator destFld = dest.begin() + stride_;
  // Whether it is x, y, or z face field. So its 
  // face index will start at zero, a face for which we cannot compute the flux. 
  // So we add stride to it. In x direction, it will be face 1. 
  // In y direction, it will be nx. In z direction, it will be nx*ny
  typename PhiFaceT::const_iterator advVel = advectiveVelocity_->begin() + stride_;

  for (size_t k=1; k<=faceCount_[2]; k++) { // count zCount times
    
    for (size_t j=1; j<=faceCount_[1]; j++) { // count yCount times
      
      for (size_t i =1; i<=faceCount_[0]; i++) { // count xCount times
        if ((*advVel) > 0.0) *destFld = *srcFieldMinus;
        else if ((*advVel) < 0.0) *destFld = *srcFieldPlus;
        else *destFld = 0.0; // may need a better condition here to account for
                             // a tolerance value for example.
        
        ++srcFieldMinus;
        ++srcFieldPlus;
        ++destFld;
        ++advVel;
      }
      
      srcFieldMinus += volIncr_[1];      
      srcFieldPlus  += volIncr_[1];
      destFld += faceIncr_[1];   
      advVel  += faceIncr_[1];
    }

    srcFieldMinus += volIncr_[2];
    srcFieldPlus  += volIncr_[2];
    destFld += faceIncr_[2];    
    advVel  += faceIncr_[2];
  }
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
namespace SS = SpatialOps::structured;

template class UpwindInterpolant< SS::SVolField, SS::SSurfXField >;
template class UpwindInterpolant< SS::SVolField, SS::SSurfYField >;
template class UpwindInterpolant< SS::SVolField, SS::SSurfZField >;

template class UpwindInterpolant< SS::XVolField, SS::XSurfXField >;
template class UpwindInterpolant< SS::XVolField, SS::XSurfYField >;
template class UpwindInterpolant< SS::XVolField, SS::XSurfZField >;

template class UpwindInterpolant< SS::YVolField, SS::YSurfXField >;
template class UpwindInterpolant< SS::YVolField, SS::YSurfYField >;
template class UpwindInterpolant< SS::YVolField, SS::YSurfZField >;

template class UpwindInterpolant< SS::ZVolField, SS::ZSurfXField >;
template class UpwindInterpolant< SS::ZVolField, SS::ZSurfYField >;
template class UpwindInterpolant< SS::ZVolField, SS::ZSurfZField >;
//==================================================================
