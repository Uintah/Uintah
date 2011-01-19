#include "SuperbeeInterpolant.h"
#include "OperatorTypes.h"

#include <cmath>
#include "spatialops/SpatialOpsDefs.h"
#include "spatialops/structured/FVTools.h"
#include "spatialops/structured/FVStaggeredIndexHelper.h"

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
SuperbeeInterpolant<PhiVolT,PhiFaceT>::
SuperbeeInterpolant( const std::vector<int>& dim,
                  const std::vector<bool> hasPlusFace )
{
  const SpatialOps::structured::IndexHelper<PhiVolT,PhiFaceT> indexHelper( dim, hasPlusFace[0], hasPlusFace[1], hasPlusFace[2] );
  stride_ = indexHelper.calculate_stride();
  
  xyzCount_.resize(3);
  xyzVolIncr_.resize(3);
  xyzFaceIncr_.resize(3);
  
  xyzCountBnd_.resize(3);
  xyzVolIncrBnd_.resize(3);
  xyzFaceIncrBnd_.resize(3);

  int nGhost = 2*SrcGhost::NGHOST;
  for (int i=0;i<=2;i++) {
    xyzFaceIncr_[i] = 0;
    xyzVolIncr_[i] = 0;
    xyzCount_[i] = dim[i] + nGhost;
    //
    xyzCountBnd_[i] = dim[i] + nGhost;
    xyzVolIncrBnd_[i] = dim[i] + nGhost;
    xyzFaceIncrBnd_[i] = dim[i] + nGhost;
  }
  
  const size_t direction = PhiFaceT::Location::FaceDir::value;
  switch (direction) {
      
    case SpatialOps::XDIR::value:
      xyzFaceIncr_[1] = 3;
      xyzFaceIncr_[2] = 0;
      xyzVolIncr_[1] = 3;
      xyzVolIncr_[2] = 0;
      if (hasPlusFace[0]) xyzFaceIncr_[1] += 1;
      xyzCount_[0] = dim[0]-1;
      
      // boundary counters
      //bndPlusStrideCoef_ = dim[0];
      bndPlusStrideCoef_ = dim[0] + nGhost - 1;
      
      xyzCountBnd_[0] = 1;
      xyzCountBnd_[1] = dim[1] + nGhost;
      xyzCountBnd_[2] = dim[2] + nGhost; 
      
      xyzVolIncrBnd_[0] = 0;
      xyzVolIncrBnd_[1] = dim[0] + nGhost;
      xyzVolIncrBnd_[2] = 0;
      
      xyzFaceIncrBnd_[0] = 0;
      xyzFaceIncrBnd_[1] = dim[0] + nGhost;
      xyzFaceIncrBnd_[2] = 0;
      
      if (hasPlusFace[0]) xyzFaceIncrBnd_[1] += 1;      
      break;
      
    case SpatialOps::YDIR::value:
      xyzFaceIncr_[1] = 0;
      xyzFaceIncr_[2] = 3*stride_;
      xyzVolIncr_[1] = 0;
      xyzVolIncr_[2] = 3*stride_;      
      if (hasPlusFace[1]) xyzFaceIncr_[2] += stride_;
      xyzCount_[1] = dim[1] -1;
      
      // boundary counters
      bndPlusStrideCoef_ = dim[1] + nGhost -1;
      
      xyzCountBnd_[0] = dim[0] + nGhost;
      xyzCountBnd_[1] = 1;
      xyzCountBnd_[2] = dim[2] + nGhost;
      
      xyzVolIncrBnd_[0] = 1;
      xyzVolIncrBnd_[1] = 0;
      xyzVolIncrBnd_[2] = (dim[0]+nGhost)*(dim[1]+nGhost -1);
      
      xyzFaceIncrBnd_[0] = 1;
      xyzFaceIncrBnd_[1] = 0;
      xyzFaceIncrBnd_[2] = (dim[0]+nGhost)*(dim[1]+nGhost -1);
      
      if (hasPlusFace[1]) xyzFaceIncrBnd_[2] += stride_;      
      break;
      
    case SpatialOps::ZDIR::value:
      // NOTE: for the z direction, xyzVolIncr & xyzFaceIncr are all zero.
      // no need to set them here as they are initialized to zero previously.
      xyzFaceIncr_[1] = 0;
      xyzFaceIncr_[2] = 0;
      xyzVolIncr_[1] = 0;
      xyzVolIncr_[2] = 0;
      if (hasPlusFace[2]) xyzFaceIncr_[2] += stride_;
      xyzCount_[0] = dim[0] + nGhost;
      //xyzCount_[2] = 1;
      xyzCount_[2] -= 1;
            
      // boundary counters
      bndPlusStrideCoef_ = dim[2] + nGhost - 1;
      
      xyzCountBnd_[0] = dim[0] + nGhost;
      xyzCountBnd_[1] = dim[1] + nGhost;
      xyzCountBnd_[2] = 1;
      
      xyzVolIncrBnd_[0] = 1;
      xyzVolIncrBnd_[1] = 0;
      xyzVolIncrBnd_[2] = 0;
      
      xyzFaceIncrBnd_[0] = 1;
      xyzFaceIncrBnd_[1] = 0;
      xyzFaceIncrBnd_[2] = 0;
      
      if (hasPlusFace[2]) xyzFaceIncrBnd_[2] += stride_;      
      break;
  }  
}

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
void
SuperbeeInterpolant<PhiVolT,PhiFaceT>::
set_advective_velocity( const PhiFaceT &theAdvectiveVelocity )
{
  // !!! NOT THREAD SAFE !!! USE LOCK
  advectiveVelocity_ = &theAdvectiveVelocity;  
}

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
SuperbeeInterpolant<PhiVolT,PhiFaceT>::
~SuperbeeInterpolant()
{}  

//--------------------------------------------------------------------

template< typename PhiVolT, typename PhiFaceT >
void 
SuperbeeInterpolant<PhiVolT,PhiFaceT>::
apply_to_field( const PhiVolT &src, PhiFaceT &dest ) const
{   
  /* Algorithm: TSAAD - TODO - DESCRIBE ALGORITHM IN DETAIL
   * Loop over faces
   */
  //
  double psi = 0.0; // this is the limiter function
  double r = 0.0;   // this is the ratio of successive gradients
  
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
  for (size_t k=1; k<=xyzCountBnd_[2]; k++) { // count zCount times
    
    for (size_t j=1; j<=xyzCountBnd_[1]; j++) { // count yCount times
      
      for (size_t i =1; i<=xyzCountBnd_[0]; i++) { // count xCount times
        
        if ((*advVel) > 0.0) {
          // for a minus face, if the velocity if positive, then use central approximation
          *destFld = 0.5*(*srcFieldPlus + *srcFieldMinus);
          //*destFld = *srcFieldMinus;
        }
        
        else if ((*advVel) < 0.0) { 
          // calculate the gradient between successive cells
          r = (*srcFieldPlusPlus - *srcFieldPlus)/(*srcFieldPlus - *srcFieldMinus);
          psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
          psi = std::max( 0.0, psi );
          *destFld = *srcFieldPlus + 0.5*psi*(*srcFieldMinus - *srcFieldPlus);
          //*destFld = *srcFieldPlus;
        }
        
        else *destFld = 0.0; // may need a better condition here to account for
        // a tolerance value for example.
                
        srcFieldMinus += xyzVolIncrBnd_[0];
        srcFieldPlus += xyzVolIncrBnd_[0];
        srcFieldPlusPlus += xyzVolIncrBnd_[0];
        destFld += xyzFaceIncrBnd_[0];
        advVel += xyzFaceIncrBnd_[0];
      }
      
      srcFieldMinus += xyzVolIncrBnd_[1];
      srcFieldPlus  += xyzVolIncrBnd_[1];
      srcFieldPlusPlus += xyzVolIncrBnd_[1];
      destFld += xyzFaceIncrBnd_[1];   
      advVel  += xyzFaceIncrBnd_[1];
    }
    
    srcFieldMinus += xyzVolIncrBnd_[2];
    srcFieldPlus  += xyzVolIncrBnd_[2];
    srcFieldPlusPlus += xyzVolIncrBnd_[2];
    destFld += xyzFaceIncrBnd_[2];    
    advVel  += xyzFaceIncrBnd_[2];
  }
  
  //
  // now for the plus side (i.e. x+, y+, z+).
  destFld = dest.begin() + bndPlusStrideCoef_*stride_;
  advVel = advectiveVelocity_->begin() + bndPlusStrideCoef_*stride_;
  srcFieldMinus = src.begin() + (bndPlusStrideCoef_-1)*stride_;
  // Source field on the minus, minus side of a face
  typename PhiVolT::const_iterator srcFieldMinusMinus = src.begin() + (bndPlusStrideCoef_ - 2)*stride_;
  srcFieldPlus = src.begin() + bndPlusStrideCoef_*stride_;
  
  for (size_t k=1; k<=xyzCountBnd_[2]; k++) { // count zCount times
    
    for (size_t j=1; j<=xyzCountBnd_[1]; j++) { // count yCount times
      
      for (size_t i =1; i<=xyzCountBnd_[0]; i++) { // count xCount times
        
        if ((*advVel) < 0.0) {
          // for a minus face, if the velocity if positive, then use central approximation
          *destFld = 0.5*(*srcFieldPlus + *srcFieldMinus);
          //*destFld = *srcFieldPlus;
        }
        
        else if ((*advVel) > 0.0) { 
          // calculate the gradient between successive cells
          r = (*srcFieldMinus - *srcFieldMinusMinus)/(*srcFieldPlus - *srcFieldMinus);
          psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
          psi = std::max( 0.0, psi );
          *destFld = *srcFieldMinus + 0.5*psi*(*srcFieldPlus - *srcFieldMinus);
          //*destFld = *srcFieldMinus;
        }
        
        else *destFld = 0.0; // may need a better condition here to account for
        // a tolerance value for example.
        
        srcFieldMinus += xyzVolIncrBnd_[0];
        srcFieldPlus += xyzVolIncrBnd_[0];
        srcFieldMinusMinus += xyzVolIncrBnd_[0];
        destFld += xyzFaceIncrBnd_[0];
        advVel += xyzFaceIncrBnd_[0];
      }
      
      srcFieldMinus += xyzVolIncrBnd_[1];
      srcFieldPlus  += xyzVolIncrBnd_[1];
      srcFieldMinusMinus += xyzVolIncrBnd_[1];
      destFld += xyzFaceIncrBnd_[1];   
      advVel  += xyzFaceIncrBnd_[1];
    }
    
    srcFieldMinus += xyzVolIncrBnd_[2];
    srcFieldPlus  += xyzVolIncrBnd_[2];
    srcFieldMinusMinus += xyzVolIncrBnd_[2];
    destFld += xyzFaceIncrBnd_[2];    
    advVel  += xyzFaceIncrBnd_[2];
  }
  
  //
  // now for the internal faces
  srcFieldMinus = src.begin() + stride_;
  srcFieldMinusMinus = src.begin();
  srcFieldPlus = src.begin() + stride_ + stride_;
  srcFieldPlusPlus = src.begin() + stride_ + stride_ + stride_;
  advVel = advectiveVelocity_->begin() + stride_ + stride_;
  destFld = dest.begin() + stride_ + stride_;
  
  for (size_t k=1; k<=xyzCount_[2]; k++) { // count zCount times
    
    for (size_t j=1; j<=xyzCount_[1]; j++) { // count yCount times
      
      for (size_t i=1; i<=xyzCount_[0]; i++) { // count xCount times
        
        if ((*advVel) > 0.0) {
          // calculate the gradient between successive cells
          r = (*srcFieldMinus - *srcFieldMinusMinus)/(*srcFieldPlus - *srcFieldMinus);
          psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
          psi = std::max( 0.0, psi );
          *destFld = *srcFieldMinus + 0.5*psi*(*srcFieldPlus - *srcFieldMinus);
        }
        
        else if ((*advVel) < 0.0) { 
          // calculate the gradient between successive cells
          r = (*srcFieldPlusPlus - *srcFieldPlus)/(*srcFieldPlus - *srcFieldMinus);
          psi = std::max( std::min(2.0*r, 1.0), std::min(r, 2.0) );
          psi = std::max( 0.0, psi );
          *destFld = *srcFieldPlus + 0.5*psi*(*srcFieldMinus - *srcFieldPlus);
        }
        
        else *destFld = 0.0; // may need a better condition here to account for
        // a tolerance value for example.
        
        ++srcFieldMinus;
        ++srcFieldMinusMinus;
        ++srcFieldPlus;
        ++srcFieldPlusPlus;
        ++destFld;
        ++advVel;
      }
      
      srcFieldMinus += xyzVolIncr_[1];
      srcFieldMinusMinus += xyzVolIncr_[1];
      srcFieldPlus  += xyzVolIncr_[1];
      srcFieldPlusPlus += xyzVolIncr_[1];
      destFld += xyzFaceIncr_[1];   
      advVel  += xyzFaceIncr_[1];
    }
    
    srcFieldMinus += xyzVolIncr_[2];
    srcFieldMinusMinus += xyzVolIncr_[2];
    srcFieldPlus  += xyzVolIncr_[2];
    srcFieldPlusPlus += xyzVolIncr_[2];
    destFld += xyzFaceIncr_[2];    
    advVel  += xyzFaceIncr_[2];
  }
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
namespace SS = SpatialOps::structured;

template class SuperbeeInterpolant< SS::SVolField, SS::SSurfXField >;
template class SuperbeeInterpolant< SS::SVolField, SS::SSurfYField >;
template class SuperbeeInterpolant< SS::SVolField, SS::SSurfZField >;

template class SuperbeeInterpolant< SS::XVolField, SS::XSurfXField >;
template class SuperbeeInterpolant< SS::XVolField, SS::XSurfYField >;
template class SuperbeeInterpolant< SS::XVolField, SS::XSurfZField >;

template class SuperbeeInterpolant< SS::YVolField, SS::YSurfXField >;
template class SuperbeeInterpolant< SS::YVolField, SS::YSurfYField >;
template class SuperbeeInterpolant< SS::YVolField, SS::YSurfZField >;

template class SuperbeeInterpolant< SS::ZVolField, SS::ZSurfXField >;
template class SuperbeeInterpolant< SS::ZVolField, SS::ZSurfYField >;
template class SuperbeeInterpolant< SS::ZVolField, SS::ZSurfZField >;
//==================================================================