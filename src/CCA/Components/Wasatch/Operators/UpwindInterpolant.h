/* ----------------------------------------------------------------------------------------------
 %%%%%%  %%%%%%  %%%%%%%%%%    %%%%%%  %%%%%%  %%%%%% %%%%%%  %%%%    %%%%%%  %%%%%%%%%%    
   %%      %%      %%      %%    %%      %%      %%     %%      %%      %%      %%      %%  
   %%      %%      %%      %%    %%      %%      %%     %%      %%%%    %%      %%        %%
   %%      %%      %%      %%      %%    %%    %%       %%      %%  %%  %%      %%        %%
   %%      %%      %%%%%%%%        %%  %%  %%  %%       %%      %%  %%  %%      %%        %%
   %%      %%      %%              %%  %%  %%  %%       %%      %%    %%%%      %%        %%
   %%      %%      %%                %%      %%         %%      %%      %%      %%      %%  
     %%%%%%      %%%%%%              %%      %%       %%%%%%  %%%%%%    %%    %%%%%%%%%%    
 -------------------------------------------------------------------------------------------------*/
#include <iostream>
#include <cmath>
#include "spatialops/SpatialOpsDefs.h"
#include "spatialops/structured/FVTools.h"
#include "spatialops/structured/FVStaggeredIndexHelper.h"

/**
 *  \class     UpwindInterpolant
 *  \author    Tony Saad
 *  \date      July 2010
 *
 *  \brief     Calculates convective flux using upwind interpolation.
 *  
 *  This class is a lightweight operator, i.e. it does NOT implement a
 *  matvec operation. The UpwindInterplant will interpolate the
 *  convective flux \phi u_i where \phi denotes a staggered or
 *  non-staggered field. For example, if \phi denotes the temperature
 *  T, then, \phi is a scalar volume field. On the other hand, if \phi
 *  denotes the momentum \rho u_i, then, \phi is staggered volume
 *  field. The UpwindInterpolant will interpolate the volume field to
 *  its corresponding face field. Thus, if \phi denotes a scalar
 *  field, then the UpwindInterpolant will produce a scalar face
 *  field. Similarly, if \phi was an X-Staggered volume field, then
 *  the UpwindInterpolant will produce an X-Staggered face field.
 * 
 *  Based on this convention, the UpwindInterpolant class is templated
 *  on two types o fields: the phi field type (PhiVolT) and its
 *  corresponding face field type (PhiFaceT).
 */

template < typename PhiVolT, typename PhiFaceT >
class UpwindInterpolant {
  
private:
  
  // Here, the advectivevelocity has been already interpolated to the phi cell 
  // faces. The destination field should be of the same type as the advective 
  // velocity, i.e. a staggered, cell centered field.
  const PhiFaceT* advectiveVelocity_; 
  // An integer denoting the offset for the face index owned by the control 
  // volume in question. For the x direction, theStride = 0. 
  // For the y direction, stride_ = nx. For the z direction, stride_=nx*ny.
  size_t stride_; 
  // IndexHelper object - help in calculating stride
  SpatialOps::structured::IndexHelper<PhiVolT,PhiFaceT> indexHelper_;
  // some counters to help in the evaluate member function
  std::vector<size_t> xyzCount_;
  std::vector<size_t> xyzVolIncr_;
  std::vector<size_t> xyzFaceIncr_;
  
public:
  
  typedef typename PhiVolT::Ghost      SrcGhost;
  typedef typename PhiVolT::Location   SrcLocation;
  typedef typename PhiFaceT::Ghost     DestGhost;
  typedef typename PhiFaceT::Location  DestLocation;
  typedef PhiVolT SrcFieldType;
  typedef PhiFaceT DestFieldType;
  
  /**
   *  \brief Constructor for upwind interpolant.
   *  \param dim: A 3D vector containing the number of control volumes
   *         in each direction.
   *  \param hasPlusFace: Determines if this patch has a physical
   *         boundary on its plus side.
   */
  UpwindInterpolant( const std::vector<int>& dim,
                     const std::vector<bool> hasPlusFace );
  
  /**
   *  \brief Destructor for upwind interpolant.
   */  
  ~UpwindInterpolant();
  
  /**
   *  \brief Sets the advective velocity field for the Upwind interpolator.
   *
   *  \param theAdvectiveVelocity: A reference to the advective
   *         velocity field. This must be of the same type as the
   *         destination field, i.e. a face centered field.
   */
  void set_advective_velocity (const PhiFaceT &theAdvectiveVelocity);
  
  /**
   *  \brief Applies the Upwind interpolation to the source field.
   *
   *  \param src: A constant reference to the source field. In this
   *         case this is a scalar field usually denoted by phi.
   *
   *  \param dest: A reference to the destination field. This will
   *         hold the convective flux \phi*u_i in the direction i. It
   *         will be stored on the staggered cell centers.
   */
  void apply_to_field(const PhiVolT &src, PhiFaceT &dest) const; 
  
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// I M P L E M E N T A T I O N
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template <typename PhiVolT, typename PhiFaceT>
UpwindInterpolant<PhiVolT,PhiFaceT>::
UpwindInterpolant( const std::vector<int>& dim, const std::vector<bool> hasPlusFace )
  : indexHelper_(dim,hasPlusFace[0],hasPlusFace[1],hasPlusFace[2])
{
  
  stride_ = indexHelper_.calculate_stride();
  xyzCount_.resize(3);
  xyzVolIncr_.resize(3);
  xyzFaceIncr_.resize(3);  
  
  int nGhost = 2*SrcGhost::NGHOST;
  for (int i=0;i<=2;i++) {
    xyzFaceIncr_[i] = 0;
    xyzVolIncr_[i] = 0;
    xyzCount_[i] = dim[i] + nGhost;
  }
  
  const size_t direction = PhiFaceT::Location::FaceDir::value;
  switch (direction) {
      
    case SpatialOps::XDIR::value:
      xyzFaceIncr_[1] = 1;
      xyzVolIncr_[1] = 1;
      if (hasPlusFace[0]) xyzFaceIncr_[1] += 1;
      xyzCount_[0] -= 1;
      break;
      
    case SpatialOps::YDIR::value:
      xyzFaceIncr_[2] = stride_;
      xyzVolIncr_[2] = stride_;      
      if (hasPlusFace[1]) xyzFaceIncr_[2] += stride_;
      xyzCount_[1] -= 1;
      break;
      
    case SpatialOps::ZDIR::value:
      // NOTE: for the z direction, xyzVolIncr & xyzFaceIncr are all zero.
      // no need to set them here as they are initialized to zero previously.
      if (hasPlusFace[2]) xyzFaceIncr_[2] += stride_;
      xyzCount_[2] -= 1;
      break;
  }  
}

//--------------------------------------------------------------------

template<typename PhiVolT, typename PhiFaceT>
void
UpwindInterpolant<PhiVolT,PhiFaceT>::
set_advective_velocity (const PhiFaceT &theAdvectiveVelocity) {
  // !!! NOT THREAD SAFE !!! USE LOCK
  advectiveVelocity_ = &theAdvectiveVelocity;  
}

//--------------------------------------------------------------------

template<typename PhiVolT, typename PhiFaceT>
UpwindInterpolant<PhiVolT,PhiFaceT>::
~UpwindInterpolant()
{
}  

//--------------------------------------------------------------------

template<typename PhiVolT, typename PhiFaceT> 
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

  for (size_t k=1; k<=xyzCount_[2]; k++) { // count zCount times
    
    for (size_t j=1; j<=xyzCount_[1]; j++) { // count yCount times
      
      for (size_t i =1; i<=xyzCount_[0]; i++) { // count xCount times
        if ((*advVel) > 0.0) *destFld = *srcFieldMinus;
        else if ((*advVel) < 0.0) *destFld = *srcFieldPlus;
        else *destFld = 0.0; // may need a better condition here to account for
                             // a tolerance value for example.
        
        ++srcFieldMinus;
        ++srcFieldPlus;
        ++destFld;
        ++advVel;
      }
      
      srcFieldMinus += xyzVolIncr_[1];      
      srcFieldPlus  += xyzVolIncr_[1];
      destFld += xyzFaceIncr_[1];   
      advVel  += xyzFaceIncr_[1];
    }

    srcFieldMinus += xyzVolIncr_[2];
    srcFieldPlus  += xyzVolIncr_[2];
    destFld += xyzFaceIncr_[2];    
    advVel  += xyzFaceIncr_[2];
  }
}
