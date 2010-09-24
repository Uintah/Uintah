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
/*!
    @class     UpwindInterpolant
    @author    Tony Saad
    @date      July 2010
 
    @brief     Calculates convective flux using upwind interpolation.
    
    This class is a lightweight operator, i.e. it does NOT implement a matvec
 operation. The UpwindInterplant will interpolate the convective flux \phi u_i
 where \phi denotes a staggered or non-staggered field. For example, if \phi
 denotes the temperature T, then, \phi is a scalar volume field. On the other
 hand, if \phi denotes the momentum \rho u_i, then, \phi is staggered volume
 field. The UpwindInterpolant will interpolate the volume field to its
 corresponding face field. Thus, if \phi denotes a scalar field, then the
 UpwindInterpolant will produce a scalar face field. Similarly, if \phi
 was an X-Staggered volume field, then the UpwindInterpolant will produce an
 X-Staggered face field.
 
 Based on this convention, the UpwindInterpolant class is templated on two types
 o fields: the phi field type (PhiFieldT) and its corresponding face field type
 (PhiFaceFieldT).
*/
using namespace SpatialOps;
using namespace structured;

template < typename PhiFieldT, typename PhiFaceFieldT >
class UpwindInterpolant {
   
private:
  // Here, the advectivevelocity has been already interpolated to the phi cell 
  // faces. The destination field should be of the same type as the advective 
  // velocity, i.e. a staggered, cell centered field.
  const PhiFaceFieldT* advectiveVelocity_; 
  // An integer denoting the offset for the face index owned by the control 
  // volume in question. For the x direction, theStride = 0. 
  // For the y direction, stride_ = nx. For the z direction, stride_=nx*ny.
  int stride_; 
  // dimension of the domain
  const std::vector<int> dim_;
   
public:
  
  typedef typename PhiFieldT::Ghost      SrcGhost;
  typedef typename PhiFieldT::Location   SrcLocation;
  typedef typename PhiFaceFieldT::Ghost     DestGhost;
  typedef typename PhiFaceFieldT::Location  DestLocation;
  typedef PhiFieldT SrcFieldType;
  typedef PhiFaceFieldT DestFieldType;
  
  UpwindInterpolant(const std::vector<int>& dim, const bool hasPlusFace);
  void set_advective_velocity (const PhiFaceFieldT &theAdvectiveVelocity);
  void apply_to_field(const PhiFieldT &src, PhiFaceFieldT &dest) const; 

};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// I M P L E M E N T A T I O N
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*!
    @brief Constructor for upwind interpolant.
    @param dim: A 3D vector containing the number of control volumes in each 
 direction.
    @param hasPlusFace: Determines if this patch has a physical boundary on its 
 plus side.
*/
template <typename PhiFieldT, typename PhiFaceFieldT>
UpwindInterpolant<PhiFieldT,PhiFaceFieldT>::
UpwindInterpolant(const std::vector<int>& dim,const bool hasPlusFace):dim_(dim) {
  // TSAAD - TODO: MOVE THIS TO FVTOOLS IN SPATIALOPS
  const size_t direction = PhiFaceFieldT::Location::FaceDir::value;
  switch (direction) {
  
    case XDIR::value:
      stride_ = 1;
      break;
    
    case YDIR::value:
      stride_ = get_nx<PhiFieldT>(dim,hasPlusFace);
      break;
    
    case ZDIR::value:
      stride_ = get_nx<PhiFieldT>(dim,hasPlusFace)*get_ny<PhiFieldT>(dim,hasPlusFace);
      break;
    
    default:
      break;
  }
   //stride_ = get_stride<DestFieldT>(dim, hasPlusFace)
}

/*!
    @brief Sets the advective velocity field for the Upwind interpolator.
    @param theAdvectiveVelocity: A reference to the advective velocity field. 
           This must be of the same type as the destination field, i.e.
           a face centered field.
*/
template<typename PhiFieldT, typename PhiFaceFieldT>
void
UpwindInterpolant<PhiFieldT,PhiFaceFieldT>::
set_advective_velocity (const PhiFaceFieldT &theAdvectiveVelocity) {
   // !!! NOT THREAD SAFE !!! USE LOCK
   advectiveVelocity_ = &theAdvectiveVelocity;
}

/*!
    @brief Applies the Upwind interpolation to the source field.
    @param src: A constant reference to the source field. In this case this is 
 a scalar field usually denoted by phi.
    @param dest: A reference to the destination field. This will hold the 
 convective flux \phi*u_i in the direction i. It will be stored on the 
 staggered cell centers.
 */
template<typename PhiFieldT, typename PhiFaceFieldT> 
void 
UpwindInterpolant<PhiFieldT,PhiFaceFieldT>::
apply_to_field(const PhiFieldT &src, PhiFaceFieldT &dest) const {   
  
   /* This will compute the convective flux in a given direction using upwind 
    interpolation. The source field will represent the scalar variable, i.e. 
    \phi while the destination field will correspond to the convective flux: 
    \phi*u_i. */
   int xCount = dim_[0];
   int yCount = dim_[1];
   int zCount = dim_[2];
   //
   size_t direction = PhiFaceFieldT::Location::FaceDir::value;
   switch (direction) {
   case XDIR::value:
       xCount -= 1;
       break;
   case YDIR::value:
       yCount -= 1;
       break;
   case ZDIR::value:
       zCount -= 1;
       break;
   }
  
   /* Algorithm: TSAAD - TODO - DESCRIBE ALGORITHM IN DETAIL
    * Loop over faces
    */
   //
   typename PhiFieldT::const_iterator theSrcFieldIteratorMinus = src.begin(); // Source field on the minus side of a face
   typename PhiFieldT::const_iterator theSrcFieldIteratorPlus = src.begin() + stride_; // Source field on the plus side of a face
   typename PhiFaceFieldT::iterator theDestFieldIterator = dest.begin() + stride_; // Destination field (face). Starts on the first face
   // for that particular field. Whether it is x, y, or z face field. So its face index will start at zero, a face for which we cannot
   // compute the flux. So we add stride to it. In x direction, it will be face 1. In y direction, it will be nx. In z direction, it will be nx*ny
   typename PhiFaceFieldT::const_iterator theAdvectiveVelocityIterator = advectiveVelocity_->begin() + stride_;
   //
   //std::vec<int> theCount = get_count(patchSize); // Calculate the number of faces we have to go through.
   //int xCount = theCount[0];
   //int yCount = theCount[1];
   //int zCount = theCount[2];
   //
   for (int k=0; k<zCount; k++) { // count zCount times
      for (int j=0; j<yCount; j++) { // count yCount times
         for (int i =0; i<xCount; i++) { // count xCount times
            const double& theAdvectiveVelocity = *theAdvectiveVelocityIterator;
           *theDestFieldIterator =  std::max(theAdvectiveVelocity,0.0)*(*theSrcFieldIteratorMinus) + std::max(-theAdvectiveVelocity,0.0)*(*theSrcFieldIteratorPlus);
            theDestFieldIterator++;
            theSrcFieldIteratorPlus++;
            theSrcFieldIteratorMinus++;
         }
         theSrcFieldIteratorPlus += 2;
         theSrcFieldIteratorMinus += 2;
         theDestFieldIterator += 2;
      }
   }

}



