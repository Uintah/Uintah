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
using namespace SpatialOps;
using namespace structured;

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
  int stride_; 
  // extra face
  bool hasPlusFace_;
  // dimension of the domain
  const std::vector<int> dim_;
  
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
  UpwindInterpolant(const std::vector<int>& dim, const bool hasPlusFace);
  
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
UpwindInterpolant(const std::vector<int>& dim,const bool hasPlusFace):
dim_(dim),hasPlusFace_(hasPlusFace) {
  // TSAAD - TODO: MOVE THIS TO FVTOOLS IN SPATIALOPS
  //stride_ = get_stride<DestFieldT>(dim, hasPlusFace)
  const size_t direction = PhiFaceT::Location::FaceDir::value;
  switch (direction) {
      
    case XDIR::value:
      stride_ = 1;
      break;
      
    case YDIR::value:
      stride_ = get_nx<SrcFieldType>(dim,hasPlusFace);
      break;
      
    case ZDIR::value:
      stride_ = get_nx<SrcFieldType>(dim,hasPlusFace)*get_ny<SrcFieldType>(dim,hasPlusFace);
      break;
      
    default:
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
apply_to_field(const PhiVolT &src, PhiFaceT &dest) const {   
  int xCount, yCount, zCount, incrFY=0, incrFZ=0, incrVolY=0, incrVolZ=0;
  //
  size_t direction = PhiFaceT::Location::FaceDir::value;
  switch (direction) {
      
    case XDIR::value:
      incrFY = 1;
      incrFZ = 1;
      incrVolY = 1;
      incrVolZ = 1;
      if (hasPlusFace_) incrFY++;
      xCount = dim_[0] + 2*SrcGhost::NGHOST -1;
      yCount = dim_[1] + 2*SrcGhost::NGHOST;
      zCount = dim_[2] + 2*SrcGhost::NGHOST;
      break;
      
    case YDIR::value:
      incrVolY = 0;
      incrVolZ = stride_;
      incrFY = 0;
      incrFZ = stride_;
      if (hasPlusFace_) incrFZ += stride_;
      xCount = dim_[0] + 2*SrcGhost::NGHOST;
      yCount = dim_[1] + 2*SrcGhost::NGHOST-1;
      zCount = dim_[2] + 2*SrcGhost::NGHOST;
      break;
      
    case ZDIR::value:
      incrVolY = 0;
      incrVolZ = 0;
      incrFY = 0;
      incrFZ = 0;
      if (hasPlusFace_) incrFZ += stride_;      
      xCount = dim_[0] + 2*SrcGhost::NGHOST;
      yCount = dim_[1] + 2*SrcGhost::NGHOST;
      zCount = dim_[2] + 2*SrcGhost::NGHOST-1;
      break;
  }  

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
  //
  //std::vec<int> theCount = get_count(patchSize);
  //int xCount = theCount[0];
  //int yCount = theCount[1];
  //int zCount = theCount[2];
  //
  for (int k=1; k<=zCount; k++) { // count zCount times
    
    for (int j=1; j<=yCount; j++) { // count yCount times
      
      for (int i =1; i<=xCount; i++) { // count xCount times
        if ((*advVel) > 0) *destFld = *srcFieldMinus;
        else if ((*advVel) < 0) *destFld = *srcFieldPlus;
        else *destFld = 0.0; // may need a better condition here to account
                             // a tolerance value for example.
        
        destFld++;
        srcFieldPlus++;
        srcFieldMinus++;
        advVel++;
      }
      
      srcFieldPlus += incrVolY;
      srcFieldMinus += incrVolY;
      advVel += incrFY;
      destFld += incrFY;   
    }
    
    srcFieldPlus += incrVolZ - incrVolY;
    srcFieldMinus += incrVolZ - incrVolY;
    advVel += incrFZ - incrFZ;
    destFld += incrFZ - incrFZ;    
  }
}