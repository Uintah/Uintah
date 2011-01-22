#ifndef FluxLimiterInterpolant_h
#define FluxLimiterInterpolant_h
/* -------------------------------------------------------
%%       %%%% %%     %% %%%% %%%%%%%% %%%%%%%% %%%%%%%%  
%%        %%  %%%   %%%  %%     %%    %%       %%     %% 
%%        %%  %%%% %%%%  %%     %%    %%       %%     %% 
%%        %%  %% %%% %%  %%     %%    %%%%%%   %%%%%%%%  
%%        %%  %%     %%  %%     %%    %%       %%   %%   
%%        %%  %%     %%  %%     %%    %%       %%    %%  
%%%%%%%% %%%% %%     %% %%%%    %%    %%%%%%%% %%     %% 
----------------------------------------------------------*/

#include <vector>
#include <string>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>

/**
 *  \class     FluxLimiterInterpolant
 *  \author    Tony Saad
 *  \date      January 2011
 *
 *  \brief     Calculates convective flux using a flux limiter.
 *  
 *  This class is a lightweight operator, i.e. it does NOT implement a
 *  matvec operation. The FluxLimiterInterpolant will interpolate the
 *  convective flux \phi u_i where \phi denotes a staggered or
 *  non-staggered field. For example, if \phi denotes the temperature
 *  T, then, \phi is a scalar volume field. On the other hand, if \phi
 *  denotes the momentum \rho u_i, then, \phi is staggered volume
 *  field. The FluxLimiterInterpolant will interpolate the volume field to
 *  its corresponding face field. Thus, if \phi denotes a scalar
 *  field, then the FluxLimiterInterpolant will produce a scalar face
 *  field. Similarly, if \phi was an X-Staggered volume field, then
 *  the FluxLimiterInterpolant will produce an X-Staggered face field.
 * 
 *  Based on this convention, the FluxLimiterInterpolant class is templated
 *  on two types o fields: the phi field type (PhiVolT) and its
 *  corresponding face field type (PhiFaceT).
 */

template < typename PhiVolT, typename PhiFaceT >
class FluxLimiterInterpolant {
  
private:
  
  // Here, the advectivevelocity has been already interpolated to the phi cell 
  // faces. The destination field should be of the same type as the advective 
  // velocity, i.e. a staggered, cell centered field.
  const PhiFaceT* advectiveVelocity_; 
  
  // holds the limiter type to be used, i.e. SUPERBEE, VANLEER, etc...
  Wasatch::ConvInterpMethods limiterType_;
  
  // An integer denoting the offset for the face index owned by the control 
  // volume in question. For the x direction, theStride = 0. 
  // For the y direction, stride_ = nx. For the z direction, stride_=nx*ny.
  size_t stride_; 
  
  // for the plus face, bndPlusStrideCoef is useful to define how many strides
  // one will have to move
  size_t bndPlusStrideCoef_;  
  
  // some counters to help in the evaluate member function
  std::vector<size_t> faceCount_;
  std::vector<size_t> volIncr_;
  std::vector<size_t> faceIncr_;
  
  // boundary counters
  std::vector<size_t> bndFaceCount_; // counter for faces at the boundary  
  std::vector<size_t> bndVolIncr_;
  std::vector<size_t> bndFaceIncr_;
  
  
public:
  
  typedef typename PhiVolT::Ghost      SrcGhost;
  typedef typename PhiVolT::Location   SrcLocation;
  typedef typename PhiFaceT::Ghost     DestGhost;
  typedef typename PhiFaceT::Location  DestLocation;
  typedef PhiVolT SrcFieldType;
  typedef PhiFaceT DestFieldType;
  
  /**
   *  \brief Constructor for flux limiter interpolant.
   *  \param dim: A 3D vector containing the number of control volumes
   *         in each direction.
   *  \param hasPlusFace: Determines if this patch has a physical
   *         boundary on its plus side.
   */
  FluxLimiterInterpolant( const std::vector<int>& dim,
                      const std::vector<bool> hasPlusFace );
  
  /**
   *  \brief Destructor for flux limiter interpolant.
   */  
  ~FluxLimiterInterpolant();
  
  /**
   *  \brief Sets the advective velocity field for the flux limiter interpolator.
   *
   *  \param theAdvectiveVelocity: A reference to the advective
   *         velocity field. This must be of the same type as the
   *         destination field, i.e. a face centered field.
   */
  void set_advective_velocity (const PhiFaceT &theAdvectiveVelocity);
  
  /**
   *   \brief Sets the flux limiter type to be used by this interpolant.
   *   \param limiterType: An enum that holds the limiter name.
   */
  void set_flux_limiter_type (Wasatch::ConvInterpMethods limiterType);
  
  /**
   *  \brief Applies the flux limiter interpolation to the source field.
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

#endif // FluxLimiterInterpolant_h