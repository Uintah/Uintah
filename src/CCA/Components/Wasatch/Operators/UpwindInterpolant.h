#ifndef UpwindInterpolant_h
#define UpwindInterpolant_h

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

#include <vector>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>

/**
 *  \class     UpwindInterpolant
 *  \author    Tony Saad
 *  \date      July 2010
 *  \ingroup   WasatchOperators
 *
 *  \brief     Calculates convective flux using upwind interpolation.
 *
 *  \todo Parallelize apply_to_field() method
 *
 *  \todo Add mutex when set_advective_velocity() is set.  Release
 *	  when apply_to_field() is done.
 *  
 *  This class is a lightweight operator, i.e. it does NOT implement a
 *  matvec operation. The UpwindInterplant will interpolate the
 *  convective flux \f$\phi u_i\f$ where \f$\phi\f$ denotes a
 *  staggered or non-staggered field. For example, if \f$\phi\f$
 *  denotes the temperature T, then, \f$\phi\f$ is a scalar volume
 *  field. On the other hand, if \f$\phi\f$ denotes the momentum
 *  \f$\rho u_i\f$, then, \f$\phi\f$ is staggered volume field. The
 *  UpwindInterpolant will interpolate the volume field to its
 *  corresponding face field. Thus, if \f$\phi\f$ denotes a scalar
 *  field, then the UpwindInterpolant will produce a scalar face
 *  field. Similarly, if \f$\phi\f$ was an X-Staggered volume field,
 *  then the UpwindInterpolant will produce an X-Staggered face field.
 * 
 *  Based on this convention, the UpwindInterpolant class is templated
 *  on two types o fields: the \f$phi\f$ field type (SrcT) and its
 *  corresponding face field type (DestT).
 */

template < typename SrcT, typename DestT >
class UpwindInterpolant {
  
  // Here, the advectivevelocity has been already interpolated to the phi cell 
  // faces. The destination field should be of the same type as the advective 
  // velocity, i.e. a staggered, cell centered field.
  const DestT* advectiveVelocity_;
  
public:
  
  typedef SrcT  SrcFieldType;
  typedef DestT DestFieldType;
  
  /**
   *  \brief Constructor for upwind interpolant.
   */
  UpwindInterpolant();
  
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
  void set_advective_velocity( const DestT &theAdvectiveVelocity );
  
  /**
   *   \brief Sets the flux limiter type to be used by this interpolant.
   *   \param limiterType: An enum that holds the limiter name.
   */
  void set_flux_limiter_type( Wasatch::ConvInterpMethods limiterType ){}
  
  /**
   *  \brief Applies the Upwind interpolation to the source field.
   *
   *  \param src: A constant reference to the source field. In this
   *         case this is a scalar field usually denoted by phi.
   *
   *  \param dest: A reference to the destination field. This will
   *         hold the convective flux \f$\phi*u_i\f$ in the direction
   *         i. It will be stored on the staggered cell centers.
   */
  void apply_to_field( const SrcT &src, DestT &dest );
  
};

#endif // UpwindInterpolant_h

