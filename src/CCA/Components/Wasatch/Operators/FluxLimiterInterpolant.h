/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#include <spatialops/SpatialOpsConfigure.h>
#ifdef ENABLE_THREADS
#include <boost/thread/mutex.hpp>
#endif

#include <spatialops/SpatialOpsDefs.h>
#include <spatialops/structured/SpatialFieldStore.h>
/**
 *  \class     FluxLimiterInterpolant
 *  \author    Tony Saad
 *  \date      January 2011
 *  \ingroup   WasatchOperators
 *
 *  \todo Consider basing this on the SpatialOps::Stencil2 stuff.
 *  \todo Parallelize apply_to_field() method
 *
 *  \brief     Calculates convective flux using a flux limiter.
 *
 *  The FluxLimiterInterpolant will interpolate the
 *  convective flux \f$\phi u_i\f$ where \f$\phi\f$ denotes a staggered or
 *  non-staggered field. For example, if \f$\phi\f$ denotes the temperature
 *  T, then, \f$\phi\f$ is a scalar volume field. On the other hand, if \f$\phi\f$
 *  denotes the momentum \f$\rho u_i\f$, then, \f$\phi\f$ is staggered volume
 *  field. The FluxLimiterInterpolant will interpolate the volume field to
 *  its corresponding face field. Thus, if \f$\phi\f$ denotes a scalar
 *  field, then the FluxLimiterInterpolant will produce a scalar face
 *  field. Similarly, if \f$\phi\f$ was an X-Staggered volume field, then
 *  the FluxLimiterInterpolant will produce an X-Staggered face field.
 *
 *  Based on this convention, the FluxLimiterInterpolant class is templated
 *  on two types o fields: the phi field type (PhiVolT) and its
 *  corresponding face field type (PhiFaceT).
 */

template < typename PhiVolT, typename PhiFaceT >
class FluxLimiterInterpolant {
  
private:
  
  // Here, the advective velocity has been already interpolated to the phi cell
  // faces. The destination field should be of the same type as the advective
  // velocity, i.e. a staggered, cell centered field.
  const PhiFaceT* advectiveVelocity_;
  
  // holds the limiter type to be used, i.e. SUPERBEE, VANLEER, etc...
  WasatchCore::ConvInterpMethods limiterType_;
  
  SpatialOps::IntVec unitNormal_;

  // boundary information
  bool hasPlusBoundary_, hasMinusBoundary_;
  
# ifdef ENABLE_THREADS
  boost::mutex mutex_;
# endif

  void build_src_fields( const PhiVolT& src, std::vector<PhiVolT>& ) const;

public:
  
  typedef typename PhiVolT::Location   SrcLocation;
  typedef typename PhiFaceT::Location  DestLocation;
  typedef PhiVolT SrcFieldType;
  typedef PhiFaceT DestFieldType;
  
  /**
   *  \brief Constructor for flux limiter interpolant.
   *  \param dim: A 3D vector containing the number of control volumes
   *         in each direction.
   *  \param hasPlusFace: Determines if this patch has a physical
   *         boundary on its plus side.
   *  \param hasMinusBoundary true if a physical boundary is present on the (-) side of the patch.
   */
  FluxLimiterInterpolant( const std::vector<int>& dim,
                          const std::vector<bool>& hasPlusFace,
                          const std::vector<bool>& hasMinusBoundary);
  
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
  void set_flux_limiter_type (WasatchCore::ConvInterpMethods limiterType);
  
  /**
   *  \brief Applies the flux limiter interpolation to the source field.
   *
   *  \param src: A constant reference to the source field. In this
   *         case this is a scalar field usually denoted by phi.
   *
   *  \param dest: A reference to the destination field. This will
   *         hold the convective flux \f$\phi*u_i\f$ in the direction
   *         i. It will be stored on the staggered cell centers.
   */
  void apply_to_field(const PhiVolT &src, PhiFaceT &dest);
  
  void apply_embedded_boundaries(const PhiVolT &src, PhiFaceT &dest) const;
};

#endif // FluxLimiterInterpolant_h
