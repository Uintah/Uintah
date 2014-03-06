
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

#ifndef Wasatch_MMS_Functions
#define Wasatch_MMS_Functions
#include <Core/IO/UintahZlibUtil.h>

#include <expression/Expression.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <limits>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <ctime>            // std::time

// boost includes
#include <boost/random/linear_congruential.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/special_functions/bessel.hpp>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

//--------------------------------------------------------------------

/**
 *  \class SineTime
 *  \author Tony Saad
 *  \date September, 2011
 *  \brief Implements a sin(t) function. This is useful for testing time integrators
           with ODEs. Note that we can't pass time as a argument to the functions
					 provided by ExprLib at the moment.
 */
template< typename ValT >
class SineTime : public Expr::Expression<ValT>
{
public:

  /**
   *  \brief Builds a Sin(t) expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
             const Expr::Tag& tTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag tt_;
  };

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

private:
  typedef typename SpatialOps::structured::SingleValueField TimeField;
  SineTime( const Expr::Tag& tTag );
  const Expr::Tag tTag_;
  const TimeField* t_;
};

//====================================================================
//--------------------------------------------------------------------

template<typename ValT>
SineTime<ValT>::
SineTime( const Expr::Tag& ttag )
: Expr::Expression<ValT>(),
  tTag_( ttag )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename ValT >
void
SineTime<ValT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( tTag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
SineTime<ValT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  t_ = &fml.field_ref<TimeField>( tTag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
SineTime<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  phi <<= sin( *t_ );
}

//--------------------------------------------------------------------

template< typename ValT >
SineTime<ValT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& ttag )
: ExpressionBuilder(result),
  tt_( ttag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
SineTime<ValT>::Builder::build() const
{
  return new SineTime<ValT>( tt_ );
}

//--------------------------------------------------------------------

/**
 *  \class ReadFromFileExpression
 *  \author Tony Saad
 *  \date July, 2012
 *  \brief Implementes an expression that reads data from a file.
 */
template< typename FieldT >
class ReadFromFileExpression : public Expr::Expression<FieldT>
{
public:

  /**
   *  \brief Save pointer to the patch associated with this expression. This
   *          is needed to set boundary conditions and extract other mesh info.
   */    
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder(const Expr::Tag& result,
            const Expr::Tag& xTag,
            const Expr::Tag& yTag,
            const Expr::Tag& zTag,
            const std::string fileName);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag xtag_, ytag_, ztag_;    
    const std::string filename_;    
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

private:
  ReadFromFileExpression( const Expr::Tag& xTag,
                          const Expr::Tag& yTag,
                          const Expr::Tag& zTag,
                          const std::string fileName );
  const Expr::Tag xtag_, ytag_, ztag_;
  const std::string filename_;  
  const FieldT* x_;
  const FieldT* y_;  
  const FieldT* z_;    
};

//--------------------------------------------------------------------

template<typename FieldT>
ReadFromFileExpression<FieldT>::
ReadFromFileExpression( const Expr::Tag& xTag,
                        const Expr::Tag& yTag,
                        const Expr::Tag& zTag,                       
                        const std::string fileName )
: Expr::Expression<FieldT>(),
  xtag_( xTag ),
  ytag_( yTag ),
  ztag_( zTag	),
  filename_(fileName)
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
ReadFromFileExpression<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xtag_ );
  exprDeps.requires_expression( ytag_ );
  exprDeps.requires_expression( ztag_ );  
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ReadFromFileExpression<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  x_ = &fm.field_ref( xtag_ );
  y_ = &fm.field_ref( ytag_ );  
  z_ = &fm.field_ref( ztag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ReadFromFileExpression<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  using namespace SpatialOps::structured;
  FieldT& phi = this->value();
  phi <<= 0.0;

  // gzFile utilities as they can handle both gzip and ascii files.
  gzFile inputFile = gzopen( filename_.c_str(), "r" );
  
  if(inputFile == NULL) {
    std::ostringstream warn;
    warn << "ERROR: Wasatch::ReadFromFileExpresssion: \n Unable to open the given input file " << filename_;
    throw Uintah::ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  const double xMax = field_max_interior(*x_);
  const double xMin = field_min_interior(*x_);
  const double yMax = field_max_interior(*y_);
  const double yMin = field_min_interior(*y_);
  const double zMax = field_max_interior(*z_);
  const double zMin = field_min_interior(*z_);    
  typename FieldT::interior_iterator phiiter = phi.interior_begin();

  double x,y,z,val;

  // to take care of comparing doubles, use a tolerance value for min & max (see below)
  const double eps = 2.0*std::numeric_limits<double>::epsilon();
//  int size = 0;
  while ( !gzeof( inputFile ) ) { // check for end of file
    x   = Uintah::getDouble(inputFile);
    y   = Uintah::getDouble(inputFile);
    z   = Uintah::getDouble(inputFile);
    val = Uintah::getDouble(inputFile);  
    const bool containsValue = x >= (xMin - eps) && x <= (xMax + eps)
                            && y >= (yMin - eps) && y <= (yMax + eps)
                            && z >= (zMin - eps) && z <= (zMax + eps);
    if( containsValue && phiiter != phi.interior_end() ){
      // this assumes that the list of data in the input file is ordered according to x, y, z locations...
      *phiiter = val;
      ++phiiter;
//      size++;
    }        
  }
//  const int pid =  Uintah::Parallel::getMPIRank();
//  std::cout << "Processor: " << pid << " collected total cells: "<< size << std::endl;
  gzclose( inputFile );     
}

//--------------------------------------------------------------------

template< typename FieldT >
ReadFromFileExpression<FieldT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& xTag,
         const Expr::Tag& yTag,
         const Expr::Tag& zTag,
         const std::string fileName )
: ExpressionBuilder(result),
  xtag_( xTag ),
  ytag_( yTag ),
  ztag_( zTag	),
  filename_(fileName)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ReadFromFileExpression<FieldT>::Builder::
build() const
{
  return new ReadFromFileExpression<FieldT>( xtag_, ytag_, ztag_, filename_ );
}

//--------------------------------------------------------------------

/**
 *  \class StepFunction
 *  \author Tony Saad
 *  \date July, 2012
 *  \brief Implements a StepFunction for initialization purposes among other things.
 */
template< typename FieldT >
class StepFunction : public Expr::Expression<FieldT>
{
public:
  
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     * @param result Tag of the resulting expression.
     * @param indepVarTag   Tag of the independent variable.
     * @param transitionPoint Location where the stepFunction switches values. This is the independent variable location.
     * @param lowValue  Value of the step function for independentVar <  transitionPoint.
     * @param highValue	Value of the step function for independentVar >= transitionPoint.
     */
    Builder(const Expr::Tag& result,
            const Expr::Tag& indepVarTag,
            const double transitionPoint=0.1,
            const double lowValue = 1.0,
            const double highValue = 0.0);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag indepVarTag_;
    const double transitionPoint_, lowValue_, highValue_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  StepFunction( const Expr::Tag& indepVarTag,
               const double transitionPoint,
               const double lowValue,
               const double highValue);
  const Expr::Tag indepVarTag_;
  const double transitionPoint_, lowValue_, highValue_;
  const FieldT* indepVar_;
};

//--------------------------------------------------------------------

template<typename FieldT>
StepFunction<FieldT>::
StepFunction( const Expr::Tag& indepVarTag,
             const double transitionPoint,
             const double lowValue,
             const double highValue)
: Expr::Expression<FieldT>(),
  indepVarTag_(indepVarTag),
  transitionPoint_(transitionPoint),
  lowValue_(lowValue),
  highValue_(highValue)
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
StepFunction<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( indepVarTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
StepFunction<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  indepVar_ = &fm.field_ref( indepVarTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
StepFunction<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= cond( (*indepVar_ < transitionPoint_), lowValue_ )
                 ( highValue_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
StepFunction<FieldT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& indepVarTag,
         const double transitionPoint,
         const double lowValue,
         const double highValue )
: ExpressionBuilder(result),
  indepVarTag_(indepVarTag),
  transitionPoint_(transitionPoint),
  lowValue_(lowValue),
  highValue_(highValue)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
StepFunction<FieldT>::Builder::
build() const
{
  return new StepFunction<FieldT>( indepVarTag_, transitionPoint_, lowValue_, highValue_);
}

//--------------------------------------------------------------------

/**
 *  \class RayleighTaylor
 *  \author Tony Saad
 *  \date October, 2013
 *  \brief Implements a stepfunction with a perturbed interface that can used to initialize a 
 Rayleigh-Taylor instability simulation. The formula used is:
 if y < y0 + A * sin(2*pi*f*x1) * sin(2*pi*f*x2): result = lowValue
 else                                             result = highValue
 */
template< typename FieldT >
class RayleighTaylor : public Expr::Expression<FieldT>
{
public:
  
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     * @param result Tag of the resulting expression.
     * @param indepVarTag   Tag of the independent variable.
     * @param x1Tag Tag of the x1 coordinate.
     * @param x2Tag Tag of the x2 coordinate.
     * @param transitionPoint Location where the RayleighTaylor switches values. This is the independent variable location.
     * @param lowValue  Value of the step function for independentVar <  transitionPoint.
     * @param highValue	Value of the step function for independentVar >= transitionPoint.
     * @param frequency Frequency of the interface perturbation, given as a multiple of 2*Pi.
     * @param amplitude	Amplitude of the perturbation.
     */
    Builder(const Expr::Tag& result,
            const Expr::Tag& indepVarTag,
            const Expr::Tag& x1Tag,
            const Expr::Tag& x2Tag,
            const double transitionPoint=0.1,
            const double lowValue = 1.0,
            const double highValue = 0.0,
            const double frequency=2.0*PI,
            const double amplitude=1.0);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag indepVarTag_, x1Tag_, x2Tag_;
    const double trans_, lo_, hi_, f_, amp_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  RayleighTaylor( const Expr::Tag& indepVarTag,
                  const Expr::Tag& x1Tag,
                  const Expr::Tag& x2Tag,
                  const double transitionPoint,
                  const double lowValue,
                  const double highValue,
                  const double frequency=2.0*PI,
                  const double amplitude=1.0);
  const Expr::Tag indepVarTag_, x1Tag_, x2Tag_;
  const double trans_, lo_, hi_, f_, amp_;
  const FieldT *indepVar_, *x1_, *x2_;
};

//--------------------------------------------------------------------

template<typename FieldT>
RayleighTaylor<FieldT>::
RayleighTaylor( const Expr::Tag& indepVarTag,
                const Expr::Tag& x1Tag,
                const Expr::Tag& x2Tag,
                const double transitionPoint,
                const double lowValue,
                const double highValue,
                const double frequency,
                const double amplitude)
: Expr::Expression<FieldT>(),
indepVarTag_(indepVarTag),
x1Tag_(x1Tag),
x2Tag_(x2Tag),
trans_(transitionPoint),
lo_(lowValue),
hi_(highValue),
f_(frequency),
amp_(amplitude)
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
RayleighTaylor<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( indepVarTag_ );
  exprDeps.requires_expression( x1Tag_       );
  exprDeps.requires_expression( x2Tag_       );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
RayleighTaylor<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  indepVar_ = &fm.field_ref( indepVarTag_ );
  x1_ = &fm.field_ref( x1Tag_ );
  x2_ = &fm.field_ref( x2Tag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
RayleighTaylor<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= cond( *indepVar_ < trans_ + amp_ * sin(2*PI*f_ * *x1_) * sin(2*PI*f_ * *x2_), lo_ )
                 ( hi_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
RayleighTaylor<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& indepVarTag,
        const Expr::Tag& x1Tag,
        const Expr::Tag& x2Tag,
        const double transitionPoint,
        const double lowValue,
        const double highValue,
        const double frequency,
        const double amplitude )
: ExpressionBuilder(result),
indepVarTag_(indepVarTag),
x1Tag_(x1Tag),
x2Tag_(x2Tag),
trans_(transitionPoint),
lo_(lowValue),
hi_(highValue),
f_(frequency),
amp_(amplitude)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
RayleighTaylor<FieldT>::Builder::
build() const
{
  return new RayleighTaylor<FieldT>( indepVarTag_, x1Tag_, x2Tag_, trans_, lo_, hi_, f_, amp_);
}

//--------------------------------------------------------------------

/**
 *  \class  RandomField
 *  \author Tony Saad
 *  \date   July, 2012
 *  \brief  Generates a pseudo-random field based.
 */
template< typename ValT >
class RandomField : public Expr::Expression<ValT>
{
public:
  
  /**
   *  \brief Builds a RandomField expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
             const double lo,
             const double hi,
             const double seed );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double lo_, hi_, seed_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  const double lo_, hi_, seed_;
  RandomField( const double lo,
               const double hi,
               const double seed );
};

//====================================================================

template<typename ValT>
RandomField<ValT>::
RandomField(const double lo, 
            const double hi, 
            const double seed )
: Expr::Expression<ValT>(),
  lo_(lo),
  hi_(hi),
  seed_(seed)
{}

//--------------------------------------------------------------------

template< typename ValT >
void
RandomField<ValT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{}

//--------------------------------------------------------------------

template< typename ValT >
void
RandomField<ValT>::
bind_fields( const Expr::FieldManagerList& fml )
{}

//--------------------------------------------------------------------

template< typename ValT >
void
RandomField<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  typename ValT::iterator phiIter = phi.begin();
  
  // This is a typedef for a random number generator.
  typedef boost::mt19937 base_generator_type; // mersenne twister
  // Define a random number generator and initialize it with a seed.
  // (The seed is unsigned, otherwise the wrong overload may be selected
  // when using mt19937 as the base_generator_type.)
  // seed the random number generator based on the MPI rank
  const int pid =  Uintah::Parallel::getMPIRank();
  base_generator_type generator((unsigned) ( (pid+1) * seed_ * std::time(0) ));

  boost::uniform_real<> rand_dist(lo_,hi_);
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > boost_rand(generator, rand_dist);  
  
  while ( phiIter != phi.end() ) {
    *phiIter = boost_rand();
    ++phiIter;
  }  
}

//--------------------------------------------------------------------

template< typename ValT >
RandomField<ValT>::Builder::
Builder( const Expr::Tag& result,
         const double lo,
         const double hi,
         const double seed )
: ExpressionBuilder(result),
  lo_(lo),
  hi_(hi),
  seed_(seed)
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
RandomField<ValT>::Builder::build() const
{
  return new RandomField<ValT>(lo_, hi_, seed_ );
}

//--------------------------------------------------------------------

/**
 *  \class ExponentialVortex
 *  \author Tony Saad
 *  \date July, 2012
 *  \brief Implements an exponential vortex with the streamfunction given
           by \f$ \psi = C_0 \exp \left( - \frac{r^2}{2 R_0^2} \right) + U_0 y  \f$.
    Here, \f$ C_0 \f$ is the vortex strength, \f$ r = \sqrt{(x-x_0)^2 + (y-y_0)^2} \f$
    with \f$ x_0 \f$ and \f$ y_0 \f$ corresponding to the vortex center, \f$ R_0 \f$
    is the vortex radius, and \f$ U_0 \f$ is the free stream velocity. NOTE: when the vortex
    strength is larger than 0.001, you will observe skewness in the streamlines because as the
    vortex strength corresponds to a rotation of the vortex (rotating cylinder).
 */
template< typename FieldT >
class ExponentialVortex : public Expr::Expression<FieldT>
{
public:
  enum VelocityComponent{
    X1,
    X2
  };
  
public:
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     * @param result Tag of the resulting expression.
     * @param xTag   Tag of the first coordinate.
     * @param yTag   Tag of the second coordinate.
     * @param xCenter Vortex center.
     * @param yCenter Vortex center.
     * @param vortexStrength Vortex strength.
     * @param vortexRadius Vortex radius.
     * @param freeStreamVelocity  Free stream velocity.
     * @param velocityComponent	Velocity component to return in a right-handed Cartesian 
              coordinate system. - use Stokes' streamfunction definition to 
              figure out which component you want.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const double xCenter,
             const double yCenter,
             const double vortexStrength,
             const double vortexRadius,
             const double u,
             const double v,
             const VelocityComponent velocityComponent);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag xTag_, yTag_;
    const double xCenter_, yCenter_, vortexStrength_, vortexRadius_, u_, v_;
    const VelocityComponent velocityComponent_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  ExponentialVortex( const Expr::Tag& xTag,
                     const Expr::Tag& yTag,
                     const double xCenter,
                     const double yCenter,
                     const double vortexStrength,
                     const double vortexRadius,
                     const double U,
                     const double V,
                     const VelocityComponent velocityComponent);
  const Expr::Tag xTag_, yTag_;
  const double xCenter_, yCenter_, vortexStrength_,  vortexRadius_, u_, v_;
  const VelocityComponent velocityComponent_;
  const FieldT *x_, *y_;
};

//--------------------------------------------------------------------

template<typename FieldT>
ExponentialVortex<FieldT>::
ExponentialVortex( const Expr::Tag& xTag,
                   const Expr::Tag& yTag,
                   const double xCenter,
                   const double yCenter,
                   const double vortexStrength,
                   const double vortexRadius,
                   const double U,
                   const double V,
                   const VelocityComponent velocityComponent)
: Expr::Expression<FieldT>(),
  xTag_   ( xTag    ),
  yTag_   ( yTag    ),
  xCenter_( xCenter ),
  yCenter_( yCenter ),
  vortexStrength_( vortexStrength ),
  vortexRadius_  ( vortexRadius   ),
  u_( U ),
  v_( V ),
  velocityComponent_( velocityComponent )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ExponentialVortex<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ExponentialVortex<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ExponentialVortex<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const double denom = 2.0*vortexRadius_*vortexRadius_;
  const double expFactor = 2.0 * vortexStrength_/denom;
  SpatFldPtr<FieldT> tmp = SpatialFieldStore::get<FieldT>( result );

  *tmp <<= (*x_ - xCenter_)*(*x_ - xCenter_) + (*y_ - yCenter_)*(*y_- yCenter_);

  result <<= expFactor * exp(- *tmp/denom );

  switch (velocityComponent_) {
    case X1:
      // jcs why do the work above if we only reset it here?
      result <<= u_ - (*y_ - yCenter_)*result;
      break;
    case X2:
      // jcs why do the work above if we only reset it here?
      result <<= v_ + (*x_ - xCenter_)*result;
      break;
    default:
      break;
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
ExponentialVortex<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& xTag,
        const Expr::Tag& yTag,
        const double xCenter,
        const double yCenter,                  
        const double vortexStrength,
        const double vortexRadius,
        const double u,
        const double v,
        const VelocityComponent velocityComponent )
: ExpressionBuilder(result),
  xTag_   ( xTag    ),
  yTag_   ( yTag    ),
  xCenter_( xCenter ),
  yCenter_( yCenter ),
  vortexStrength_( vortexStrength ),
  vortexRadius_  ( vortexRadius   ),
  u_( u ),
  v_( v ),
  velocityComponent_( velocityComponent )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ExponentialVortex<FieldT>::Builder::
build() const
{
  return new ExponentialVortex<FieldT>( xTag_, yTag_, xCenter_, yCenter_, vortexStrength_, vortexRadius_, u_, v_, velocityComponent_);
}

//--------------------------------------------------------------------

/**
 *  \class LambsDipole
 *  \author Tony Saad
 *  \date August, 2012
 *  \brief Implements a Lamb's dipole vortex.
 */
template< typename FieldT >
class LambsDipole : public Expr::Expression<FieldT>
{
public:
  enum VelocityComponent{
    X1,
    X2
  };
  
public:
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     * @param result Tag of the resulting expression.
     * @param xTag   Tag of the first coordinate.
     * @param yTag   Tag of the second coordinate.
     * @param xCenter Vortex center.
     * @param yCenter Vortex center.
     * @param vortexStrength Vortex strength.
     * @param vortexRadius Vortex radius.
     * @param freeStreamVelocity  Free stream velocity.
     * @param velocityComponent	Velocity component to return in a right-handed Cartesian 
     coordinate system. - use Stokes' streamfunction definition to 
     figure out which component you want.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const double xCenter,
             const double yCenter,
             const double vortexStrength,
             const double vortexRadius,
             const double U,
             const VelocityComponent velocityComponent );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag xTag_, yTag_;
    const double x0_, y0_, G_, R_, U_;
    const VelocityComponent velocityComponent_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  LambsDipole( const Expr::Tag& xTag,
               const Expr::Tag& yTag,
               const double xCenter,
               const double yCenter,
               const double vortexStrength,
               const double vortexRadius,
               const double U,
               const VelocityComponent velocityComponent );
  const Expr::Tag xTag_, yTag_;
  const double x0_, y0_, g_, r_, u_;
  const VelocityComponent velocityComponent_;
  const FieldT *x_, *y_;
};

//--------------------------------------------------------------------

template<typename FieldT>
LambsDipole<FieldT>::
LambsDipole( const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const double x0,
             const double y0,
             const double g,
             const double r,
             const double u,
             const VelocityComponent velocityComponent )
: Expr::Expression<FieldT>(),
  xTag_( xTag ),
  yTag_( yTag ),
  x0_  ( x0 ),
  y0_  ( y0 ),
  g_   ( g  ),
  r_   ( r  ),
  u_   ( u  ),
  velocityComponent_( velocityComponent )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
LambsDipole<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
LambsDipole<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
LambsDipole<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= 0.0;
  
  const double kR = 3.831705970207515;
  const double k = kR/r_;
  const double denom = boost::math::cyl_bessel_j(0, kR);

  SpatFldPtr<FieldT> xx0 = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> yy0 = SpatialFieldStore::get<FieldT>( result );

  *xx0 <<= *x_ - x0_;
  *yy0 <<= *y_ - y0_;
  
  SpatFldPtr<FieldT> r = SpatialFieldStore::get<FieldT>( result );
  *r <<= sqrt(*xx0 * *xx0 + *yy0 * *yy0);
  
  SpatFldPtr<FieldT> tmp0 = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> tmp1 = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> tmp2 = SpatialFieldStore::get<FieldT>( result );
  
  typename FieldT::iterator riter = r->begin();
  typename FieldT::iterator tmp0iter = tmp0->begin();
  typename FieldT::iterator tmp1iter = tmp1->begin();
  typename FieldT::iterator tmp2iter = tmp2->begin();  
  double kr;
  while (riter != r->end()) {
    kr = k * *riter;
    
    *tmp0iter = boost::math::cyl_bessel_j(0, kr);
    *tmp1iter = boost::math::cyl_bessel_j(1, kr);
    *tmp2iter = boost::math::cyl_bessel_j(2, kr);    
    
    ++riter;
    ++tmp0iter;
    ++tmp1iter;
    ++tmp2iter;
  }

  switch (velocityComponent_) {
    case X1:
      result <<= u_ + cond ( *r <= r_,
                       2.0*g_/(k*denom) * ( k * *yy0 * *yy0 * *tmp0 / (*r * *r) + (*xx0 * *xx0 - *yy0 * *yy0) * *tmp1/(*r * *r * *r) ) )
                      ( g_ + g_*r_*r_/(*r * *r) - 2.0*g_*r_*r_* (*xx0 * *xx0)/(*r * *r * *r * *r) );
      break;
    case X2:
      result <<= cond ( *r <= r_,
                        2.0*g_/denom * *xx0 * *yy0 * *tmp2 /(*r * *r) )
                      ( - 2.0*r_*r_*g_* *xx0 * *yy0/(*r * *r * *r * *r) );
      break;
    default:
      // jcs why do all of the work above if we are not using the result?
      break;
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
LambsDipole<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& xTag,
        const Expr::Tag& yTag,
        const double xCenter,
        const double yCenter,     
        const double vortexStrength,
        const double vortexRadius,
        const double U,
        const VelocityComponent velocityComponent)
: ExpressionBuilder(result),
xTag_     ( xTag      ), 
yTag_     ( yTag      ), 
x0_  ( xCenter   ),
y0_  ( yCenter   ),
G_ ( vortexStrength),
R_ ( vortexRadius ),
U_( U ),
velocityComponent_  ( velocityComponent )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
LambsDipole<FieldT>::Builder::
build() const
{
  return new LambsDipole<FieldT>( xTag_, yTag_, x0_, y0_, G_, R_, U_, velocityComponent_);
}

//--------------------------------------------------------------------

/**
 *  \class BurnsChristonAbskg
 */
template< typename FieldT >
class BurnsChristonAbskg
: public Expr::Expression<FieldT>
{
  const Expr::Tag xTag_, yTag_, zTag_;
  const FieldT* x_;
  const FieldT* y_;
  const FieldT* z_;
  
  /* declare operators associated with this expression here */
  
  BurnsChristonAbskg( const Expr::Tag& xTag,
                     const Expr::Tag& yTag,
                     const Expr::Tag& zTag );
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a BurnsChristonAbskg expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& xTag,
            const Expr::Tag& yTag,
            const Expr::Tag& zTag );
    
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag xTag_, yTag_, zTag_;
  };
  
  ~BurnsChristonAbskg();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
BurnsChristonAbskg<FieldT>::
BurnsChristonAbskg( const Expr::Tag& xTag,
                   const Expr::Tag& yTag,
                   const Expr::Tag& zTag )
: Expr::Expression<FieldT>(),
xTag_( xTag ),
yTag_( yTag ),
zTag_( zTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
BurnsChristonAbskg<FieldT>::
~BurnsChristonAbskg()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
BurnsChristonAbskg<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( zTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
BurnsChristonAbskg<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  x_ = &fml.template field_ref< FieldT >( xTag_ );
  y_ = &fml.template field_ref< FieldT >( yTag_ );
  z_ = &fml.template field_ref< FieldT >( zTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
BurnsChristonAbskg<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
BurnsChristonAbskg<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= 0.9 * ( 1.0 - 2.0 * abs(*x_) )
                 * ( 1.0 - 2.0 * abs(*y_) )
                 * ( 1.0 - 2.0 * abs(*z_) ) + 0.1;
}

//--------------------------------------------------------------------

template< typename FieldT >
BurnsChristonAbskg<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                 const Expr::Tag& xTag,
                 const Expr::Tag& yTag,
                 const Expr::Tag& zTag )
: ExpressionBuilder( resultTag ),
xTag_( xTag ),
yTag_( yTag ),
zTag_( zTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
BurnsChristonAbskg<FieldT>::
Builder::build() const
{
  return new BurnsChristonAbskg<FieldT>( xTag_,yTag_,zTag_ );
}

//--------------------------------------------------------------------

#endif // Wasatch_MMS_Functions
