/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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
#include <spatialops/OperatorDatabase.h>
#include <CCA/Components/Wasatch/PatchInfo.h>
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
#include <boost/math/special_functions/next.hpp>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif
//--------------------------------------------------------------------

/**
 *  \class MultiplicativeInverse
 *  \author Tony Saad
 *  \date October, 2016
 *  \brief Implements a reciprocal or multiplicative inverse function, $a/x + b$.
 */
template< typename ValT >
class MultiplicativeInverse : public Expr::Expression<ValT>
{
public:
  
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
            const Expr::Tag& indepVarTag,
            const double& a=1.0,
            const double& b=0.0);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag xTag_;
    const double a_, b_;
  };
  
  void evaluate();
  
private:
  MultiplicativeInverse( const Expr::Tag& xTag, const double& a, const double& b );
  const double a_, b_;
  DECLARE_FIELD(ValT, x_)
};

//====================================================================
//--------------------------------------------------------------------

template<typename ValT>
MultiplicativeInverse<ValT>::
MultiplicativeInverse( const Expr::Tag& xTag,
                       const double& a,
                       const double& b)
: Expr::Expression<ValT>(),
  a_(a),
  b_(b)
{
  this->set_gpu_runnable( true );
  x_ = this->template create_field_request<ValT>(xTag);
}

//--------------------------------------------------------------------

template< typename ValT >
void
MultiplicativeInverse<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& result = this->value();
  const ValT& x = x_->field_ref();
  result <<= a_ * 1.0/x + b_;
}

//--------------------------------------------------------------------

template< typename ValT >
MultiplicativeInverse<ValT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& xTag,
        const double& a,
        const double& b)
: ExpressionBuilder(result),
xTag_( xTag ),
a_(a),
b_(b)
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
MultiplicativeInverse<ValT>::Builder::build() const
{
  return new MultiplicativeInverse<ValT>( xTag_, a_, b_ );
}

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

  void evaluate();

private:
  typedef typename SpatialOps::SingleValueField TimeField;
  SineTime( const Expr::Tag& tTag );
  DECLARE_FIELD(TimeField, t_)
};

//====================================================================
//--------------------------------------------------------------------

template<typename ValT>
SineTime<ValT>::
SineTime( const Expr::Tag& ttag )
: Expr::Expression<ValT>()
{
  this->set_gpu_runnable( true );
   t_ = this->template create_field_request<TimeField>(ttag);
}

//--------------------------------------------------------------------

template< typename ValT >
void
SineTime<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  const TimeField& t = t_->field_ref();
  phi <<= sin( t );
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
    Builder(const Expr::Tag& result,
            const std::string fileName);

    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag xtag_, ytag_, ztag_;    
    const std::string filename_;    
  };
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

private:
  ReadFromFileExpression( const Expr::Tag& xTag,
                          const Expr::Tag& yTag,
                          const Expr::Tag& zTag,
                          const std::string fileName );
  const std::string filename_;
  DECLARE_FIELDS(FieldT, x_, y_, z_)
  WasatchCore::UintahPatchContainer* patchContainer_;
};

//--------------------------------------------------------------------

template<typename FieldT>
ReadFromFileExpression<FieldT>::
ReadFromFileExpression( const Expr::Tag& xTag,
                        const Expr::Tag& yTag,
                        const Expr::Tag& zTag,                       
                        const std::string fileName )
: Expr::Expression<FieldT>(),
  filename_(fileName)
{
   x_ = this->template create_field_request<FieldT>(xTag);
   y_ = this->template create_field_request<FieldT>(yTag);
   z_ = this->template create_field_request<FieldT>(zTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ReadFromFileExpression<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  patchContainer_ = opDB.retrieve_operator<WasatchCore::UintahPatchContainer>();
}

//--------------------------------------------------------------------
// function that can be used to determine if a floating point x is between xMin and xMax
// based on a specified ULP, i.e. how many floats are allowed in between x and xMin.
template< typename T >
bool in_range(const T& x, const T& xMin, const T& xMax, const int ULP)
{
  using namespace boost::math;
  bool valid = false;
  if ( fabs(float_distance(xMin,x)) < ULP ) {
    valid = true;
  } else if (fabs(float_distance(xMax,x)) < ULP ) {
    valid = true;
  } else if ( x >= xMin && x <= xMax ) {
    valid = true;
  } else {
    valid = false;
  }
  return valid;
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ReadFromFileExpression<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& phi = this->value();
  const Uintah::Patch* const patch = patchContainer_->get_uintah_patch();
  phi <<= 0.0;
  
  const FieldT& x = x_->field_ref();
  const FieldT& y = y_->field_ref();
  const FieldT& z = z_->field_ref();
  
  // use gzFile utilities as they can handle both gzip and ascii files.
  gzFile inputFile = gzopen( filename_.c_str(), "r" );
  
  if(inputFile == nullptr) {
    std::ostringstream warn;
    warn << "ERROR: WasatchCore::ReadFromFileExpresssion: \n Unable to open the given input file " << filename_;
    throw Uintah::ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  typename FieldT::iterator phiiter = phi.interior_begin();

  double val;  
  const std::string inputFormat = Uintah::getString(inputFile);
  
  if (inputFormat == "FLAT") {
    int nx, ny, nz;
    nx   = Uintah::getInt(inputFile);
    ny   = Uintah::getInt(inputFile);
    nz   = Uintah::getInt(inputFile);
    for (int k=0; k<nz; k++) {
      for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
          Uintah::IntVector p(i,j,k);
          val = Uintah::getDouble(inputFile);
          const bool containsCell = patch->containsIndex(patch->getCellLowIndex(), patch->getCellHighIndex(), p);
          if( containsCell && phiiter != phi.interior_end() ) {
            // this assumes that the input file data is structured in the x, y, and z directions, respectively.
            // Note also the assumption here that the memory layout of SpatialField iterators is also in x, y, and z
            *phiiter = val;
            ++phiiter;
          }
        }
      }
    }
  } else if (inputFormat == "XYZ") {
    const double xMax = field_max_interior(x);
    const double xMin = field_min_interior(x);
    const double yMax = field_max_interior(y);
    const double yMin = field_min_interior(y);
    const double zMax = field_max_interior(z);
    const double zMin = field_min_interior(z);
    typename FieldT::iterator phiiter = phi.interior_begin();
    
    const double dx = x(IntVec(1,0,0)) - x(IntVec(0,0,0));
    const double dy = y(IntVec(0,1,0)) - y(IntVec(0,0,0));
    const double dz = z(IntVec(0,0,1)) - z(IntVec(0,0,0));
    double xp,yp,zp;
    // to take care of comparing doubles, use a tolerance value for min & max (see below)
    const double epsx = dx/2.0;
    const double epsy = dy/2.0;
    const double epsz = dz/2.0;
    while ( !gzeof( inputFile ) ) { // check for end of file
      xp   = Uintah::getDouble(inputFile);
      yp   = Uintah::getDouble(inputFile);
      zp   = Uintah::getDouble(inputFile);
      val = Uintah::getDouble(inputFile);
      
      const bool containsValue =     xp >= (xMin - epsx) && xp <= (xMax + epsx)
      && yp >= (yMin - epsy) && yp <= (yMax + epsy)
      && zp >= (zMin - epsz) && zp <= (zMax + epsz);
      
      if( containsValue && phiiter != phi.interior_end() ){
        // this assumes that the input file data is structured in the x, y, and z directions, respectively.
        // Note also the assumption here that the memory layout of SpatialField iterators is also in x, y, and z
        *phiiter = val;
        ++phiiter;
      }
    }
  } else {
    std::ostringstream warn;
    warn << "ERROR: WasatchCore::ReadFromFileExpresssion: \n unsupported file format. Supported file formats are FLAT and XYZ. You must include that in the first line of your data." << filename_;
    throw Uintah::ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
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
ReadFromFileExpression<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const std::string fileName )
: ExpressionBuilder(result),
xtag_( Expr::Tag() ),
ytag_( Expr::Tag() ),
ztag_( Expr::Tag()	),
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
 *  \brief Implements a Heaviside step function for initialization purposes among other things.
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
  
  void evaluate();
  
private:
  
  StepFunction( const Expr::Tag& indepVarTag,
               const double transitionPoint,
               const double lowValue,
               const double highValue);
  const double transitionPoint_, lowValue_, highValue_;
  DECLARE_FIELD(FieldT, indepVar_)
};

//--------------------------------------------------------------------

template<typename FieldT>
StepFunction<FieldT>::
StepFunction( const Expr::Tag& indepVarTag,
             const double transitionPoint,
             const double lowValue,
             const double highValue)
: Expr::Expression<FieldT>(),
  transitionPoint_(transitionPoint),
  lowValue_(lowValue),
  highValue_(highValue)
{
  this->set_gpu_runnable( true );
   indepVar_ = this->template create_field_request<FieldT>(indepVarTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
StepFunction<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& indepVar = indepVar_->field_ref();
  result <<= cond( (indepVar < transitionPoint_), lowValue_ )
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
  
  const double trans_, lo_, hi_, f_, amp_;
  DECLARE_FIELDS(FieldT, indepVar_, x1_, x2_)
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
trans_(transitionPoint),
lo_(lowValue),
hi_(highValue),
f_(frequency),
amp_(amplitude)
{
  this->set_gpu_runnable( true );
   indepVar_ = this->template create_field_request<FieldT>(indepVarTag);
   x1_ = this->template create_field_request<FieldT>(x1Tag);
   x2_ = this->template create_field_request<FieldT>(x2Tag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
RayleighTaylor<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& indepVar = indepVar_->field_ref();
  const FieldT& x1 = x1_->field_ref();
  const FieldT& x2 = x2_->field_ref();
  result <<= cond( indepVar < trans_ + amp_ * sin(2*PI*f_ * x1) * sin(2*PI*f_ * x2), lo_ )
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
  
  const double xCenter_, yCenter_, vortexStrength_,  vortexRadius_, u_, v_;
  const VelocityComponent velocityComponent_;
  DECLARE_FIELDS(FieldT, x_, y_)
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
  xCenter_( xCenter ),
  yCenter_( yCenter ),
  vortexStrength_( vortexStrength ),
  vortexRadius_  ( vortexRadius   ),
  u_( U ),
  v_( V ),
  velocityComponent_( velocityComponent )
{
  this->set_gpu_runnable( true );
   x_ = this->template create_field_request<FieldT>(xTag);
   y_ = this->template create_field_request<FieldT>(yTag);
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

  const FieldT& x = x_->field_ref();
  const FieldT& y = y_->field_ref();
  
  *tmp <<= (x - xCenter_)*(x - xCenter_) + (y - yCenter_)*(y- yCenter_);

  result <<= expFactor * exp(- *tmp/denom );

  switch (velocityComponent_) {
    case X1:
      // jcs why do the work above if we only reset it here?
      result <<= u_ - (y - yCenter_)*result;
      break;
    case X2:
      // jcs why do the work above if we only reset it here?
      result <<= v_ + (x - xCenter_)*result;
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
  const double x0_, y0_, g_, r_, u_;
  const VelocityComponent velocityComponent_;
  DECLARE_FIELDS(FieldT, x_, y_)
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
  x0_  ( x0 ),
  y0_  ( y0 ),
  g_   ( g  ),
  r_   ( r  ),
  u_   ( u  ),
  velocityComponent_( velocityComponent )
{
   x_ = this->template create_field_request<FieldT>(xTag);
   y_ = this->template create_field_request<FieldT>(yTag);
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
  
  const FieldT& x = x_->field_ref();
  const FieldT& y = y_->field_ref();
  
  const double kR = 3.831705970207515;
  const double k = kR/r_;
  const double denom = boost::math::cyl_bessel_j(0, kR);

  SpatFldPtr<FieldT> xx0 = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> yy0 = SpatialFieldStore::get<FieldT>( result );

  *xx0 <<= x - x0_;
  *yy0 <<= y - y0_;
  
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
  DECLARE_FIELDS(FieldT, x_, y_, z_)
  
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
: Expr::Expression<FieldT>()
{
   x_ = this->template create_field_request<FieldT>(xTag);
   y_ = this->template create_field_request<FieldT>(yTag);
   z_ = this->template create_field_request<FieldT>(zTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
BurnsChristonAbskg<FieldT>::
~BurnsChristonAbskg()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
BurnsChristonAbskg<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& x = x_->field_ref();
  const FieldT& y = y_->field_ref();
  const FieldT& z = z_->field_ref();
  
  result <<= 0.9 * ( 1.0 - 2.0 * abs(x) )
                 * ( 1.0 - 2.0 * abs(y) )
                 * ( 1.0 - 2.0 * abs(z) ) + 0.1;
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
