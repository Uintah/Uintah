
/*
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
#include <spatialops/FieldReductions.h>

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

  SineTime( const Expr::Tag& tTag );
  const Expr::Tag tTag_;
  const double* t_;
};

//====================================================================
//--------------------------------------------------------------------

template<typename ValT>
SineTime<ValT>::
SineTime( const Expr::Tag& ttag )
: Expr::Expression<ValT>(),
  tTag_( ttag )
{}

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
  t_ = &fml.template field_manager<double>().field_ref( tTag_ );
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
 *  \class CylinderPatch
 *  \author Tony Saad
 *  \date April, 2012
 *  \brief Implements a cylindrical patch for initialization purposes among other things.
 By patch here we mean "patching a region of a domain with a specific value".
 For a given field, This class allows users to "patch" a cylindrical region with
 a specific value for that field and also set its value outside the cylinder.
 The user specifies two coordinates (XSVOL, YSVOL for example) through
 the input file along with an inside and outside values. The coordinates
 determine the plane perpendicular to the axis of the cylinder. The inside
 value specifies the desired value inside the cylinder while the outside
 value corresponds to that outside the cylinder. The user must also provide
 an origin and a radius. Note that the first two coordinates in the origin
 are of important as they correspond to the cylinder axis intersection with the
 coordinate plane. For example, if the user specifies (x,y) as coordinates,
 then the axis of the cylinder is aligned with the z-axis.
 */
template< typename FieldT >
class CylinderPatch : public Expr::Expression<FieldT>
{
public:

  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     * @param result Tag of the resulting expression.
     * @param tag1   Tag of the first coordinate of the plane perpendicular
                     to the cylinder axis.
     * @param tag2   Tag of the second coordinate of the plane perpendicular
                     to the cylinder axis. Note  that the order of coordinates
                     is not important (tag1 = X, tag2 = Y) is equivalent to
                     (tag1 = Y, tag2 = X).
     * @param origin Coordinates of the intersection point between the cylinder axis
                     and the plane of the supplied coordinates (tag1, tag2). Note,
                     the order of the coordinates is important here and must match
                     the order supplied in tag1 and tag2. The user may be required
                     to set a third coordinate from the input file, but that will
                     not affect the evaluation.
     * @param insideValue Desired value of the resulting field inside the cylinder.
     * @param outsideValue Desired value of the resulting field outside the cylinder.
     * @param radius	Radius of the cylinder.
     */
    Builder(const Expr::Tag& result,
            const Expr::Tag& tag1,
            const Expr::Tag& tag2,
            const std::vector<double> origin,
            const double insideValue = 1.0,
            const double outsideValue = 0.0,
            const double radius=0.1);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag tag1_, tag2_;
    const std::vector<double> origin_;
    const double insidevalue_, outsidevalue_, radius_;
  };

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();

private:

  CylinderPatch( const Expr::Tag& tag1,
                 const Expr::Tag& tag2,
                 const std::vector<double> origin,  // origin on the minus face perpendicular to axis
                 const double insideValue,
                 const double outsideValue,
                 const double radius);
  const Expr::Tag tag1_, tag2_;
  const std::vector<double> origin_;
  const double insidevalue_, outsidevalue_, radius_;
  const FieldT* field1_;
  const FieldT* field2_;
};

//--------------------------------------------------------------------

template<typename FieldT>
CylinderPatch<FieldT>::
CylinderPatch( const Expr::Tag& tag1,
               const Expr::Tag& tag2,
               const std::vector<double> origin,
               const double insideValue,
               const double outsideValue,
               const double radius)
: Expr::Expression<FieldT>(),
  tag1_(tag1), tag2_(tag2), origin_(origin), insidevalue_(insideValue),
  outsidevalue_(outsideValue), radius_(radius)
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
CylinderPatch<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( tag1_ );
  exprDeps.requires_expression( tag2_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
CylinderPatch<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  field1_ = &fm.field_ref( tag1_ );
  field2_ = &fm.field_ref( tag2_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
CylinderPatch<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  const double orig0 = origin_[0];
  const double orig1 = origin_[1];
  FieldT& result = this->value();
  result <<= cond( (*field1_ - orig0) * (*field1_ - orig0) + (*field2_ - orig1)*(*field2_ - orig1) - radius_*radius_ <= 0,
                   insidevalue_)
                 ( outsidevalue_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
CylinderPatch<FieldT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& tag1,
         const Expr::Tag& tag2,
         const std::vector<double> origin,
         const double insideValue,
         const double outsideValue,
         const double radius )
: ExpressionBuilder(result),
  tag1_(tag1),
  tag2_(tag2),
  origin_(origin),
  insidevalue_ (insideValue ),
  outsidevalue_(outsideValue),
  radius_(radius)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
CylinderPatch<FieldT>::Builder::
build() const
{
  return new CylinderPatch<FieldT>( tag1_, tag2_, origin_, insidevalue_, outsidevalue_, radius_ );
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
  while ( gzeof(inputFile) == 0 ) { // check for end of file
    x   = Uintah::getDouble(inputFile);
    y   = Uintah::getDouble(inputFile);
    z   = Uintah::getDouble(inputFile);
    val = Uintah::getDouble(inputFile);  
    const bool contains_value = x >= (xMin - eps) && x <= (xMax + eps) && y >= (yMin - eps) && y <= (yMax + eps) && z >= (zMin - eps) && z <= (zMax + eps);
    if (contains_value && phiiter != phi.interior_end()) { // this assumes that the list of data in the input file is ordered according to x, y, z locations...      
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
{}

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
Builder(const Expr::Tag& result, 
        const Expr::Tag& indepVarTag,
        const double transitionPoint,
        const double lowValue,
        const double highValue)
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
 *  \class PlusProfile
 *  \author Amir Biglari
 *  \date July, 2012
 *  \brief Implements a PlusProfile for initialization purposes among other things.
 */
template< typename FieldT >
class PlusProfile : public Expr::Expression<FieldT>
{
public:
  
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     * @param result Tag of the resulting expression.
     * @param xTag   Tag of the first coordinate.
     * @param yTag   Tag of the second coordinate.
     * @param xStart Location where the plusProfile starts in x direction. This is the independent variable location.
     * @param yStart Location where the plusProfile starts in y direction. This is the independent variable location.
     * @param xWidth the width of the plusProfile starts in x direction.
     * @param xWidth the width of the plusProfile starts in y direction.
     * @param lowValue  Value of the step function for independentVar <  transitionPoint.
     * @param highValue	Value of the step function for independentVar >= transitionPoint.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const double xStart,
             const double yStart,
             const double xWidth,
             const double yWidth,
             const double lowValue = 1.0,
             const double highValue = 0.0);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag xTag_, yTag_;
    const double xStart_, yStart_, xWidth_, yWidth_, lowValue_, highValue_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  PlusProfile( const Expr::Tag& xTag,
               const Expr::Tag& yTag,              
               const double xStart,
               const double yStart,
               const double xWidth,
               const double yWidth,
               const double lowValue,
               const double highValue );
  const Expr::Tag xTag_, yTag_;;
  const double xStart_, yStart_, xWidth_, yWidth_, lowValue_, highValue_;
  const FieldT *x_, *y_;
};

//--------------------------------------------------------------------

template<typename FieldT>
PlusProfile<FieldT>::
PlusProfile( const Expr::Tag& xTag,
             const Expr::Tag& yTag,
             const double xStart,
             const double yStart,
             const double xWidth,
             const double yWidth,
             const double lowValue,
             const double highValue )
: Expr::Expression<FieldT>(),
xTag_     ( xTag      ), 
yTag_     ( yTag      ), 
xStart_   ( xStart    ), 
yStart_   ( yStart    ), 
xWidth_   ( xWidth    ), 
yWidth_   ( yWidth    ), 
lowValue_ ( lowValue  ),
highValue_( highValue )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PlusProfile<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PlusProfile<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PlusProfile<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= cond( ( ((*x_ > xStart_)&&(*x_ < xStart_+xWidth_)) || ((*y_ > yStart_)&&(*y_ < yStart_+yWidth_)) ), highValue_ )
  ( lowValue_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
PlusProfile<FieldT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& xTag,
         const Expr::Tag& yTag,
         const double xStart,
         const double yStart,
         const double xWidth,
         const double yWidth,
         const double lowValue,
         const double highValue)
: ExpressionBuilder(result),
xTag_     ( xTag     ), 
yTag_     ( yTag     ), 
xStart_   ( xStart   ), 
yStart_   ( yStart   ), 
xWidth_   ( xWidth   ), 
yWidth_   ( yWidth   ), 
lowValue_ ( lowValue ),
highValue_( highValue)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
PlusProfile<FieldT>::Builder::
build() const
{
  return new PlusProfile<FieldT>( xTag_, yTag_, xStart_, yStart_, xWidth_, yWidth_, lowValue_, highValue_);
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
            const double seed);
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
  RandomField(const double lo, 
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
        const double seed)
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
           by \f$ \psi = C_0 \text{Exp}\left( - \frac{r^2}{2 R_0^2} \right) + U_0 y  \f$.
    here, \f$ C_0 \f$ is the vortex strength, \f$ r = \sqrt{(x-x_0)^2 + (y-y0)^2} \f$
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
            const double U,
            const double V,
            const VelocityComponent velocityComponent);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag xTag_, yTag_;
    const double xCenter_, yCenter_, vortexStrength_, vortexRadius_, U_, V_;
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
  const double xCenter_, yCenter_, vortexStrength_,  vortexRadius_, U_, V_;
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
xTag_     ( xTag      ), 
yTag_     ( yTag      ), 
xCenter_  ( xCenter   ),
yCenter_  ( yCenter   ),
vortexStrength_   ( vortexStrength    ), 
vortexRadius_ ( vortexRadius ),
U_ ( U ),
V_ ( V ),
velocityComponent_  ( velocityComponent )
{}

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
  result <<= 0.0;
  const double denom = 2.0*vortexRadius_*vortexRadius_;
  const double expFactor = 2.0 * vortexStrength_/denom;
  SpatFldPtr<FieldT> tmp = SpatialFieldStore::get<FieldT>( result );
  *tmp <<= 0.0;
  *tmp <<= (*x_ - xCenter_)*(*x_ - xCenter_) + (*y_ - yCenter_)*(*y_- yCenter_);
  result <<= expFactor * exp(- *tmp/denom );

  switch (velocityComponent_) {
    case X1:
      result <<=U_ - (*y_ - yCenter_)*result;
      break;
    case X2:
      result <<=V_ + (*x_ - xCenter_)*result;
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
        const double U,
        const double V,
        const VelocityComponent velocityComponent)
: ExpressionBuilder(result),
xTag_     ( xTag      ), 
yTag_     ( yTag      ), 
xCenter_  ( xCenter   ),
yCenter_  ( yCenter   ),
vortexStrength_   ( vortexStrength    ), 
vortexRadius_ ( vortexRadius ),
U_( U ),
V_( V ),
velocityComponent_  ( velocityComponent )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ExponentialVortex<FieldT>::Builder::
build() const
{
  return new ExponentialVortex<FieldT>( xTag_, yTag_, xCenter_, yCenter_, vortexStrength_, vortexRadius_, U_, V_, velocityComponent_);
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
            const VelocityComponent velocityComponent);
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
                    const VelocityComponent velocityComponent);
  const Expr::Tag xTag_, yTag_;
  const double x0_, y0_, G_, R_, U_;
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
                  const double G,
                  const double R,
                  const double U,
                  const VelocityComponent velocityComponent)
: Expr::Expression<FieldT>(),
xTag_( xTag ), 
yTag_( yTag ), 
x0_  ( x0 ),
y0_  ( y0 ),
G_   ( G  ),
R_   ( R  ),
U_   ( U  ),
velocityComponent_  ( velocityComponent )
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
  const double k = kR/R_;
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
      result <<= U_ + cond ( *r <= R_, 
                       2.0*G_/(k*denom) * ( k * *yy0 * *yy0 * *tmp0 / (*r * *r) + (*xx0 * *xx0 - *yy0 * *yy0) * *tmp1/(*r * *r * *r) ) )
                      ( G_ + G_*R_*R_/(*r * *r) - 2.0*G_*R_*R_* (*xx0 * *xx0)/(*r * *r * *r * *r) );
      break;
    case X2:
      result <<= cond ( *r <= R_, 
                        2.0*G_/denom * *xx0 * *yy0 * *tmp2 /(*r * *r) )
                      ( - 2.0*R_*R_*G_* *xx0 * *yy0/(*r * *r * *r * *r) );      
      break;
    default:
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


#endif // Wasatch_MMS_Functions
