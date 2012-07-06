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

#include <expression/Expression.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <spatialops/FieldReductions.h>

#include <iostream>
#include <fstream>
#include <algorithm>


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
  
  std::ifstream fd(filename_.c_str(), std::ifstream::in);
  
  if(fd.fail()) {
    std::ostringstream warn;
    warn << "ERROR: Wasatch::ReadFromFileExpresssion: \n Unable to open the given input file " << filename_;
    throw Uintah::ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  

  typename FieldT::const_interior_iterator xiter = x_->interior_begin();
  typename FieldT::const_interior_iterator yiter = y_->interior_begin();
  typename FieldT::const_interior_iterator ziter = z_->interior_begin();
  typename FieldT::interior_iterator     phiiter = phi.interior_begin();    
  
  const double xMax = *(std::max_element( x_->interior_begin(), x_->interior_end() ) ); //field_max(*x_) does not seem to work...
  const double xMin = *(std::min_element( x_->interior_begin(), x_->interior_end() ) );
  const double yMax = *(std::max_element( y_->interior_begin(), y_->interior_end() ) );
  const double yMin = *(std::min_element( y_->interior_begin(), y_->interior_end() ) );
  const double zMax = *(std::max_element( z_->interior_begin(), z_->interior_end() ) );
  const double zMin = *(std::min_element( z_->interior_begin(), z_->interior_end() ) );
  
  double x,y,z,val;  
  while (fd.good()) {
    fd >> x >> y >> z >> val;
    const bool contains_value = x >= xMin && x <= xMax && y >= yMin && y <= yMax && z >= zMin && z <= zMax;
    if (contains_value) { // this assumes that the list of data in the input file is ordered according to x, y, z locations...
      *phiiter = val;
      ++phiiter;
    }    
  }
  
  fd.close();
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

#endif // Wasatch_MMS_Functions
