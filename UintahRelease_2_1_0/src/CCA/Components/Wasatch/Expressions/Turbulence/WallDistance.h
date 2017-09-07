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

#ifndef Wall_Distance_Expr_h
#define Wall_Distance_Expr_h

#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <CCA/Components/Wasatch/Expressions/PoissonExpression.h>

#include <expression/Expression.h>

/**
 *  \class WallDistance
 *  \author Tony Saad
 *  \date   June, 2012
 *  \ingroup Expressions
 *  \brief Calculates the distance to the nearest wall based on Spalding's 
           differential equation for wall distance. NOTE: you must solve a Poisson
           system of equations \f$\nabla^2\phi = -1\f$ with Dirichlet conditions
           on walls (\f$\phi = 0\f$) and Neumann conditions on all other boundary
           types (\f$\frac{\partial \phi}{\partial n} = 0\f$.
 *
 */

class WallDistance
: public Expr::Expression<SVolField>
{
  DECLARE_FIELD(SVolField, phi_)
   
  // gradient operators are only here to extract spacing information out of them
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, SVolField, XVolField >::type GradXT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, SVolField, YVolField >::type GradYT;
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, SVolField, ZVolField >::type GradZT;
  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, XVolField, SVolField >::type XtoSInterpT;  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, YVolField, SVolField >::type YtoSInterpT;  
  typedef SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, ZVolField, SVolField >::type ZtoSInterpT;  
  
  const GradXT*  gradXOp_;            ///< x-component of the gradient operator
  const GradYT*  gradYOp_;            ///< y-component of the gradient operator  
  const GradZT*  gradZOp_;            ///< z-component of the gradient operator

  const XtoSInterpT*  xToSInterpOp_;            ///< x-component of the gradient operator
  const YtoSInterpT*  yToSInterpOp_;            ///< y-component of the gradient operator  
  const ZtoSInterpT*  zToSInterpOp_;            ///< z-component of the gradient operator    
  
  WallDistance( const Expr::Tag& phitag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  private:
    const Expr::Tag phit_;
    
  public:
    Builder( const Expr::Tag& result,
            const Expr::Tag& phit )
    : ExpressionBuilder(result),
      phit_         ( phit )
    {}    
    
    Expr::ExpressionBase* build() const 
    {
      return new WallDistance(phit_);
    }
    
    ~Builder(){}
  };
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

WallDistance::
WallDistance( const Expr::Tag& phitag )
: Expr::Expression<SVolField>()
{
   phi_ = create_field_request<SVolField>(phitag);
}

//--------------------------------------------------------------------

void
WallDistance::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradXOp_ = opDB.retrieve_operator<GradXT>();
  gradYOp_ = opDB.retrieve_operator<GradYT>();
  gradZOp_ = opDB.retrieve_operator<GradZT>();
  
  xToSInterpOp_ = opDB.retrieve_operator<XtoSInterpT>();
  yToSInterpOp_ = opDB.retrieve_operator<YtoSInterpT>();
  zToSInterpOp_ = opDB.retrieve_operator<ZtoSInterpT>();  
}

//--------------------------------------------------------------------

void
WallDistance::
evaluate()
{
  using namespace SpatialOps;
  SVolField& walld = this->value();
  walld <<= 0.0;
  const SVolField& phi = phi_->field_ref();
  SpatialOps::SpatFldPtr<SVolField> gradPhiSq = SpatialOps::SpatialFieldStore::get<SVolField>( walld );
  *gradPhiSq <<=   (*xToSInterpOp_)((*gradXOp_)(phi)) * (*xToSInterpOp_)((*gradXOp_)(phi))
                 + (*yToSInterpOp_)((*gradYOp_)(phi)) * (*yToSInterpOp_)((*gradYOp_)(phi))
                 + (*zToSInterpOp_)((*gradZOp_)(phi)) * (*zToSInterpOp_)((*gradZOp_)(phi));
  
  walld <<= sqrt( *gradPhiSq + 2.0 * phi ) - sqrt(*gradPhiSq);
}

//--------------------------------------------------------------------

#endif // Wall_Distance_Expr_h
