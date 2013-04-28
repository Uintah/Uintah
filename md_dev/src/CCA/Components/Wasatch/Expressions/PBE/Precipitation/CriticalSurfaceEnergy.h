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

#ifndef CriticalSurfaceEnergy_Expr_h
#define CriticalSurfaceEnergy_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class CriticalSurfaceEnergy
 *  \author Alex Abboud
 *  \date April 2013
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief calculates the critical surface energy for nucleation
 *  \f$ \sigma = \sigma_\infty * ( 1 - 2*\delta_T/ r) \f$
 *  where \f$ \sigma_\infty \f$ is the bulk surface energy, \f$ \delta_T \f$ is the Tolman lenght
 *  and \f$ r \f$ is the nucleate size
 *  since \f$ r \f$ is non-constant the expression can be sustituted in to give a quadratic equation for \f$ \sigma \f$
 *  \f$ \sigma^2 - \sigma_\infty \sigma + 0.2 R T \ln (S) \f$
 *  the highest value is used here for physical reasons
 */
template< typename FieldT >
class CriticalSurfaceEnergy
: public Expr::Expression<FieldT>
{
  /* declare private variables such as fields, operators, etc. here */
  const Expr::Tag superSatTag_;
  const double bulkSurfaceEnergy_;
  const double sqrtCoef_;
  const FieldT* superSat_; //field from table of supersaturation

  CriticalSurfaceEnergy( const Expr::Tag& superSatTag,
                         const double bulkSurfaceEnergy_,
                         const double sqrtCoef_);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& superSatTag,
             const double bulkSurfaceEnergy,
             const double sqrtCoef)
    : ExpressionBuilder(result),
    supersatt_(superSatTag),
    bulksurfaceenergy_(bulkSurfaceEnergy),
    sqrtcoef_(sqrtCoef)
    {}
    
    ~Builder(){}
    
    Expr::ExpressionBase* build() const
    {
      return new CriticalSurfaceEnergy<FieldT>( supersatt_, bulksurfaceenergy_, sqrtcoef_ );
    }
    
  private:
    const Expr::Tag supersatt_;
    const double bulksurfaceenergy_;
    const double sqrtcoef_;
  };
  
  ~CriticalSurfaceEnergy();
  
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
CriticalSurfaceEnergy<FieldT>::
CriticalSurfaceEnergy( const Expr::Tag& superSatTag,
                       const double bulkSurfaceEnergy,
                       const double sqrtCoef)
: Expr::Expression<FieldT>(),
superSatTag_(superSatTag),
bulkSurfaceEnergy_(bulkSurfaceEnergy),
sqrtCoef_(sqrtCoef)
{}

//--------------------------------------------------------------------

template< typename FieldT >
CriticalSurfaceEnergy<FieldT>::
~CriticalSurfaceEnergy()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
CriticalSurfaceEnergy<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( superSatTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
CriticalSurfaceEnergy<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  superSat_ = &fml.template field_manager<FieldT>().field_ref( superSatTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
CriticalSurfaceEnergy<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
CriticalSurfaceEnergy<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= cond( *superSat_ > 1.0, (bulkSurfaceEnergy_ + sqrt(bulkSurfaceEnergy_*bulkSurfaceEnergy_ - sqrtCoef_ * log(*superSat_)) )*0.5 )
                 ( bulkSurfaceEnergy_ ); //make equal to bulk
}

//--------------------------------------------------------------------

#endif // CriticalSurfaceEnergy_Expr_h

