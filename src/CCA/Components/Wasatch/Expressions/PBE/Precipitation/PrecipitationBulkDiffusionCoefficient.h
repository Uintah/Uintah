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

#ifndef PrecipitationBulkDiffusionCoefficient_Expr_h
#define PrecipitationBulkDiffusionCoefficient_Expr_h
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class PrecipitationBulkDiffusionCoefficient
 *  \author Alex Abboud
 *  \date January 2012
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief calculates the expression containing the coefficient used in a
 *  precipitation reaction with bulk diffusion growth
 *  \f$ g_0 = \nu D C_{eq} (S-1) \f$ or \f$ (S - \bar{S}) \f$
 *  \f$ g(r) = 1/r \f$
 *
 */
template< typename FieldT >
class PrecipitationBulkDiffusionCoefficient
: public Expr::Expression<FieldT>
{
  const Expr::Tag superSatTag_, eqConcTag_, sBarTag_;
  const double growthCoefVal_;
  const FieldT* superSat_; //field from table of supersaturation
  const FieldT* eqConc_;   //field form table of equilibrium concentration
  const FieldT* sBar_;     //S Bar term for ostwald ripening

  PrecipitationBulkDiffusionCoefficient( const Expr::Tag& superSatTag,
                                         const Expr::Tag& eqConcTag,
                                         const Expr::Tag& sBarTag,
                                         const double growthCoefVal );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
            const Expr::Tag& superSatTag,
            const Expr::Tag& eqConcTag,
            const Expr::Tag& sBarTag,
            const double growthCoefVal)
    : ExpressionBuilder(result),
    supersatt_(superSatTag),
    eqconct_(eqConcTag),
    sbart_(sBarTag),
    growthcoefval_(growthCoefVal)
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const
    {
      return new PrecipitationBulkDiffusionCoefficient<FieldT>( supersatt_, eqconct_, sbart_, growthcoefval_ );
    }

  private:
    const Expr::Tag supersatt_, eqconct_, sbart_;
    const double growthcoefval_;
  };

  ~PrecipitationBulkDiffusionCoefficient();

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
PrecipitationBulkDiffusionCoefficient<FieldT>::
PrecipitationBulkDiffusionCoefficient( const Expr::Tag& superSatTag,
                                       const Expr::Tag& eqConcTag,
                                       const Expr::Tag& sBarTag,
                                       const double growthCoefVal )
: Expr::Expression<FieldT>(),
superSatTag_(superSatTag),
eqConcTag_(eqConcTag),
sBarTag_(sBarTag),
growthCoefVal_(growthCoefVal)
{}

//--------------------------------------------------------------------

template< typename FieldT >
PrecipitationBulkDiffusionCoefficient<FieldT>::
~PrecipitationBulkDiffusionCoefficient()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationBulkDiffusionCoefficient<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( superSatTag_ );
  exprDeps.requires_expression( eqConcTag_ );
  if ( sBarTag_ != Expr::Tag() )
    exprDeps.requires_expression( sBarTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationBulkDiffusionCoefficient<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  superSat_ = &fm.field_ref( superSatTag_ );
  eqConc_   = &fm.field_ref( eqConcTag_   );
  if ( sBarTag_ != Expr::Tag() )
    sBar_ = &fm.field_ref( sBarTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationBulkDiffusionCoefficient<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationBulkDiffusionCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  if ( sBarTag_ != Expr::Tag() ) {
    result <<= growthCoefVal_ * *eqConc_ * ( *superSat_ - *sBar_);
  } else {
    result <<= growthCoefVal_ * *eqConc_ * ( *superSat_ - 1.0 );  // this is g0  
  }
}

//--------------------------------------------------------------------

#endif // PrecipitationBulkDiffusionCoefficient_Expr_h
