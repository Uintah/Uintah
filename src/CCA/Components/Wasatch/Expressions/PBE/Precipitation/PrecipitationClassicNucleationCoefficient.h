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

#ifndef PrecipitationClassicNucleationCoefficient_Expr_h
#define PrecipitationClassicNucleationCoefficient_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class PrecipitationClassicNucleationCoefficient
 *  \author Alex Abboud
 *  \date January, 2012
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief Nucleation Coeffcient Source term for use in QMOM
 *  classic nucleation refers to this value as
 *  \f$ B_0 = \exp ( 16 \pi /3 ( \gamma /K_B T)^3( \nu /N_A/ \ln(S)^2  \f$
 */
template< typename FieldT >
class PrecipitationClassicNucleationCoefficient
: public Expr::Expression<FieldT>
{
  const double expConst_;
  DECLARE_FIELD(FieldT, superSat_)

  PrecipitationClassicNucleationCoefficient( const Expr::Tag& superSatTag,
                                             const double expConst);

  public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& superSatTag,
             const double expConst )
    : ExpressionBuilder(result),
      supersatt_(superSatTag),
      expconst_(expConst)
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const
    {
      return new PrecipitationClassicNucleationCoefficient<FieldT>( supersatt_, expconst_);
    }

  private:
    const Expr::Tag supersatt_;
    const double expconst_;
  };

  ~PrecipitationClassicNucleationCoefficient();
  void evaluate();

};

// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
PrecipitationClassicNucleationCoefficient<FieldT>::
PrecipitationClassicNucleationCoefficient( const Expr::Tag& superSatTag,
                                           const double expConst)
: Expr::Expression<FieldT>(),
  expConst_(expConst)
{
  this->set_gpu_runnable( true );
   superSat_ = this->template create_field_request<FieldT>(superSatTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
PrecipitationClassicNucleationCoefficient<FieldT>::
~PrecipitationClassicNucleationCoefficient()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationClassicNucleationCoefficient<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& S = superSat_->field_ref();
  result <<= cond( S > 1.0, exp(expConst_ / log(S) / log(S) ) )
                 ( 0.0 );
}

//--------------------------------------------------------------------

#endif // PrecipitationClassicNucleationCoefficient_Expr_h
