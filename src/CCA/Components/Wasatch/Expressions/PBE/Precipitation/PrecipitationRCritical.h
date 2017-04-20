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

#ifndef PrecipitationRCritical_Expr_h
#define PrecipitationRCritical_Expr_h

#include <expression/Expression.h>

/**
 *  \ingroup WasatchExpressions
 *  \class PrecipitationRCritical
 *  \author Alex Abboud
 *  \date February 2012
 *
 *  \tparam FieldT the type of field.
 *
 *  \brief calculates the critical radius for Nucleation
 *  \f$ R^* = R_0 / \ln (S) \f$
 *  \f$ R_0 = 2*\sigma*\nu / R / T \f$
 */
template< typename FieldT >
class PrecipitationRCritical
: public Expr::Expression<FieldT>
{
  /* declare private variables such as fields, operators, etc. here */
  const double rKnotVal_;
  const bool doSurfEng_;
//  const FieldT* superSat_;    //field from table of supersaturation
//  const FieldT* surfaceEng_;  //critcal value of surface energy for small radii
  
  DECLARE_FIELDS(FieldT, superSat_, surfaceEng_)
  
  PrecipitationRCritical( const Expr::Tag& superSatTag,
                          const Expr::Tag& surfaceEng,
                          const double rKnotVal_);

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& result,
             const Expr::Tag& superSatTag,
             const Expr::Tag& surfaceEngTag,
             const double rKnotVal)
    : ExpressionBuilder(result),
      supersatt_(superSatTag),
      surfaceengt_(surfaceEngTag),
      rknotval_(rKnotVal)
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const
    {
      return new PrecipitationRCritical<FieldT>( supersatt_, surfaceengt_, rknotval_ );
    }

  private:
    const Expr::Tag supersatt_;
    const Expr::Tag surfaceengt_;
    const double rknotval_;
  };

  ~PrecipitationRCritical();
  void evaluate();

};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
PrecipitationRCritical<FieldT>::
PrecipitationRCritical( const Expr::Tag& superSatTag,
                        const Expr::Tag& surfaceEngTag,
                        const double rKnotVal)
: Expr::Expression<FieldT>(),
  rKnotVal_(rKnotVal),
  doSurfEng_(surfaceEngTag != Expr::Tag())
{
  this->set_gpu_runnable( true );
   superSat_ = this->template create_field_request<FieldT>(superSatTag);
  if (doSurfEng_)  surfaceEng_ = this->template create_field_request<FieldT>(surfaceEngTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
PrecipitationRCritical<FieldT>::
~PrecipitationRCritical()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
PrecipitationRCritical<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& S = superSat_->field_ref();
  
  if (doSurfEng_) {
    const FieldT& surfEng = surfaceEng_->field_ref();
    result <<= cond( S > 1.0, rKnotVal_ * surfEng / log(S) )
                   ( 0.0 );
  } else {
    result <<= cond( S > 1.0, rKnotVal_ / log(S) )
                   ( 0.0 ); //this is r*
  }
}

//--------------------------------------------------------------------

#endif // PrecipitationRCritical_Expr_h
