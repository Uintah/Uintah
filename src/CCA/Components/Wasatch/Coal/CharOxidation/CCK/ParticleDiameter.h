/*
 * The MIT License
 *
 * Copyright (c) 2015-2018 The University of Utah
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

#ifndef ParticleDiameter_Expr_h
#define ParticleDiameter_Expr_h

#include <expression/Expression.h>
#include "CCKData.h"

namespace CCK{
/**
 *  \class ParticleDiameter
 */
template< typename FieldT >
class ParticleDiameter
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, charMass_, initPrtMass_, devolDensity_,
                          initDevolDensity_, initPrtDiameter_  )

  const CCKData& cckData_;
  
  ParticleDiameter( const Expr::Tag& charMassTag,
                    const Expr::Tag& initPrtMassTag,
                    const Expr::Tag& devolDensityTag,
                    const Expr::Tag& initDevolDensityTag,
                    const Expr::Tag& initPrtDiameterTag,
                    const CCKData&   cckData )
    : Expr::Expression<FieldT>(),
      cckData_( cckData )
  {
    this->set_gpu_runnable(true);
    charMass_         = this->template create_field_request<FieldT>( charMassTag         );
    initPrtMass_      = this->template create_field_request<FieldT>( initPrtMassTag      );
    devolDensity_     = this->template create_field_request<FieldT>( devolDensityTag     );
    initDevolDensity_ = this->template create_field_request<FieldT>( initDevolDensityTag );
    initPrtDiameter_  = this->template create_field_request<FieldT>( initPrtDiameterTag  );
  }

public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a ParticleDiameter expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& charMassTag,
             const Expr::Tag& initPrtMassTag,
             const Expr::Tag& devolDensityTag,
             const Expr::Tag& initDevolDensityTag,
             const Expr::Tag& initPrtDiameterTag,
             const CCKData&   cckData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
        charMassTag_        ( charMassTag         ),
        initPrtMassTag_     ( initPrtMassTag      ),
        devolDensityTag_    ( devolDensityTag     ),
        initDevolDensityTag_( initDevolDensityTag ),
        initPrtDiameterTag_ ( initPrtDiameterTag  ),
        cckData_            ( cckData             )
    {}


    Expr::ExpressionBase* build() const{
      return new ParticleDiameter<FieldT>( charMassTag_,initPrtMassTag_,devolDensityTag_,
                                           initDevolDensityTag_,initPrtDiameterTag_,
                                           cckData_ );
    }


  private:
    const Expr::Tag charMassTag_, initPrtMassTag_, devolDensityTag_,
                    initDevolDensityTag_, initPrtDiameterTag_;
    const CCKData& cckData_;
  };

  ~ParticleDiameter(){};
  void evaluate()
  {
    FieldT& result = this->value();

    const FieldT& mC      = charMass_        ->field_ref();
    const FieldT& mP0     = initPrtMass_     ->field_ref();
    const FieldT& rhoP_d  = devolDensity_    ->field_ref();
    const FieldT& rhoP0_d = initDevolDensity_->field_ref();
    const FieldT& dP0     = initPrtDiameter_ ->field_ref();

    const double xA0 = cckData_.get_ash();
    const double xC0 = cckData_.get_fixed_C();

    /* The first term in the argument of pow is
     * (mass of char + ash)/(initial mass of char + ash);
     */

    result <<= dP0 *pow( (mC + mP0*xA0)
                         / ( mP0*(xA0 + xC0) )
                         * rhoP0_d/rhoP_d, 1.0/3.0 );
  }
};

}// namespace CCK

#endif // ParticleDiameter_Expr_h
