/*
 * The MIT License
 *
 * Copyright (c) 2015-2017 The University of Utah
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

#ifndef DevolatilizedMassFracs_Expr_h
#define DevolatilizedMassFracs_Expr_h

#include <expression/Expression.h>
#include "CCKData.h"

namespace CCK{
/**
 *  \class DevolatilizedMassFracs
 *  \author Josh McConnell
 *  \date   June 2015
 *
 *  \brief Calculates the mass fractions of char and ash without moisture and volatiles.
 *         The calculations below are equivalent to
 *         xChar_d  = xChar/(xChar + xAsh) and
 *         xAsh_d   = xAsh/(xChar + xAsh).
 */
template< typename FieldT >
class DevolatilizedMassFracs
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, initPrtMass_,prtMass_,charMass_ )
  const CCKData& cckData_;
  
  DevolatilizedMassFracs( const Expr::Tag& initPrtMassTag,
                  const Expr::Tag& prtMassTag,
                  const Expr::Tag& charMassTag,
                  const CCKData&   cckData )
    : Expr::Expression<FieldT>(),
      cckData_( cckData )
  {
    initPrtMass_ = this->template create_field_request<FieldT>( initPrtMassTag );
    prtMass_     = this->template create_field_request<FieldT>( prtMassTag     );
    charMass_    = this->template create_field_request<FieldT>( charMassTag    );
  }


public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a DevolatilizedMassFracs expression
     */
    Builder( const Expr::TagList& resultTags,
             const Expr::Tag& initPrtMassTag,
             const Expr::Tag& prtMassTag,
             const Expr::Tag& charMassTag,
             const CCKData&   cckData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTags, nghost ),
        initPrtMassTag_( initPrtMassTag ),
        prtMassTag_    ( prtMassTag     ),
        charMassTag_   ( charMassTag    ),
        cckData_       ( cckData        )
    {}

    Expr::ExpressionBase* build() const{
      return new DevolatilizedMassFracs<FieldT>( initPrtMassTag_,prtMassTag_,charMassTag_,cckData_ );
    }


  private:
    const Expr::Tag initPrtMassTag_, prtMassTag_, charMassTag_;
    const CCKData& cckData_;
  };

  ~DevolatilizedMassFracs(){};

  void evaluate()
  {
    typename Expr::Expression<FieldT>::ValVec&  result = this->get_value_vec();
    FieldT& xChar   = *result[0];
    FieldT& xAsh    = *result[1];
    FieldT& xChar_d = *result[2];
    FieldT& xAsh_d  = *result[3];

    const FieldT& mP0   = initPrtMass_->field_ref();
    const FieldT& mP    = prtMass_    ->field_ref();
    const FieldT& mChar = charMass_   ->field_ref();

    const double xA0    = cckData_.get_ash();

    xAsh    <<= min( 1.0, max( xA0*mP0/mP, 0.0) );
    xChar   <<= 1.0 - xAsh;
    xAsh_d  <<= min( 1.0, max( xA0*mP0/(mChar + xA0*mP0), 0.0) );
    xChar_d <<= 1.0 - xAsh_d;
  }
};

}// namespace CCK

#endif // DevolatilizedMassFracs_Expr_h
