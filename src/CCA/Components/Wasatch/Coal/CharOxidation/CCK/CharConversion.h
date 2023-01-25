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

#ifndef CharConversion_Expr_h
#define CharConversion_Expr_h

#include <expression/Expression.h>
#include "CCKData.h"

namespace CCK{

/**
 *  \class  CharConversion
 *  \author Josh McConnell
 *  \date   June 2015
 *
 *  \brief Calculates the fraction of char reacted.
 */
template< typename FieldT >
class CharConversion
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, charMass_, pMass0_ )

  const CCKData& cckData_;
  
  CharConversion( const Expr::Tag& charMassTag,
                  const Expr::Tag& initialPrtMassTag,
                  const CCKData&   cckData )
    : Expr::Expression<FieldT>(),
      cckData_( cckData )
  {
    charMass_ = this->template create_field_request<FieldT>( charMassTag       );
    pMass0_   = this->template create_field_request<FieldT>( initialPrtMassTag );
  }

public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a CharConversion expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& charMassTag,
             const Expr::Tag& initialPrtMassTag,
             const CCKData& cckData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
        charMassTag_      ( charMassTag       ),
        initialPrtMassTag_( initialPrtMassTag ),
        cckData_          ( cckData           )
    {}

    Expr::ExpressionBase* build() const{
      return new CharConversion<FieldT>( charMassTag_, initialPrtMassTag_, cckData_ );
    }


  private:
    const Expr::Tag charMassTag_, initialPrtMassTag_;

    const CCKData& cckData_;
  };

  ~CharConversion(){};

  void evaluate()
  {
    FieldT& result = this->value();

    const FieldT& charMass = charMass_->field_ref();
    const FieldT& pMass0   = pMass0_->field_ref();

    //const double xc   = cckData_.get_fixed_C() + cckData_.get_vm()*c0;
    const double xc   = cckData_.get_fixed_C();

    result <<= 1.0 - charMass/(xc*pMass0);
    result <<= min( max( 0.0, result ), 1.0 );
  }

};

} //namespace CCK

#endif // CharConversion_Expr_h
