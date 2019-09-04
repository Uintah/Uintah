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

#ifndef InitialDevolatilizedDensity_Expr_h
#define InitialDevolatilizedDensity_Expr_h

#include <expression/Expression.h>
#include "CCKData.h"

namespace CCK{

/**
 *  \class InitialDevolatilizedDensity
 *  \author Josh McConnell
 *  \date   June 2015
 *
 *  \brief Calculates initial value of (mass char + ash)/(particle volume).
 *         This calculation assumes that the volatiles contribute to the mass
 *         of the particle but not the volume of the enclosing surface of the
 *         particle (think of fluid within a sponge).
 */
template< typename FieldT >
class InitialDevolatilizedDensity
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELD( FieldT, initPrtDensity_ )

  const CCKData& cckData_;
  
  InitialDevolatilizedDensity( const Expr::Tag& initPrtDensityTag,
                               const CCKData&   cckData )
    : Expr::Expression<FieldT>(),
      cckData_( cckData )
  {
    this->set_gpu_runnable(true);
    initPrtDensity_ = this->template create_field_request<FieldT>( initPrtDensityTag );
  }


public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a InitialDevolatilizedDensity expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& initPrtDensityTag,
             const CCKData&   cckData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
        initPrtDensityTag_( initPrtDensityTag ),
        cckData_          ( cckData           )
    {}


    Expr::ExpressionBase* build() const{
      return new InitialDevolatilizedDensity<FieldT>( initPrtDensityTag_,cckData_ );
    }

  private:
    const Expr::Tag initPrtDensityTag_;
    const CCKData& cckData_;
  };

  ~InitialDevolatilizedDensity(){};
  void evaluate()
  {
    FieldT& result = this->value();

    const FieldT& rhoP0 = initPrtDensity_->field_ref();

    const double xA0 = cckData_.get_ash();
    const double xC0 = cckData_.get_fixed_C();

    result <<= (xA0 + xC0)*rhoP0;
  }

};


}// namespace CCK

#endif // InitialDevolatilizedDensity_Expr_h
