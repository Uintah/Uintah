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

#ifndef CoreDensity_Expr_h
#define CoreDensity_Expr_h

#include <expression/Expression.h>
#include "CCKData.h"
namespace CCK{
/**
 *  \class  CCKodel
 *  \author Josh McConnell
 *  \date   June 2015
 *
 *  \brief Calculates the particle core density for use with the
 *         the CCK model. The particle "core" is the carbonaceous
 *         center of a char particle with an ash film.
 */
template< typename FieldT >
class CoreDensity
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, charConversion_, initCoreDensity_ )

  const CCKData& cckData_;

    CoreDensity( const Expr::Tag& charConversionTag,
               const Expr::Tag& initCoreDensityTag,
               const CCKData&   cckData )
    : Expr::Expression<FieldT>(),
      cckData_( cckData )
  {
    this->set_gpu_runnable(true);
    charConversion_  = this->template create_field_request<FieldT>( charConversionTag  );
    initCoreDensity_ = this->template create_field_request<FieldT>( initCoreDensityTag );
  }


public:

  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag initCoreDensityTag_, charConversionTag_;
    const CCKData& cckData_;
  public:
    /**
     *  @brief Build a CoreDensity expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& charConversionTag,
             const Expr::Tag& initCoreDensityTag,
             const CCKData&   cckData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : Expr::ExpressionBuilder( resultTag, nghost ),
        initCoreDensityTag_( initCoreDensityTag ),
        charConversionTag_ ( charConversionTag  ),
        cckData_           ( cckData            )
    {}

    Expr::ExpressionBase* build() const{
      return new CoreDensity<FieldT>( charConversionTag_,initCoreDensityTag_, cckData_ );
    }
  };

  ~CoreDensity(){}

  void evaluate()
  {
    const FieldT& x     = charConversion_ ->field_ref();
    const FieldT& rhoC0 = initCoreDensity_->field_ref();

    const double alpha = cckData_.get_mode_of_burning_param();

    this->value() <<= rhoC0 * pow( 1.0 - x, alpha );
  }

};

} // namespace CCK
#endif // CoreDensity_Expr_h
