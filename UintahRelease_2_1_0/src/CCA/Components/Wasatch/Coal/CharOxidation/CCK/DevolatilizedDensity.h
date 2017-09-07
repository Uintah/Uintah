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

#ifndef DevolatilizedDensity_Expr_h
#define DevolatilizedDensity_Expr_h

#include <expression/Expression.h>
#include "CCKData.h"

namespace CCK{
/**
 *  \class  DevolatilizedDensity
 *  \author Josh McConnell
 *  \date   June 2015
 *
 *  \brief Calculates (mass char + ash)/(particle volume) from eq A9 of [1].
 *         This calculation assumes that the volatiles contribute to the mass
 *         of the particle but not the volume of the enclosing surface of the
 *         particle (think of fluid within a sponge).
 */
template< typename FieldT >
class DevolatilizedDensity
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, coreDensity_,ashMassFrac_,ashMassFrac_d_,charMassFrac_,ashDensity_ )

  const CCKData& cckData_;

  /*
   * \param coreDensity   : density of carbonacious core
   * \param ashMassFrac_d : (mass of ash)/(mass of ash + char)
   * \param ashDensity    : apparent density of ash in the particle
   */
  
  DevolatilizedDensity( const Expr::Tag& coreDensityTag,
                        const Expr::Tag& ashMassFracTag,
                        const Expr::Tag& ashMassFrac_dTag,
                        const Expr::Tag& charMassFracTag,
                        const Expr::Tag& ashDensityTag,
                        const CCKData& cckData )
    : Expr::Expression<FieldT>(),
      cckData_( cckData )
  {
    this->set_gpu_runnable(true);
    coreDensity_   = this->template create_field_request<FieldT>( coreDensityTag   );
    ashMassFrac_d_ = this->template create_field_request<FieldT>( ashMassFrac_dTag );
    ashMassFrac_   = this->template create_field_request<FieldT>( ashMassFracTag   );
    charMassFrac_  = this->template create_field_request<FieldT>( charMassFracTag  );
    ashDensity_    = this->template create_field_request<FieldT>( ashDensityTag    );
  }

public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a DevolatilizedDensity expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& coreDensityTag,
             const Expr::Tag& ashMassFracTag,
             const Expr::Tag& ashMassFrac_dTag,
             const Expr::Tag& charMassFracTag,
             const Expr::Tag& ashDensityTag,
             const CCKData& cckData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
        coreDensityTag_  ( coreDensityTag   ),
        ashMassFracTag_  ( ashMassFracTag   ),
        ashMassFrac_dTag_( ashMassFrac_dTag ),
        charMassFracTag_ ( charMassFracTag  ),
        ashDensityTag_   ( ashDensityTag    ),
        cckData_         ( cckData          )
    {}

    Expr::ExpressionBase* build() const{
      return new DevolatilizedDensity<FieldT>( coreDensityTag_, ashMassFracTag_,ashMassFrac_dTag_,
                                               charMassFracTag_,ashDensityTag_, cckData_ );
    }

  private:
    const Expr::Tag coreDensityTag_,ashMassFracTag_, ashMassFrac_dTag_, charMassFracTag_, ashDensityTag_;
    const CCKData& cckData_;
  };

  ~DevolatilizedDensity(){};

  void evaluate()
  {
    using namespace SpatialOps;
    FieldT& result = this->value();

    const FieldT& rhoC = coreDensity_  ->field_ref();
    const FieldT& xA_d = ashMassFrac_d_->field_ref();
    const FieldT& xA   = ashMassFrac_  ->field_ref();
    const FieldT& xC   = charMassFrac_ ->field_ref();
    const FieldT& rhoA = ashDensity_   ->field_ref();

    result <<= (xA+xC)/((1 - xA_d)/rhoC + xA_d/rhoA );
    /* [1]  Robert Hurt, Jian-Kuan Sun, Melissa Lunden. A Kinetic Model of
     *      Carbon Burnout in Pulverized Coal Combustion. Combustion and
     *      Flame 113:181-197 (1998)
     */
  }

};

}// namespace CCK

#endif // DevolatilizedDensity_Expr_h

