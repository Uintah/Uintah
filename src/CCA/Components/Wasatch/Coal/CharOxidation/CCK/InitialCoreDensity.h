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

#ifndef InitialCoreDensity_Expr_h
#define InitialCoreDensity_Expr_h

#include <expression/Expression.h>
#include <spatialops/particles/ParticleFieldTypes.h>
#include "CCKData.h"

namespace CCK{
/**
 *  \class  InitialCoreDensity
 *  \author Josh McConnell
 *  \date   June 2015
 *
 *  \brief Calculates the initial density of the carbonacious core of a char particle.
 *
 *  The CCK model assumes a growing ash film encapsulates a burning carbonacious core.
 *  In this implementation, it is assumed that the coal particle is composed of a porous
 *  char matrix and that volatiles and moisture contribute to mass of the particle but
 *  not the volume that encapsulates it. This means that consumption of volatiles and
 *  moisture would result in a decrease in the apparent density of the particle.
 *
 */
template< typename FieldT >
class InitialCoreDensity
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, initPrtDensity_d_, initPrtDiameter_ )

  const CCKData& cckData_;

  /*
   * \param initPrtDensity_d: initial (mass ash + char)/(particle volume)
   * \param initPrtDiameter : initial particle diameter
   */
  InitialCoreDensity( const Expr::Tag& initPrtDensity_dTag,
                      const Expr::Tag& initPrtDiameterTag,
                      const CCKData&   cckData )
    : Expr::Expression<FieldT>(),
      cckData_ ( cckData  )
  {
    this->set_gpu_runnable(true);
    initPrtDensity_d_ = this->template create_field_request<FieldT>( initPrtDensity_dTag );
    initPrtDiameter_  = this->template create_field_request<FieldT>( initPrtDiameterTag  );
  }


public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a InitialCoreDensity expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& initPrtDensity_dTag,
             const Expr::Tag& initPrtDiameterTag,
             const CCKData&   cckData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
        initPrtDensity_dTag_ ( initPrtDensity_dTag ),
        initPrtDiameterTag_  ( initPrtDiameterTag  ),
        cckData_             ( cckData             )
    {}

    Expr::ExpressionBase* build() const{
      return new InitialCoreDensity<FieldT>( initPrtDensity_dTag_, initPrtDiameterTag_, cckData_ );
    }

  private:
    const Expr::Tag initPrtDensity_dTag_, initPrtDiameterTag_;
    const CCKData &cckData_;
  };

  ~InitialCoreDensity(){};

  void evaluate()
  {
    using namespace SpatialOps;
    FieldT& result = this->value();

    const FieldT& rhoP0_d = initPrtDensity_d_->field_ref();

    const double rhoA     = cckData_.get_nonporous_ash_density()
                              *(1 - cckData_.get_min_ash_porosity() );
    const double xA0      = cckData_.get_ash();                    // initial ash mass fraction in coal
    const double xC0      = cckData_.get_fixed_C();                // initial char mass fraction in coal
    const double xA0_d    = xA0/(xC0 + xA0);

    result <<= (1.0 -xA0_d)/( 1.0/rhoP0_d - xA0_d/(rhoA) );
  }
};

}// namespace CCK

#endif // InitialCoreDensity_Expr_h
