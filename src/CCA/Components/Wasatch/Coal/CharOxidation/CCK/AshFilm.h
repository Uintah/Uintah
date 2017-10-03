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

#ifndef AshFilm_Expr_h
#define AshFilm_Expr_h

#include <expression/Expression.h>
#include "CCKData.h"

namespace CCK{

/**
 *  \class  AshFilm
 *  \author Josh McConnell
 *  \date   June 2015
 *
 *  \brief Calculates ash film thickness and porosity and core diameter. These
 *         are calculated together because the values are interdependent.
 */
template< typename FieldT >
class AshFilm
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, prtDiameter_, prtDensity_d_, initPrtDensity_d_, ashMassFrac_d_ )

  const CCKData&  cckData_;

  /*
   * \param prtDiameter     :  particle diameter
   * \param prtDensity_d    : (mass of ash + char)/(particle volume)
   * \param initPrtDensity_d: initial value of prtDensity_d
   * \param ashMassFrac_d   : (ash mass)/(ash mass + char mass)
   */
  
  AshFilm( const Expr::Tag& prtDiameterTag,
           const Expr::Tag& prtDensity_dTag,
           const Expr::Tag& initPrtDensity_dTag,
           const Expr::Tag& ashMassFrac_dTag,
           const CCKData&   cckData )
    : Expr::Expression<FieldT>(),
      cckData_ ( cckData  )
  {
    prtDiameter_      = this->template create_field_request<FieldT>( prtDiameterTag      );
    prtDensity_d_     = this->template create_field_request<FieldT>( prtDensity_dTag     );
    initPrtDensity_d_ = this->template create_field_request<FieldT>( initPrtDensity_dTag );
    ashMassFrac_d_    = this->template create_field_request<FieldT>( ashMassFrac_dTag    );
  }


public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a AshFilm expression
     *  @param resultTags the tag for the value that this expression computes
     */
    Builder( const Expr::TagList& resultTags,
             const Expr::Tag& prtDiameterTag,
             const Expr::Tag& prtDensity_dTag,
             const Expr::Tag& initPrtDensity_dTag,
             const Expr::Tag& ashMassFrac_dTag,
             const CCKData&   cckData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTags, nghost ),
        prtDiameterTag_     ( prtDiameterTag      ),
        prtDensity_dTag_    ( prtDensity_dTag     ),
        initPrtDensity_dTag_( initPrtDensity_dTag ),
        ashMassFrac_dTag_   ( ashMassFrac_dTag    ),
        cckData_            ( cckData             )
    {}

    Expr::ExpressionBase* build() const{
      return new AshFilm<FieldT>( prtDiameterTag_,prtDensity_dTag_,initPrtDensity_dTag_,ashMassFrac_dTag_,cckData_ );
    }

  private:
    const Expr::Tag prtDiameterTag_, prtDensity_dTag_, initPrtDensity_dTag_, ashMassFrac_dTag_;
    const CCKData&  cckData_;
  };

  ~AshFilm(){};
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################


template< typename FieldT >
void
AshFilm<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typedef typename FieldT::const_iterator FieldIter;
  typename Expr::Expression<FieldT>::ValVec& result = this->get_value_vec();

  typename FieldT::iterator itheta  = result[0]->begin();  // ash film porosity
  typename FieldT::iterator idelta  = result[1]->begin();  // ash film thickness
  typename FieldT::iterator idC     = result[2]->begin();  // carbonacious core diameter

  FieldIter idP       = prtDiameter_     ->field_ref().begin();
  FieldIter irhoP_d   = prtDensity_d_    ->field_ref().begin();
  FieldIter irhoP0_d  = initPrtDensity_d_->field_ref().begin();
  FieldIter ixA_d     = ashMassFrac_d_   ->field_ref().begin();

  const typename FieldT::const_iterator iEnd = prtDiameter_->field_ref().end();

  const double xA0      = cckData_.get_ash();
  const double xC0      = cckData_.get_fixed_C();
  const double xA0_d    = xA0/(xA0 + xC0);
  const double thetaMin = cckData_.get_min_ash_porosity();
  const double deltaMin = cckData_.get_min_ash_thickness();
  const double rhoA_np  = cckData_.get_nonporous_ash_density();


  for(; idP != iEnd; ++idP, ++irhoP_d, ++irhoP0_d, ++ixA_d,
                     ++itheta, ++idelta, ++idC ){

   // Calculate trial value of particle core diameter and ash film thickness.
    const double dC_t =  pow(
                              (*ixA_d * *irhoP_d   - rhoA_np*( 1.0 - thetaMin ) )
                            / ( xA0_d * *irhoP0_d  - rhoA_np*( 1.0 - thetaMin ) )
                            ,  1.0/3.0 )* *idP;

    const double delta_t = 0.5*(*idP - dC_t);


    /* Calculate ash film porosity. If the ash film thickness is larger than its minimum
     * value, the film porosity will be set to its minimum value and the core diameter and
     * film thickness will be set to their trial values...
     */
    if( delta_t >= deltaMin ){

      *idelta = delta_t;

      *itheta = thetaMin;

      *idC    = dC_t;
    }

    /* ...otherwise, film thickness is set to its minimum value, film porosity is calculated
     *  using the minimum film thickness, and core diameter is calculated with the new value
     *  for film porosity.
     */
    else{
      *idelta = deltaMin;

      *itheta =  deltaMin > 0?
                 1.0 - ( *ixA_d * *irhoP_d - xA0_d * *irhoP0_d*pow(1 - 2*deltaMin/(*idP), 3.0) )
                     / ( rhoA_np*( 1            -              pow(1 - 2*deltaMin/(*idP), 3.0) ) ):
                 thetaMin;

      *idC    = *idP - 2*deltaMin;

    }

  }//for
}

//--------------------------------------------------------------------

}// namespace CCK

#endif // AshFilm_Expr_h
