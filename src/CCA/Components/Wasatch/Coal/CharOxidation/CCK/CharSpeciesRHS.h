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

#ifndef CharSpeciesRHS_Expr_h
#define CharSpeciesRHS_Expr_h

#include <expression/Expression.h>
#include "CCKData.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

namespace CCK{
/**
 *  \class CharSpeciesRHS
 *  \author Josh McConnell
 *  \date   June 2015
 *
 *  \brief  Calculates the RHSs of species reacting with and being produced by
 *          char gasification by CO2, H2O and H2, and oxidation by O2.
 *
 *  The CCK model considers the following global reactions with carbon (C) in char:
 *
 *  C + 0.5*(1 + fCO2 )*O2 --> fCO2*CO2 + (1 - fCO2)*CO
 *
 *  C + CO2                --> 2*CO
 *
 *  C + H2O                --> CO + H2
 *
 *  C + 2*H2               --> CH4
 *
 */
template< typename FieldT >
class CharSpeciesRHS
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, rC_CO2_, rC_CO_, rC_O2_, rC_H2_, rC_H2O_, co2_co_ratio_)

  const CCKData& cckData_;

  CharSpeciesRHS( const Expr::TagList& charDepletionTags,
                  const CCKData&       cckData );

public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a CharSpeciesRHS expression
     *  @param resultTags the tag for the value that this expression computes
     */
    Builder( const Expr::TagList& resultTags,
             const Expr::TagList& charDepletionTags,
             const CCKData&       cckData,
             const int nghost =   DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTags, nghost ),
        charDepletionTags_( charDepletionTags ),
        cckData_          ( cckData           )
    {}

    Expr::ExpressionBase* build() const{
      return new CharSpeciesRHS<FieldT>( charDepletionTags_, cckData_ );
    }

  private:
    const Expr::TagList charDepletionTags_;
    const CCKData& cckData_;
  };

  ~CharSpeciesRHS(){};
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
CharSpeciesRHS<FieldT>::
CharSpeciesRHS( const Expr::TagList& charDepletionTags,
                const CCKData&       cckData )
  : Expr::Expression<FieldT>(),
    cckData_( cckData )
{
  /* Species consumption rates have the following units (kg char consumed by i)/sec
   * and are in the following order:
   * *****************     [ CO2, CO, O2, H2, H2O ]     ************************/

  rC_CO2_ = this->template create_field_request<FieldT>( charDepletionTags[0] );
  rC_CO_  = this->template create_field_request<FieldT>( charDepletionTags[1] );
  rC_O2_  = this->template create_field_request<FieldT>( charDepletionTags[2] );
  rC_H2_  = this->template create_field_request<FieldT>( charDepletionTags[3] );
  rC_H2O_ = this->template create_field_request<FieldT>( charDepletionTags[4] );

  co2_co_ratio_ = this->template create_field_request<FieldT>( charDepletionTags[5] );

}

//--------------------------------------------------------------------

template< typename FieldT >
void
CharSpeciesRHS<FieldT>::
evaluate()
{
  /* There will be 6 results (in this order) for the following species:
   **************     [ CO2, CO, O2, H2, H2O, char }.    *******************
   *
   * All will be in kg/s produced.
   */

  typename Expr::Expression<FieldT>::ValVec& results = this->get_value_vec();

  FieldT& rhs_CO2  = *results[0];
  FieldT& rhs_CO   = *results[1];
  FieldT& rhs_O2   = *results[2];
  FieldT& rhs_H2   = *results[3];
  FieldT& rhs_H2O  = *results[4];
  FieldT& rhs_CH4  = *results[5];
  FieldT& rhs_Char = *results[6];


  const FieldT& rC_CO2  = rC_CO2_->field_ref();
  const FieldT& rC_CO   = rC_CO_ ->field_ref();
  const FieldT& rC_O2   = rC_O2_ ->field_ref();
  const FieldT& rC_H2   = rC_H2_ ->field_ref();
  const FieldT& rC_H2O  = rC_H2O_->field_ref();

  const FieldT& phi  = co2_co_ratio_->field_ref();

  typename FieldT::const_iterator irC_CO = rC_CO.begin();

  /* Check if consumption rate of char by CO is zero. If it is,
   * thow an error
   */
  const typename FieldT::const_iterator iEnd =rC_CO.end();
  for(; irC_CO != iEnd; ++irC_CO ){
    if(*irC_CO != 0.0 ){
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "Consumption rate by of char by CO is nonzero!!!" << std::endl
          <<"rC_CO:\t"<<*irC_CO<<std::endl;
      throw std::runtime_error( msg.str() );
    }
  }

  /* Species RHS calculations are analogous to calculations of depletion
   * fluxes by eqs 6.24-6.31 in [1].
   *
   */
  rhs_CO2 <<=   ( phi/(1.0 + phi)*rC_O2 - rC_CO2 )
              * 44.01/12.01;

  rhs_CO  <<=   ( 1.0/(1.0 + phi)*rC_O2 + 2*rC_CO2 + rC_H2O )
              * 28.01/12.01;

  rhs_O2  <<=   -(0.5 + phi)/(1.0 + phi)*rC_O2
              * 32.0/12.01;

  rhs_H2  <<=   ( rC_H2O - 2*rC_H2 )
              * 2.0/12.01;

  rhs_H2O <<=   -rC_H2O
              * 18.01/12.01;

  rhs_CH4 <<=   rC_H2
              * 16.01/12.01;

  rhs_Char <<= (rC_CO2 + rC_O2 + rC_H2 + rC_H2O );

}

//--------------------------------------------------------------------
}// namespace CCK

#endif // CharSpeciesRHS_Expr_h
