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

#ifndef FirstOrderArrhenius_Expr_h
#define FirstOrderArrhenius_Expr_h

#include <expression/Expression.h>
#include <CCA/Components/Wasatch/Coal/CharOxidation/CharData.h>
#include "FirstOrderData.h"

/**
 *  \class  FirstOrderArrhenius
 *  \date   May 2015
 *  \author Josh McConnell
 */
namespace FOA{
typedef CHAR::CharOxidationData CharData;
typedef FirstOrderData          FOAData;

using CHAR::Vec;
using CHAR::Array2D;

template< typename FieldT >
class FirstOrderArrhenius
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, charMass_,  pressure_, gTemp_, pTemp_, pDiameter_, mixMw_ )
  DECLARE_VECTOR_OF_FIELDS( FieldT, yi_ )

  const CharData& charData_;
  const FOAData&  foaData_;
  const double    pi_, gasConst_, shNum_;
  double          ea_, aH2o_, aCo2_;
  CHAR::Vec       mwVec_;

  std::vector<typename FieldT::const_iterator> yIterVec_;

  /* declare any operators associated with this expression here */
  
  FirstOrderArrhenius( const Expr::TagList& yiTags,
                       const Expr::Tag&     pressureTag,
                       const Expr::Tag&     mixMwTag,
                       const Expr::Tag&     gTempTag,
                       const Expr::Tag&     pTempTag,
                       const Expr::Tag&     charMassTag,
                       const Expr::Tag&     pDiameterTag,
                       const CharData&      charData,
                       const FOAData&       foaData );

public:

  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::TagList yiTags_;
    const Expr::Tag charMassTag_, pressureTag_, pDiameterTag_, mixMwTag_, gTempTag_, pTempTag_;
    const CharData& charData_;
    const FOAData&  foaData_;

  public:
    /**
     *  @brief Build a FirstOrderArrhenius expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::TagList& resultTags,
             const Expr::TagList& yiTags,
             const Expr::Tag&     pressureTag,
             const Expr::Tag&     mixMwTag,
             const Expr::Tag&     gTempTag,
             const Expr::Tag&     pTempTag,
             const Expr::Tag&     charMassTag,
             const Expr::Tag&     pDiameterTag,
             const CharData&      charData,
             const FOAData&       foaData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
    : ExpressionBuilder( resultTags, nghost ),
      yiTags_      ( yiTags       ),
      charMassTag_ ( charMassTag  ),
      pressureTag_ ( pressureTag  ),
      pDiameterTag_( pDiameterTag ),
      mixMwTag_    ( mixMwTag     ),
      gTempTag_    ( gTempTag     ),
      pTempTag_    ( pTempTag     ),
      charData_    ( charData     ),
      foaData_     ( foaData      )
    {}

    Expr::ExpressionBase* build() const{
      return new FirstOrderArrhenius<FieldT>( yiTags_,   pressureTag_, mixMwTag_,     gTempTag_,
                                              pTempTag_, charMassTag_, pDiameterTag_, charData_,
                                              foaData_ );
    }
  };

  ~FirstOrderArrhenius(){};
  void evaluate();

};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
FirstOrderArrhenius<FieldT>::

FirstOrderArrhenius( const Expr::TagList& yiTags,
                     const Expr::Tag&     pressureTag,
                     const Expr::Tag&     mixMwTag,
                     const Expr::Tag&     gTempTag,
                     const Expr::Tag&     pTempTag,
                     const Expr::Tag&     charMassTag,
                     const Expr::Tag&     pDiameterTag,
                     const CharData&      charData,
                     const FOAData&       foaData )
  : Expr::Expression<FieldT>(),
    charData_( charData ),
    foaData_ ( foaData  ),
    pi_      ( 3.14159265 ),
    gasConst_( 8.3144621  ),
    shNum_   ( 2.0        )
{
  mwVec_.clear();
  mwVec_.push_back( charData_.get_mw(CHAR::CO2) );
  mwVec_.push_back( charData_.get_mw(CHAR::CO ) );
  mwVec_.push_back( charData_.get_mw(CHAR::O2 ) );
  mwVec_.push_back( charData_.get_mw(CHAR::H2 ) );
  mwVec_.push_back( charData_.get_mw(CHAR::H2O) );
  mwVec_.push_back( charData_.get_mw(CHAR::CH4) );

  ea_   = foaData_.get_ea();
  aH2o_ = foaData_.get_a_h2o();
  aCo2_ = foaData_.get_a_co2();

  this->template create_field_vector_request<FieldT>( yiTags, yi_ );

  pressure_  = this->template create_field_request<FieldT>( pressureTag  );
  mixMw_     = this->template create_field_request<FieldT>( mixMwTag     );
  gTemp_     = this->template create_field_request<FieldT>( gTempTag     );
  pTemp_     = this->template create_field_request<FieldT>( pTempTag     );
  charMass_  = this->template create_field_request<FieldT>( charMassTag  );
  pDiameter_ = this->template create_field_request<FieldT>( pDiameterTag );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
FirstOrderArrhenius<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  typedef typename FieldT::const_iterator FieldIter;
  typename Expr::Expression<FieldT>::ValVec&  result = this->get_value_vec();


  typename FieldT::iterator irH2o = result[0]->begin();
  typename FieldT::iterator irCo2 = result[1]->begin();

  FieldIter ipress  = pressure_ ->field_ref().begin();
  FieldIter imixMw  = mixMw_    ->field_ref().begin();
  FieldIter igTemp  = gTemp_    ->field_ref().begin();
  FieldIter ipTemp  = pTemp_    ->field_ref().begin();
  FieldIter icMass  = charMass_ ->field_ref().begin();
  FieldIter ipDiam  = pDiameter_->field_ref().begin();

  // pack a vector with mass fraction iterators.
  yIterVec_.clear();
  for( size_t i=0; i<yi_.size(); ++i ){
    yIterVec_.push_back( yi_[i]->field_ref().begin() );
  }

  const typename FieldT::const_iterator iEnd = pressure_->field_ref().end();

  for( ; ipress != iEnd; ++ipress, ++imixMw, ++igTemp, ++ipTemp,
        ++icMass, ++ipDiam, ++irH2o, ++irCo2 )
  {
    Vec xInf;  xInf.clear();  // bulk species mole fractions
    Vec ppInf; ppInf.clear(); // bulk species partial pressures

    Vec::const_iterator imw = mwVec_.begin();

    // Assemble vectors for bulk mole fractions and partial pressures;
    for( typename std::vector<typename FieldT::const_iterator>::iterator iy = yIterVec_.begin();
        iy!=yIterVec_.end(); ++iy, ++imw )
    {

      const double moleFrac = **iy * *imixMw / *imw;
      xInf.push_back( moleFrac );
      ppInf.push_back( *ipress * moleFrac );

      ++(*iy); // increment iterators for mass fractions to the next particle
    }

    if( *icMass>0){
    // exterior surface area of particle
    const double sa = pi_*pow(*ipDiam, 2);

    // assemble matrix of binary diffusivities and vector of effective diffusivities
    const CHAR::Array2D dFick = CHAR::binary_diff_coeffs( *ipress, *igTemp );
    const CHAR::Vec    dMix  = CHAR::effective_diff_coeffs(dFick, xInf );

    // rate constants
    const double kH2o = aH2o_*exp(-ea_/(gasConst_ * *ipTemp));
    const double kCo2 = aCo2_*exp(-ea_/(gasConst_ * *ipTemp));

    // mass transfer coefficients for CO2 and H2O
    const double mtcH2o = dMix[4]*shNum_/(*ipDiam);
    const double mtcCo2 = dMix[0]*shNum_/(*ipDiam);

    // partial pressures of CO2 and H2O at particle surface
    const double pH2oSurf =  (mtcH2o*ppInf[4])
                           / ( gasConst_*(*igTemp)*kH2o + mtcH2o*(*ipTemp)/(*igTemp) );

    const double pCo2Surf =  (mtcCo2*ppInf[0])
                           / ( gasConst_*(*igTemp)*kCo2 + mtcCo2*(*ipTemp)/(*igTemp) );

    // rates of char consumption by gasification (kg/s)
    *irH2o = -sa*kH2o*pH2oSurf;
    *irCo2 = -sa*kCo2*pCo2Surf;
    }
    else{
      *irH2o = 0;
      *irCo2 = 0;
    }
  }
}

//--------------------------------------------------------------------

}// namespace FOA

#endif // FirstOrderArrhenius_Expr_h
