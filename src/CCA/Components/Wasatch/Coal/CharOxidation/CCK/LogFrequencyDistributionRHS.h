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

#ifndef LogFrequencyDistributionRHS_Expr_h
#define LogFrequencyDistributionRHS_Expr_h

#include <expression/Expression.h>
#include "CCKData.h"

#include <stdexcept>
#include <sstream>
namespace CCK{
/**
 *  \class  LogFrequencyDistributionRHS
 *  \author Josh McConnell
 *  \date   June 2015
 *
 *  \brief Calculates the RHSs for d(nf)/dt = -aD*exp(-eD/RT) where nf  = ln(f)
 *         and f = f(t,eD) is the frequency distribution for the fraction of
 *         active sites at time t. Please see eqs 2-8 of [1] for more details.
 */
template< typename FieldT >
class LogFrequencyDistributionRHS
 : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS( FieldT, prtTemp_, initPrtMass_, prtMass_, vMass_ )
  const CCKData& cckData_;
  const double gasConst_;
  
  LogFrequencyDistributionRHS( const Expr::Tag& prtTempTag,
                               const Expr::Tag& initPrtMassTag,
                               const Expr::Tag& prtMassTag,
                               const Expr::Tag& vMassTag,
                               const CCKData&   cckData )
    : Expr::Expression<FieldT>(),
      cckData_ ( cckData       ),
      gasConst_( 1.98720413e-3 ) // gas constant in kCal/mol-K
  {
    this->set_gpu_runnable(true);
    prtTemp_     = this->template create_field_request<FieldT>( prtTempTag     );
    initPrtMass_ = this->template create_field_request<FieldT>( initPrtMassTag );
    prtMass_     = this->template create_field_request<FieldT>( prtMassTag     );
    vMass_       = this->template create_field_request<FieldT>( vMassTag       );
  }


public:

  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag prtTempTag_, initPrtMassTag_, prtMassTag_, vMassTag_;
    const CCKData&   cckData_;
  public:
    /**
     *  @brief Build a LogFrequencyDistributionRHS expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::TagList& resultTags,
             const Expr::Tag& prtTempTag,
             const Expr::Tag& initPrtMassTag,
             const Expr::Tag& prtMassTag,
             const Expr::Tag& vMassTag,
             const CCKData&   cckData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTags, nghost ),
        prtTempTag_    ( prtTempTag     ),
        initPrtMassTag_( initPrtMassTag ),
        prtMassTag_    ( prtMassTag     ),
        vMassTag_      ( vMassTag       ),
        cckData_       ( cckData        )
    {}

    Expr::ExpressionBase* build() const{
      return new LogFrequencyDistributionRHS<FieldT>( prtTempTag_, initPrtMassTag_, prtMassTag_, vMassTag_, cckData_ );
    }
  };

  ~LogFrequencyDistributionRHS(){};
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################


//--------------------------------------------------------------------

template< typename FieldT >
void
LogFrequencyDistributionRHS<FieldT>::
evaluate()
{
  typename Expr::Expression<FieldT>::ValVec& result = this->get_value_vec();

  const FieldT& pTemp  = prtTemp_    ->field_ref();
  const FieldT& pMass0 = initPrtMass_->field_ref();
  const FieldT& pMass  = prtMass_    ->field_ref();
  const FieldT& vMass  = vMass_      ->field_ref();

  const CHAR::Vec eDVec = cckData_.get_eD_vec();
  const double aD = cckData_.get_aD();
  const double xV = cckData_.get_vm();

  /* The parameter 'a' controls the behavior of the inhibiting effect
   * volatiles have on thermal annealing of the char.
   */
  const double a  = 30;

  if( result.size() != eDVec.size() ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << std::endl
        << "Number of result tags does not match number of"
        << "elements in vector of Activation energies"
        << std::endl
        << std::endl
        << " result.size(): "<<result.size()
        << " eDVec.size(): "<<eDVec.size()
        << std::endl
        << std::endl;
    throw std::runtime_error( msg.str() );
  }

  for( size_t i = 0; i<result.size(); ++i ){
    *result[i] <<= cond( pMass0*xV <= vMass, 0.0 )
                   (
                     -aD*exp(-eDVec[i]/(gasConst_*pTemp))
                     *exp(-a*vMass/(xV*pMass0 - vMass))
                   );
  }
}

}// namespace CCK

#endif // LogFrequencyDistributionRHS_Expr_h

/* [1] Robert Hurt, Jian-Kuan Sun, and Melissa Lunden. A Kinetic Model
 *     of Carbon Burnout in Pulverized Coal Combustion. Combustion and
 *     Flame 113:181-197 (1998).
 *
 */
