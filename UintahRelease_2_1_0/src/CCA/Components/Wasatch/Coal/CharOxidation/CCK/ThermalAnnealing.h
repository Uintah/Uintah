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

#ifndef ThermalAnnealing_Expr_h
#define ThermalAnnealing_Expr_h

#include <expression/Expression.h>
#include "CCKData.h"

#include <stdexcept>
#include <sstream>

namespace CCK{
/**
 *  \class  ThermalAnnealing
 *  \author Josh McConnell
 *  \date   June 2015
 *
 *  \brief Calculates A_i/A_i0 according to eq 8 of [1]. Evaluation uses the trapezoid rule
 *         for integration over a lognormal distribution.
 */
template< typename FieldT >
class ThermalAnnealing
 : public Expr::Expression<FieldT>
{
  DECLARE_VECTOR_OF_FIELDS( FieldT, logDist_ )
  const CCKData& cckData_;

  ThermalAnnealing( const Expr::TagList& logFreqDistTags,
                    const CCKData&       cckData )
    : Expr::Expression<FieldT>(),
      cckData_ ( cckData )
  {
    this->set_gpu_runnable( true );
    this->template create_field_vector_request<FieldT>( logFreqDistTags, logDist_ );
  }

public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a ThermalAnnealing expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag&     resultTag,
             const Expr::TagList& logFreqDistTags,
             const CCKData&       cckData,
             const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
          logFreqDistTags_( logFreqDistTags ),
          cckData_        ( cckData         )
    {}


    Expr::ExpressionBase* build() const
    {
      return new ThermalAnnealing<FieldT>( logFreqDistTags_, cckData_ );
    }

  private:
    const Expr::TagList logFreqDistTags_;
    const CCKData&   cckData_;
  };

  ~ThermalAnnealing(){};
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
ThermalAnnealing<FieldT>::
evaluate()
{
  FieldT& result = this->value();

  const CHAR::Vec edVec = cckData_.get_eD_vec();

  if( logDist_.size() != edVec.size() ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << std::endl
        << " Number oElements in logDistags does not match"
        << " that in vector of Activation energies"
        << std::endl
        << std::endl
        << " logDist_.size(): "<<logDist_.size()
        << " EdVec.size()   : "<<edVec.size()
        << std::endl
        << std::endl;
    throw std::runtime_error( msg.str() );
  }

  const FieldT& lnf_0 = logDist_[0]->field_ref();


  /* Calculate the first segment of the distribution. The lower bound for
   * Ed is zero and f(t, Ed = 0) = 0 for all t>0;
   */
  result <<= 0.5 * exp(lnf_0) * edVec[0];

  /* Calculate the remaining segments of the distribution. using the trapezoid
   * rule for integration.
   */
  for( size_t i = 0; i<logDist_.size() - 1; ++i ){

    const FieldT& lnf_i   = logDist_[i  ]->field_ref();
    const FieldT& lnf_ip1 = logDist_[i+1]->field_ref();

    result <<= result + 0.5*( exp(lnf_i) + exp(lnf_ip1) )
                           *( edVec[i+1] - edVec[i] );
  }

  result <<= sqrt(result);
}

//--------------------------------------------------------------------

}// namespace CCK

#endif // ThermalAnnealing_Expr_h

/* [1] Robert Hurt, Jian-Kuan Sun, and Melissa Lunden. A Kinetic Model
 *     of Carbon Burnout in Pulverized Coal Combustion. Combustion and
 *     Flame 113:181-197 (1998).
 *
 * [2] Randy C. Shurtz. Effects of Pressure on the Properties of Coal
 *     Char Under Gasification Conditions at High Initial Heating Rates.
 *     (2011). All Theses and Dissertations. Paper 2877.
 *     http://scholarsarchive.byu.edu/etd/2877/
 *
 */
