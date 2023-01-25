/*
 * The MIT License
 *
 * Copyright (c) 2010-2018 The University of Utah
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

#include <expression/Expression.h>
#include <pokitt/CanteraObjects.h>

#ifndef AdiabaticIndex_h
#define AdiabaticIndex_h

namespace WasatchCore{

/**
 *  \class AdiabaticIndex
 *  \author Josh McConnell
 *  \date   October 2018
 *
 *  \brief Computes \f$ \left[\frac{M c_p}{R}\right-1]^{-1} \f$
 *
 *  where \f$\M\f$ is the mixture molecular weight, \f$\c_p\f$ is the
 *  mixture heat capacity, and \f$\R\f$ is the universal gas constant
 *  The resulting quantity is used to calculate  \f$\frac{DP}{Dt}\f$
 *  (material derivative of pressure) when species transport is enabled.
 */
template<typename FieldT>
class AdiabaticIndex : public Expr::Expression<FieldT>
{
  DECLARE_FIELDS(FieldT, cp_, mixMW_ )

  const double gasConstant_;

    AdiabaticIndex( const Expr::Tag& cpTag,
                    const Expr::Tag& mixMWTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a AdiabaticIndex expression
     *  @param resultTag the tag for the value that this expression computes
     */

    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& cpTag,
             const Expr::Tag& mixMWTag );

    Expr::ExpressionBase* build() const{
      return new AdiabaticIndex( cpTag_, mixMWTag_ );
    }

  private:
    const Expr::Tag cpTag_, mixMWTag_;
  };

  void evaluate();

};


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template<typename FieldT>
AdiabaticIndex<FieldT>::
AdiabaticIndex( const Expr::Tag& cpTag,
                const Expr::Tag& mixMWTag )
: Expr::Expression<FieldT>(),
  gasConstant_( CanteraObjects::gas_constant() )
{
  this->set_gpu_runnable(true);

  cp_    = this->template create_field_request<FieldT>( cpTag    );
  mixMW_ = this->template create_field_request<FieldT>( mixMWTag );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
AdiabaticIndex<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const FieldT& cp  = cp_->field_ref();
  const FieldT& mmw = mixMW_  ->field_ref();

  result <<= (cp*mmw)/(cp*mmw - gasConstant_);
}

//--------------------------------------------------------------------

template<typename FieldT>
AdiabaticIndex<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& cpTag,
                  const Expr::Tag& mixMWTag )
  : ExpressionBuilder( resultTag ),
    cpTag_   ( cpTag    ),
    mixMWTag_( mixMWTag )
{}

//====================================================================
}
#endif // AdiabaticIndex_h

