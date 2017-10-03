/*
 * The MIT License
 *
 * Copyright (c) 2013-2017 The University of Utah
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

#include "TabPropsHeatLossEvaluator.h"

//===================================================================

template< typename FieldT >
TabPropsHeatLossEvaluator<FieldT>::
TabPropsHeatLossEvaluator( const InterpT* const adEnthInterp,
                           const InterpT* const sensEnthInterp,
                           const InterpT* const enthInterp,
                           const size_t hlIx,
                           const Expr::TagList& ivarNames )
  : Expr::Expression<FieldT>(),
    hlIx_( hlIx ),
    enthEval_    ( enthInterp     ),
    adEnthEval_  ( adEnthInterp   ),
    sensEnthEval_( sensEnthInterp )
{
  this->template create_field_vector_request<FieldT>(ivarNames, indepVars_);
}

//--------------------------------------------------------------------

template< typename FieldT >
TabPropsHeatLossEvaluator<FieldT>::
~TabPropsHeatLossEvaluator()
{
  delete enthEval_;
  delete adEnthEval_;
  delete sensEnthEval_;
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TabPropsHeatLossEvaluator<FieldT>::
evaluate()
{
  FieldT& heatLoss = this->value();

  typedef std::vector< typename FieldT::const_iterator > IVarIter;
  IVarIter ivarIters;

  for( size_t i=0; i<indepVars_.size(); ++i ){
    const FieldT& iVar = indepVars_[i]->field_ref();
    ivarIters.push_back( iVar.begin() );
  }
  
  // loop over grid points.
  const typename FieldT::iterator ihle = heatLoss.end();
  for( typename FieldT::iterator ihl=heatLoss.begin(); ihl!=ihle; ++ihl ){

    // extract indep vars at this grid point
    ivarsPoint_.clear();
    for( size_t i=0; i<ivarIters.size(); ++i ){
      if( i == hlIx_ ){
        // heat loss is a separate independent variable for our purposes
        // insert it at this location.  Value is arbitrary...
        ivarsPoint_.push_back(0.0);
      }
      ivarsPoint_.push_back( *ivarIters[i] );
    }

    // calculate the result
    const double h  = enthEval_    ->value( ivarsPoint_ );
    const double ha = adEnthEval_  ->value( ivarsPoint_ );
    const double hs = sensEnthEval_->value( ivarsPoint_ );
    *ihl = (h-ha)/hs;

    // increment all iterators to the next grid point
    for( typename IVarIter::iterator i=ivarIters.begin(); i!=ivarIters.end(); ++i )  ++(*i);

  } // grid loop
}

//--------------------------------------------------------------------

template< typename FieldT >
TabPropsHeatLossEvaluator<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const InterpT& adEnthInterp,
                  const InterpT& sensEnthInterp,
                  const InterpT& enthInterp,
                  const size_t hlIx,
                  const Expr::TagList& ivarNames )
  : ExpressionBuilder(result),
    adEnthInterp_  ( adEnthInterp   ),
    sensEnthInterp_( sensEnthInterp ),
    enthInterp_    ( enthInterp     ),
    hlIx_     ( hlIx      ),
    ivarNames_( ivarNames )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
TabPropsHeatLossEvaluator<FieldT>::
Builder::build() const
{
  return new TabPropsHeatLossEvaluator<FieldT>(
      adEnthInterp_.clone(),
      sensEnthInterp_.clone(),
      enthInterp_.clone(),
      hlIx_,
      ivarNames_ );
}

//===================================================================

//====================================================================
// Explicit template instantiation
#include <spatialops/structured/FVStaggered.h>
template class TabPropsHeatLossEvaluator<SpatialOps::SVolField>;
