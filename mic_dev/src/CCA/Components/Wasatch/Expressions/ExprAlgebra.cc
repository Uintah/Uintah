/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <spatialops/structured/FVStaggered.h>

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
ExprAlgebra<FieldT>::
ExprAlgebra( const Expr::TagList srcTagList,
            const OperationType algebraicOperation,
            const bool isModifierExpr)
: Expr::Expression<FieldT>(),
  srcTagList_ (srcTagList),
  algebraicOperation_( algebraicOperation ),
  isModifierExpr_( isModifierExpr )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
ExprAlgebra<FieldT>::
~ExprAlgebra()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
ExprAlgebra<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  for( Expr::TagList::const_iterator iSrcTag=srcTagList_.begin();
      iSrcTag!=srcTagList_.end();
      ++iSrcTag ){
    exprDeps.requires_expression( *iSrcTag );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ExprAlgebra<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.field_manager<FieldT>();
  srcFields_.clear();
  for( Expr::TagList::const_iterator iSrcTag=srcTagList_.begin();
      iSrcTag!=srcTagList_.end();
      ++iSrcTag ){
    srcFields_.push_back( &fm.field_ref(*iSrcTag) );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ExprAlgebra<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  if( !isModifierExpr_ ){
    if( algebraicOperation_ == PRODUCT ) result <<= 1.0;
    else                                 result <<= 0.0;
  }
  
  typename std::vector<const FieldT*>::const_iterator srcFieldsIter = srcFields_.begin();
  while (srcFieldsIter != srcFields_.end()) {
    switch( algebraicOperation_ ){
      case SUM       : result <<= result + **srcFieldsIter;  break;
      case DIFFERENCE: result <<= result - **srcFieldsIter;  break;
      case PRODUCT   : result <<= result * **srcFieldsIter;  break;
    }
    ++srcFieldsIter;
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
ExprAlgebra<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                 const Expr::TagList srcTagList,
                 const OperationType algebraicOperation,
                 const bool isModifierExpr)
: ExpressionBuilder( resultTag ),
  srcTagList_( srcTagList ),
  algebraicOperation_( algebraicOperation ),
  isModifierExpr_( isModifierExpr )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
ExprAlgebra<FieldT>::
Builder::build() const
{
  return new ExprAlgebra<FieldT>( srcTagList_,algebraicOperation_,isModifierExpr_ );
}

//====================================================================
template class ExprAlgebra<SVolField>;
template class ExprAlgebra<XVolField>;
template class ExprAlgebra<YVolField>;
template class ExprAlgebra<ZVolField>;
template class ExprAlgebra<ParticleField>;
