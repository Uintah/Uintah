/**
 *  \file   MonolithicRHS.cpp
 *
 *  \date   Apr 5, 2012
 *  \author James C. Sutherland
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

#include "MonolithicRHS.h"

#include <spatialops/structured/FVStaggered.h>

template< typename FieldT >
MonolithicRHS<FieldT>::
MonolithicRHS( const Expr::Tag& tcondTag,
                const Expr::Tag& tempTag )
  : Expr::Expression<FieldT>(),
    tcondTag_( tcondTag ),
    tempTag_( tempTag )
{
# ifdef ENABLE_CUDA
  this->set_gpu_runnable( true );
# endif
}

//--------------------------------------------------------------------

template< typename FieldT >
MonolithicRHS<FieldT>::
~MonolithicRHS()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MonolithicRHS<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( tcondTag_ );
  exprDeps.requires_expression( tempTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MonolithicRHS<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
  tcond_ = &fm.field_ref( tcondTag_ );
  temp_  = &fm.field_ref( tempTag_  );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MonolithicRHS<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpX_ = opDB.retrieve_operator<InterpX>();
  interpY_ = opDB.retrieve_operator<InterpY>();
  interpZ_ = opDB.retrieve_operator<InterpZ>();
  gradX_   = opDB.retrieve_operator<GradX  >();
  gradY_   = opDB.retrieve_operator<GradY  >();
  gradZ_   = opDB.retrieve_operator<GradZ  >();
  divX_    = opDB.retrieve_operator<DivX   >();
  divY_    = opDB.retrieve_operator<DivY   >();
  divZ_    = opDB.retrieve_operator<DivZ   >();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MonolithicRHS<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= (*divX_)( (*interpX_)(*tcond_) * (*gradX_)(*temp_) )
           + (*divY_)( (*interpY_)(*tcond_) * (*gradY_)(*temp_) )
           + (*divZ_)( (*interpZ_)(*tcond_) * (*gradZ_)(*temp_) );
}

//--------------------------------------------------------------------

template< typename FieldT >
MonolithicRHS<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& tcondTag,
                  const Expr::Tag& tempTag )
  : ExpressionBuilder( resultTag ),
    tcondTag_( tcondTag ),
    tempTag_( tempTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
MonolithicRHS<FieldT>::
Builder::build() const
{
  return new MonolithicRHS<FieldT>( tcondTag_,tempTag_ );
}

//--------------------------------------------------------------------



//--- Explicit template instantiations ---
template class MonolithicRHS<SpatialOps::SVolField>;
