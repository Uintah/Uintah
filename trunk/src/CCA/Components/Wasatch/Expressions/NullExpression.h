/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#ifndef NullExpression_Expr_h
#define NullExpression_Expr_h

#include <expression/Expression.h>

/**
 *  \class     NullExpression
 *  \ingroup   Expressions
 *  \author 	 Tony Saad
 *  \date 	   January, 2014
 *
 *  \brief An expression that does "nothing" but acts as a computes/modifies in the taskgraph. This
 is needed for example when there is a modifies (but no computes) downstream that will populate this variable.
 See the RMCRT benchmark interface for a use-case on this.
 */

template< typename SrcT, typename TargetT >
class NullExpression
 : public Expr::Expression<TargetT>
{
  DECLARE_VECTOR_OF_FIELDS( SrcT, f_ )
  /* declare operators associated with this expression here */

    NullExpression( const Expr::TagList& VarNameTags );
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a NullExpression expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::TagList& VarNameTags );

    Expr::ExpressionBase* build() const;

  private:
    const Expr::TagList VarNameTags_;
  };

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename SrcT, typename TargetT >
NullExpression<SrcT, TargetT>::
NullExpression( const Expr::TagList& VarNameTags )
  : Expr::Expression<TargetT>()
{
  this->set_gpu_runnable(true);
  this->template create_field_vector_request<SrcT>(VarNameTags, f_);
}

//--------------------------------------------------------------------

template< typename SrcT, typename TargetT >
void
NullExpression<SrcT, TargetT>::
evaluate()
{}

//--------------------------------------------------------------------

template< typename SrcT, typename TargetT >
NullExpression<SrcT, TargetT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::TagList& VarNameTags )
  : ExpressionBuilder( resultTag ),
    VarNameTags_( VarNameTags )
{}

//--------------------------------------------------------------------

template< typename SrcT, typename TargetT >
Expr::ExpressionBase*
NullExpression<SrcT, TargetT>::
Builder::build() const
{
  return new NullExpression<SrcT, TargetT>( VarNameTags_ );
}


#endif // NullExpression_Expr_h
