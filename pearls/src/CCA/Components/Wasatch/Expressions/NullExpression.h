/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

template< typename FieldT >
class NullExpression
 : public Expr::Expression<FieldT>
{
  const Expr::TagList VarNameTags_;

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

  ~NullExpression();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
NullExpression<FieldT>::
NullExpression( const Expr::TagList& VarNameTags )
  : Expr::Expression<FieldT>(),
    VarNameTags_( VarNameTags )
{}

//--------------------------------------------------------------------

template< typename FieldT >
NullExpression<FieldT>::
~NullExpression()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
NullExpression<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( VarNameTags_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
NullExpression<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
NullExpression<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
NullExpression<FieldT>::
evaluate()
{}

//--------------------------------------------------------------------

template< typename FieldT >
NullExpression<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::TagList& VarNameTags )
  : ExpressionBuilder( resultTag ),
    VarNameTags_( VarNameTags )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
NullExpression<FieldT>::
Builder::build() const
{
  return new NullExpression<FieldT>( VarNameTags_ );
}


#endif // NullExpression_Expr_h
