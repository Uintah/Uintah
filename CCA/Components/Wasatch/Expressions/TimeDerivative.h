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

#ifndef TimeDerivativeExpr_h
#define TimeDerivativeExpr_h

#include <expression/Expression.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename ValT >
class TimeDerivative : public Expr::Expression<ValT>
{
public:
  
  /**
   *  \class   TimeDerivative
   *  \author  Tony Saad
   *  \date    March, 2013
   *  \ingroup Expressions
   *
   *  \brief Calculates the time derivative of any field.
   *
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
            const Expr::Tag& newVarTag,
            const Expr::Tag& oldVarTag,
            const Expr::Tag& timestepTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag newvartag_;
    const Expr::Tag oldvartag_;
    const Expr::Tag timesteptag_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  TimeDerivative( const Expr::Tag& newVarTag,
                 const Expr::Tag& oldVarTag,
                 const Expr::Tag& timestepTag );
  const Expr::Tag newvartag_;
  const Expr::Tag oldvartag_;
  const Expr::Tag timesteptag_;
  const ValT* newvar_;
  const ValT* oldvar_;
  const double* dt_;
};

//====================================================================

//--------------------------------------------------------------------

template<typename ValT>
TimeDerivative<ValT>::
TimeDerivative( const Expr::Tag& newVarTag,
               const Expr::Tag& oldVarTag,
               const Expr::Tag& timestepTag )
: Expr::Expression<ValT>(),
newvartag_  ( newVarTag ),
oldvartag_  ( oldVarTag ),
timesteptag_( timestepTag )
{}

//--------------------------------------------------------------------

template< typename ValT >
void
TimeDerivative<ValT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( newvartag_ );
  exprDeps.requires_expression( oldvartag_ );  
  exprDeps.requires_expression( timesteptag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
TimeDerivative<ValT>::
bind_fields( const Expr::FieldManagerList& fml )
{  
  const typename Expr::FieldMgrSelector<ValT>::type& valtfm = fml.template field_manager<ValT>();
  newvar_ = &valtfm.field_ref( newvartag_ );
  oldvar_ = &valtfm.field_ref( oldvartag_ );
  dt_     = &fml.template field_manager<double>().field_ref( timesteptag_ );
}

//--------------------------------------------------------------------

template< typename ValT >
void
TimeDerivative<ValT>::
evaluate()
{
  using namespace SpatialOps;
  ValT& phi = this->value();
  phi <<= (*newvar_ - *oldvar_)/ *dt_;
}

//--------------------------------------------------------------------

template< typename ValT >
TimeDerivative<ValT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& newVarTag,
        const Expr::Tag& oldVarTag,
        const Expr::Tag& timestepTag )
: ExpressionBuilder(result),
newvartag_  ( newVarTag ),
oldvartag_  ( oldVarTag ),
timesteptag_( timestepTag )
{}

//--------------------------------------------------------------------

template< typename ValT >
Expr::ExpressionBase*
TimeDerivative<ValT>::Builder::build() const
{
  return new TimeDerivative<ValT>( newvartag_, oldvartag_, timesteptag_ );
}

//--------------------------------------------------------------------

#endif // TimeDerivativeExpr_h
