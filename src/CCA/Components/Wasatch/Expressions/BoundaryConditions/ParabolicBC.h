/*
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

#ifndef ParabolicBC_Expr_h
#define ParabolicBC_Expr_h

#include <expression/Expression.h>

template< typename FieldT >
class ParabolicBC
: public BoundaryConditionBase<FieldT>
{ 
  ParabolicBC( const Expr::Tag& indepVarTag,
               const double a,
               const double b,
               const double c,
               const double x0) :
  indepVarTag_ (indepVarTag),
  a_(a), b_(b), c_(c), x0_(x0)
  {}
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag, 
            const Expr::Tag& indepVarTag,   
            const double a,
            const double b,
            const double c,
            const double x0) :
    ExpressionBuilder(resultTag), 
    indepVarTag_ (indepVarTag),
    a_(a), b_(b), c_(c), x0_(x0)
    {}
    Expr::ExpressionBase* build() const{ return new ParabolicBC(indepVarTag_, a_, b_, c_, x0_); }
  private:
    const Expr::Tag indepVarTag_;
    const double a_, b_, c_, x0_;
  };
  
  ~ParabolicBC(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){  exprDeps.requires_expression( indepVarTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){
    const typename Expr::FieldMgrSelector<FieldT>::type& phifm = fml.template field_manager<FieldT>();
    x_    = &phifm.field_ref( indepVarTag_    );    
  }
  void evaluate();
private:
  const FieldT* x_;
  const Expr::Tag indepVarTag_;
  const double a_, b_, c_, x0_;
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################


//--------------------------------------------------------------------

template< typename FieldT >
void
ParabolicBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  std::vector<int>::const_iterator ia = this->flatGhostPoints_.begin(); // ia is the ghost flat index
  std::vector<int>::const_iterator ib = this->flatInteriorPoints_.begin(); // ib is the interior flat index
  double x = 0.0;
  for( ; ia != this->flatGhostPoints_.end(); ++ia, ++ib ){
    x = (*x_)[*ia] - x0_;
    f[*ia] = ( (a_ * x*x + b_ * x + c_) - ci*f[*ib] ) / cg;
  }
}

#endif // ParabolicBC_Expr_h