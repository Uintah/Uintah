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

#ifndef BCCopier_Expr_h
#define BCCopier_Expr_h

#include <expression/Expression.h>

template< typename FieldT >
class BCCopier
: public BoundaryConditionBase<FieldT>
{
  BCCopier( const Expr::Tag& srcTag ) :
  srcTag_ (srcTag)
  {}
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& srcTag) :
    ExpressionBuilder(resultTag),
    srcTag_ (srcTag)
    {}
    Expr::ExpressionBase* build() const{ return new BCCopier(srcTag_); }
  private:
    const Expr::Tag srcTag_;
  };
  
  ~BCCopier(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){  exprDeps.requires_expression( srcTag_ );}
  void bind_fields( const Expr::FieldManagerList& fml ){
    const typename Expr::FieldMgrSelector<FieldT>::type& phifm = fml.template field_manager<FieldT>();
    src_    = &phifm.field_ref( srcTag_    );
  }
  void evaluate();
private:
  const FieldT* src_;
  const Expr::Tag srcTag_;
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################


//--------------------------------------------------------------------

template< typename FieldT >
void
BCCopier<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  std::vector<int>::const_iterator ia = this->flatGhostPoints_.begin(); // ia is the ghost flat index
  std::vector<int>::const_iterator ib = this->flatInteriorPoints_.begin(); // ib is the interior flat index
  for( ; ia != this->flatGhostPoints_.end(); ++ia, ++ib )
    f[*ia] = (*src_)[*ia];
}

#endif // BCCopier_Expr_h
