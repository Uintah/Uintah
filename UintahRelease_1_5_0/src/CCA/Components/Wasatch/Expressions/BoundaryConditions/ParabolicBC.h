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
: public Expr::Expression<FieldT>
{ 
  ParabolicBC( const Expr::Tag& indepVarTag,
               const double a,
               const double b,
               const double c,
               const double cghost,
               const std::vector<int> flatGhostPoints,              
               const double cinterior,
               const std::vector<int> flatInteriorPoints) : 
  indepVarTag_ (indepVarTag),
  a_(a), b_(b), c_(c),
  cghost_(cghost),
  flatGhostPoints_ (flatGhostPoints),  
  cinterior_(cinterior),
  flatInteriorPoints_ ( flatInteriorPoints )
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
            const double cghost,
            const std::vector<int> flatGhostPoints,            
            const double cinterior,            
            const std::vector<int> flatInteriorPoints) : 
    ExpressionBuilder(resultTag), 
    indepVarTag_ (indepVarTag),
    a_(a), b_(b), c_(c),    
    cghost_(cghost),
    flatGhostPoints_ (flatGhostPoints),    
    cinterior_(cinterior),    
    flatInteriorPoints_ ( flatInteriorPoints )
    {}
    Expr::ExpressionBase* build() const{ return new ParabolicBC(indepVarTag_, a_, b_, c_, cghost_, flatGhostPoints_, cinterior_, flatInteriorPoints_); }
  private:
    const Expr::Tag indepVarTag_;
    const double a_, b_, c_;
    const double cghost_;
    const std::vector<int> flatGhostPoints_;    
    const double cinterior_;
    const std::vector<int> flatInteriorPoints_;  
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
  const double a_, b_, c_;  
  const double cghost_;
  const std::vector<int> flatGhostPoints_;  
  const double cinterior_;
  const std::vector<int> flatInteriorPoints_;  
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
  
  std::vector<int>::const_iterator ia = flatGhostPoints_.begin(); // ia is the ghost flat index
  std::vector<int>::const_iterator ib = flatInteriorPoints_.begin(); // ib is the interior flat index
  for( ; ia != flatGhostPoints_.end(); ++ia, ++ib ){
    f[*ia] = ( (a_ * (*x_)[*ia] * (*x_)[*ia] + b_ * (*x_)[*ia] + c_) - cinterior_*f[*ib] ) / cghost_;
  }
}

#endif // ParabolicBC_Expr_h