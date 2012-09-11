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

#ifndef BasicBoundaryCondition_Expr_h
#define BasicBoundaryCondition_Expr_h

#include <expression/Expression.h>
/**
 *  \class 	BasicBoundaryCondition
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    September, 2012
 *
 *  \brief Provides an expression to set basic Dirichlet and Neumann boundary
 *  conditions. Given a BCValue, we set the ghost value such that
 *  f[ghost] = \alpha f[interior] + \beta BCValue
 *
 *  \tparam FieldT - the type of field for the RHS.
 *
 */

template< typename FieldT >
class BasicBoundaryCondition
: public Expr::Expression<FieldT>
{
  BasicBoundaryCondition( const double bcValue,
                         const double cghost,
                         const std::vector<int> flatGhostPoints,                         
                         const double cinterior,
                         const std::vector<int> flatInteriorPoints) : 
  bcValue_(bcValue),
  cghost_(cghost),
  flatGhostPoints_ (flatGhostPoints),  
  cinterior_(cinterior),
  flatInteriorPoints_ ( flatInteriorPoints )
  {}
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     * @param result Tag of the resulting expression.
     * @param bcValue   constant boundary condition value.
     * @param cghost ghost coefficient. This is usually provided by an operator.
     * @param flatGhostPoints  flat indices of the ghost points in which BC is being set.
     * @param cinterior interior coefficient. This is usually provided by an operator.
     * @param flatInteriorPoints  flat indices of the interior points that are used to set the ghost value.
     */    
    Builder( const Expr::Tag& resultTag, 
            const double bcValue,            
            const double cghost,
            const std::vector<int> flatGhostPoints,            
            const double cinterior,            
            const std::vector<int> flatInteriorPoints) : 
    ExpressionBuilder(resultTag), 
    bcValue_(bcValue),
    cghost_(cghost),
    flatGhostPoints_ (flatGhostPoints),    
    cinterior_(cinterior),    
    flatInteriorPoints_ ( flatInteriorPoints )
    {}
    Expr::ExpressionBase* build() const{ return new BasicBoundaryCondition(bcValue_, cghost_, flatGhostPoints_, cinterior_, flatInteriorPoints_); }
  private:
    const double bcValue_;
    const double cghost_;
    const std::vector<int> flatGhostPoints_;    
    const double cinterior_;
    const std::vector<int> flatInteriorPoints_;  
  };
  
  ~BasicBoundaryCondition(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){}
  void bind_fields( const Expr::FieldManagerList& fml ){}
  void evaluate();
private:
  const double bcValue_;
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

template< typename FieldT >
void
BasicBoundaryCondition<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  
  std::vector<int>::const_iterator ig = flatGhostPoints_.begin();    // ig is the ghost flat index
  std::vector<int>::const_iterator ii = flatInteriorPoints_.begin(); // ii is the interior flat index
  for( ; ig != flatGhostPoints_.end(); ++ig, ++ii ){
    f[*ig] = ( bcValue_- cinterior_ * f[*ii] ) / cghost_;    
  }
}

#endif // BasicBoundaryCondition_Expr_h