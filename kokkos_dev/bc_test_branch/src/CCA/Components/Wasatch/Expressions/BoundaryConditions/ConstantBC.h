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

#ifndef ConstantBC_Expr_h
#define ConstantBC_Expr_h
#include "BoundaryConditionBase.h"
#include <expression/Expression.h>
/**
 *  \class 	ConstantBC
 *  \ingroup 	Expressions
 *  \author 	Tony Saad
 *  \date    September, 2012
 *
 *  \brief Provides an expression to set basic Dirichlet and Neumann boundary
 *  conditions. Given a BCValue, we set the ghost value such that
 *  \f$ f[ghost] = \alpha f[interior] + \beta BCValue \f$
 *
 *  \tparam FieldT - the type of field for the RHS.
 */

template< typename FieldT >
class ConstantBC
: public BoundaryConditionBase<FieldT>
{
public:
  ConstantBC( const double bcValue) :
  bcValue_(bcValue)
  {}

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
            const double bcValue ) :
    ExpressionBuilder(resultTag),
    bcValue_(bcValue)
    {}
    Expr::ExpressionBase* build() const{ return new ConstantBC(bcValue_); }
  private:
    const double bcValue_;
  };
  
  ~ConstantBC(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps ){}
  void bind_fields( const Expr::FieldManagerList& fml ){}
  void evaluate();
  
private:
  const double bcValue_;
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
void
ConstantBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  std::vector<int>::const_iterator ig = (this->flatGhostPoints_).begin();    // ig is the ghost flat index
  std::vector<int>::const_iterator ii = (this->flatInteriorPoints_).begin(); // ii is the interior flat index
//  if (this->isStaggered_) {
//    std::cout << "--------------------------------------------------- \n";
//    std::cout << "setting bc on " << this->name().name() << std::endl;
//  }
//  for( ; ig != (this->flatGhostPoints_).end(); ++ig, ++ii ){
//    f[*ig] = ( bcValue_- ci * f[*ii] ) / cg;
//    if (this->isStaggered_) {
//      std::cout << "ci = " << ci << " cg = " << cg  << " fi = " << f[*ii] << std::endl;
//      std::cout <<" bcvalue = " << bcValue_  <<" actual value = " << f[*ig] << std::endl;
//    }
//  }
  if(this->isStaggered_) {
    for( ; ig != (this->flatGhostPoints_).end(); ++ig, ++ii ){
      f[*ig] = bcValue_;
    }
  } else {
    for( ; ig != (this->flatGhostPoints_).end(); ++ig, ++ii ){
      f[*ig] = ( bcValue_- ci * f[*ii] ) / cg;
    }
  }
}

#endif // ConstantBC_Expr_h
